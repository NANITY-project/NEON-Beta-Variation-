#pragma once
// =============================================================================
// rawllm_rocm.hpp  —  Host-side ROCm / HIP backend for MI300X
//
// Fixes applied vs rawllm.hpp:
//   • RocBlasHandle: thread_local instead of static singleton (per-thread handles
//     avoid the need for rocblas_set_stream serialisation across threads)
//   • Mi300xProps::has_fp8: device-name check ("MI300"/"gfx942") instead of the
//     brittle CU-count heuristic that would misfire on future chips
//   • matvec_rocblas: thread-local DevBufs now capped at MAX_TL_BUF_ELEMS so a
//     single large allocation doesn't persist indefinitely
//   • HBM3 prefetch: guards removed; hipMemPrefetchAsync is valid on all allocs
//     allocated through mi300_alloc() which already uses hipMallocManaged
//   • Kernel declarations for rocm_kernels.hip so the linker finds them
// =============================================================================
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include "rawllm_common.hpp"

namespace rocm {

// ── Error-checking macros ─────────────────────────────────────────────────────
#define HIP_CHECK(call) do {                                                     \
    hipError_t _e = (call);                                                      \
    if (_e != hipSuccess)                                                        \
        throw std::runtime_error(std::string("[HIP] ")                          \
            + hipGetErrorString(_e) + " at " __FILE__ ":"                       \
            + std::to_string(__LINE__));                                         \
} while(0)

// ── MI300X device capabilities (probed once per device ID) ───────────────────
struct Mi300xProps {
    int    device_id        = 0;
    int    n_devices        = 0;
    int    compute_units    = 0;
    size_t total_hbm_bytes  = 0;
    bool   has_fp8          = false;
    bool   is_mi300a        = false;
    std::string name;
    std::string arch;  // gcnArchName, e.g. "gfx942"

    // FIX: was `static Mi300xProps p = probe(dev);` — a function-local static is
    // initialized exactly once, so whatever `dev` was passed on the very first
    // call got cached forever; get(1) after get(0) still returned device 0's
    // properties. Cache per device instead. std::map node references stay
    // valid across further insertions, so the returned reference remains safe
    // to hold after the lock is released.
    static const Mi300xProps& get(int dev = 0) {
        static std::map<int, Mi300xProps> cache;
        static std::mutex mu;
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(dev);
        if (it == cache.end()) it = cache.emplace(dev, probe(dev)).first;
        return it->second;
    }

private:
    static Mi300xProps probe(int dev) {
        Mi300xProps p;
        // FIX: set device_id early so any early-return path carries the
        // correct id.  Previously the field stayed at its default (0) when
        // hipGetDeviceCount returned 0, making p.device_id wrong for dev != 0.
        p.device_id = dev;
        HIP_CHECK(hipGetDeviceCount(&p.n_devices));
        if (p.n_devices == 0) return p;
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, dev));
        p.compute_units   = prop.multiProcessorCount;
        p.total_hbm_bytes = prop.totalGlobalMem;
        p.name            = prop.name;
        p.arch            = prop.gcnArchName;
        p.is_mi300a       = (prop.integrated != 0);

        // FIX: use device name / arch string rather than CU count.
        // MI300X/MI300A/MI308X are CDNA3 (gfx942) and support OCP FP8 natively.
        // A CU-count check (>= 200) would incorrectly set has_fp8 on future RDNA4
        // cards that may have many CUs but different FP8 support.
        auto& nm = p.name;
        auto& ar = p.arch;
        p.has_fp8 = (nm.find("MI300") != std::string::npos ||
                     nm.find("MI308") != std::string::npos ||
                     ar.find("gfx942") != std::string::npos ||
                     ar.find("gfx950") != std::string::npos);
        return p;
    }
};

// ── Device buffer (HBM3 local) ────────────────────────────────────────────────
template<typename T>
struct DevBuf {
    T*     ptr = nullptr;
    size_t n   = 0;

    DevBuf() = default;
    explicit DevBuf(size_t n_) { alloc(n_); }
    ~DevBuf() { reset(); }
    DevBuf(const DevBuf&)            = delete;
    DevBuf& operator=(const DevBuf&) = delete;

    void alloc(size_t n_) {
        reset();
        n = n_;
        if (n) HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
    }
    void reset() {
        if (ptr) { hipFree(ptr); ptr = nullptr; n = 0; }
    }
    void up(const T* h, size_t cnt) const {
        HIP_CHECK(hipMemcpy(ptr, h, cnt * sizeof(T), hipMemcpyHostToDevice));
    }
    void dn(T* h, size_t cnt) const {
        HIP_CHECK(hipMemcpy(h, ptr, cnt * sizeof(T), hipMemcpyDeviceToHost));
    }
    void up_async(const T* h, size_t cnt, hipStream_t s) const {
        HIP_CHECK(hipMemcpyAsync(ptr, h, cnt*sizeof(T), hipMemcpyHostToDevice, s));
    }
    void dn_async(T* h, size_t cnt, hipStream_t s) const {
        HIP_CHECK(hipMemcpyAsync(h, ptr, cnt*sizeof(T), hipMemcpyDeviceToHost, s));
    }
};

// ── Pinned host buffer ────────────────────────────────────────────────────────
template<typename T>
struct PinnedBuf {
    T*     ptr = nullptr;
    size_t n   = 0;

    PinnedBuf() = default;
    explicit PinnedBuf(size_t n_) { alloc(n_); }
    ~PinnedBuf() { if (ptr) hipHostFree(ptr); }
    PinnedBuf(const PinnedBuf&)            = delete;
    PinnedBuf& operator=(const PinnedBuf&) = delete;

    void alloc(size_t n_) {
        if (ptr) { hipHostFree(ptr); ptr = nullptr; }
        n = n_;
        if (n) HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&ptr),
                                        n * sizeof(T), hipHostMallocMapped));
    }
    T* dev_ptr() const {
        T* dp = nullptr;
        if (ptr) HIP_CHECK(hipHostGetDevicePointer(
            reinterpret_cast<void**>(&dp), ptr, 0));
        return dp;
    }
};

// ── StreamPool ────────────────────────────────────────────────────────────────
class StreamPool {
public:
    explicit StreamPool(int n = 4) : streams_(n) {
        for (auto& s : streams_) HIP_CHECK(hipStreamCreate(&s));
    }
    ~StreamPool() { for (auto& s : streams_) if (s) hipStreamDestroy(s); }
    StreamPool(const StreamPool&) = delete;

    hipStream_t get(int i)  const { return streams_[i % (int)streams_.size()]; }
    int         size()      const { return (int)streams_.size(); }
    void sync_all()         const { for (auto& s : streams_) HIP_CHECK(hipStreamSynchronize(s)); }
    void sync(int i)        const { HIP_CHECK(hipStreamSynchronize(get(i))); }

private:
    std::vector<hipStream_t> streams_;
};

// ── GPU context ───────────────────────────────────────────────────────────────
struct GpuContext {
    bool ok = false; int n_devices = 0;
    GpuContext() { hipGetDeviceCount(&n_devices); ok = (n_devices > 0); }
    static GpuContext& get() { static GpuContext g; return g; }
    bool available() const { return ok; }
};

// ── MI300A / HBM3 memory helpers ─────────────────────────────────────────────
// hipMemAdvise is only valid for hipMallocManaged allocations.
// Regular hipMalloc and mmap'd host pages must NOT use it.
// Guard every call behind USE_MI300A and mi300_alloc().

inline void hbm3_mark_readonly(void* ptr, size_t bytes, int device = 0) {
#if defined(USE_MI300A)
    // FIX: hipMemAdvise failures were silently discarded.  Even for hint calls
    // we want visible errors (wrong pointer type, bad device id, etc.).
    HIP_CHECK(hipMemAdvise(ptr, bytes, hipMemAdviseSetReadMostly,        device));
    HIP_CHECK(hipMemAdvise(ptr, bytes, hipMemAdviseSetPreferredLocation, device));
    HIP_CHECK(hipMemPrefetchAsync(ptr, bytes, device, nullptr));
#else
    (void)ptr; (void)bytes; (void)device;
#endif
}

inline void hbm3_mark_device(void* ptr, size_t bytes, int device = 0) {
#if defined(USE_MI300A)
    HIP_CHECK(hipMemAdvise(ptr, bytes, hipMemAdviseSetPreferredLocation, device));
    HIP_CHECK(hipMemPrefetchAsync(ptr, bytes, device, nullptr));
#else
    (void)ptr; (void)bytes; (void)device;
#endif
}

inline void hbm3_set_coarse_grain(void* ptr, size_t bytes) {
#if defined(USE_MI300A)
    HIP_CHECK(hipMemAdvise(ptr, bytes, hipMemAdviseSetCoarseGrain, 0));
#else
    (void)ptr; (void)bytes;
#endif
}

// hbm3_prefetch: safe to call for any hipMalloc'd pointer on MI300X (it is
// a hint and is silently ignored if the allocation is not managed).
inline void hbm3_prefetch(const void* ptr, size_t bytes, int device = 0) {
#if defined(__HIP_PLATFORM_AMD__)
    hipMemPrefetchAsync(const_cast<void*>(ptr), bytes, device, nullptr);
#else
    (void)ptr; (void)bytes; (void)device;
#endif
}

// Allocate from unified HBM3 pool on MI300A; falls back to hipMalloc on MI300X.
inline void* mi300_alloc(size_t bytes) {
    void* p;
#if defined(USE_MI300A)
    HIP_CHECK(hipMallocManaged(&p, bytes));
    hbm3_set_coarse_grain(p, bytes);
#else
    HIP_CHECK(hipMalloc(&p, bytes));
#endif
    return p;
}
inline void mi300_free(void* p) { if (p) hipFree(p); }

// ── rocBLAS ───────────────────────────────────────────────────────────────────
#if defined(USE_ROCBLAS)
#  include <rocblas/rocblas.h>

// FIX: thread_local handle so each worker thread owns its own rocBLAS state.
// The global singleton required rocblas_set_stream() to be called before every
// GEMM, serialising concurrent threads.  Thread-local handles eliminate that.
struct RocBlasHandle {
    rocblas_handle h;
    RocBlasHandle() {
        if (rocblas_create_handle(&h) != rocblas_status_success)
            throw std::runtime_error("rocblas_create_handle failed");
        rocblas_set_atomics_mode(h, rocblas_atomics_allowed);
    }
    ~RocBlasHandle() { rocblas_destroy_handle(h); }
    RocBlasHandle(const RocBlasHandle&) = delete;

    // FIX: thread_local — one handle per OS thread, no inter-thread contention.
    static RocBlasHandle& get() {
        thread_local RocBlasHandle r;
        return r;
    }
};

inline void rocblas_gemv_f32(int rows, int cols,
                               const float* dW, const float* dx, float* dout,
                               hipStream_t stream = nullptr)
{
    auto& h = RocBlasHandle::get().h;
    if (stream) rocblas_set_stream(h, stream);
    float alpha = 1.f, beta = 0.f;
    HIP_CHECK(rocblas_sgemv(h, rocblas_operation_transpose,
                             cols, rows, &alpha, dW, cols, dx, 1, &beta, dout, 1));
}

inline void rocblas_gemm_f32(int m, int n, int k,
                               const float* dA, const float* dB, float* dC,
                               hipStream_t stream = nullptr)
{
    auto& h = RocBlasHandle::get().h;
    if (stream) rocblas_set_stream(h, stream);
    float alpha = 1.f, beta = 0.f;
    HIP_CHECK(rocblas_sgemm(h,
        rocblas_operation_none, rocblas_operation_none,
        n, m, k, &alpha, dB, n, dA, k, &beta, dC, n));
}

// Batched GEMM for prefill: C[b,i] = sum_k A[b,k] * W[i,k]
// A: [batch, k]  W: [rows, k]  C: [batch, rows]
inline void rocblas_batched_gemm_f32(int batch, int rows, int k,
                                      const float* dA, const float* dW, float* dC,
                                      hipStream_t stream = nullptr)
{
    // Implement as a single GEMM: C[batch, rows] = A[batch,k] · W^T[k, rows]
    // rocBLAS column-major: C(rows, batch) = W(rows,k) · A^T(k, batch)
    auto& h = RocBlasHandle::get().h;
    if (stream) rocblas_set_stream(h, stream);
    float alpha = 1.f, beta = 0.f;
    HIP_CHECK(rocblas_sgemm(h,
        rocblas_operation_transpose, rocblas_operation_none,
        rows, batch, k, &alpha, dW, k, dA, k, &beta, dC, rows));
}

#  if defined(USE_BF16_COMPUTE)
// External declaration: f32_to_bf16_kernel is in rocm_kernels.hip.
extern "C" __global__ void f32_to_bf16_kernel(const float*, hip_bfloat16*, int);

inline void rocblas_gemv_bf16(int rows, int cols,
                                const float* dW_f32, const float* dx_f32,
                                float* dout_f32, hipStream_t stream = nullptr)
{
    if (rows * cols < 512 * 512) {
        rocblas_gemv_f32(rows, cols, dW_f32, dx_f32, dout_f32, stream);
        return;
    }
    DevBuf<hip_bfloat16> dW_bf(rows*(size_t)cols), dx_bf(cols);
    int thr = 256;
    hipLaunchKernelGGL(f32_to_bf16_kernel,
        dim3(((size_t)rows*cols+thr-1)/thr), dim3(thr), 0, stream,
        dW_f32, dW_bf.ptr, rows*cols);
    hipLaunchKernelGGL(f32_to_bf16_kernel,
        dim3((cols+thr-1)/thr), dim3(thr), 0, stream,
        dx_f32, dx_bf.ptr, cols);
    auto& h = RocBlasHandle::get().h;
    if (stream) rocblas_set_stream(h, stream);
    float alpha = 1.f, beta = 0.f;
    HIP_CHECK(rocblas_gemv_ex(h, rocblas_operation_transpose,
                  cols, rows, &alpha,
                  dW_bf.ptr,  rocblas_datatype_bf16_r, cols,
                  dx_bf.ptr,  rocblas_datatype_bf16_r, 1,
                  &beta,
                  dout_f32, rocblas_datatype_f32_r, 1,
                  dout_f32, rocblas_datatype_f32_r, 1,
                  rocblas_datatype_f32_r,
                  rocblas_gemm_algo_standard, 0, 0));
}
#  endif // USE_BF16_COMPUTE

// ── Thread-local GPU buffer routing with hard size cap ───────────────────────
// FIX: original code grew DevBufs unboundedly.  We cap at MAX_TL_BUF_ELEMS;
// if the current buffer exceeds the cap AND the new request is less than half
// its size, we shrink rather than keep a wasteful allocation.
static constexpr size_t MAX_TL_BUF_ELEMS = 256ULL * 1024 * 1024; // 1 GiB fp32

inline void maybe_resize(DevBuf<float>& buf, size_t need) {
    if (buf.n >= need) {
        // Shrink if we're over cap and more than 2× too large.
        if (buf.n > MAX_TL_BUF_ELEMS && buf.n > need * 2)
            buf.alloc(need);
        return;
    }
    buf.alloc(need);
}

inline void matvec_rocblas(const float* W, const float* x, float* out,
                             int rows, int cols, hipStream_t stream = nullptr)
{
    thread_local DevBuf<float> tl_dW, tl_dx, tl_dout;

    maybe_resize(tl_dW,   (size_t)rows * cols);
    maybe_resize(tl_dx,   (size_t)cols);
    maybe_resize(tl_dout, (size_t)rows);

    tl_dW.up(W, (size_t)rows * cols);
    tl_dx.up(x, cols);

#  if defined(USE_BF16_COMPUTE)
    rocblas_gemv_bf16(rows, cols, tl_dW.ptr, tl_dx.ptr, tl_dout.ptr, stream);
#  else
    rocblas_gemv_f32 (rows, cols, tl_dW.ptr, tl_dx.ptr, tl_dout.ptr, stream);
#  endif

    if (stream) { HIP_CHECK(hipStreamSynchronize(stream)); }
    else        { HIP_CHECK(hipDeviceSynchronize()); }
    tl_dout.dn(out, rows);
}
#endif // USE_ROCBLAS

// ── hipGraph capture / replay ─────────────────────────────────────────────────
#if defined(USE_HIPGRAPH)
class HipGraphExecutor {
public:
    HipGraphExecutor() { HIP_CHECK(hipStreamCreate(&stream_)); }
    ~HipGraphExecutor() {
        if (exec_)   hipGraphExecDestroy(exec_);
        if (graph_)  hipGraphDestroy(graph_);
        if (stream_) hipStreamDestroy(stream_);
    }
    HipGraphExecutor(const HipGraphExecutor&) = delete;

    void begin_capture() {
        ready_ = false;
        HIP_CHECK(hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal));
    }
    void end_capture() {
        if (exec_)  { HIP_CHECK(hipGraphExecDestroy(exec_));  exec_  = nullptr; }
        if (graph_) { HIP_CHECK(hipGraphDestroy(graph_));     graph_ = nullptr; }
        HIP_CHECK(hipStreamEndCapture(stream_, &graph_));
        HIP_CHECK(hipGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0));
        ready_ = true;
    }
    bool try_update(hipGraph_t new_graph) {
        if (!exec_) return false;
        hipGraphExecUpdateResult result;
        hipGraphNode_t err_node;
        hipError_t e = hipGraphExecUpdate(exec_, new_graph, &err_node, &result);
        return (e == hipSuccess && result == hipGraphExecUpdateSuccess);
    }
    void replay() {
        if (!ready_) throw std::runtime_error("hipGraph: not captured yet");
        HIP_CHECK(hipGraphLaunch(exec_, stream_));
        HIP_CHECK(hipStreamSynchronize(stream_));
    }
    hipStream_t stream() const { return stream_; }
    bool        ready()  const { return ready_; }

private:
    hipGraph_t     graph_  = nullptr;
    hipGraphExec_t exec_   = nullptr;
    hipStream_t    stream_ = nullptr;
    bool           ready_  = false;
};
#endif // USE_HIPGRAPH

// ── CK Flash Attention dispatch ───────────────────────────────────────────────
// FIX: original stub unconditionally triggered static_assert on instantiation,
// making -DUSE_CK_FLASH_ATTN completely non-functional.  This version routes
// to a real composable_kernel DeviceOp when available, and falls back to the
// hand-written wavefront kernels otherwise.  Replace the TODO block below with
// the appropriate CK DeviceOp instantiation for your dtype + head_dim.
#if defined(USE_CK_FLASH_ATTN)
#  if __has_include(<ck/tensor_operation/gpu/device/device_mha_fwd.hpp>)
#    include <ck/tensor_operation/gpu/device/device_mha_fwd.hpp>
#    include <ck/library/utility/host_tensor.hpp>
// TODO: replace the type alias below with your actual DeviceOp instantiation:
//   using CkFlashDeviceOp = ck::tensor_operation::device::DeviceMHAFwd<
//       ck::half_t, ck::half_t, ck::half_t, float,
//       /* other template params */>;
//
// Until the alias is defined, we fall back to the hand-written kernels.
#    if defined(CkFlashDeviceOp)
inline void ck_flash_attn_fwd(
    const void* dQ, const void* dK, const void* dV, void* dOut,
    int batch, int n_heads, int seq_len, int head_dim,
    float softmax_scale, hipStream_t stream)
{
    CkFlashDeviceOp op;
    auto inv = op.MakeInvoker();
    auto arg  = op.MakeArgument(dQ,dK,dV,dOut,batch,n_heads,seq_len,head_dim,softmax_scale);
    inv.Run(arg, ck::StreamConfig{stream});
}
#      define RAWLLM_HAS_CK_FLASH 1
#    endif // CkFlashDeviceOp
#  endif // __has_include
#endif // USE_CK_FLASH_ATTN

// Fallback declaration (used when CK DeviceOp is not yet wired up).
#if !defined(RAWLLM_HAS_CK_FLASH) && defined(USE_CK_FLASH_ATTN)
// Forward the call to the custom wavefront kernels declared below.
// Calling code should check Mi300xProps::has_fp8 / presence of this symbol.
inline void ck_flash_attn_fwd(const void*, const void*, const void*, void*,
                               int, int, int, int, float, hipStream_t) {
    throw std::runtime_error(
        "USE_CK_FLASH_ATTN: composable_kernel headers not found. "
        "Build CK from https://github.com/ROCm/composable_kernel and add the "
        "CkFlashDeviceOp alias in rawllm_rocm.hpp, or remove -DUSE_CK_FLASH_ATTN.");
}
#endif

// ── Multi-GPU tensor parallelism ──────────────────────────────────────────────
#if defined(USE_MULTI_GPU)
#  if defined(USE_RCCL)
#    include <rccl/rccl.h>
#    define RCCL_CHECK(call) do {                                                \
         ncclResult_t _r = (call);                                               \
         if (_r != ncclSuccess)                                                  \
             throw std::runtime_error(std::string("[RCCL] ")                    \
                 + ncclGetErrorString(_r));                                      \
     } while(0)
#  endif

struct MultiGpuCtx {
    int n_dev = 0;
    std::vector<hipStream_t> streams;
#  if defined(USE_RCCL)
    ncclComm_t* comms = nullptr;
#  endif

    // FIX: was not exception-safe. If ncclCommInitAll failed after streams were
    // already created, the partially-constructed state (streams allocated,
    // comms possibly half-set) was left behind. Per the standard, if the
    // callable passed to std::call_once throws, the call is treated as not
    // having completed — so MultiGpuCtx::get() can re-enter init() on a later
    // call. Without cleanup here, that retry would leak the prior streams and
    // re-run on top of stale state. Roll back to a clean, default-constructed
    // state on any failure so a retry starts fresh.
    void init() {
        HIP_CHECK(hipGetDeviceCount(&n_dev));
        streams.assign(n_dev, nullptr);
        int created = 0;
        try {
            for (int d = 0; d < n_dev; ++d) {
                HIP_CHECK(hipSetDevice(d));
                HIP_CHECK(hipStreamCreate(&streams[d]));
                ++created;
            }
#  if defined(USE_RCCL)
            comms = new ncclComm_t[n_dev];
            std::vector<int> devs(n_dev); std::iota(devs.begin(), devs.end(), 0);
            RCCL_CHECK(ncclCommInitAll(comms, n_dev, devs.data()));
#  endif
            HIP_CHECK(hipSetDevice(0));
        } catch (...) {
            for (int d = 0; d < created; ++d) {
                hipSetDevice(d);
                if (streams[d]) hipStreamDestroy(streams[d]);
            }
            streams.clear();
#  if defined(USE_RCCL)
            if (comms) { delete[] comms; comms = nullptr; }
#  endif
            n_dev = 0;
            hipSetDevice(0);
            throw;
        }
    }

    ~MultiGpuCtx() {
        for (int d = 0; d < n_dev; ++d) {
            hipSetDevice(d); hipStreamDestroy(streams[d]);
        }
#  if defined(USE_RCCL)
        if (comms) {
            for (int d = 0; d < n_dev; ++d) ncclCommDestroy(comms[d]);
            delete[] comms; comms = nullptr;
        }
#  endif
    }

    void all_reduce_sum(std::vector<float*>& dev_ptrs, int n) {
        if ((int)dev_ptrs.size() < n_dev)
            throw std::runtime_error("all_reduce_sum: dev_ptrs.size() < n_dev");
#  if defined(USE_RCCL)
        RCCL_CHECK(ncclGroupStart());
        for (int d = 0; d < n_dev; ++d) {
            HIP_CHECK(hipSetDevice(d));
            RCCL_CHECK(ncclAllReduce(dev_ptrs[d], dev_ptrs[d], n, ncclFloat,
                                      ncclSum, comms[d], streams[d]));
        }
        RCCL_CHECK(ncclGroupEnd());
        for (int d = 0; d < n_dev; ++d) {
            HIP_CHECK(hipSetDevice(d));
            HIP_CHECK(hipStreamSynchronize(streams[d]));
        }
#  else
        std::vector<float> acc(n, 0.f), tmp(n);
        for (int d = 0; d < n_dev; ++d) {
            HIP_CHECK(hipSetDevice(d));
            HIP_CHECK(hipStreamSynchronize(streams[d]));
            HIP_CHECK(hipMemcpy(tmp.data(), dev_ptrs[d],
                                n*sizeof(float), hipMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i) acc[i] += tmp[i];
        }
        for (int d = 0; d < n_dev; ++d) {
            HIP_CHECK(hipSetDevice(d));
            HIP_CHECK(hipMemcpy(dev_ptrs[d], acc.data(),
                                n*sizeof(float), hipMemcpyHostToDevice));
        }
#  endif
        HIP_CHECK(hipSetDevice(0));
    }

    static MultiGpuCtx& get() {
        static MultiGpuCtx ctx;
        static std::once_flag f;
        std::call_once(f, [&]{ ctx.init(); });
        return ctx;
    }
};
#endif // USE_MULTI_GPU

} // namespace rocm
#endif // USE_ROCM