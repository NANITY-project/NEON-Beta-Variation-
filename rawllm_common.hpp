#pragma once
// =============================================================================
// rawllm_common.hpp  —  Shared types, Config, and compile-flag documentation
//
// All other rawllm_*.hpp headers include this file.
// Compile flags (pass on hipcc command line):
//   -DUSE_ROCM              HIP/ROCm GPU backend
//   -DUSE_ROCBLAS           rocBLAS SGEMM / GEMV
//   -DUSE_BF16_COMPUTE      BF16 HGEMM (2× FP32 throughput on MI300X)
//   -DUSE_HIPGRAPH          hipGraph capture/replay + ExecUpdate
//   -DUSE_MULTI_GPU         Tensor-parallel across MI300X XCDs
//   -DUSE_RCCL              RCCL all-reduce over Infinity Fabric
//   -DUSE_CK_FLASH_ATTN     composable_kernel flash attention
//   -DUSE_FP8_KV            FP8 e4m3 KV cache (MI300X native)
//   -DUSE_MI300A            MI300A unified HBM3 (hipMallocManaged)
//   -DUSE_NUMA              NUMA-aware thread + memory binding
//   -DUSE_AVX512            AVX-512 on host
// =============================================================================

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ── SIMD ──────────────────────────────────────────────────────────────────────
#if defined(__AVX512F__) && defined(USE_AVX512)
#  include <immintrin.h>
#  define RAWLLM_AVX512 1
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define RAWLLM_AVX2 1
#endif

// ── POSIX ─────────────────────────────────────────────────────────────────────
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
// -- BUG FIX
#ifndef RAWLLM_ALWAYS_INLINE
#  ifdef _MSC_VER
#    define RAWLLM_ALWAYS_INLINE __forceinline
#  else
#    define RAWLLM_ALWAYS_INLINE __attribute__((always_inline)) inline
#  endif
#endif

// ── NUMA (optional) ───────────────────────────────────────────────────────────
#ifdef USE_NUMA
#  include <numa.h>
#  include <numaif.h>
#endif

namespace engine {

// FIX: fast_expf — replaces std::exp in hot host paths (sampling softmax).
// Uses the standard bit-manipulation trick: 2^x = exp(x·ln2), approximated via
// FP32 exponent field.  Max relative error ≈ 0.07 %.  Full std::expf is used
// for edge cases (|x| > 88).
RAWLLM_ALWAYS_INLINE static float fast_expf(float x) noexcept {
#if defined(__HIP_DEVICE_COMPILE__)
    return __expf(x);
#else
    if (__builtin_expect(x < -88.f || x > 88.f, 0)) return std::exp(x);
    // Schraudolph approximation:  2^(x/ln2) via integer add on FP32 bits.
    static constexpr float kA = 12102203.9f;   // (1 << 23) / ln2
    static constexpr float kB = 1064872507.f;  // 127 * (1 << 23) - 0.5 * kA * ln2  (bias)
    union { float f; int32_t i; } u;
    u.i = (int32_t)(kA * x + kB);
    return u.f;
#endif
}

// ── NANITY architecture identity ────────────────────────────────────────────
// NANITY is a fixed, published transformer architecture (see
// NANITY_ARCHITECTURE_SPEC.md). This engine no longer tries to guess what
// architecture a GGUF file contains — it loads exactly one, and these two
// constants are the single source of truth for what "conforming" means.
// A file is a NANITY model iff general.architecture == kArchName and
// nanity.spec_version == kSpecVersion; rawllm_loader.hpp's validate_config()
// checks both before trusting anything else in the file.
constexpr const char* kArchName    = "nanity";
constexpr uint32_t    kSpecVersion = 1;

struct Config {
    uint32_t n_vocab          = 0;
    uint32_t n_embd           = 0;
    uint32_t n_layer          = 0;
    uint32_t n_head           = 0;
    uint32_t n_kv_head        = 0;
    uint32_t head_dim         = 0;
    uint32_t n_ff             = 0;
    uint32_t ctx_len          = 4096;
    uint32_t kv_window        = 0;
    uint32_t prefill_chunk_size = 512;  // 0 = no chunking; >0 = tokens per chunk
    uint32_t max_seq_slots    = 8;      // concurrent-sequence pool size
    float    rope_base        = 10000.f;
    float    rope_scale       = 1.f;
    uint32_t rope_dim_count   = 0;     // 0 means full head_dim rotation
    float    norm_eps         = 1e-5f;
    int      n_threads        = 4;
    int      n_gpu_layers     = 0;
    int      tensor_par       = 1;
    bool     numa_threads     = false;

    // FFN variant flag — set by validate_config() after scanning the tensor
    // manifest; never set by callers directly.
    //
    //   true  (default) → SwiGLU: ff_act = silu(ffn_gate(x)) * ffn_up(x)
    //                     Requires blk.{i}.ffn_gate.weight in every layer.
    //
    //   false           → plain SiLU-FFN: ff_act = silu(ffn_up(x))
    //                     ffn_gate.weight is absent from the file; models
    //                     like Nanity X1 (32-layer, n_embd=3072) export this
    //                     variant. The down-projection is otherwise identical.
    bool     use_swiglu       = true;
};

} // namespace engine

