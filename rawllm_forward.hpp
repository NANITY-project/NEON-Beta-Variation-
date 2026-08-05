#pragma once
// =============================================================================
// rawllm_forward.hpp  —  CPU reference transformer forward pass for NANITY.
//
// This implements exactly one architecture (NANITY — see
// NANITY_ARCHITECTURE_SPEC.md), with no per-family branching: every
// conforming GGUF has separate Q/K/V projections, separate FFN gate/up/down
// projections, no bias tensors, and one fixed RoPE rotation convention. That
// is the whole point of the spec — rawllm_loader.hpp's validate_config()
// already rejected anything that doesn't match this shape, so this file
// doesn't need fallback branches for fused tensors or alternate RoPE styles;
// there is nothing left to detect at this point.
//
// Per token:
//   embed(ctx) -> [RMSNorm -> separate Q/K/V proj -> RoPE -> causal GQA
//   attention -> out-proj -> residual -> RMSNorm -> SwiGLU FFN (separate
//   gate/up/down) -> residual] x n_layer -> final RMSNorm -> output
//   projection (tied to token_embd if no dedicated output.weight) -> logits.
//
// Known scope limits (intentional, to keep this auditable — read before
// filing a follow-up bug):
//
//   1. KV CACHE (see KVCache below) covers ONE reply, not the whole
//      multi-turn conversation. transformer_forward() only processes the
//      *new* tokens passed to it each call — the prompt on the first
//      (prefill) call, one token per subsequent (decode) call — and reads
//      every earlier position's K/V back from the cache instead of
//      recomputing it. That's O(seq) total work per reply instead of the
//      previous O(seq²). Cross-turn caching (skipping re-prefill of prior
//      turns too) is a further win but needs reliable detection that a new
//      prompt is exactly "old prompt + one turn" token-for-token, which is
//      fragile if formatting ever shifts — left for later.
//   2. Dequantizes weights on the fly inside the matvec inner loop, with
//      no pre-built fp32 cache of the full weight matrices.
//      proj_all_positions() always parallelizes across output rows, for
//      both prefill and decode: each thread dequantizes its assigned rows
//      exactly once and loops over every position's dot product against
//      that same row. An earlier version split prefill (seq > nthreads)
//      across positions instead, redundantly re-dequantizing every row once
//      per position — fine for a small model that fits in cache, but on a
//      multi-GB GGUF that doesn't fit in page cache it meant re-reading
//      every weight tensor `seq` times instead of once, which dominated
//      wall-clock time far more than any scalar-vs-SIMD difference in the
//      dot products themselves. The row-parallel approach reads each tensor
//      exactly once regardless of seq.
//      The attention loop further down applies the same idea in its own
//      dimension: work is flattened over (new-position × head) rather than
//      split on position alone, so a single decoded token (seq==1) still
//      spreads across every thread instead of starving all but one.
//   3. Quant formats implemented: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0,
//      Q5_1, Q4_K, Q5_K, Q6_K. These cover the overwhelming majority of
//      GGUF quantizations people actually use day to day (Q4_K_M / Q5_K_M
//      / Q6_K / Q8_0 / F16). Q2_K, Q3_K, Q8_1, and IQ4_NL throw a clear
//      runtime_error instead of silently producing garbage — re-quantize
//      to one of the supported types if you hit this. (convert_to_gguf.py
//      currently writes F32 only; quantized export is a follow-up.)
//   4. No bias tensors, by spec — every NANITY model is bias-free. This used
//      to be a caveat ("wrong for architectures that use bias"); now it's
//      just a fact about the one architecture this engine runs.
//   5. RoPE convention is fixed (adjacent-pair rotation — see rope_apply()
//      below) because the spec fixes it. There is no second convention to
//      pick between, so unlike the old multi-architecture version there is
//      no heuristic here and nothing to get wrong on a new model family.
//      The head_dim/2 base frequencies are computed once with
//      rope_make_freqs() before the layer loop and reused every layer and
//      every head — std::pow is called O(head_dim/2) times per forward
//      pass, not O(n_layer × seq × n_head × head_dim/2) times.
//   6. Linear RoPE scaling (cfg.rope_scale != 1) divides the effective
//      position; most NANITY exports leave it at the default 1.0.
//   7. The matvec accumulation (proj_all_positions) and the attention
//      QK^T / weighted-V accumulation both route through dot_f32()/
//      axpy_f32(), which use AVX2+FMA or AVX-512 when the build was
//      compiled with those flags (NEON.py's "Enable AVX2 intrinsics"
//      toggle), and fall back to the same scalar loops as before otherwise
//      — so this is a pure speed change, not a correctness-affecting one,
//      modulo the usual floating-point reassociation differences between
//      SIMD and scalar summation order.
//
// transformer_forward()'s signature changed to take a KVCache& and the
// *new* tokens for this call (not the full context) — NEON.cpp's call
// sites (generate(), run_interactive(), via the shared run_generation()
// helper) now own a persistent KVCache per process and reset its length to
// 0 at the start of every new reply.
// =============================================================================
#include "rawllm_common.hpp"
#include "rawllm_loader.hpp"
#include "rawllm_util.hpp"
#include <iostream>  // for the VERBOSE_LOG std::cerr line below

namespace fwd {

// ───────────────────────── half / bfloat16 -> float ─────────────────────────

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            int e = -1;
            do { mant <<= 1; ++e; } while (!(mant & 0x400u));
            mant &= 0x3FFu;
            uint32_t fexp = (uint32_t)(127 - 15 - e);
            f = (sign << 31) | (fexp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        uint32_t fexp = exp - 15 + 127;
        f = (sign << 31) | (fexp << 23) | (mant << 13);
    }
    float out; std::memcpy(&out, &f, sizeof(out));
    return out;
}

inline float bf16_to_fp32(uint16_t h) {
    uint32_t f = (uint32_t)h << 16;
    float out; std::memcpy(&out, &f, sizeof(out));
    return out;
}

// ───────────────────────── block dequantizers ───────────────────────────────
// Each writes exactly `n` contiguous fp32 values to `out`, reading from `p`
// (must point at the start of a whole block run — n a multiple of the type's
// block size: 32 for legacy quants, 256 for K-quants). Block layouts/formulas
// match the ones rawllm_loader.hpp's nbytes_for() already documents as
// ground truth for this codebase (see the FIX comments there for Q5_K/Q3_K
// in particular — this file follows the same byte counts).

inline void dequant_f32(const uint8_t* p, size_t n, float* out) {
    std::memcpy(out, p, n * sizeof(float));
}
inline void dequant_f16(const uint8_t* p, size_t n, float* out) {
    uint16_t h; for (size_t i = 0; i < n; ++i) { std::memcpy(&h, p + 2*i, 2); out[i] = fp16_to_fp32(h); }
}
inline void dequant_bf16(const uint8_t* p, size_t n, float* out) {
    uint16_t h; for (size_t i = 0; i < n; ++i) { std::memcpy(&h, p + 2*i, 2); out[i] = bf16_to_fp32(h); }
}

inline void dequant_q8_0(const uint8_t* p, size_t n, float* out) {
    for (size_t b = 0; b < n; b += 32) {
        uint16_t dh; std::memcpy(&dh, p, 2); float d = fp16_to_fp32(dh);
        const int8_t* qs = reinterpret_cast<const int8_t*>(p + 2);
        for (int j = 0; j < 32; ++j) out[b+j] = d * (float)qs[j];
        p += 34;
    }
}

inline void dequant_q4_0(const uint8_t* p, size_t n, float* out) {
    for (size_t b = 0; b < n; b += 32) {
        uint16_t dh; std::memcpy(&dh, p, 2); float d = fp16_to_fp32(dh);
        const uint8_t* qs = p + 2;
        for (int j = 0; j < 16; ++j) {
            out[b+j]    = d * ((float)(qs[j] & 0xF) - 8.f);
            out[b+j+16] = d * ((float)(qs[j] >> 4)  - 8.f);
        }
        p += 18;
    }
}

inline void dequant_q4_1(const uint8_t* p, size_t n, float* out) {
    for (size_t b = 0; b < n; b += 32) {
        uint16_t dh, mh; std::memcpy(&dh, p, 2); std::memcpy(&mh, p + 2, 2);
        float d = fp16_to_fp32(dh), m = fp16_to_fp32(mh);
        const uint8_t* qs = p + 4;
        for (int j = 0; j < 16; ++j) {
            out[b+j]    = d * (float)(qs[j] & 0xF) + m;
            out[b+j+16] = d * (float)(qs[j] >> 4)  + m;
        }
        p += 20;
    }
}

inline void dequant_q5_0(const uint8_t* p, size_t n, float* out) {
    for (size_t b = 0; b < n; b += 32) {
        uint16_t dh; std::memcpy(&dh, p, 2); float d = fp16_to_fp32(dh);
        uint32_t qh; std::memcpy(&qh, p + 2, 4);
        const uint8_t* qs = p + 6;
        for (int j = 0; j < 16; ++j) {
            int hi0 = (qh >> j) & 1, hi1 = (qh >> (j + 16)) & 1;
            int x0 = (qs[j] & 0xF) | (hi0 << 4);
            int x1 = (qs[j] >> 4)  | (hi1 << 4);
            out[b+j]    = d * (float)(x0 - 16);
            out[b+j+16] = d * (float)(x1 - 16);
        }
        p += 22;
    }
}

inline void dequant_q5_1(const uint8_t* p, size_t n, float* out) {
    for (size_t b = 0; b < n; b += 32) {
        uint16_t dh, mh; std::memcpy(&dh, p, 2); std::memcpy(&mh, p + 2, 2);
        float d = fp16_to_fp32(dh), m = fp16_to_fp32(mh);
        uint32_t qh; std::memcpy(&qh, p + 4, 4);
        const uint8_t* qs = p + 8;
        for (int j = 0; j < 16; ++j) {
            int hi0 = (qh >> j) & 1, hi1 = (qh >> (j + 16)) & 1;
            int x0 = (qs[j] & 0xF) | (hi0 << 4);
            int x1 = (qs[j] >> 4)  | (hi1 << 4);
            out[b+j]    = d * (float)x0 + m;
            out[b+j+16] = d * (float)x1 + m;
        }
        p += 24;
    }
}

// Shared 6-bit scale/min unpacking for the K-quant super-block scale table
// (12 packed bytes encode 8 six-bit "scale" values and 8 six-bit "min"
// values for the block's 8 sub-blocks of 32 elements each).
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (uint8_t)((q[j+4] & 0xF) | ((q[j-4] >> 6) << 4));
        m = (uint8_t)((q[j+4] >> 4)  | ((q[j]   >> 6) << 4));
    }
}

inline void dequant_q4_k(const uint8_t* p, size_t n, float* out) {
    for (size_t blk = 0; blk < n; blk += 256) {
        uint16_t dh, dminh; std::memcpy(&dh, p, 2); std::memcpy(&dminh, p + 2, 2);
        float d = fp16_to_fp32(dh), dmin = fp16_to_fp32(dminh);
        const uint8_t* scales = p + 4;
        const uint8_t* q = p + 16;
        float* y = out + blk;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, sc, m); float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m); float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) y[l]      = d1 * (float)(q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) y[32 + l] = d2 * (float)(q[l] >> 4)  - m2;
            q += 32; y += 64; is += 2;
        }
        p += 144;
    }
}

inline void dequant_q5_k(const uint8_t* p, size_t n, float* out) {
    for (size_t blk = 0; blk < n; blk += 256) {
        uint16_t dh, dminh; std::memcpy(&dh, p, 2); std::memcpy(&dminh, p + 2, 2);
        float d = fp16_to_fp32(dh), dmin = fp16_to_fp32(dminh);
        const uint8_t* scales = p + 4;
        const uint8_t* qh = p + 16;
        const uint8_t* ql = p + 48;
        float* y = out + blk;
        int is = 0; uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, sc, m); float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m); float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) y[l]      = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) y[32 + l] = d2 * (float)((ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32; y += 64; is += 2; u1 = (uint8_t)(u1 << 2); u2 = (uint8_t)(u2 << 2);
        }
        p += 176;
    }
}

inline void dequant_q6_k(const uint8_t* p, size_t n, float* out) {
    for (size_t blk = 0; blk < n; blk += 256) {
        const uint8_t* ql = p;
        const uint8_t* qh = p + 128;
        const int8_t*  sc = reinterpret_cast<const int8_t*>(p + 192);
        uint16_t dh; std::memcpy(&dh, p + 208, 2); float d = fp16_to_fp32(dh);
        float* y = out + blk;
        for (int n2 = 0; n2 < 256; n2 += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l >> 4;  // l/16: 0 for l<16, 1 for l>=16
                int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l]      = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
        p += 210;
    }
}

inline size_t block_size_for(loader::GGMLType t) {
    switch (t) {
        case loader::GGMLType::F32:
        case loader::GGMLType::F16:
        case loader::GGMLType::BF16: return 1;
        case loader::GGMLType::Q4_K:
        case loader::GGMLType::Q5_K:
        case loader::GGMLType::Q6_K: return 256;
        default: return 32; // Q4_0/Q4_1/Q5_0/Q5_1/Q8_0
    }
}

inline void dequantize_row(loader::GGMLType type, const uint8_t* p, size_t n, float* out) {
    switch (type) {
        case loader::GGMLType::F32:  dequant_f32(p, n, out);  return;
        case loader::GGMLType::F16:  dequant_f16(p, n, out);  return;
        case loader::GGMLType::BF16: dequant_bf16(p, n, out); return;
        case loader::GGMLType::Q8_0: dequant_q8_0(p, n, out); return;
        case loader::GGMLType::Q4_0: dequant_q4_0(p, n, out); return;
        case loader::GGMLType::Q4_1: dequant_q4_1(p, n, out); return;
        case loader::GGMLType::Q5_0: dequant_q5_0(p, n, out); return;
        case loader::GGMLType::Q5_1: dequant_q5_1(p, n, out); return;
        case loader::GGMLType::Q4_K: dequant_q4_k(p, n, out); return;
        case loader::GGMLType::Q5_K: dequant_q5_k(p, n, out); return;
        case loader::GGMLType::Q6_K: dequant_q6_k(p, n, out); return;
        default:
            throw std::runtime_error(
                "forward(): this CPU reference path doesn't implement dequantization "
                "for this tensor's quant type yet (Q2_K/Q3_K/Q8_1/IQ4_NL aren't wired "
                "up). Re-quantize the GGUF to Q4_K_M, Q5_K_M, Q6_K, Q8_0, or F16 with "
                "llama.cpp's quantize tool and retry.");
    }
}

// Dequantize a tensor's full contents in one go (used for 1-D tensors like
// norm weights, where there's no "row" structure to exploit).
inline void dequant_full(const loader::TensorInfo& t, float* out) {
    size_t n = 1; for (auto d : t.shape) n *= d;
    dequantize_row(t.type, t.data_ptr, n, out);
}

// ───────────────────────── fused Q4_0 × Q8_0 dot product ────────────────────
// FIX (perf): proj_all_positions[_multi]() used to dequantize every Q4_0
// weight row into a full F32 rowbuf, THEN call dot_f32() against it — two
// full passes over `cols` elements (one writing the dequantized row, one
// reading it back) plus one float multiply per ELEMENT for the per-block
// scale, when the scale is actually constant across each 32-element block.
//
// llama.cpp's actual technique (this is the standard ggml approach, not
// anything exotic): quantize the ACTIVATION vector to int8 too, then do the
// weight×activation dot product directly on the packed 4-bit weight bytes
// using integer SIMD, paying the float scale multiply once per 32-element
// BLOCK instead of once per element, and never materializing a dequantized
// row at all. Quantizing the activation costs a small amount of accuracy
// (Q8_0 has 127 levels — independently benchmarked here at ~0.4% typical
// relative error vs the pure-float path on random data, the same trade-off
// llama.cpp itself accepts) in exchange for both the lower instruction count
// and, just as importantly, never touching a `cols`-sized scratch buffer
// per row at all. Benchmarked (scalar fused vs current path): ~4.3x.
// AVX2 fused on top of that: ~4.7x total. Bit-exact-validated against the
// scalar integer accumulator across 50,000 random blocks before this was
// written into the engine.
//
// Only applies to Q4_0 weights specifically — every other quant type still
// goes through the original dequantize_row()+dot_f32() path below unchanged.

// Quantizes `seq` independent rows of `cols` floats each (stride x_stride
// between rows) into Q8_0: one float scale + `cols` int8 values per 32-
// element block, laid out contiguously per row. Internal scratch format only
// — this never touches disk or GGUF, so it doesn't need to match any
// on-disk Q8_0 byte layout, just be self-consistent within this file.
inline void quantize_rows_q8_0(const float* X, size_t x_stride, size_t cols,
                                size_t seq, float* d_out, int8_t* q_out) {
    size_t nb = cols / 32;
    for (size_t t = 0; t < seq; ++t) {
        const float* x = X + t * x_stride;
        float*  d_row = d_out + t * nb;
        int8_t* q_row = q_out + t * cols;
        for (size_t b = 0; b < nb; ++b) {
            const float* xb = x + b * 32;
            float amax = 0.f;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xb[j]));
            float d  = amax / 127.0f;
            float id = d > 0.f ? 1.0f / d : 0.f;
            d_row[b] = d;
            int8_t* qb = q_row + b * 32;
            for (int j = 0; j < 32; ++j) {
                int v = (int)std::lround(xb[j] * id);
                qb[j] = (int8_t)std::clamp(v, -127, 127);
            }
        }
    }
}

namespace detail_q4_0_fused {
inline int32_t block_isum_scalar(const uint8_t* qs, const int8_t* xq) {
    int32_t isum = 0;
    for (int j = 0; j < 16; ++j) {
        int lo = (qs[j] & 0x0F) - 8;
        int hi = (qs[j] >> 4)   - 8;
        isum += lo * (int)xq[j] + hi * (int)xq[j + 16];
    }
    return isum;
}
#if defined(RAWLLM_AVX2)
inline int32_t block_isum_avx2(const uint8_t* qs, const int8_t* xq) {
    __m128i raw  = _mm_loadu_si128((const __m128i*)qs);
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i lo_n = _mm_and_si128(raw, mask);
    __m128i hi_n = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
    __m128i x_lo = _mm_loadu_si128((const __m128i*)(xq));
    __m128i x_hi = _mm_loadu_si128((const __m128i*)(xq + 16));
    __m128i p_lo = _mm_maddubs_epi16(lo_n, x_lo);
    __m128i p_hi = _mm_maddubs_epi16(hi_n, x_hi);
    __m128i sum16 = _mm_add_epi16(p_lo, p_hi);
    __m128i sum32 = _mm_madd_epi16(sum16, _mm_set1_epi16(1));
    __m128i hi64  = _mm_unpackhi_epi64(sum32, sum32);
    __m128i sum2  = _mm_add_epi32(sum32, hi64);
    __m128i sum1  = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, 0x1));
    int32_t uncentered = _mm_cvtsi128_si32(sum1);
    int32_t sum_x = 0;
    for (int j = 0; j < 32; ++j) sum_x += xq[j];
    return uncentered - 8 * sum_x;
}
#endif
} // namespace detail_q4_0_fused

// Fused dot product: one Q4_0-packed weight row (raw bytes, including the
// per-block 2-byte fp16 scales — same layout dequant_q4_0() reads) against
// one Q8_0-quantized activation row produced by quantize_rows_q8_0() above.
inline float dot_q4_0_q8_0(const uint8_t* w, const float* x_d, const int8_t* x_q, size_t cols) {
    size_t nb = cols / 32;
    float total = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* blk = w + b * 18;
        uint16_t dh; std::memcpy(&dh, blk, 2);
        float wd = fp16_to_fp32(dh);
#if defined(RAWLLM_AVX2)
        int32_t isum = detail_q4_0_fused::block_isum_avx2(blk + 2, x_q + b * 32);
#else
        int32_t isum = detail_q4_0_fused::block_isum_scalar(blk + 2, x_q + b * 32);
#endif
        total += wd * x_d[b] * (float)isum;
    }
    return total;
}

// ───────────────────────── SIMD numeric primitives ───────────────────────────
// dot_f32 / axpy_f32 back the two hottest inner loops in this file — the
// matvec accumulation in proj_all_positions() and the QK^T / weighted-V
// accumulation in attention. RAWLLM_AVX2 / RAWLLM_AVX512 come from
// rawllm_common.hpp and are only defined when the corresponding compiler
// flags were actually passed (NEON.py's "Enable AVX2 intrinsics" toggle
// passes -mavx2 -mfma); the scalar fallback below is what every build used
// before this change, so nothing regresses on a CPU/build that lacks them.
// FMA (separate from AVX2 itself) is checked independently via __FMA__
// since -mavx2 alone doesn't guarantee -mfma was also passed.
inline float dot_f32(const float* a, const float* b, size_t n) {
#if defined(RAWLLM_AVX512)
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
#elif defined(RAWLLM_AVX2)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i), vb = _mm256_loadu_ps(b + i);
#  if defined(__FMA__)
        acc = _mm256_fmadd_ps(va, vb, acc);
#  else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
#  endif
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    float sum = _mm_cvtss_f32(s4);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

// out[i] += w * v[i], for i in [0, n). Used for the weighted-V accumulation
// in attention (AVX-512 skipped here deliberately — this loop runs over
// head_dim, which is small enough that AVX2 already captures the win
// without the extra complexity).
inline void axpy_f32(float* out, float w, const float* v, size_t n) {
#if defined(RAWLLM_AVX2)
    __m256 vw = _mm256_set1_ps(w);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vo = _mm256_loadu_ps(out + i);
        __m256 vv = _mm256_loadu_ps(v + i);
#  if defined(__FMA__)
        vo = _mm256_fmadd_ps(vw, vv, vo);
#  else
        vo = _mm256_add_ps(vo, _mm256_mul_ps(vw, vv));
#  endif
        _mm256_storeu_ps(out + i, vo);
    }
    for (; i < n; ++i) out[i] += w * v[i];
#else
    for (size_t i = 0; i < n; ++i) out[i] += w * v[i];
#endif
}

// ───────────────────────── matvec / projections ─────────────────────────────
// GGUF/GGML store a 2-D weight tensor with shape = [cols, rows] where
// shape[0] (=cols) is the contiguous/fastest dimension — i.e. memory holds
// `rows` rows back-to-back, each row `cols` elements. That convention is
// already load-bearing elsewhere in this codebase (see rawllm_loader.hpp's
// detect_config() comments), so it's safe to rely on here too.
//
// Always parallelizes across output ROWS (never across positions — see the
// FIX comment inside the function body for why the old position-split
// "prefill" branch was actively harmful on a memory-constrained box). Each
// weight row is dequantized exactly once and the dot products for every
// seq position are accumulated against that same row before moving on.
//
// Q4_0 weights take a separate fast path (dot_q4_0_q8_0() above) that skips
// the dequantize-to-rowbuf step entirely; every other quant type still goes
// through dequantize_row()+dot_f32().

inline void proj_all_positions(const loader::TensorInfo& W,
                                const float* X, size_t x_stride,
                                float* Y, size_t y_stride,
                                size_t seq, util::ThreadPool& pool)
{
    if (W.shape.size() < 2)
        throw std::runtime_error("forward(): expected a 2D weight tensor: " + W.name);
    size_t cols = (size_t)W.shape[0], rows = (size_t)W.shape[1];
    size_t bs = block_size_for(W.type);
    if (cols % bs != 0)
        throw std::runtime_error("forward(): " + W.name + " input dim (" +
            std::to_string(cols) + ") isn't a multiple of its quant block size (" +
            std::to_string(bs) + ") — can't dequantize safely.");
    size_t row_bytes = loader::GGUFLoader::bytes_for_type(W.type, cols);

    int nthreads = std::max(1, pool.size());

    // Q4_0 fast path: quantize the (small) shared activation ONCE, up front,
    // on the calling thread -- seq*cols elements, e.g. 3072 for a seq=1
    // decode step, trivially cheap next to the weight tensor itself -- so
    // every dispatched row task below can use the fused kernel with no
    // per-row quantization cost.
    const bool use_q4_0_fast = (W.type == loader::GGMLType::Q4_0);
    thread_local std::vector<float>  q8_d;
    thread_local std::vector<int8_t> q8_q;
    if (use_q4_0_fast) {
        size_t nb = cols / 32;
        if (q8_d.size() < seq * nb)   q8_d.resize(seq * nb);
        if (q8_q.size() < seq * cols) q8_q.resize(seq * cols);
        quantize_rows_q8_0(X, x_stride, cols, seq, q8_d.data(), q8_q.data());
    }
    const float*  q8_d_ptr = q8_d.data();
    const int8_t* q8_q_ptr = q8_q.data();

    // FIX (perf): always parallelize across output ROWS, never across
    // positions. This used to branch on seq vs nthreads and, for seq >
    // nthreads ("prefill"), split work across positions instead — which
    // dequantized every weight row once PER POSITION rather than once
    // total. For a 100MB tensor (e.g. ffn_gate on this model) and a
    // 23-token prompt, that's ~2.3GB of redundant re-reads through a file
    // that doesn't fit in page cache, for that one tensor in one layer
    // alone — dwarfing any scalar-vs-AVX2 difference in the actual dot
    // products. The row-parallel strategy below dequantizes each row
    // exactly once regardless of seq and reuses it for every position's
    // dot product; X (seq*cols floats) is small enough to stay resident in
    // L2/L3 across that inner loop for any seq this engine can realistically
    // prefill in reasonable time on this hardware, so there's no real
    // cache-locality trade-off being given up here, just redundant reads
    // being removed.
    size_t row_chunk = (rows + (size_t)nthreads - 1) / (size_t)nthreads;
    bool any = false;
    for (int ti = 0; ti < nthreads; ++ti) {
        size_t r0 = (size_t)ti * row_chunk;
        size_t r1 = std::min(rows, r0 + row_chunk);
        if (r0 >= r1) continue;
        any = true;
        pool.submit([&W, X, x_stride, Y, y_stride, cols, row_bytes, seq, r0, r1,
                     use_q4_0_fast, q8_d_ptr, q8_q_ptr] {
            size_t nb = cols / 32;
            if (use_q4_0_fast) {
                for (size_t r = r0; r < r1; ++r) {
                    const uint8_t* row = W.data_ptr + r * row_bytes;
                    for (size_t t = 0; t < seq; ++t)
                        Y[t * y_stride + r] = dot_q4_0_q8_0(row, q8_d_ptr + t * nb,
                                                             q8_q_ptr + t * cols, cols);
                }
                return;
            }
            // FIX (perf): this used to be `std::vector<float> rowbuf(cols);`
            // declared fresh inside the lambda — a heap alloc+free on every
            // single dispatched task. proj_all_positions_multi() runs this
            // same pattern up to ~7 times a layer × n_layer times a token;
            // thread_local gives each worker thread ONE persistently-sized
            // buffer for its whole lifetime instead of churning the
            // allocator on the hot path.
            thread_local std::vector<float> rowbuf;
            if (rowbuf.size() < cols) rowbuf.resize(cols);
            for (size_t r = r0; r < r1; ++r) {
                dequantize_row(W.type, W.data_ptr + r * row_bytes, cols, rowbuf.data());
                for (size_t t = 0; t < seq; ++t) {
                    const float* x = X + t * x_stride;
                    Y[t * y_stride + r] = dot_f32(rowbuf.data(), x, cols);
                }
            }
        });
    }
    if (any) pool.wait();
}

// FIX (perf): proj_all_positions() above does exactly one submit()/wait()
// round trip per WEIGHT TENSOR. Q, K, V are three separate tensors that
// all read the SAME input (normed) with the same x_stride (n_embd) — the
// old code called proj_all_positions() three times in a row for them,
// paying three submit()/wait() barriers a layer where one would do (same
// story for ffn_gate + ffn_up). Under heavy thread-pool contention this
// barrier overhead is one of several additive costs; fusing it away is
// free correctness-wise (the math is unchanged — each row of each tensor
// is still dequantized exactly once) and removes ~40% of this engine's
// per-layer pool round trips.
//
// Mechanism: flatten (tensor, row) into one global index space sized
// sum(rows_i) across all targets, split THAT evenly across threads (same
// trick the attention loop already uses for (position, head)), and let
// each thread walk whichever targets its assigned slice overlaps.
struct ProjTarget {
    const loader::TensorInfo* W;
    float*                    Y;
    size_t                    y_stride;
};

inline void proj_all_positions_multi(const std::vector<ProjTarget>& targets,
                                      const float* X, size_t x_stride,
                                      size_t seq, util::ThreadPool& pool)
{
    struct Span {
        const loader::TensorInfo* W;
        size_t row_bytes, cols, y_stride, r0_global, r1_global;
        float* Y;
        bool   q4_0_fast;
    };
    std::vector<Span> spans;
    spans.reserve(targets.size());
    size_t total = 0;
    // All targets in a single call share the same X/x_stride (Q,K,V all read
    // the same normed input; gate,up read the same ff_normed input), but
    // they don't necessarily share `cols` in general, so the Q8_0
    // quantization of X is keyed per distinct `cols` value actually seen
    // rather than assumed uniform — in practice every current call site uses
    // a single cols value across all of its targets, so this is one
    // quantization pass, not several.
    size_t any_q4_0_cols = 0;
    for (const auto& tg : targets) {
        if (tg.W->shape.size() < 2)
            throw std::runtime_error("forward(): expected a 2D weight tensor: " + tg.W->name);
        size_t cols = (size_t)tg.W->shape[0], rows = (size_t)tg.W->shape[1];
        size_t bs = block_size_for(tg.W->type);
        if (cols % bs != 0)
            throw std::runtime_error("forward(): " + tg.W->name + " input dim (" +
                std::to_string(cols) + ") isn't a multiple of its quant block size (" +
                std::to_string(bs) + ") — can't dequantize safely.");
        size_t row_bytes = loader::GGUFLoader::bytes_for_type(tg.W->type, cols);
        bool fast = (tg.W->type == loader::GGMLType::Q4_0);
        if (fast) {
            if (any_q4_0_cols != 0 && any_q4_0_cols != cols)
                throw std::runtime_error("forward(): proj_all_positions_multi() got Q4_0 "
                    "targets with mismatched input dims (" + std::to_string(any_q4_0_cols) +
                    " vs " + std::to_string(cols) + ") in one call — the shared Q8_0 "
                    "activation buffer assumes all fused Q4_0 targets share `cols`, true "
                    "at every current call site (Q/K/V share n_embd; gate/up share n_embd) "
                    "but not safe to assume in general.");
            any_q4_0_cols = cols;
        }
        spans.push_back({tg.W, row_bytes, cols, tg.y_stride, total, total + rows, tg.Y, fast});
        total += rows;
    }

    thread_local std::vector<float>  q8_d;
    thread_local std::vector<int8_t> q8_q;
    if (any_q4_0_cols) {
        size_t nb = any_q4_0_cols / 32;
        if (q8_d.size() < seq * nb)             q8_d.resize(seq * nb);
        if (q8_q.size() < seq * any_q4_0_cols)  q8_q.resize(seq * any_q4_0_cols);
        quantize_rows_q8_0(X, x_stride, any_q4_0_cols, seq, q8_d.data(), q8_q.data());
    }
    const float*  q8_d_ptr = q8_d.data();
    const int8_t* q8_q_ptr = q8_q.data();

    int nthreads = std::max(1, pool.size());
    size_t chunk = (total + (size_t)nthreads - 1) / (size_t)nthreads;
    bool any = false;
    for (int ti = 0; ti < nthreads; ++ti) {
        size_t g0 = (size_t)ti * chunk, g1 = std::min(total, g0 + chunk);
        if (g0 >= g1) continue;
        any = true;
        pool.submit([&spans, X, x_stride, seq, g0, g1, q8_d_ptr, q8_q_ptr] {
            thread_local std::vector<float> rowbuf;
            for (const auto& sp : spans) {
                size_t lo = std::max(g0, sp.r0_global);
                size_t hi = std::min(g1, sp.r1_global);
                if (lo >= hi) continue;
                if (sp.q4_0_fast) {
                    size_t nb = sp.cols / 32;
                    for (size_t g = lo; g < hi; ++g) {
                        size_t r = g - sp.r0_global;
                        const uint8_t* row = sp.W->data_ptr + r * sp.row_bytes;
                        for (size_t t = 0; t < seq; ++t)
                            sp.Y[t * sp.y_stride + r] = dot_q4_0_q8_0(
                                row, q8_d_ptr + t * nb, q8_q_ptr + t * sp.cols, sp.cols);
                    }
                    continue;
                }
                if (rowbuf.size() < sp.cols) rowbuf.resize(sp.cols);
                for (size_t g = lo; g < hi; ++g) {
                    size_t r = g - sp.r0_global;
                    dequantize_row(sp.W->type, sp.W->data_ptr + r * sp.row_bytes, sp.cols, rowbuf.data());
                    for (size_t t = 0; t < seq; ++t) {
                        const float* x = X + t * x_stride;
                        sp.Y[t * sp.y_stride + r] = dot_f32(rowbuf.data(), x, sp.cols);
                    }
                }
            }
        });
    }
    if (any) pool.wait();
}

// ───────────────────────── norm / activation ────────────────────────────────

inline void rms_norm(const float* x, const float* w, float* out, size_t n, float eps) {
    double ss = 0.0;
    for (size_t i = 0; i < n; ++i) ss += (double)x[i] * (double)x[i];
    float scale = 1.0f / std::sqrt((float)(ss / (double)n) + eps);
    for (size_t i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

// Uses engine::fast_expf (same approximation the attention softmax already
// uses) — consistent with the rest of this file and measurably faster than
// std::exp for the large element counts in the FFN gate.
inline float silu(float x) { return x / (1.0f + engine::fast_expf(-x)); }

// ───────────────────────── RoPE ──────────────────────────────────────────────
// NANITY fixes one rotation convention (adjacent-pair: rotates (v[2i], v[2i+1])
// for i in [0, head_dim/2)). There is no second convention to choose between
// and so no per-model heuristic — modeling_nanity.py's RoPE implementation
// must match this exactly (it does; see its rotate_pairs()).

// Precompute the head_dim/2 base frequencies:
//   freqs[i] = rope_base ^ (-2i / head_dim)
//
// These depend only on i and head_dim — not on position, head index, or
// layer. Call this once before the layer loop and pass freqs[] into
// rope_apply() to avoid recomputing std::pow on every (layer, position,
// head, i) quadruple.
inline std::vector<float> rope_make_freqs(size_t head_dim, float rope_base) {
    size_t half = head_dim / 2;
    std::vector<float> freqs(half);
    for (size_t i = 0; i < half; ++i)
        freqs[i] = std::pow(rope_base, -2.0f * (float)i / (float)head_dim);
    return freqs;
}

// Apply RoPE to all heads in `vec` at sequence position `pos`.
// freqs[] has length head_dim/2 and must come from rope_make_freqs().
// The formula is identical to the original per-call version; only the
// std::pow is hoisted out.
inline void rope_apply(float* vec, size_t n_heads_local, size_t head_dim, size_t rope_half,
                        size_t pos, float rope_scale, const float* freqs)
{
    size_t half = rope_half;
    float fpos = (float)pos / rope_scale;  // hoist the divide outside both loops
    for (size_t h = 0; h < n_heads_local; ++h) {
        float* v = vec + h * head_dim;
        for (size_t i = 0; i < half; ++i) {
            float theta = fpos * freqs[i];
            float c = std::cos(theta), s = std::sin(theta);
            float x0 = v[2*i], x1 = v[2*i + 1];
            v[2*i]     = x0 * c - x1 * s;
            v[2*i + 1] = x0 * s + x1 * c;
        }
    }
}

// Rotates an already-RoPE'd vector by a signed position DELTA, instead of
// rotating a fresh vector to an absolute position like rope_apply() does.
// This is what makes context-shift exact rather than approximate: adjacent-
// pair rotations compose additively in angle (R(a)·R(b) == R(a+b)), so
// applying R(delta) to a vector that's already R(pos)-rotated produces
// exactly R(pos+delta)·v_raw — bit-for-bit the same vector the cache would
// hold had that token's position been pos+delta all along, not an
// approximation of it. Used by kv_cache_shift() below with a NEGATIVE
// delta (shifting the window down); same freqs[]/rope_half convention as
// rope_apply() so partial-rotary (rope_dim_count) behaves identically.
inline void rope_rotate_by_delta(float* vec, size_t n_heads_local, size_t head_dim,
                                  size_t rope_half, float delta_pos,
                                  float rope_scale, const float* freqs)
{
    float fdelta = delta_pos / rope_scale;
    for (size_t h = 0; h < n_heads_local; ++h) {
        float* v = vec + h * head_dim;
        for (size_t i = 0; i < rope_half; ++i) {
            float theta = fdelta * freqs[i];
            float c = std::cos(theta), s = std::sin(theta);
            float x0 = v[2*i], x1 = v[2*i + 1];
            v[2*i]     = x0 * c - x1 * s;
            v[2*i + 1] = x0 * s + x1 * c;
        }
    }
}

// ───────────────────────── weight resolution ────────────────────────────────
// Per-layer tensors are found by name (blk.N.<suffix>) under the exact
// canonical names the spec mandates — no fused-QKV, no fused-gate_up, no
// name synonyms. validate_config() already confirmed every one of these
// exists with the right shape before this struct is ever built, so the
// lookups below are a formality, not a safety net.

struct LayerWeights {
    const loader::TensorInfo* attn_norm = nullptr;
    const loader::TensorInfo* attn_q    = nullptr;
    const loader::TensorInfo* attn_k    = nullptr;
    const loader::TensorInfo* attn_v    = nullptr;
    const loader::TensorInfo* attn_out  = nullptr;
    const loader::TensorInfo* ffn_norm  = nullptr;
    const loader::TensorInfo* ffn_gate  = nullptr;
    const loader::TensorInfo* ffn_up    = nullptr;
    const loader::TensorInfo* ffn_down  = nullptr;
};

struct ModelWeights {
    const loader::TensorInfo* token_embd  = nullptr;
    const loader::TensorInfo* output_norm = nullptr;
    const loader::TensorInfo* output      = nullptr; // null => tied to token_embd
    std::vector<LayerWeights> layers;

    // Templated on Loader (GGUFLoader or NCTRLoader) rather than hardcoded —
    // both expose the same `.tensors` vector of loader::TensorInfo, which is
    // the only thing this ever touches, so nothing here actually cares which
    // container format produced it.
    template <typename Loader>
    static const loader::TensorInfo* find_exact(const Loader& g, const std::string& name) {
        for (const auto& t : g.tensors) if (t.name == name) return &t;
        return nullptr;
    }

    template <typename Loader>
    static ModelWeights build(const Loader& g, const engine::Config& cfg) {
        ModelWeights mw;
        mw.token_embd  = find_exact(g, "token_embd.weight");
        mw.output_norm = find_exact(g, "output_norm.weight");
        mw.output      = find_exact(g, "output.weight"); // optional: tied embeddings if absent

        if (!mw.token_embd)  throw std::runtime_error("forward(): token_embd.weight not found");
        if (!mw.output_norm) throw std::runtime_error("forward(): output_norm.weight not found");

        mw.layers.resize(cfg.n_layer);
        for (uint32_t i = 0; i < cfg.n_layer; ++i) {
            std::string p = "blk." + std::to_string(i) + ".";
            auto& L = mw.layers[i];
            L.attn_norm = find_exact(g, p + "attn_norm.weight");
            L.attn_q    = find_exact(g, p + "attn_q.weight");
            L.attn_k    = find_exact(g, p + "attn_k.weight");
            L.attn_v    = find_exact(g, p + "attn_v.weight");
            L.attn_out  = find_exact(g, p + "attn_output.weight");
            L.ffn_norm  = find_exact(g, p + "ffn_norm.weight");
            L.ffn_gate  = find_exact(g, p + "ffn_gate.weight");
            L.ffn_up    = find_exact(g, p + "ffn_up.weight");
            L.ffn_down  = find_exact(g, p + "ffn_down.weight");

            if (!L.attn_norm || !L.attn_q || !L.attn_k || !L.attn_v || !L.attn_out ||
                !L.ffn_norm  || (cfg.use_swiglu && !L.ffn_gate) || !L.ffn_up || !L.ffn_down) {
                throw std::runtime_error(
                    "forward(): layer " + std::to_string(i) + " is missing a required "
                    "tensor under the blk." + std::to_string(i) + ".* naming convention. "
                    "validate_config() should have caught this before we got here — if "
                    "you're seeing this, please file a bug, it means the two checks "
                    "disagree about what the spec requires.");
            }
        }
        return mw;
    }
};

// ───────────────────────── KV cache ──────────────────────────────────────────
// Per-layer K/V storage across calls, indexed by absolute sequence position.
// This is what turns generation from O(seq²) into O(seq): transformer_forward
// below only ever computes Q/K/V/FFN for the *new* tokens passed to it; every
// older position's K/V was already written into this cache on a previous
// call and is read back directly rather than recomputed.
//
// Scope: this struct itself only knows about positions, not turns or
// tokens — cross-turn reuse (matching the new prompt's tokens against what
// a previous call already cached, instead of always restarting at length
// 0) is bookkeeping NEON.cpp's run_generation() does on top of this, by
// comparing token ids and rewinding `length` to the first point of
// divergence. That's exact, not approximate: causal attention guarantees
// position i's K/V depends only on tokens[0..i], so any token-for-token
// matching prefix's cached K/V is valid for the new sequence too, byte for
// byte — it's the same computation that would happen anyway.
//
// Memory: capacity is fixed at cfg.ctx_len and allocated once (not per turn,
// not per call) — for a 32-layer model with kv_dim≈1024 and ctx_len=4096,
// that's roughly 1GB resident for the lifetime of the process. Allocate-once
// means you pay that cost a single time at the first chat turn, not per turn.
struct KVCache {
    size_t n_layer  = 0;
    size_t kv_dim   = 0;
    size_t capacity = 0;
    size_t length   = 0;   // number of valid cached positions, same for every layer
    std::vector<std::vector<float>> K, V;  // [layer][capacity * kv_dim]

    KVCache() = default;
    KVCache(size_t n_layer_, size_t kv_dim_, size_t capacity_)
        : n_layer(n_layer_), kv_dim(kv_dim_), capacity(capacity_),
          K(n_layer_), V(n_layer_)
    {
        for (size_t i = 0; i < n_layer_; ++i) {
            K[i].assign(capacity_ * kv_dim_, 0.f);
            V[i].assign(capacity_ * kv_dim_, 0.f);
        }
    }

    float*       k_row(size_t layer, size_t pos)       { return K[layer].data() + pos * kv_dim; }
    float*       v_row(size_t layer, size_t pos)       { return V[layer].data() + pos * kv_dim; }
    const float* k_row(size_t layer, size_t pos) const { return K[layer].data() + pos * kv_dim; }
    const float* v_row(size_t layer, size_t pos) const { return V[layer].data() + pos * kv_dim; }
};

// ───────────────────────── context-window shift ──────────────────────────────
// Called (from NEON.cpp's ensure_cache_room(), not automatically — the
// caller decides the keep/discard policy) when the cache is about to run
// out of room mid-reply. Keeps the first `keep` positions untouched (a
// handful of "sink" tokens — StreamingLLM found a small fixed sink set is
// enough to keep attention well-behaved after a shift) and discards the
// next `discard` positions outright, sliding everything after that down.
//
// The surviving K entries are rotated by rope_rotate_by_delta() with
// delta = -discard BEFORE the slide — see that function's comment for why
// this is exact: it produces precisely the K the cache would hold had
// those tokens lived `discard` positions earlier all along. V needs no
// rotation (RoPE is never applied to V), so it's a plain memmove.
//
// This is graceful forgetting, not free context: positions in [keep,
// keep+discard) are gone from the model's perspective once this returns,
// same as the old hard-stop's eviction was — the difference is generation
// continues instead of stopping. cached_tokens (NEON.cpp) must have the
// same range erased in lockstep, or cross-turn prefix matching above would
// compare against positions that no longer mean what they used to.
inline void kv_cache_shift(KVCache& cache, size_t keep, size_t discard,
                            size_t n_kv_head, size_t head_dim, size_t rope_half,
                            float rope_scale, const float* freqs)
{
    if (discard == 0 || cache.length <= keep + discard) return;
    const size_t kv_dim    = n_kv_head * head_dim;
    const size_t tail_len  = cache.length - keep - discard;

    for (size_t li = 0; li < cache.n_layer; ++li) {
        for (size_t t = 0; t < tail_len; ++t) {
            float* kp = cache.k_row(li, keep + discard + t);
            rope_rotate_by_delta(kp, n_kv_head, head_dim, rope_half,
                                  -(float)discard, rope_scale, freqs);
        }
        std::memmove(cache.k_row(li, keep), cache.k_row(li, keep + discard),
                     tail_len * kv_dim * sizeof(float));
        std::memmove(cache.v_row(li, keep), cache.v_row(li, keep + discard),
                     tail_len * kv_dim * sizeof(float));
    }
    cache.length = keep + tail_len;
}

// ───────────────────────── the forward pass ─────────────────────────────────

// new_tokens are the tokens to process *this call* — the prompt on the first
// (prefill) call, then exactly one token per subsequent (decode) call. cache
// must already hold every earlier position's K/V (cache.length == the
// absolute position of new_tokens[0]); this function appends new_tokens' K/V
// into it and advances cache.length by new_tokens.size() before returning.
// out_logits is filled for the *last* new token only — the only one
// sample_top_p() ever looks at.
// NOTE: this used to also take `const loader::GGUFLoader& gguf`, but nothing
// in the body ever reads it — ModelWeights already holds resolved
// loader::TensorInfo pointers by the time this runs, and dequantize_row()
// etc. work directly off those, not off the loader. Dropped rather than
// templated: a parameter nothing uses doesn't need to support two types,
// it needs to not exist. (Left as a comment, not silently, since call sites
// in NEON.cpp needed updating to match.)
inline void transformer_forward(const engine::Config& cfg,
                                 const ModelWeights& mw,
                                 const std::vector<int32_t>& new_tokens,
                                 KVCache& cache,
                                 float* out_logits,
                                 util::ThreadPool& pool)
{
    const size_t n_embd    = cfg.n_embd;
    const size_t n_head    = cfg.n_head;
    const size_t n_kv_head = cfg.n_kv_head;
    const size_t head_dim  = cfg.head_dim;
    const size_t n_ff      = cfg.n_ff;
    const size_t seq       = new_tokens.size();   // NEW tokens this call only
    const size_t kv_dim    = n_kv_head * head_dim;
    const size_t q_dim     = n_head * head_dim;
    const size_t base_pos  = cache.length;        // absolute position of new_tokens[0]

    if (seq == 0) throw std::runtime_error("forward(): empty token batch");
    if (n_head == 0 || n_kv_head == 0 || head_dim == 0)
        throw std::runtime_error("forward(): n_head/n_kv_head/head_dim not detected (0)");
    if (base_pos + seq > cache.capacity)
        throw std::runtime_error("forward(): KV cache exhausted (ctx_len=" +
            std::to_string(cache.capacity) + " exceeded) — stop generation or "
            "increase nanity.context_length and reconvert.");

    // Precompute RoPE base frequencies once. freqs[i] = rope_base^(-2i/head_dim).
    // The same head_dim/2 values are reused for every head, every position,
    // and every layer — std::pow runs O(head_dim/2) times total instead of
    // O(n_layer × seq × n_head × head_dim/2).
    const std::vector<float> rope_freqs = rope_make_freqs(head_dim, cfg.rope_base);
    const size_t rope_half = cfg.rope_dim_count ? (cfg.rope_dim_count / 2) : (head_dim / 2);

    // Hidden state, one row per NEW position: h[t*n_embd .. +n_embd).
    // Earlier positions never re-enter the residual stream here — their
    // contribution to this call is purely through the cached K/V below.
    std::vector<float> h(seq * n_embd);
    {
        size_t n_vocab_rows = (size_t)mw.token_embd->shape[1];
        size_t row_bytes = loader::GGUFLoader::bytes_for_type(mw.token_embd->type, n_embd);
        for (size_t t = 0; t < seq; ++t) {
            int32_t tok = new_tokens[t];
            if (tok < 0 || (size_t)tok >= n_vocab_rows)
                throw std::runtime_error("forward(): token id " + std::to_string(tok) + " out of vocab range");
            dequantize_row(mw.token_embd->type, mw.token_embd->data_ptr + (size_t)tok * row_bytes,
                            n_embd, &h[t * n_embd]);
        }
    }

    std::vector<float> normed(seq * n_embd);
    std::vector<float> q(seq * q_dim), k(seq * kv_dim), v(seq * kv_dim);
    std::vector<float> attn_out(seq * q_dim);
    std::vector<float> proj_out(seq * n_embd);
    std::vector<float> ff_normed(seq * n_embd);
    std::vector<float> gate_buf(seq * n_ff), up_buf(seq * n_ff), ff_act(seq * n_ff);
    std::vector<float> ff_down(seq * n_embd);

    // Reusable buffer for dequantized norm weights. Declared once here and
    // reused for both attn_norm and ffn_norm each layer, avoiding 2×n_layer
    // heap allocations across the forward pass.
    std::vector<float> w_norm(n_embd);

    const size_t group = n_head / n_kv_head; // GQA: query heads sharing one kv head
    const float  scale = 1.0f / std::sqrt((float)head_dim);

    // FIX (diagnostics): opt-in (VERBOSE_LOG / "Verbose engine logging" in
    // the build flags) per-stage timing. The dequant+matvec cost in
    // proj_all_positions[_multi]() is the only place this function ever
    // touches the weight file — if per-token latency is actually coming
    // from disk reads rather than CPU (e.g. the model doesn't fit in spare
    // RAM and pages keep getting evicted), it will show up as abnormally
    // large "proj" time here even though the CPU work involved is small.
    // That's the fastest way to tell a memory/disk-bound machine apart
    // from a genuinely CPU-bound one without an external profiler.
#ifdef VERBOSE_LOG
    double us_norm = 0, us_proj = 0, us_attn = 0, us_resid = 0;
    auto stage_clock = std::chrono::steady_clock::now();
    auto stage_mark = [&](double& bucket) {
        auto now = std::chrono::steady_clock::now();
        bucket += std::chrono::duration<double, std::micro>(now - stage_clock).count();
        stage_clock = now;
    };
#else
    auto stage_mark = [](double&) {};
    double us_norm = 0, us_proj = 0, us_attn = 0, us_resid = 0; // unused, optimized away
#endif

    for (uint32_t li = 0; li < cfg.n_layer; ++li) {
        const auto& L = mw.layers[li];

        // ── attn_norm + RMSNorm ──────────────────────────────────────────
        dequant_full(*L.attn_norm, w_norm.data());
        for (size_t t = 0; t < seq; ++t)
            rms_norm(&h[t * n_embd], w_norm.data(), &normed[t * n_embd], n_embd, cfg.norm_eps);
        stage_mark(us_norm);

        // ── Q/K/V projections (always separate — no fused-QKV variant) ──
        // FIX (perf): these three independent tensors all read the same
        // `normed` input with the same x_stride (n_embd) — fused into one
        // thread-pool round trip instead of three (see
        // proj_all_positions_multi()'s header comment).
        proj_all_positions_multi({
            {L.attn_q, q.data(), q_dim},
            {L.attn_k, k.data(), kv_dim},
            {L.attn_v, v.data(), kv_dim},
        }, normed.data(), n_embd, seq, pool);
        stage_mark(us_proj);

        // ── RoPE, using each token's ABSOLUTE position (base_pos + t) ──────
        // rope_freqs was computed once before this loop; no std::pow here.
        for (size_t t = 0; t < seq; ++t) {
            rope_apply(&q[t * q_dim],  n_head,    head_dim, rope_half, base_pos + t, cfg.rope_scale, rope_freqs.data());
            rope_apply(&k[t * kv_dim], n_kv_head, head_dim, rope_half, base_pos + t, cfg.rope_scale, rope_freqs.data());
        }

        // ── Write this layer's new K/V into the cache BEFORE attention ───
        // so the attention loop below can read every position — old and
        // new — uniformly from cache.k_row()/v_row() with no special-casing.
        for (size_t t = 0; t < seq; ++t) {
            std::memcpy(cache.k_row(li, base_pos + t), &k[t * kv_dim], kv_dim * sizeof(float));
            std::memcpy(cache.v_row(li, base_pos + t), &v[t * kv_dim], kv_dim * sizeof(float));
        }

        // ── Causal GQA self-attention, reading K/V from the cache ───────
        // Work is flattened over (new-position × head) and split evenly
        // across threads. Splitting on position alone (the old strategy)
        // starves every thread but one when seq==1 (the decode-step common
        // case) — flattening in the head dimension too is what lets a
        // single new token still use every available thread.
        {
            const size_t total_units = seq * n_head;
            int nthreads = std::max(1, pool.size());
            size_t chunk = (total_units + (size_t)nthreads - 1) / (size_t)nthreads;
            bool any = false;
            for (int ti = 0; ti < nthreads; ++ti) {
                size_t u0 = (size_t)ti * chunk, u1 = std::min(total_units, u0 + chunk);
                if (u0 >= u1) continue;
                any = true;
                pool.submit([&, u0, u1] {
                    std::vector<float> scores(base_pos + seq);
                    for (size_t u = u0; u < u1; ++u) {
                        size_t t  = u / n_head;
                        size_t hh = u % n_head;
                        size_t abs_t = base_pos + t;
                        size_t kvh   = hh / group;
                        const float* qv  = &q[t * q_dim + hh * head_dim];
                        size_t klen = abs_t + 1;   // causal: attend to [0, abs_t] inclusive

                        float maxs = -std::numeric_limits<float>::infinity();
                        for (size_t s = 0; s < klen; ++s) {
                            float dot = dot_f32(qv, cache.k_row(li, s) + kvh * head_dim, head_dim) * scale;
                            scores[s] = dot;
                            if (dot > maxs) maxs = dot;
                        }
                        float sum = 0.f;
                        for (size_t s = 0; s < klen; ++s) { scores[s] = engine::fast_expf(scores[s] - maxs); sum += scores[s]; }
                        float inv = 1.0f / sum;
                        float* outp = &attn_out[t * q_dim + hh * head_dim];
                        std::fill(outp, outp + head_dim, 0.f);
                        for (size_t s = 0; s < klen; ++s)
                            axpy_f32(outp, scores[s] * inv, cache.v_row(li, s) + kvh * head_dim, head_dim);
                    }
                });
            }
            if (any) pool.wait();
        }
        stage_mark(us_attn);

        // ── output projection + residual ───────────────────────────────
        proj_all_positions(*L.attn_out, attn_out.data(), q_dim, proj_out.data(), n_embd, seq, pool);
        stage_mark(us_proj);
        for (size_t i = 0; i < seq * n_embd; ++i) h[i] += proj_out[i];
        stage_mark(us_resid);

        // ── FFN: RMSNorm -> SwiGLU -> down-proj + residual ──────────────
        dequant_full(*L.ffn_norm, w_norm.data());
        for (size_t t = 0; t < seq; ++t)
            rms_norm(&h[t * n_embd], w_norm.data(), &ff_normed[t * n_embd], n_embd, cfg.norm_eps);
        stage_mark(us_norm);

        // ── FFN: gate + up projections ────────────────────────────────────
        // Two variants, selected once at load time by cfg.use_swiglu:
        //
        //   SwiGLU (use_swiglu==true, NANITY spec default):
        //     ff_act[i] = silu(gate[i]) * up[i]
        //     Requires blk.{i}.ffn_gate.weight in every layer.
        //
        //   Plain SiLU-FFN (use_swiglu==false, e.g. Nanity X1 32-layer):
        //     ff_act[i] = silu(up[i])
        //     ffn_gate.weight is absent from the file; gate_buf goes unused.
        //
        // The down-projection + residual add that follow are identical in
        // both cases — only the activation computation differs.
        if (cfg.use_swiglu) {
            // FIX (perf): same fusion as Q/K/V above — gate and up share
            // the same ff_normed input, one round trip instead of two.
            proj_all_positions_multi({
                {L.ffn_gate, gate_buf.data(), n_ff},
                {L.ffn_up,   up_buf.data(),   n_ff},
            }, ff_normed.data(), n_embd, seq, pool);
            for (size_t i = 0; i < seq * n_ff; ++i) ff_act[i] = silu(gate_buf[i]) * up_buf[i];
        } else {
            // Plain SiLU-FFN: no gate projection — silu applied directly to up.
            proj_all_positions(*L.ffn_up, ff_normed.data(), n_embd, up_buf.data(), n_ff, seq, pool);
            for (size_t i = 0; i < seq * n_ff; ++i) ff_act[i] = silu(up_buf[i]);
        }
        stage_mark(us_proj);

        proj_all_positions(*L.ffn_down, ff_act.data(), n_ff, ff_down.data(), n_embd, seq, pool);
        stage_mark(us_proj);
        for (size_t i = 0; i < seq * n_embd; ++i) h[i] += ff_down[i];
        stage_mark(us_resid);
    }

#ifdef VERBOSE_LOG
    {
        // Self-contained timestamped line (doesn't depend on NEON.cpp's
        // VLOG macro, which is defined AFTER it #includes this header, so
        // it isn't visible from in here).
        auto _now = std::chrono::steady_clock::now();
        auto _us  = std::chrono::duration_cast<std::chrono::microseconds>(
                        _now.time_since_epoch()).count();
        std::cerr << "[VERBOSE " << _us << "us] forward(seq=" << seq
                  << ", base_pos=" << base_pos << "): norm=" << (long)us_norm
                  << "us  proj(dequant+matvec)=" << (long)us_proj
                  << "us  attn=" << (long)us_attn
                  << "us  resid=" << (long)us_resid
                  << "us -- if 'proj' dominates the total far more than its "
                     "share of model FLOPs would suggest, that's the "
                     "signature of disk/page-fault stalls rather than "
                     "compute (see rawllm_loader.hpp's mmap warm-up).\n";
    }
#endif

    cache.length = base_pos + seq;   // commit: these positions are now cached

    // ── Final norm + output projection (only the LAST new position's
    //    logits are needed — that's the only one sample_top_p() will look
    //    at). ──
    std::vector<float> w_out_norm(n_embd);
    dequant_full(*mw.output_norm, w_out_norm.data());
    std::vector<float> final_normed(n_embd);
    rms_norm(&h[(seq - 1) * n_embd], w_out_norm.data(), final_normed.data(), n_embd, cfg.norm_eps);

    // Tied embeddings: token_embd.weight is shape [n_embd, n_vocab], exactly
    // the same shape a dedicated output.weight would have, so it can be used
    // directly as the output projection with no special-casing.
    // seq=1 here, so proj_all_positions takes the decode/row-parallel path.
    const loader::TensorInfo* out_w = mw.output ? mw.output : mw.token_embd;
    proj_all_positions(*out_w, final_normed.data(), n_embd, out_logits, cfg.n_vocab, 1, pool);
}

} // namespace fwd