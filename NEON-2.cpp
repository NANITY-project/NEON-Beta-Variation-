#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rawllm_common.hpp"
#include "rawllm_util.hpp"
#include "rawllm_json.hpp"
#include "rawllm_loader.hpp"
#include "rawllm_nctr_loader.hpp"
#include "rawllm_rocm.hpp"
#include "rawllm_forward.hpp"
#include "nectar_diskmem.hpp"

// =============================================================================
// FIX (bug 2): VERBOSE_LOG timing macros.
// Previously the Python flag passed -DVERBOSE_LOG=1 but no C++ code read it,
// so the "per-stage timing" advertised in the UI was a no-op.  These macros
// write microsecond-stamped lines to stderr whenever VERBOSE_LOG is defined.
// =============================================================================
#ifdef VERBOSE_LOG
#  define VLOG(msg) \
    do { \
        auto _now = std::chrono::steady_clock::now(); \
        auto _us  = std::chrono::duration_cast<std::chrono::microseconds>( \
                        _now.time_since_epoch()).count(); \
        std::cerr << "[VERBOSE " << _us << "us] " << (msg) << "\n"; \
    } while(0)
#  define VLOG_ELAPSED(label, t0) \
    do { \
        auto _dt = std::chrono::steady_clock::now() - (t0); \
        std::cerr << "[VERBOSE] " << (label) << " " \
                  << std::chrono::duration_cast<std::chrono::milliseconds>(_dt).count() \
                  << "ms\n"; \
    } while(0)
#else
#  define VLOG(msg)             do {} while(0)
#  define VLOG_ELAPSED(lbl, t0) do {} while(0)
#endif


// =============================================================================
// FIX (bug 9 — tokenizer family mismatch): Phi-4-mini-instruct's config.json
// declares "AutoTokenizer": "Xenova/gpt-4o" — a byte-level BPE tokenizer
// (the GPT-2 / GPT-4o / tiktoken family), NOT SentencePiece. There is no ▁
// (U+2581) word-start marker anywhere in that vocab; instead every raw byte
// is remapped through a fixed byte<->unicode table before BPE ever sees it
// (so byte 0x20 "space" becomes the vocab character "Ġ", and a long tail of
// other bytes map to obscure Latin-1/printable Unicode codepoints that look
// like ordinary text but each stand in for exactly one raw byte).
//
// The previous Tokenizer searched for ▁-prefixed substrings (which never
// match) and greedily matched literal text against the vocab (which mostly
// finds nothing, because the vocab strings are byte-remapped, not literal
// text) — and on decode, it stripped a ▁ sequence that doesn't exist while
// leaving every byte-remapped vocab string un-translated, which is exactly
// what produces output like "_l<5256155535353535...": real vocab strings
// emitted as literal characters instead of being run through the inverse
// byte<->unicode mapping.
//
// This replacement is a real byte-level BPE codec:
//   1. ByteCodec: the canonical GPT-2-style byte<->unicode table.
//   2. pretokenize(): splits input into word/number/punctuation/whitespace
//      chunks. This is a hand-rolled approximation of the regex GPT-2/
//      tiktoken-family tokenizers pretokenize with ('s|'t|'re|...| ?\p{L}+|
//      ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+) — std::regex has no Unicode
//      property classes, so non-ASCII bytes are simply treated as "letter"
//      class. This is NOT byte-exact for every script, but it is the right
//      shape of pretokenizer for this vocab family, vs. none at all.
//   3. bpe_merge(): real merge-rule application over tok_meta.merges
//      (rank = list position, lowest rank wins, repeat to fixed point) —
//      the part the old comment admitted was missing.
//   4. decode_one(): inverse byte<->unicode mapping per vocab string, fed
//      through a small streaming buffer that withholds any trailing bytes
//      that don't yet form a complete UTF-8 codepoint — necessary because
//      a single printed character can be split across two adjacent BPE
//      tokens, and the call sites JSON-escape/print each piece immediately
//      as it's produced. flush() drains anything still buffered once
//      generation stops (EOS or max_tokens).
// =============================================================================
struct ByteCodec {
    std::array<std::string, 256> byte_to_str;          // raw byte -> vocab-alphabet UTF-8 string
    std::unordered_map<uint32_t, uint8_t> cp_to_byte;   // vocab-alphabet codepoint -> raw byte

    static std::string encode_cp(uint32_t cp) {
        // All codepoints this table produces are < 0x800 (verified by
        // construction below), so 1- or 2-byte UTF-8 is always sufficient.
        std::string s;
        if (cp < 0x80) {
            s += (char)cp;
        } else {
            s += (char)(0xC0 | (cp >> 6));
            s += (char)(0x80 | (cp & 0x3F));
        }
        return s;
    }

    ByteCodec() {
        std::vector<int> bs;
        for (int b = (int)'!'; b <= (int)'~'; ++b) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
        std::vector<int> cs = bs;

        std::array<bool, 256> present{};
        for (int b : bs) present[b] = true;

        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!present[b]) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }

        for (size_t i = 0; i < bs.size(); ++i) {
            uint8_t  byte = (uint8_t)bs[i];
            uint32_t cp   = (uint32_t)cs[i];
            byte_to_str[byte] = encode_cp(cp);
            cp_to_byte[cp]    = byte;
        }
    }
};

// Decode one UTF-8 codepoint starting at s[i]. Returns false (and advances
// by exactly 1 byte) on malformed input so callers can pass the raw byte
// through rather than getting stuck.
static bool utf8_decode_one(const std::string& s, size_t i, uint32_t& cp, size_t& len) {
    uint8_t c = (uint8_t)s[i];
    if ((c & 0x80) == 0x00) { cp = c; len = 1; return true; }
    if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
        cp = (uint32_t)(c & 0x1F) << 6 | (uint8_t(s[i+1]) & 0x3F);
        len = 2; return true;
    }
    if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
        cp = (uint32_t)(c & 0x0F) << 12 | (uint32_t)(uint8_t(s[i+1]) & 0x3F) << 6
                                         | (uint8_t(s[i+2]) & 0x3F);
        len = 3; return true;
    }
    if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
        cp = (uint32_t)(c & 0x07) << 18 | (uint32_t)(uint8_t(s[i+1]) & 0x3F) << 12
                                         | (uint32_t)(uint8_t(s[i+2]) & 0x3F) << 6
                                         | (uint8_t(s[i+3]) & 0x3F);
        len = 4; return true;
    }
    len = 1; return false;
}

// Length of the prefix of buf that consists only of complete UTF-8
// codepoints, i.e. where to cut so we never emit a torn multi-byte char.
static size_t complete_utf8_prefix_len(const std::string& buf) {
    size_t n = buf.size();
    if (n == 0) return 0;
    size_t back_limit = std::min<size_t>(4, n);
    for (size_t back = 1; back <= back_limit; ++back) {
        uint8_t c = (uint8_t)buf[n - back];
        if ((c & 0xC0) == 0x80) continue;   // continuation byte, keep walking back
        int need;
        if      ((c & 0x80) == 0x00) need = 1;
        else if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        else need = 1;                       // invalid lead byte: don't buffer forever
        return (back < (size_t)need) ? (n - back) : n;
    }
    return n;   // more than 4 trailing continuation bytes with no lead found: malformed, flush all
}

struct Tokenizer {
    const loader::TokenizerMeta&             meta;
    std::unordered_map<std::string, int32_t> vocab_map;   // byte-level-BPE string -> id
    std::unordered_map<std::string, int32_t> bpe_ranks;   // "left right" -> merge priority (lower = earlier)
    int32_t bos_id, eos_id;
    ByteCodec codec;
    std::string pending_bytes;   // streaming-decode buffer: bytes not yet a complete UTF-8 char

    // FIX (bug 10 — root cause of garbled output even after the byte-BPE
    // codec fix): control tokens like <|system|>, <|user|>, <|assistant|>,
    // <|end|> are single atomic vocab entries the model was trained to
    // treat as structural markers — NOT literal text. The old encode() had
    // no concept of this: literal "<|system|>" in a prompt got pretokenized
    // and BPE-merged character-by-character like ordinary text (<, |, s, y,
    // s, t, e, m, |, >...), so the model never actually saw its own control
    // tokens, just a punctuation-heavy approximation of their spelling. No
    // amount of correct RoPE/byte-codec/EOS-id fixing helps if the prompt
    // itself never contains the tokens the model was fine-tuned to expect.
    //
    // special_tokens here is auto-populated from the vocab itself (any
    // entry of the exact shape "<|...|>", with no '|' in the middle) rather
    // than a hardcoded list, so it stays correct if the vocab changes —
    // sorted longest-first so e.g. "<|end|>" can't accidentally be matched
    // as a prefix of something longer at the same position.
    std::vector<std::string> special_tokens;

    static bool looks_like_special_token(const std::string& s) {
        if (s.size() < 4) return false;
        if (s[0] != '<' || s[1] != '|' || s[s.size()-2] != '|' || s.back() != '>') return false;
        for (size_t i = 2; i + 2 < s.size(); ++i)
            if (s[i] == '|' || s[i] == '<' || s[i] == '>') return false;   // keep it a single clean tag
        return true;
    }

    explicit Tokenizer(const loader::TokenizerMeta& m)
        : meta(m), bos_id(m.bos_id), eos_id(m.eos_id)
    {
        vocab_map.reserve(m.tokens.size());
        for (size_t i = 0; i < m.tokens.size(); ++i)
            vocab_map[m.tokens[i]] = static_cast<int32_t>(i);

        // tok_meta.merges entries are "left right" (single-space separated,
        // standard GGUF tokenizer.ggml.merges convention) in priority order.
        bpe_ranks.reserve(m.merges.size());
        for (size_t i = 0; i < m.merges.size(); ++i)
            bpe_ranks[m.merges[i]] = static_cast<int32_t>(i);

        for (const auto& t : m.tokens)
            if (looks_like_special_token(t)) special_tokens.push_back(t);
        std::sort(special_tokens.begin(), special_tokens.end(),
                  [](const std::string& a, const std::string& b){ return a.size() > b.size(); });
    }

    // Returns the matching special token string at text[pos], or empty if
    // none matches there. Longest-match since special_tokens is presorted.
    const std::string* match_special_token(const std::string& text, size_t pos) const {
        for (const auto& s : special_tokens)
            if (text.compare(pos, s.size(), s) == 0) return &s;
        return nullptr;
    }

    // Split text into pretokenization chunks (word / number / punctuation /
    // whitespace runs), each independently fed through BPE merges below.
    static std::vector<std::string> pretokenize(const std::string& text) {
        static const char* kContractions[] = { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };
        auto is_space       = [](unsigned char c){ return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\v'||c=='\f'; };
        auto is_ascii_letter = [](unsigned char c){ return std::isalpha(c) != 0; };
        auto is_ascii_digit  = [](unsigned char c){ return std::isdigit(c) != 0; };
        auto utf8_len        = [](unsigned char c)->size_t {
            if ((c & 0x80) == 0x00) return 1;
            if ((c & 0xE0) == 0xC0) return 2;
            if ((c & 0xF0) == 0xE0) return 3;
            if ((c & 0xF8) == 0xF0) return 4;
            return 1;
        };

        std::vector<std::string> chunks;
        size_t i = 0, n = text.size();
        while (i < n) {
            unsigned char c = (unsigned char)text[i];

            // Contractions bind to the preceding word in real GPT-2 BPE, but
            // splitting them off as their own chunk still segments correctly
            // against this vocab (each contraction piece is itself a vocab
            // entry), so treat them as their own chunk for simplicity.
            if (c == '\'') {
                bool matched = false;
                for (const char* ctok : kContractions) {
                    size_t clen = std::strlen(ctok);
                    if (text.compare(i, clen, ctok) == 0) {
                        chunks.push_back(ctok);
                        i += clen; matched = true; break;
                    }
                }
                if (matched) continue;
            }

            if (is_space(c)) {
                size_t j = i;
                while (j < n && is_space((unsigned char)text[j])) ++j;
                size_t wslen = j - i;
                if (j == n) {
                    // Trailing whitespace at end of input: its own chunk.
                    chunks.push_back(text.substr(i, wslen));
                    i = j;
                    continue;
                } else if (wslen > 1) {
                    // All but the last space is its own chunk; the final
                    // space is left for the next chunk to pick up as a
                    // leading space (mirrors " ?\p{L}+" etc. in the regex).
                    chunks.push_back(text.substr(i, wslen - 1));
                    i = j - 1;
                    continue;
                }
                // wslen == 1 and not at end: fall through below so the
                // word/number/punct branch picks this up as a leading
                // space (its own `c == ' '` check handles it). Looping
                // back here with i unchanged would spin forever.
            }

            size_t start = i;
            bool leading_space = false;
            if (c == ' ') {
                leading_space = true;
                ++i;
                if (i >= n) { chunks.push_back(" "); break; }
                c = (unsigned char)text[i];
            }
            (void)leading_space;

            if (is_ascii_letter(c) || c >= 0x80) {
                size_t j = i;
                while (j < n) {
                    unsigned char cj = (unsigned char)text[j];
                    if (is_ascii_letter(cj))      { ++j; }
                    else if (cj >= 0x80)          { j += utf8_len(cj); }
                    else break;
                }
                chunks.push_back(text.substr(start, j - start));
                i = j;
            } else if (is_ascii_digit(c)) {
                size_t j = i;
                while (j < n && is_ascii_digit((unsigned char)text[j])) ++j;
                chunks.push_back(text.substr(start, j - start));
                i = j;
            } else {
                size_t j = i;
                while (j < n) {
                    unsigned char cj = (unsigned char)text[j];
                    if (is_space(cj) || is_ascii_letter(cj) || is_ascii_digit(cj) || cj >= 0x80) break;
                    ++j;
                }
                if (j == i) j = i + 1;   // guarantee forward progress
                chunks.push_back(text.substr(start, j - start));
                i = j;
            }
        }
        return chunks;
    }

    // Repeatedly merge the lowest-rank adjacent pair until none apply.
    std::vector<std::string> bpe_merge(std::vector<std::string> symbols) const {
        while (symbols.size() > 1) {
            int    best_rank = std::numeric_limits<int>::max();
            size_t best_i    = SIZE_MAX;
            for (size_t k = 0; k + 1 < symbols.size(); ++k) {
                auto it = bpe_ranks.find(symbols[k] + " " + symbols[k + 1]);
                if (it != bpe_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i    = k;
                }
            }
            if (best_i == SIZE_MAX) break;
            symbols[best_i] += symbols[best_i + 1];
            symbols.erase(symbols.begin() + best_i + 1);
        }
        return symbols;
    }

    // Encode text -> token-id sequence via real byte-level BPE, with
    // special control tokens (<|system|>, <|end|>, etc.) recognized as
    // atomic ids rather than being shredded through BPE as literal text.
    std::vector<int32_t> encode(const std::string& text, bool add_bos = true) const {
        std::vector<int32_t> ids;
        if (add_bos && bos_id >= 0) ids.push_back(bos_id);

        auto encode_literal_chunk = [&](const std::string& literal) {
            for (const std::string& chunk : pretokenize(literal)) {
                std::vector<std::string> symbols;
                symbols.reserve(chunk.size());
                for (unsigned char b : chunk) symbols.push_back(codec.byte_to_str[b]);

                for (const std::string& sym : bpe_merge(std::move(symbols))) {
                    auto it = vocab_map.find(sym);
                    if (it != vocab_map.end()) {
                        ids.push_back(it->second);
                        continue;
                    }
                    // Vocab/merges mismatch (shouldn't happen with a consistent
                    // GGUF, but don't silently drop input): fall back to
                    // per-byte lookup of this merged symbol's original bytes.
                    size_t bi = 0;
                    while (bi < sym.size()) {
                        uint32_t cp; size_t len;
                        if (!utf8_decode_one(sym, bi, cp, len)) { ++bi; continue; }
                        auto bit = codec.cp_to_byte.find(cp);
                        if (bit != codec.cp_to_byte.end()) {
                            char hex[8];
                            std::snprintf(hex, sizeof(hex), "<0x%02X>", bit->second);
                            auto hit = vocab_map.find(hex);
                            if (hit != vocab_map.end()) ids.push_back(hit->second);
                        }
                        bi += len;
                    }
                }
            }
        };

        size_t i = 0, n = text.size(), literal_start = 0;
        while (i < n) {
            const std::string* tag = special_tokens.empty() ? nullptr : match_special_token(text, i);
            if (tag) {
                if (i > literal_start) encode_literal_chunk(text.substr(literal_start, i - literal_start));
                ids.push_back(vocab_map.at(*tag));   // present by construction: tag came from the vocab itself
                i += tag->size();
                literal_start = i;
            } else {
                ++i;
            }
        }
        if (literal_start < n) encode_literal_chunk(text.substr(literal_start, n - literal_start));

        return ids;
    }

    // Decode a single token id -> UTF-8 bytes, run through the inverse
    // byte<->unicode mapping, withholding any trailing incomplete UTF-8
    // sequence (it'll complete once the next token's bytes are appended).
    std::string decode_one(int32_t id) {
        if (id < 0 || id >= (int32_t)meta.tokens.size()) return "";
        const std::string& t = meta.tokens[id];

        std::string raw;
        raw.reserve(t.size());
        size_t i = 0;
        while (i < t.size()) {
            uint32_t cp; size_t len;
            if (!utf8_decode_one(t, i, cp, len)) { raw += t[i]; ++i; continue; }
            auto it = codec.cp_to_byte.find(cp);
            if (it != codec.cp_to_byte.end()) {
                raw += (char)it->second;
            } else {
                // Not part of the byte-remap alphabet — this is a literal
                // special-token character (e.g. from "<|im_start|>"), not a
                // remapped byte. Emit it verbatim.
                raw += t.substr(i, len);
            }
            i += len;
        }

        pending_bytes += raw;
        size_t emit_len = complete_utf8_prefix_len(pending_bytes);
        std::string out = pending_bytes.substr(0, emit_len);
        pending_bytes.erase(0, emit_len);
        return out;
    }

    // Drain any bytes still withheld by decode_one (call once generation
    // stops — EOS or max_tokens — so a trailing torn character isn't lost).
    std::string flush() {
        std::string out = std::move(pending_bytes);
        pending_bytes.clear();
        return out;
    }
};


// =============================================================================
// FIX (bug 8): Top-p (nucleus) sampler using fast_expf from rawllm_common.hpp.
// Previously the only sampling call was commented out; logits were never used.
// =============================================================================
int32_t sample_top_p(const float* logits, int n_vocab,
                     float temperature, float top_p,
                     std::mt19937& rng)
{
    // Greedy decode when temperature ≈ 0 or top_p ≈ 0.
    if (temperature < 1e-5f || top_p < 1e-5f) {
        return static_cast<int32_t>(
            std::max_element(logits, logits + n_vocab) - logits);
    }

    // Compute softmax with temperature via fast_expf.
    std::vector<float> probs(n_vocab);
    float max_l = *std::max_element(logits, logits + n_vocab);
    float sum   = 0.f;
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] = engine::fast_expf((logits[i] - max_l) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < n_vocab; ++i) probs[i] /= sum;

    // Sort indices by probability descending.
    std::vector<int32_t> idx(n_vocab);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(),
                      idx.begin() + std::min(n_vocab, 200), // cap sort for speed
                      idx.end(),
                      [&](int a, int b){ return probs[a] > probs[b]; });

    // Accumulate until nucleus covers top_p of probability mass.
    float cumsum = 0.f;
    int   cutoff = 1;
    for (int i = 0; i < std::min(n_vocab, 200); ++i) {
        cumsum += probs[idx[i]];
        cutoff  = i + 1;
        if (cumsum >= top_p) break;
    }

    // Sample uniformly within the nucleus.
    std::uniform_real_distribution<float> dist(0.f, cumsum);
    float r   = dist(rng);
    float acc = 0.f;
    for (int i = 0; i < cutoff; ++i) {
        acc += probs[idx[i]];
        if (r <= acc) return idx[i];
    }
    return idx[0];
}


// =============================================================================
// FIX (bug 8 — root cause of garbled output): forward() used to be a stub
// that filled out_logits with zeros, so sample_top_p() sampled uniformly at
// random over the whole vocab every step. The real transformer math lives
// in rawllm_forward.hpp — the fixed NANITY architecture (RMSNorm -> separate
// Q/K/V proj -> RoPE -> causal GQA attention -> out-proj -> SwiGLU FFN), per
// layer. See that file's header comment for scope/limitations before
// reporting follow-up bugs.
//
// FIX (perf): forward() used to recompute the whole context from scratch on
// every generated token (no KV cache) — O(seq²) total work per reply. Both
// ModelWeights AND a KVCache are now built once per process (function-local
// statics — safe because main() loads exactly one GGUF per process
// lifetime, so cfg/gguf never change between calls) and reused across every
// chat turn — including across turns now: cached_tokens records which token
// ids are currently sitting in the cache, and run_generation() below
// matches that against each new turn's freshly-tokenized prompt to find how
// much can be reused instead of re-prefilled. The underlying ~1GB of
// allocated K/V storage is only paid for once, not per turn, regardless.
// =============================================================================
struct EngineState {
    fwd::ModelWeights    mw;
    fwd::KVCache          cache;
    std::vector<int32_t>  cached_tokens;   // invariant: cached_tokens.size() == cache.length
};

template <typename Loader>
static EngineState& get_engine_state(const engine::Config& cfg,
                                      const Loader& gguf)
{
    // FIX (OOM crash on first chat turn — "Engine closed stdout unexpectedly"):
    // this used to size the cache off cfg.ctx_len (the model's full trained
    // context length — 131072 for Phi-4-mini). KVCache::assign() zero-fills
    // every element immediately, so that's not a lazy reservation; it forces
    // the OS to commit ~32GB on the spot the moment the first request comes
    // in, regardless of how short that request is, and the kernel OOM-kills
    // the process before it can write a single byte back. cfg.kv_window
    // (rawllm_loader.hpp's validate_config(), defaulted to min(ctx_len,4096))
    // exists specifically to bound this and was never wired in here.
    // ensure_cache_room()/kv_cache_shift() already handle sliding the window
    // forward once it fills, so capping the allocation at kv_window doesn't
    // lose correctness — it just means conversations beyond kv_window tokens
    // gracefully evict instead of the cache ever trying to be ctx_len long.
    static EngineState state{
        fwd::ModelWeights::build(gguf, cfg),
        fwd::KVCache(cfg.n_layer, (size_t)cfg.n_kv_head * cfg.head_dim, cfg.kv_window),
        {}
    };
    return state;
}


// =============================================================================
// Graceful context-window overflow: called before every forward() call with
// the number of NEW tokens about to be processed. If there isn't room,
// evicts the oldest tokens (keeping a small fixed "sink" prefix, à la
// StreamingLLM/llama.cpp context-shift) and RoPE-rotates the survivors down
// by the discarded amount via fwd::kv_cache_shift() — see that function's
// comment for why the rotation makes this exact rather than approximate.
// cached_tokens is trimmed in lockstep so cross-turn prefix matching never
// compares against positions that no longer mean what they used to.
//
// This replaces the old behavior of just stopping generation once ctx_len
// was hit mid-reply. It's still graceful FORGETTING, not free context — the
// discarded token range is genuinely gone from the model's perspective —
// the difference is generation continues instead of halting.
// =============================================================================
// `keep` defaults to 4 (the original hardcoded StreamingLLM sink count) for
// every existing caller (chat/answer path). The idle loop (run_idle_loop(),
// below) passes its own keep count instead — the size of its pinned
// seed+sink region — since §1 of the idle-loop doc requires that region to
// survive eviction regardless of how many dedicated sink tokens it does or
// doesn't also carry.
//
// Returns the number of positions actually discarded (0 if no shift
// happened this call). run_idle_loop() uses this to pop exactly that many
// entries off its own token-aligned text buffer before handing them to
// nectar_diskmem — the two are guaranteed to line up because both are
// FIFO over the same sliding region, discard-count-for-discard-count.
static size_t ensure_cache_room(const engine::Config& cfg, EngineState& st, size_t seq,
                                 size_t keep = 4) {
    if (st.cache.length + seq <= st.cache.capacity) return 0;

    keep = std::min(keep, st.cache.length);

    if (keep + seq > st.cache.capacity) {
        throw std::runtime_error(
            "ensure_cache_room(): " + std::to_string(seq) + " new token(s) plus " +
            std::to_string(keep) + " sink token(s) exceed kv_window=" +
            std::to_string(st.cache.capacity) + " even after evicting everything "
            "else. Reduce the prompt/max_tokens, or raise the kv_window default "
            "in rawllm_loader.hpp's validate_config() (currently min(ctx_len, "
            "4096)) and rebuild -- note memory scales as n_layer * kv_window * "
            "kv_dim * 4 bytes * 2, so raise it deliberately.");
    }

    size_t need        = (st.cache.length + seq) - st.cache.capacity;
    size_t cushion     = st.cache.capacity / 4;   // avoid re-shifting on every next token
    size_t max_discard = st.cache.length - keep;
    size_t discard      = std::min(max_discard, need + cushion);
    if (discard == 0) return 0;

    VLOG("context shift: keep=" + std::to_string(keep) +
         " discard=" + std::to_string(discard) +
         " length_before=" + std::to_string(st.cache.length));

    const std::vector<float> freqs = fwd::rope_make_freqs(cfg.head_dim, cfg.rope_base);
    size_t rope_half = cfg.rope_dim_count ? (cfg.rope_dim_count / 2) : (cfg.head_dim / 2);
    fwd::kv_cache_shift(st.cache, keep, discard, cfg.n_kv_head, cfg.head_dim, rope_half,
                        cfg.rope_scale, freqs.data());

    st.cached_tokens.erase(st.cached_tokens.begin() + keep,
                            st.cached_tokens.begin() + keep + discard);

    VLOG("context shift done: length_after=" + std::to_string(st.cache.length));
    return discard;
}


// =============================================================================
// FIX (bug 10): this used to emit ChatML (<|im_start|>role\n...<|im_end|>\n),
// which doesn't exist anywhere in this model's vocab (confirmed via
// check_tokenizer.py — no <|im_start|>/<|im_end|>/<|im_sep|> entries at
// all). Phi-4-mini-instruct's actual published format, straight from
// Microsoft's model card, is simpler and has no newlines:
//   <|system|>{system}<|end|><|user|>{user}<|end|><|assistant|>
// with the trailing <|assistant|> (no closing <|end|>) left as the
// generation prompt for the model to continue from.
// =============================================================================
static std::string format_chat(const json::Value& messages) {
    std::string prompt;
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& m   = messages[i];
        const auto& role = m["role"].get_str();
        const auto& body = m["content"].get_str();
        if      (role == "system")    prompt += "<|system|>"    + body + "<|end|>";
        else if (role == "user")      prompt += "<|user|>"      + body + "<|end|>";
        else if (role == "assistant") prompt += "<|assistant|>" + body + "<|end|>";
    }
    prompt += "<|assistant|>";
    return prompt;
}


// =============================================================================
// Core autoregressive generation, shared by both single-shot generate() and
// the persistent run_interactive() loop. Decodes one token at a time,
// calling on_token(id) for each piece — so the caller decides whether to
// print plain text (generate()) or NDJSON (run_interactive()) without
// duplicating the prefill/decode logic itself.
//
// Cross-turn cache reuse: NEON.py resends the full conversation as text
// every turn, so prompt_ctx is re-tokenized from scratch each call. Causal
// attention guarantees position i's K/V depends only on tokens[0..i], so
// wherever the new prompt's tokens match what's already in the cache
// (token for token — even a single divergent id, e.g. from tokenizer
// re-segmentation rather than new content, stops the match), those
// positions don't need to be reprocessed at all. Only the new suffix after
// the match point gets prefilled. This is exact, never approximate: a
// mismatch just means a shorter reused prefix, never a wrong one.
// =============================================================================
template <typename Loader, typename OnToken>
static void run_generation(const engine::Config& cfg,
                            const Loader& gguf,
                            const Tokenizer& tok,   // only eos_id is read here; decode_one (stateful) is called by the on_token callback in our callers, not here
                            const std::vector<int32_t>& prompt_ctx,
                            int max_new_tokens,
                            float temperature,
                            float top_p,
                            util::ThreadPool& pool,
                            std::mt19937& rng,
                            OnToken&& on_token)
{
    auto& st = get_engine_state(cfg, gguf);

    size_t match_len = 0;
    size_t max_check = std::min({ st.cached_tokens.size(), prompt_ctx.size(), st.cache.length });
    while (match_len < max_check && st.cached_tokens[match_len] == prompt_ctx[match_len])
        ++match_len;

    // K/V being cached doesn't cache logits — always leave at least one
    // "new" token so we get a fresh forward call to sample from, even if
    // the entire new prompt was already cached verbatim (e.g. regenerate).
    if (!prompt_ctx.empty() && match_len >= prompt_ctx.size())
        match_len = prompt_ctx.size() - 1;

    st.cache.length = match_len;
    st.cached_tokens.resize(match_len);
    std::vector<int32_t> new_prompt_tokens(prompt_ctx.begin() + match_len, prompt_ctx.end());
    VLOG("cache reuse: matched=" + std::to_string(match_len) + "/" +
         std::to_string(prompt_ctx.size()) + " new_prefill=" +
         std::to_string(new_prompt_tokens.size()));

    std::vector<float> logits(cfg.n_vocab, 0.f);

    ensure_cache_room(cfg, st, new_prompt_tokens.size());
    VLOG("prefill len=" + std::to_string(new_prompt_tokens.size()));
    auto t_pre = std::chrono::steady_clock::now();
    fwd::transformer_forward(cfg, st.mw, new_prompt_tokens, st.cache, logits.data(), pool);
    VLOG_ELAPSED("prefill", t_pre);
    st.cached_tokens.insert(st.cached_tokens.end(), new_prompt_tokens.begin(), new_prompt_tokens.end());

    int32_t next = sample_top_p(logits.data(), (int)cfg.n_vocab, temperature, top_p, rng);

    // FIX (diagnostics): NANITY_DEBUG_LOGITS=1 dumps the top-10 (logit,
    // token) candidates for the FIRST generated token only -- the one
    // produced straight from the prefill forward pass, before any
    // per-step decode/cache-reuse machinery runs. This is the single
    // cheapest way to tell "the forward pass itself is producing bad
    // logits" apart from "something about incremental decode/caching goes
    // wrong a few tokens in" -- run with --prompt, temperature=0, top_p=1
    // for a fully deterministic single sample.
    if (std::getenv("NANITY_DEBUG_LOGITS")) {
        std::vector<int32_t> idx(cfg.n_vocab);
        std::iota(idx.begin(), idx.end(), 0);
        size_t topn = std::min((size_t)10, idx.size());
        std::partial_sort(idx.begin(), idx.begin() + topn, idx.end(),
            [&](int32_t a, int32_t b){ return logits[a] > logits[b]; });
        std::cerr << "[DEBUG top-" << topn << " first-token candidates]\n";
        for (size_t i = 0; i < topn; ++i) {
            int32_t id = idx[i];
            const std::string& s = (id >= 0 && (size_t)id < gguf.tok_meta.tokens.size())
                                    ? gguf.tok_meta.tokens[id] : std::string("<oob>");
            std::cerr << "  #" << i << "  id=" << id << "  logit=" << logits[id]
                      << "  vocab_str=" << s << "\n";
        }
        std::cerr << std::flush;
    }

    for (int step = 0; step < max_new_tokens; ++step) {
        if (next == tok.eos_id) break;
        on_token(next);

        ensure_cache_room(cfg, st, 1);

        VLOG("decode step=" + std::to_string(step));
        auto t_dec = std::chrono::steady_clock::now();
        std::vector<int32_t> one_token{ next };
        fwd::transformer_forward(cfg, st.mw, one_token, st.cache, logits.data(), pool);
        VLOG_ELAPSED("decode", t_dec);
        st.cached_tokens.push_back(next);

        next = sample_top_p(logits.data(), (int)cfg.n_vocab, temperature, top_p, rng);
    }
}


// =============================================================================
// Single-shot mode: streams each decoded piece to stdout immediately.
// =============================================================================
template <typename Loader>
static void generate(const engine::Config& cfg,
                     const Loader& gguf,
                     Tokenizer& tok,
                     const std::string& prompt,
                     int   max_new_tokens,
                     float temperature,
                     float top_p,
                     util::ThreadPool& pool,
                     std::mt19937& rng)
{
    VLOG("encode prompt len=" + std::to_string(prompt.size()));
    auto t_enc = std::chrono::steady_clock::now();
    std::vector<int32_t> ctx = tok.encode(prompt, /*add_bos=*/true);
    VLOG_ELAPSED("tokenize", t_enc);

    auto t_gen = std::chrono::steady_clock::now();
    run_generation(cfg, gguf, tok, ctx, max_new_tokens, temperature, top_p, pool, rng,
        [&](int32_t id) {
            std::cout << tok.decode_one(id) << std::flush;
        });
    std::cout << tok.flush() << '\n' << std::flush;
    VLOG_ELAPSED("generate total", t_gen);
}


// =============================================================================
// FIX (bug 7, C++ side): Persistent stdin/stdout interactive loop.
//
// Protocol (one JSON object per line each way):
//   IN  → {"messages":[…], "max_tokens":256, "temperature":0.7, "top_p":0.9}
//   OUT ← {"token":"…"}   (one per generated piece, then…)
//   OUT ← {"done":true}
//
// NEON.py spawns the binary once (--interactive) and reuses the process for
// every chat turn, eliminating per-turn model-load latency and giving the
// Python layer a place to accumulate conversation history.
// =============================================================================
template <typename Loader>
static void run_interactive(const engine::Config& cfg,
                             const Loader& gguf,
                             Tokenizer& tok,
                             util::ThreadPool& pool)
{
    std::mt19937 rng(std::random_device{}());
    std::string  line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        auto req = json::try_parse(line);
        if (req.is_null()) {
            std::cout << "{\"error\":\"bad JSON\"}\n" << std::flush;
            continue;
        }

        int   max_tokens  = req["max_tokens"].get_int(256);
        float temperature = req["temperature"].get_float(0.7f);
        float top_p       = req["top_p"].get_float(0.9f);

        std::string prompt;
        const auto& msgs = req["messages"];
        if (msgs.is_array() && msgs.size() > 0) {
            prompt = format_chat(msgs);
        } else {
            prompt = req["prompt"].get_str();
        }

        // Generate and emit each token as a JSON line.
        VLOG("interactive: encode prompt");
        auto t_enc = std::chrono::steady_clock::now();
        std::vector<int32_t> ctx = tok.encode(prompt, /*add_bos=*/true);
        VLOG_ELAPSED("interactive encode", t_enc);

        run_generation(cfg, gguf, tok, ctx, max_tokens, temperature, top_p, pool, rng,
            [&](int32_t id) {
                std::string piece = tok.decode_one(id);
                // Emit as JSON so the Python side can parse it without fragile
                // substring matching.
                std::cout << "{\"token\":\"" << json::escape(piece) << "\"}\n"
                          << std::flush;
            });
        // FIX: decode_one() withholds any trailing bytes that don't yet form
        // a complete UTF-8 codepoint (it can be split across the last two
        // tokens). Drain that buffer now, or the final character of a reply
        // can silently vanish whenever generation stops mid-codepoint.
        std::string tail = tok.flush();
        if (!tail.empty()) {
            std::cout << "{\"token\":\"" << json::escape(tail) << "\"}\n"
                      << std::flush;
        }
        std::cout << "{\"done\":true}\n" << std::flush;
    }
}


// =============================================================================
// Idle loop (NECTAR idle-loop architecture v3, §8 build-order step 1):
// continuous streaming generation with StreamingLLM-style sink/window
// eviction, so idle CoT can run indefinitely on a fixed-size context
// instead of halting once the cache fills. Deliberately excludes the
// pinned implant proper, vision deltas, and eviction-to-disk (§§3-5 of the
// doc) — those are build-order steps 2-4. What this step DOES let you run
// is the open comparison §1 calls out before committing to a sink design:
//
//   --idle-seed "<text>" --idle-sink-tokens 0   → seed-block-as-sink-only
//   --idle-seed "<text>" --idle-sink-tokens N   → seed block + N dedicated
//                                                   neutral sink tokens
//
// `keep` = seed_tokens.size() + sink_tokens; both are pinned via the same
// ensure_cache_room()/kv_cache_shift() machinery the answer path already
// uses for its 4-token default, just with a caller-supplied keep count.
// Watch cache.length in the periodic [Idle] log across a long run: if
// generation degrades (repetition, incoherence) after a shift with
// sink_tokens=0 but stays stable with sink_tokens>0, that's the doc's
// signal to keep dedicated sink tokens; if seed-only holds up, they're
// redundant and step 2 (the real implant) can skip them.
//
// Own EngineState: get_engine_state()'s function-local static means this
// must run as its own process (`--idle-loop`), never in the same process
// as `--interactive`/single-shot — that's what keeps this loop's KV cache
// and cached_tokens from ever touching the answer path's, per §2's "don't
// let them share state." The separation is "two processes," not "two
// objects in one process."
//
// No natural stop: idle CoT isn't supposed to terminate, so EOS is treated
// as a soft paragraph break (emit a newline, keep generating) rather than
// end-of-reply. --idle-max-tokens caps a run for benchmarking; 0 runs until
// killed.
// =============================================================================
template <typename Loader>
static void run_idle_loop(const engine::Config& cfg,
                           const Loader& gguf,
                           Tokenizer& tok,
                           util::ThreadPool& pool,
                           const std::string& seed_text,
                           size_t sink_tokens,
                           long   max_tokens,   // 0 = infinite
                           size_t log_every,
                           float  temperature,
                           float  top_p,
                           const std::string& disk_path,
                           size_t disk_flush_every_tokens,
                           long   disk_flush_interval_s)
{
    auto& st = get_engine_state(cfg, gguf);
    std::mt19937 rng(std::random_device{}());

    // ---- Build the pinned region: implant block (§3/§4 — seed_text is now
    // the actual compressed personality distillation, not a stand-in;
    // §8 step 2) followed by optional dedicated neutral sink tokens (§1).
    std::vector<int32_t> pinned = tok.encode(seed_text, /*add_bos=*/true);
    const size_t seed_len = pinned.size();

    if (sink_tokens > 0) {
        // StreamingLLM's finding is that early *position*, not specific
        // content, is what stabilizes attention — so any bland, repeatable
        // text is fine here. Encode once, then repeat/truncate to exactly
        // sink_tokens ids.
        static const std::string kNeutralFiller =
            "The quick brown fox jumps over the lazy dog.";
        std::vector<int32_t> filler = tok.encode(kNeutralFiller, /*add_bos=*/false);
        if (filler.empty())
            throw std::runtime_error("run_idle_loop(): neutral filler encoded to zero tokens");
        pinned.reserve(pinned.size() + sink_tokens);
        for (size_t i = 0; i < sink_tokens; ++i)
            pinned.push_back(filler[i % filler.size()]);
    }

    const size_t keep = pinned.size();
    if (keep >= cfg.kv_window) {
        throw std::runtime_error(
            "run_idle_loop(): pinned region (" + std::to_string(keep) +
            " tokens: " + std::to_string(seed_len) + " seed + " +
            std::to_string(sink_tokens) + " sink) already fills or exceeds "
            "kv_window=" + std::to_string(cfg.kv_window) + " -- shorten "
            "--idle-seed, lower --idle-sink-tokens, or raise kv_window.");
    }

    // §4 "Implant Sizing Constraint": the implant is a permanent tax on
    // every cycle's budget, forever, so it should stay a small minority of
    // kv_window, not compete with rolling vision/CoT content for room.
    // 20% is a soft line, not from the doc itself -- just a sanity check
    // worth having before a long unattended run rather than after.
    if (seed_len * 5 > cfg.kv_window) {
        std::cerr << "[Idle] WARNING: implant is " << seed_len << " tokens, "
                  << (100 * seed_len / cfg.kv_window) << "% of kv_window="
                  << cfg.kv_window << " -- doc §4 wants this a small minority "
                  << "so it doesn't crowd out rolling content. Consider "
                  << "compressing the implant text.\n";
    }

    std::cerr << "[Idle] pinned region: " << keep << " tokens ("
              << seed_len << " implant + " << sink_tokens << " dedicated sink), "
              << "kv_window=" << cfg.kv_window << "\n";

    // ---- Disk write path (§5.1): batched, source-tagged. `aging` mirrors
    // exactly the sliding region's contents in FIFO order — one entry per
    // generated token, pushed the instant it's generated and popped the
    // instant ensure_cache_room() reports it as discarded. That one-to-one
    // correspondence (rather than e.g. "just buffer whatever was recently
    // generated") is what makes this a real eviction-triggered write, not
    // an approximation of one.
    diskmem::WriteBuffer idle_disk(disk_path, diskmem::Source::Idle,
                                    disk_flush_every_tokens,
                                    std::chrono::seconds(disk_flush_interval_s));
    std::deque<std::string> aging;

    std::vector<float> logits(cfg.n_vocab, 0.f);

    ensure_cache_room(cfg, st, pinned.size(), keep);
    fwd::transformer_forward(cfg, st.mw, pinned, st.cache, logits.data(), pool);
    st.cached_tokens = pinned;

    int32_t next = sample_top_p(logits.data(), (int)cfg.n_vocab, temperature, top_p, rng);

    long produced = 0;
    while (max_tokens == 0 || produced < max_tokens) {
        std::string piece;
        if (next == tok.eos_id) {
            // Soft break, not a stop: fold EOS into the stream like a
            // paragraph boundary and keep decoding.
            piece = "\n";
        } else {
            piece = tok.decode_one(next);
            ++produced;
        }
        std::cout << piece << std::flush;
        aging.push_back(piece);   // this token now occupies a live, evictable position

        size_t discard = ensure_cache_room(cfg, st, 1, keep);   // may trigger a shift
        for (size_t i = 0; i < discard && !aging.empty(); ++i) {
            idle_disk.add(aging.front());
            aging.pop_front();
        }

        std::vector<int32_t> one_token{ next };
        fwd::transformer_forward(cfg, st.mw, one_token, st.cache, logits.data(), pool);
        st.cached_tokens.push_back(next);

        if (log_every > 0 && produced > 0 && produced % (long)log_every == 0) {
            std::cerr << "[Idle] produced=" << produced
                      << " cache.length=" << st.cache.length
                      << "/" << st.cache.capacity
                      << " disk_pending=" << idle_disk.pending_tokens() << "\n";
        }

        next = sample_top_p(logits.data(), (int)cfg.n_vocab, temperature, top_p, rng);
    }

    std::cout << tok.flush() << std::flush;
    idle_disk.flush();   // don't lose a partial buffer on clean shutdown
    std::cerr << "\n[Idle] stopped after " << produced << " tokens (max_tokens reached). "
              << "still-live (never evicted, not on disk): " << aging.size() << " tokens.\n";
}


// =============================================================================
// Model-format dispatch
// =============================================================================
// Sniffed from the file's first 4 bytes, not its extension — an extension
// can be renamed; the magic can't lie about what the parser downstream is
// about to assume.
enum class ModelFormat { GGUF, NCTR };

static ModelFormat sniff_format(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open model: " + path);
    char magic[4] = {0};
    f.read(magic, 4);
    if (!std::memcmp(magic, "GGUF", 4)) return ModelFormat::GGUF;
    if (!std::memcmp(magic, "NCTR", 4)) return ModelFormat::NCTR;
    throw std::runtime_error(
        "Unrecognized model format (first 4 bytes are neither 'GGUF' nor 'NCTR'): " + path);
}

// Everything downstream of "which loader do we use" — load, validate,
// --probe, tokenizer setup, and generation — lives here once and gets
// instantiated per Loader (loader::GGUFLoader or loader::NCTRLoader) rather
// than duplicated per format. This is exactly the body that used to be the
// second half of main(); only `gguf` -> `model` (generic name, not
// format-specific) and the type changed.
struct IdleLoopArgs {
    bool        enabled     = false;
    std::string seed_text;                 // implant text (§3/§4)
    size_t      sink_tokens = 0;
    long        max_tokens  = 0;           // 0 = infinite
    size_t      log_every   = 50;
    float       temperature = 0.9f;        // idle CoT defaults a bit hotter than chat
    float       top_p       = 0.95f;
    std::string disk_path   = "nectar_idle_memory.jsonl";  // §5.1
    size_t      disk_flush_every_tokens = 300;
    long        disk_flush_interval_s   = 30;
};

template <typename Loader>
static int run_model(const std::string& model_path, engine::Config& cfg,
                      util::ThreadPool& pool, bool probe_only, bool interactive,
                      const std::string& input_payload,
                      const IdleLoopArgs& idle,
                      std::chrono::steady_clock::time_point t_start)
{
    VLOG("open: " + model_path);
    std::cerr << "[Storage] Mapping weights via mmap: " << model_path << "\n";
    Loader model;
    model.open(model_path);
    std::cerr << "[Storage] Mapped " << model.tensors.size() << " tensors.\n";

    VLOG("validate_config");
    // FIX (carried over): validate_config() throws a specific, actionable
    // message (which exact metadata key or tensor shape didn't match the
    // NANITY spec) instead of returning a bare false — main()'s catch block
    // surfaces that message directly, so there's nothing to special-case here.
    model.validate_config(cfg);
    std::cerr << "[Config]"
              << " n_vocab="   << cfg.n_vocab
              << " n_embd="    << cfg.n_embd
              << " n_layer="   << cfg.n_layer
              << " n_head="    << cfg.n_head
              << " n_kv_head=" << cfg.n_kv_head
              << " head_dim="  << cfg.head_dim
              << " n_ff="      << cfg.n_ff
              << " ctx_len="   << cfg.ctx_len
              << " rope_base=" << cfg.rope_base << "\n";

    VLOG_ELAPSED("model load+detect", t_start);

    // ── --probe: boot validation used by NEON.py's /api/boot ──────────────
    if (probe_only) {
        std::cout << "PROBE_OK\n" << std::flush;
        return 0;
    }

    // ── Tokenizer ──────────────────────────────────────────────────────────
    Tokenizer tok(model.tok_meta);
    std::cerr << "[Tokenizer] vocab=" << model.tok_meta.tokens.size()
              << " bos=" << tok.bos_id << " eos=" << tok.eos_id << "\n";

    // ── Persistent interactive mode (spawned by NEON.py on boot) ──────────
    if (interactive) {
        std::cerr << "[Mode] Interactive stdin/stdout loop — ready.\n";
        run_interactive(cfg, model, tok, pool);
        return 0;
    }

    // ── Idle loop mode (own process — see run_idle_loop()'s comment on why
    //    this must never share a process with --interactive) ──────────────
    if (idle.enabled) {
        std::cerr << "[Mode] Idle loop — ready.\n";
        run_idle_loop(cfg, model, tok, pool, idle.seed_text, idle.sink_tokens,
                      idle.max_tokens, idle.log_every, idle.temperature, idle.top_p,
                      idle.disk_path, idle.disk_flush_every_tokens, idle.disk_flush_interval_s);
        return 0;
    }

    // ── Single-shot mode (legacy / direct CLI testing) ─────────────────────
    if (input_payload.empty()) {
        std::cerr << "[Error] --prompt required in single-shot mode.\n";
        return 1;
    }

    std::cerr << "[Parser] Parsing input payload...\n";
    std::string prompt;
    if (!input_payload.empty() && input_payload.front() == '{') {
        auto req  = json::parse(input_payload);
        const auto& msgs = req["messages"];
        if (msgs.is_array() && msgs.size() > 0) {
            prompt = format_chat(msgs);
        } else {
            prompt = req["prompt"].get_str(input_payload);
        }
    } else {
        prompt = input_payload;
    }

    float temperature = 0.7f;
    float top_p       = 0.9f;
    int   max_tokens  = 256;

    std::mt19937 rng(std::random_device{}());
    std::cerr << "\n--- Generation ---\n";
    generate(cfg, model, tok, prompt, max_tokens, temperature, top_p, pool, rng);
    std::cerr << "--- End ---\n";
    return 0;
}


// =============================================================================
// main
// =============================================================================
int main(int argc, char* argv[]) {
    // FIX: was `argc < 5` which rejected valid two-flag invocations like
    // `--model path --probe`.
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " --model <path.gguf|path.nctr>"
                     " [--prompt <text_or_json> | --interactive | --probe | --idle-loop]"
                     " [--threads N]\n"
                     " Idle-loop flags (only with --idle-loop):\n"
                     "   --idle-seed <text>            implant text, inline (required unless --idle-seed-file)\n"
                     "   --idle-seed-file <path>        implant text, read from a file (overrides --idle-seed)\n"
                     "   --idle-sink-tokens <N>         dedicated neutral sink tokens appended after the implant (default 0)\n"
                     "   --idle-max-tokens <N>          stop after N generated tokens, 0 = run forever (default 0)\n"
                     "   --idle-log-every <N>           log cache/disk stats every N tokens (default 50)\n"
                     "   --idle-temperature <f>         sampling temperature (default 0.9)\n"
                     "   --idle-top-p <f>               nucleus sampling top-p (default 0.95)\n"
                     "   --idle-disk-path <path>        JSONL file evicted idle text is appended to (default nectar_idle_memory.jsonl)\n"
                     "   --idle-disk-flush-tokens <N>   flush to disk after N buffered tokens (default 300)\n"
                     "   --idle-disk-flush-seconds <N>  or after N seconds, whichever comes first (default 30)\n";
        return 1;
    }

    std::string model_path;
    std::string input_payload;
    bool interactive = false;
    bool probe_only  = false;
    int  threads_override = 0;   // 0 = auto-detect from hardware_concurrency()
    IdleLoopArgs idle;
    std::string idle_seed_file;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--model"       && i + 1 < argc) model_path    = argv[++i];
        else if (arg == "--prompt"      && i + 1 < argc) input_payload = argv[++i];
        else if (arg == "--interactive")                  interactive   = true;
        else if (arg == "--probe")                        probe_only    = true;
        else if (arg == "--threads"     && i + 1 < argc) threads_override = std::max(0, std::atoi(argv[++i]));
        else if (arg == "--idle-loop")                    idle.enabled  = true;
        else if (arg == "--idle-seed"             && i + 1 < argc) idle.seed_text  = argv[++i];
        else if (arg == "--idle-seed-file"        && i + 1 < argc) idle_seed_file  = argv[++i];
        else if (arg == "--idle-sink-tokens"      && i + 1 < argc) idle.sink_tokens = (size_t)std::max(0, std::atoi(argv[++i]));
        else if (arg == "--idle-max-tokens"       && i + 1 < argc) idle.max_tokens  = std::atol(argv[++i]);
        else if (arg == "--idle-log-every"        && i + 1 < argc) idle.log_every   = (size_t)std::max(0, std::atoi(argv[++i]));
        else if (arg == "--idle-temperature"      && i + 1 < argc) idle.temperature = std::strtof(argv[++i], nullptr);
        else if (arg == "--idle-top-p"            && i + 1 < argc) idle.top_p       = std::strtof(argv[++i], nullptr);
        else if (arg == "--idle-disk-path"        && i + 1 < argc) idle.disk_path   = argv[++i];
        else if (arg == "--idle-disk-flush-tokens" && i + 1 < argc) idle.disk_flush_every_tokens = (size_t)std::max(0, std::atoi(argv[++i]));
        else if (arg == "--idle-disk-flush-seconds"&& i + 1 < argc) idle.disk_flush_interval_s   = std::atol(argv[++i]);
    }

    if (idle.enabled && !idle_seed_file.empty()) {
        std::ifstream f(idle_seed_file, std::ios::binary);
        if (!f) {
            std::cerr << "[Error] --idle-seed-file: cannot open " << idle_seed_file << "\n";
            return 1;
        }
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        idle.seed_text = content;
    }

    if (idle.enabled && idle.seed_text.empty()) {
        std::cerr << "[Error] --idle-loop requires --idle-seed <text> or --idle-seed-file <path>.\n";
        return 1;
    }

    if (model_path.empty()) {
        std::cerr << "[Error] --model is required.\n";
        return 1;
    }

    try {
        auto t_start = std::chrono::steady_clock::now();

        // ── Hardware init (format-independent, runs once regardless of
        //    which loader ends up being picked below) ────────────────────
        engine::Config cfg;

        if (threads_override > 0) {
            cfg.n_threads = threads_override;
        } else {
            unsigned hw = std::thread::hardware_concurrency();
            cfg.n_threads = hw > 0 ? (int)hw : 4;
        }
        std::cerr << "[Hardware] using " << cfg.n_threads << " CPU thread(s)"
                  << (threads_override > 0 ? " (--threads override)\n" : " (auto-detected)\n");

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
        VLOG("probing ROCm devices");
        auto& gpu = rocm::GpuContext::get();
        if (gpu.available()) {
            std::cerr << "[Hardware] ROCm GPU: " << gpu.n_devices << " device(s)\n";
        } else {
            std::cerr << "[Hardware] ROCm: no devices found, running CPU-only.\n";
        }
#endif

#ifdef USE_NUMA
        util::ThreadPool pool(cfg.n_threads, /*numa_bind=*/true);
        std::cerr << "[Topology] NUMA-aware thread pool (" << cfg.n_threads << " threads)\n";
#else
        util::ThreadPool pool(cfg.n_threads, false);
#endif

        // ── Format dispatch ─────────────────────────────────────────────
        ModelFormat fmt = sniff_format(model_path);
        switch (fmt) {
            case ModelFormat::GGUF:
                std::cerr << "[Format] GGUF\n";
                return run_model<loader::GGUFLoader>(
                    model_path, cfg, pool, probe_only, interactive, input_payload, idle, t_start);
            case ModelFormat::NCTR:
                std::cerr << "[Format] NCTR\n";
                return run_model<loader::NCTRLoader>(
                    model_path, cfg, pool, probe_only, interactive, input_payload, idle, t_start);
        }
        return 1;   // unreachable — silences -Wreturn-type on some compilers

    } catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL ERROR] " << e.what() << "\n";
        return 1;
    }
}