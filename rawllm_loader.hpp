#pragma once
// =============================================================================
// rawllm_loader.hpp  —  GGUF mmap reader
//
// The GGUF container parsing below (open(), parse_kv, parse_array, the
// quant-type byte-size table) is format-level mechanics and is unchanged —
// it has nothing to do with which model architecture a file contains.
//
// validate_config() is the architecture-specific part, and it works
// differently than it used to: this engine targets exactly one fixed,
// published architecture (NANITY — see NANITY_ARCHITECTURE_SPEC.md), not
// "whatever GGUF someone hands us." A conforming file declares
// general.architecture="nanity" plus an explicit nanity.* hyperparameter
// block, so loading is validation against a known contract rather than
// shape-based guesswork across however many unrelated model families.
// Metadata parsing still stores all general.*/nanity.* keys (not just
// tokenizer.*) so validate_config() can read them.
// =============================================================================
#include "rawllm_common.hpp"

namespace loader {

enum class GGMLType : uint32_t {
    F32  = 0, F16  = 1,
    Q4_0 = 2, Q4_1 = 3,
    Q5_0 = 6, Q5_1 = 7,
    Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14,
    IQ4_NL = 25,
    BF16   = 30,
    COUNT  = 31,
};

struct TensorInfo {
    std::string           name;
    std::vector<uint64_t> shape;
    GGMLType              type;
    uint64_t              file_offset = 0;
    size_t                nbytes      = 0;
    const uint8_t*        data_ptr    = nullptr;
};

struct TokenizerMeta {
    std::vector<std::string> tokens;
    std::vector<std::string> merges;     // FIX: was parsed off the wire then discarded
    std::vector<float>       scores;
    std::vector<uint32_t>    token_types;
    int32_t bos_id = 1, eos_id = 2, unk_id = 0;
};

struct Cursor {
    const uint8_t* base; size_t pos; size_t size;
    template<typename T> T rd() {
        if (pos + sizeof(T) > size) throw std::runtime_error("GGUF: unexpected EOF");
        T v; std::memcpy(&v, base + pos, sizeof(T)); pos += sizeof(T); return v;
    }
    std::string rd_str() {
        uint64_t len = rd<uint64_t>();
        if (pos + len > size) throw std::runtime_error("GGUF: string overflow");
        std::string s(reinterpret_cast<const char*>(base + pos), len);
        pos += len; return s;
    }
    void skip(size_t n) { if (pos+n>size) throw std::runtime_error("GGUF: skip OOB"); pos+=n; }
};

enum class MetaType : uint32_t {
    UINT8=0,INT8,UINT16,INT16,UINT32,INT32,FLOAT32,BOOL,STRING,
    ARRAY,UINT64,INT64,FLOAT64
};

class GGUFLoader {
public:
    std::vector<TensorInfo> tensors;
    TokenizerMeta            tok_meta;

    GGUFLoader()  = default;
    ~GGUFLoader() {
        if (mapped_ != MAP_FAILED && mapped_) munmap(mapped_, file_size_);
        if (fd_ >= 0) close(fd_);
    }
    GGUFLoader(const GGUFLoader&)            = delete;
    GGUFLoader& operator=(const GGUFLoader&) = delete;

    void open(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("Cannot open model: " + path);
        struct stat st{}; fstat(fd_, &st); file_size_ = (size_t)st.st_size;
        // FIX: mmap(length=0) is UB; memcpy of 4-byte magic from a <4-byte
        // mapping reads past the mapped region.  16 bytes is the minimum needed
        // to hold magic(4)+version(4)+the two uint64 counts(8) for v2/v3.
        if (file_size_ < 16)
            throw std::runtime_error("File too small to be GGUF (<16 bytes): " + path);
        mapped_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped_ == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
        // FIX (perf — this was very likely THE dominant cause of multi-second
        // per-token latency on RAM-constrained machines): MADV_SEQUENTIAL's
        // actual Linux semantics are "aggressively read ahead, AND free
        // pages soon after they're accessed." That's the right hint for a
        // true single-pass scan, but it's the *wrong* hint here: every
        // single decode step re-reads essentially the whole weight file —
        // a GEMV has to touch every weight once per token, every token —
        // so the same pages are revisited every ~token, not once. With
        // SEQUENTIAL, the kernel drops each page behind the read cursor
        // assuming it won't be needed again, so on a box where the model
        // doesn't comfortably fit in spare RAM, every decode step degrades
        // into a fresh read from disk instead of a page-cache hit — turning
        // a ~0.1s, RAM-bandwidth-bound token into a multi-second, disk-
        // bandwidth-bound one. MADV_WILLNEED below requests one eager,
        // efficient readahead of the whole mapping at load time, with none
        // of the drop-behind behavior; ordinary kernel LRU eviction then
        // governs what stays resident afterward.
#if defined(MADV_WILLNEED)
        madvise(mapped_, file_size_, MADV_WILLNEED);
#endif
#if defined(MADV_HUGEPAGE)
        madvise(mapped_, file_size_, MADV_HUGEPAGE);
#endif
        // Belt-and-suspenders: some kernels/filesystems treat MADV_WILLNEED
        // as a soft hint and don't actually trigger readahead synchronously.
        // Touching one byte per 4KB page forces every page in unconditionally,
        // right now, at load time — so the (unavoidable) cost of pulling this
        // file off disk is paid exactly ONCE, at boot, instead of silently
        // being paid again on every decode step if a page ever gets evicted.
        // This is a plain sequential read at disk speed, same total bytes
        // either way — it just moves the cost to a place where the user
        // sees one upfront "loading model..." pause instead of a slow token
        // every single time.
        {
            volatile uint8_t sink = 0;
            const uint8_t* base = static_cast<const uint8_t*>(mapped_);
            constexpr size_t kPage = 4096;
            for (size_t off = 0; off < file_size_; off += kPage) sink ^= base[off];
            (void)sink;
        }
        // Best-effort: ask the kernel to pin the mapping in RAM so pages
        // already paged in can never be evicted again. mlock() commonly
        // fails with ENOMEM/EPERM under the default RLIMIT_MEMLOCK (often
        // just 64KB) — that failure is deliberately ignored here, not
        // fatal: it just means we fall back to ordinary demand-paged
        // behavior instead of a hard pin. Raise the limit (`ulimit -l
        // unlimited`, or the equivalent rlimit/capability) before launching
        // NEON if you want this to actually take effect on a multi-GB model.
        mlock(mapped_, file_size_);
        Cursor c{ static_cast<const uint8_t*>(mapped_), 0, file_size_ };
        char magic[4]; std::memcpy(magic, c.base, 4); c.pos=4;
        if (std::memcmp(magic, "GGUF", 4) != 0)
            throw std::runtime_error("Not a GGUF file: " + path);
        uint32_t ver = c.rd<uint32_t>();
        if (ver < 2 || ver > 3) throw std::runtime_error("Unsupported GGUF version");
        uint64_t n_tensors = c.rd<uint64_t>(), n_kv = c.rd<uint64_t>();
        for (uint64_t i = 0; i < n_kv; ++i) parse_kv(c);
        tensors.resize(n_tensors);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            auto& t = tensors[i]; t.name = c.rd_str();
            uint32_t nd = c.rd<uint32_t>(); t.shape.resize(nd);
            for (uint32_t d = 0; d < nd; ++d) t.shape[d] = c.rd<uint64_t>();
            t.type = static_cast<GGMLType>(c.rd<uint32_t>());
            t.file_offset = c.rd<uint64_t>();
        }
        // FIX: alignment was hardcoded to 32 bytes; the GGUF spec allows files to
        // override this via the optional general.alignment metadata key.
        uint64_t alignment = 32;
        {
            auto align_it = meta_num.find("general.alignment");
            if (align_it != meta_num.end() && align_it->second > 0)
                alignment = (uint64_t)align_it->second;
        }
        size_t data_start = ((c.pos + alignment - 1) / alignment) * alignment;
        const uint8_t* data_base = static_cast<const uint8_t*>(mapped_) + data_start;
        for (auto& t : tensors) {
            size_t n_elem = 1; for (auto d : t.shape) n_elem *= d;
            t.nbytes = nbytes_for(t.type, n_elem);
            if (data_start + t.file_offset + t.nbytes > file_size_)
                throw std::runtime_error("Tensor '" + t.name + "' OOB");
            t.data_ptr = data_base + t.file_offset;
        }
    }

// ── NANITY spec validation ─────────────────────────────────────────────
    // REPLACES the old shape-guessing detect_config(). This engine no longer
    // tries to support arbitrary GGUF architectures by inferring head_dim /
    // fused-QKV / fused-gate_up from tensor shapes -- that approach required
    // an ever-growing pile of per-family heuristics and still silently
    // mis-detected anything outside the families it had been taught about.
    //
    // NANITY is a fixed, published architecture (NANITY_ARCHITECTURE_SPEC.md):
    // every conforming file declares general.architecture="nanity" and an
    // explicit nanity.* hyperparameter block, so there is nothing left to
    // *guess* -- only to *validate*. Tensor shapes are still cross-checked
    // against the declared hyperparameters, so a file that lies about itself
    // fails loudly here, with a precise diff against the spec, rather than
    // producing garbage three layers into a forward pass.
    //
    // Throws std::runtime_error on any mismatch. Returns true on success
    // (kept bool-returning, rather than void, so call sites that only care
    // about success/failure don't need to change).
    bool validate_config(engine::Config& cfg) const {
        auto require_str = [&](const char* key) -> const std::string& {
            auto it = meta_str.find(key);
            if (it == meta_str.end() || it->second.empty())
                throw std::runtime_error(
                    std::string("NANITY: missing required metadata string key '") + key + "'.");
            return it->second;
        };
        auto require_num = [&](const std::string& key) -> double {
            auto it = meta_num.find(key);
            if (it == meta_num.end())
                throw std::runtime_error("NANITY: missing required metadata key '" + key + "'.");
            return it->second;
        };
        auto opt_num = [&](const std::string& key, double def) -> double {
            auto it = meta_num.find(key);
            return (it != meta_num.end()) ? it->second : def;
        };

        // -- Architecture + spec-version gate --
        // Per spec §0/§11: a file is either NANITY or it isn't. No synonym
        // architectures, no fallback prefixes, no warning-and-continue --
        // anything other than an exact match is simply not a NANITY model.
        const std::string& arch_str = require_str("general.architecture");
        if (arch_str != engine::kArchName)
            throw std::runtime_error(
                "NANITY: general.architecture=\"" + arch_str + "\", expected \"" +
                engine::kArchName + "\" -- this file is not a NANITY model.");

        uint32_t spec_ver = (uint32_t)require_num("nanity.spec_version");
        if (spec_ver != engine::kSpecVersion)
            throw std::runtime_error(
                "NANITY: nanity.spec_version=" + std::to_string(spec_ver) +
                ", this runtime only supports version " +
                std::to_string(engine::kSpecVersion) + ".");

        // -- Required hyperparameters (§3) --
        // Spec keys only. No dynamic-prefix scanning: a NANITY file declares
        // nanity.* directly, so there is nothing left to look up elsewhere.
        cfg.n_embd    = (uint32_t)require_num("nanity.embedding_length");
        cfg.n_layer   = (uint32_t)require_num("nanity.block_count");
        cfg.n_head    = (uint32_t)require_num("nanity.attention.head_count");
        cfg.n_kv_head = (uint32_t)require_num("nanity.attention.head_count_kv");
        cfg.head_dim  = (uint32_t)require_num("nanity.attention.key_length");
        cfg.n_ff      = (uint32_t)require_num("nanity.feed_forward_length");
        cfg.ctx_len   = (uint32_t)require_num("nanity.context_length");

        // -- Optional hyperparameters (§3, sane defaults) --
        cfg.norm_eps   = (float)opt_num("nanity.attention.layer_norm_rms_epsilon", cfg.norm_eps);
        cfg.rope_base  = (float)opt_num("nanity.rope.freq_base",                  cfg.rope_base);
        cfg.rope_scale = (float)opt_num("nanity.rope.scale_linear",               cfg.rope_scale);
        cfg.rope_dim_count = (uint32_t)opt_num("nanity.rope.dimension_count",     0.0);

        if (cfg.n_embd == 0 || cfg.n_layer == 0 || cfg.n_head == 0 ||
            cfg.n_kv_head == 0 || cfg.head_dim == 0 || cfg.n_ff == 0)
            throw std::runtime_error("NANITY: one or more required hyperparameters is zero.");
        if (cfg.n_head % cfg.n_kv_head != 0)
            throw std::runtime_error(
                "NANITY: attention.head_count (" + std::to_string(cfg.n_head) +
                ") must be an exact multiple of attention.head_count_kv (" +
                std::to_string(cfg.n_kv_head) + ") -- GQA grouping requires it.");

        // SwiGLU with separate gate/up/down is the only FFN variant in spec
        // v1 (§1, §4) -- every layer has all three, unconditionally.
        cfg.use_swiglu = true;

        const uint64_t q_dim  = (uint64_t)cfg.n_head    * cfg.head_dim;
        const uint64_t kv_dim = (uint64_t)cfg.n_kv_head * cfg.head_dim;

        // -- Cross-validate every required tensor's shape (§4) --
        std::unordered_map<std::string, const TensorInfo*> tmap;
        for (const auto& t : tensors) tmap[t.name] = &t;

        auto find_req = [&](const std::string& name) -> const TensorInfo& {
            auto it = tmap.find(name);
            if (it == tmap.end())
                throw std::runtime_error("NANITY: required tensor '" + name + "' is missing.");
            return *it->second;
        };
        auto check2d = [&](const std::string& name, uint64_t e0, uint64_t e1) {
            const auto& sh = find_req(name).shape;
            if (sh.size() != 2 || sh[0] != e0 || sh[1] != e1)
                throw std::runtime_error("NANITY: tensor '" + name + "' has shape " +
                    shape_str(sh) + ", expected [" + std::to_string(e0) + ", " +
                    std::to_string(e1) + "].");
        };
        auto check1d = [&](const std::string& name, uint64_t e0) {
            const auto& sh = find_req(name).shape;
            if (sh.size() != 1 || sh[0] != e0)
                throw std::runtime_error("NANITY: tensor '" + name + "' has shape " +
                    shape_str(sh) + ", expected [" + std::to_string(e0) + "].");
        };

        // token_embd carries n_vocab -- the one model property the spec
        // doesn't pre-declare in metadata, since it's a property of the
        // tokenizer/training data, not of the architecture itself.
        const auto& te = find_req("token_embd.weight");
        if (te.shape.size() != 2 || te.shape[0] != cfg.n_embd)
            throw std::runtime_error(
                "NANITY: 'token_embd.weight' shape " + shape_str(te.shape) +
                " is inconsistent with nanity.embedding_length=" +
                std::to_string(cfg.n_embd) + ".");
        cfg.n_vocab = (uint32_t)te.shape[1];
        {
            auto vv = meta_num.find("nanity.vocab_size");
            if (vv != meta_num.end() && (uint32_t)vv->second != cfg.n_vocab)
                throw std::runtime_error(
                    "NANITY: nanity.vocab_size=" + std::to_string((uint32_t)vv->second) +
                    " disagrees with token_embd.weight's vocab dimension (" +
                    std::to_string(cfg.n_vocab) + ").");
        }

        check1d("output_norm.weight", cfg.n_embd);
        // output.weight is OPTIONAL by spec: its absence means tied
        // embeddings (the output projection reuses token_embd.weight).
        if (auto it = tmap.find("output.weight"); it != tmap.end()) {
            const auto& sh = it->second->shape;
            if (sh.size() != 2 || sh[0] != cfg.n_embd || sh[1] != cfg.n_vocab)
                throw std::runtime_error(
                    "NANITY: 'output.weight' shape " + shape_str(sh) +
                    " doesn't match [embedding_length, vocab_size] = [" +
                    std::to_string(cfg.n_embd) + ", " + std::to_string(cfg.n_vocab) + "].");
        }

        for (uint32_t i = 0; i < cfg.n_layer; ++i) {
            std::string p = "blk." + std::to_string(i) + ".";
            check1d(p + "attn_norm.weight",   cfg.n_embd);
            check2d(p + "attn_q.weight",      cfg.n_embd, q_dim);
            check2d(p + "attn_k.weight",      cfg.n_embd, kv_dim);
            check2d(p + "attn_v.weight",      cfg.n_embd, kv_dim);
            check2d(p + "attn_output.weight", q_dim,      cfg.n_embd);
            check1d(p + "ffn_norm.weight",    cfg.n_embd);
            check2d(p + "ffn_gate.weight",    cfg.n_embd, cfg.n_ff);
            check2d(p + "ffn_up.weight",      cfg.n_embd, cfg.n_ff);
            check2d(p + "ffn_down.weight",    cfg.n_ff,   cfg.n_embd);
        }

        if (cfg.kv_window == 0)
            cfg.kv_window = std::min(cfg.ctx_len, 4096u);

        return true;
    }
    // Backward-compatible alias -- NEON.cpp and any external callers should
    // migrate to validate_config(), which is the same function under a name
    // that reflects what it actually does now (validate against a known
    // spec, not detect an unknown one).
    bool detect_config(engine::Config& cfg) const { return validate_config(cfg); }

    // FIX: exposed so rawllm_forward.hpp can compute a quantized row's byte
    // stride (row_bytes = bytes_for_type(type, cols)) without duplicating
    // the block-size formulas here — single source of truth for layout math.
    static size_t bytes_for_type(GGMLType t, size_t n) { return nbytes_for(t, n); }

    // Pretty-print a tensor shape for validate_config()'s error messages,
    // e.g. {4096, 32000} -> "[4096, 32000]".
    static std::string shape_str(const std::vector<uint64_t>& shape) {
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) s += ", ";
            s += std::to_string(shape[i]);
        }
        s += "]";
        return s;
    }

private:
    int fd_=-1; size_t file_size_=0; void* mapped_=MAP_FAILED;
    std::unordered_map<std::string, double>      meta_num;
    std::unordered_map<std::string, std::string>  meta_str;

    static size_t nbytes_for(GGMLType t, size_t n) {
        switch (t) {
            case GGMLType::F32:  return n*4;
            case GGMLType::F16:  return n*2;
            case GGMLType::BF16: return n*2;
            case GGMLType::Q8_0: return (n/32)*(2+32);
            case GGMLType::Q8_1: return (n/32)*(2+2+32);          // FIX: was missing (fell through to default→0)
            case GGMLType::Q4_0: return (n/32)*(2+16);
            case GGMLType::Q4_1: return (n/32)*(2+2+16);
            case GGMLType::Q5_0: return (n/32)*(2+4+16);
            case GGMLType::Q5_1: return (n/32)*(2+2+4+16);
            case GGMLType::Q4_K: return (n/256)*(2+2+12+128);
            case GGMLType::Q5_K: return (n/256)*(2+2+12+32+128);  // FIX: was (2+2+12+4+160)=180; real block_q5_K = d+dmin+scales[12]+qh[32]+qs[128] = 176
            case GGMLType::Q6_K: return (n/256)*(2+128+64+16);
            case GGMLType::Q2_K: return (n/256)*(2+2+16+64);
            case GGMLType::Q3_K: return (n/256)*(2+12+64+32);     // FIX: was (2+12+64+96)=174; real block_q3_K = hmask[32]+qs[64]+scales[12]+d[2] = 110
            case GGMLType::IQ4_NL: return (n/32)*(2+16);          // FIX: was missing (fell through to default→0)
            default:             return 0;
        }
    }

    void parse_kv(Cursor& c) {
        std::string key = c.rd_str();
        auto type = static_cast<MetaType>(c.rd<uint32_t>());
        bool is_tok = (key.size() >= 10 && key.substr(0,10) == "tokenizer.");
        bool is_meta = (!is_tok);  // capture llama.* and general.* for config
        switch (type) {
            case MetaType::UINT8:  { auto v=c.rd<uint8_t>();  if(is_tok)store_u32(key,v); if(is_meta)meta_num[key]=(double)v; break;}
            case MetaType::INT8:   { auto v=c.rd<int8_t>();   if(is_tok)store_u32(key,(uint32_t)v); if(is_meta)meta_num[key]=(double)v; break;}
            case MetaType::UINT16: { auto v=c.rd<uint16_t>(); if(is_tok)store_u32(key,v); if(is_meta)meta_num[key]=(double)v; break;}
            case MetaType::INT16:  { auto v=c.rd<int16_t>();  if(is_tok)store_u32(key,(uint32_t)v); if(is_meta)meta_num[key]=(double)v; break;}
            case MetaType::UINT32: { auto v=c.rd<uint32_t>(); if(is_tok)store_u32(key,v); if(is_meta)meta_num[key]=(double)v; break;}
            case MetaType::INT32:  { auto v=c.rd<int32_t>();  if(is_tok)store_u32(key,(uint32_t)v); if(is_meta)meta_num[key]=(double)v; break;}
            case MetaType::UINT64: { auto v=c.rd<uint64_t>(); if(is_tok)store_u32(key,(uint32_t)v); if(is_meta)meta_num[key]=(double)v; break;}
            case MetaType::INT64:  { auto v=c.rd<int64_t>();  if(is_tok)store_u32(key,(uint32_t)v); if(is_meta)meta_num[key]=(double)v; break;}
            case MetaType::FLOAT32:{ auto v=c.rd<float>();    if(is_meta)meta_num[key]=(double)v; break;}
            case MetaType::FLOAT64:{ auto v=c.rd<double>();   if(is_meta)meta_num[key]=v; break;}
            case MetaType::BOOL:   { auto v=c.rd<uint8_t>(); if(is_meta)meta_num[key]=(double)(v!=0); break;}
            case MetaType::STRING: { auto s=c.rd_str(); if(is_meta)meta_str[key]=s; break;}
            case MetaType::ARRAY:  {
                auto et=static_cast<MetaType>(c.rd<uint32_t>());
                uint64_t cnt=c.rd<uint64_t>();
                parse_array(c,key,et,cnt,is_tok); break;}
            default: throw std::runtime_error("Unknown GGUF meta type");
        }
    }

    void parse_array(Cursor& c, const std::string& key,
                     MetaType elem, uint64_t count, bool want) {
        if (elem == MetaType::STRING) {
            std::vector<std::string> arr; arr.reserve(want?count:0);
            for (uint64_t i=0;i<count;++i){ auto s=c.rd_str(); if(want) arr.push_back(std::move(s)); }
            if (want && key=="tokenizer.ggml.tokens") tok_meta.tokens=std::move(arr);
            else if (want && key=="tokenizer.ggml.merges") tok_meta.merges=std::move(arr); // FIX: was discarded
            return;
        }
        size_t w=elem_width(elem);
        if (!want){ c.skip(count*w); return; }
        std::vector<uint32_t> arr(count);
        for (uint64_t i=0;i<count;++i) arr[i]=read_u32_as(c,elem);
        if      (key=="tokenizer.ggml.token_type") tok_meta.token_types=std::move(arr);
        else if (key=="tokenizer.ggml.scores") {
            tok_meta.scores.resize(arr.size());
            for (size_t i=0;i<arr.size();++i) std::memcpy(&tok_meta.scores[i],&arr[i],4);
        }
    }

    void store_u32(const std::string& k, uint32_t v) {
        if      (k=="tokenizer.ggml.bos_token_id")     tok_meta.bos_id=(int32_t)v;
        else if (k=="tokenizer.ggml.eos_token_id")     tok_meta.eos_id=(int32_t)v;
        else if (k=="tokenizer.ggml.unknown_token_id") tok_meta.unk_id=(int32_t)v;
    }
    static size_t elem_width(MetaType t){
        switch(t){ case MetaType::UINT8: case MetaType::INT8: case MetaType::BOOL: return 1;
            case MetaType::UINT16: case MetaType::INT16: return 2;
            case MetaType::UINT32: case MetaType::INT32: case MetaType::FLOAT32: return 4;
            default: return 8; }
    }
    static uint32_t read_u32_as(Cursor& c, MetaType t){
        switch(t){
            case MetaType::UINT8:  return c.rd<uint8_t>();
            case MetaType::INT8:   return (uint32_t)c.rd<int8_t>();
            case MetaType::UINT16: return c.rd<uint16_t>();
            case MetaType::INT16:  return (uint32_t)c.rd<int16_t>();
            case MetaType::UINT32: return c.rd<uint32_t>();
            case MetaType::INT32:  return (uint32_t)c.rd<int32_t>();
            case MetaType::FLOAT32:{ uint32_t v; float f=c.rd<float>(); std::memcpy(&v,&f,4); return v; }
            case MetaType::UINT64: return (uint32_t)c.rd<uint64_t>();
            case MetaType::INT64:  return (uint32_t)c.rd<int64_t>();
            case MetaType::FLOAT64:{ c.rd<double>(); return 0; }
            case MetaType::BOOL:   return c.rd<uint8_t>()!=0;
            default: throw std::runtime_error("Unsupported array elem type");
        }
    }
};

} // namespace loader