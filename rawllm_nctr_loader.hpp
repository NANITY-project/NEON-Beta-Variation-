#pragma once
// =============================================================================
// rawllm_nctr_loader.hpp  —  NCTR mmap reader (fixed-header, NANITY-hardcoded)
//
// Companion to rawllm_loader.hpp's GGUFLoader. Unlike GGUF, .nctr has no
// generic tagged key-value stream to parse: the header below IS engine::Config,
// read positionally in one memcpy, and the tensor table has no name strings —
// tensor identity is implied by position (NANITY_ARCHITECTURE_SPEC.md §4
// ordering: token_embd, output_norm, output?, then per-layer blocks).
//
// Reuses loader::TensorInfo / loader::TokenizerMeta / loader::GGMLType /
// loader::GGUFLoader::bytes_for_type from rawllm_loader.hpp so the rest of
// the runtime doesn't need a second set of types — only a second loader
// class with the same public shape (tensors, tok_meta, validate_config()).
//
// NOT YET WIRED IN. rawllm_forward.hpp and NEON.cpp still take
// `const loader::GGUFLoader&` concretely at several call sites
// (ModelWeights::find_exact/build, transformer_forward, run_generation,
// generate, run_interactive). This header alone does not make
// `--model foo.nctr` work — those call sites need templating on the loader
// type first. See the integration notes at the bottom of this file.
// =============================================================================
#include "rawllm_common.hpp"
#include "rawllm_loader.hpp"   // TensorInfo, TokenizerMeta, GGMLType, GGUFLoader::bytes_for_type
#include "rawllm_json.hpp"

namespace loader {

// Fixed positional header — mirrors engine::Config directly, no KV tags.
// See NANITY_ARCHITECTURE_SPEC.md's "container format" section for the
// authoritative field list; keep the two in sync if either changes.
#pragma pack(push, 1)
struct NCTRHeader {
    char     magic[4];              // "NCTR"
    uint32_t format_version;        // container version (this file targets: 1)
    uint32_t nanity_spec_version;   // must equal engine::kSpecVersion
    uint32_t n_vocab;
    uint32_t n_embd;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_kv_head;
    uint32_t head_dim;
    uint32_t n_ff;
    uint32_t ctx_len;
    float    rope_base;
    float    rope_scale;
    uint32_t rope_dim_count;
    float    norm_eps;
    uint32_t flags;                 // bit0=use_swiglu, bit1=has_output_weight, bit2=has_optimizer_state
    uint32_t alignment;
    uint64_t manifest_offset;
    uint64_t manifest_length;
    uint64_t tokenizer_offset;
    uint64_t tokenizer_length;
    uint64_t tensor_table_offset;
    uint64_t data_section_offset;
};
#pragma pack(pop)

static_assert(sizeof(NCTRHeader) == 116,
    "NCTRHeader layout drifted from the spec doc — update "
    "NANITY_ARCHITECTURE_SPEC.md's container-format section alongside this struct");

enum : uint32_t {
    NCTR_FLAG_USE_SWIGLU        = 1u << 0,
    NCTR_FLAG_HAS_OUTPUT_WEIGHT = 1u << 1,
    NCTR_FLAG_HAS_OPT_STATE     = 1u << 2,
};

// Each tensor-table slot on disk, in file order: {quant_type:u32, data_offset:u64, nbytes:u64}
constexpr size_t kNctrTensorSlotBytes = 20;

class NCTRLoader {
public:
    std::vector<TensorInfo> tensors;
    TokenizerMeta            tok_meta;
    std::string              manifest_json;   // raw manifest text (see prior manifest schema), exposed as-is

    NCTRLoader()  = default;
    ~NCTRLoader() {
        if (mapped_ != MAP_FAILED && mapped_) munmap(mapped_, file_size_);
        if (fd_ >= 0) close(fd_);
    }
    NCTRLoader(const NCTRLoader&)            = delete;
    NCTRLoader& operator=(const NCTRLoader&) = delete;

    void open(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("Cannot open model: " + path);
        struct stat st{}; fstat(fd_, &st); file_size_ = (size_t)st.st_size;
        if (file_size_ < sizeof(NCTRHeader))
            throw std::runtime_error("File too small to be NCTR (<header size): " + path);
        mapped_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped_ == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
        // Same readahead/pin treatment as GGUFLoader::open() — see that
        // file's comment for why MADV_WILLNEED + a forced touch pass beats
        // MADV_SEQUENTIAL for a file every decode step re-reads in full.
#if defined(MADV_WILLNEED)
        madvise(mapped_, file_size_, MADV_WILLNEED);
#endif
#if defined(MADV_HUGEPAGE)
        madvise(mapped_, file_size_, MADV_HUGEPAGE);
#endif
        {
            volatile uint8_t sink = 0;
            const uint8_t* base = static_cast<const uint8_t*>(mapped_);
            constexpr size_t kPage = 4096;
            for (size_t off = 0; off < file_size_; off += kPage) sink ^= base[off];
            (void)sink;
        }
        mlock(mapped_, file_size_);   // best-effort; ignored failure, see GGUFLoader::open()

        const uint8_t* base = static_cast<const uint8_t*>(mapped_);
        std::memcpy(&hdr_, base, sizeof(NCTRHeader));

        if (std::memcmp(hdr_.magic, "NCTR", 4) != 0)
            throw std::runtime_error("Not an NCTR file: " + path);
        if (hdr_.nanity_spec_version != engine::kSpecVersion)
            throw std::runtime_error("NCTR: file declares spec_version " +
                std::to_string(hdr_.nanity_spec_version) + ", runtime built against " +
                std::to_string(engine::kSpecVersion));

        auto in_bounds = [&](uint64_t off, uint64_t len) {
            if (off > file_size_ || len > file_size_ - off)
                throw std::runtime_error("NCTR: section out of bounds (offset=" +
                    std::to_string(off) + " len=" + std::to_string(len) +
                    " file_size=" + std::to_string(file_size_) + ")");
        };
        in_bounds(hdr_.manifest_offset,  hdr_.manifest_length);
        in_bounds(hdr_.tokenizer_offset, hdr_.tokenizer_length);

        manifest_json.assign(reinterpret_cast<const char*>(base + hdr_.manifest_offset),
                              hdr_.manifest_length);

        parse_tokenizer_section(base);
        build_tensor_table(base, in_bounds);
    }

    // Same contract as GGUFLoader::validate_config(): fills the rest of cfg
    // and does the invariant checks that are genuinely architecture-level.
    // GGUFLoader's version spends most of its code defending against a
    // generic container where any field could be missing or mis-shaped;
    // here the header IS the declaration, so there's far less to check —
    // mainly "does the tensor count the header implies match what's
    // actually on disk," as a sanity check against a truncated/corrupt file.
    bool validate_config(engine::Config& cfg) const {
        cfg.n_vocab        = hdr_.n_vocab;
        cfg.n_embd         = hdr_.n_embd;
        cfg.n_layer        = hdr_.n_layer;
        cfg.n_head         = hdr_.n_head;
        cfg.n_kv_head      = hdr_.n_kv_head;
        cfg.head_dim       = hdr_.head_dim;
        cfg.n_ff           = hdr_.n_ff;
        cfg.ctx_len        = hdr_.ctx_len;
        cfg.rope_base      = hdr_.rope_base;
        cfg.rope_scale     = hdr_.rope_scale;
        cfg.rope_dim_count = hdr_.rope_dim_count;
        cfg.norm_eps       = hdr_.norm_eps;
        cfg.use_swiglu     = (hdr_.flags & NCTR_FLAG_USE_SWIGLU) != 0;

        if (cfg.n_head == 0 || cfg.n_kv_head == 0 || cfg.n_head % cfg.n_kv_head != 0)
            throw std::runtime_error("NCTR: n_head must be a nonzero multiple of n_kv_head (got " +
                std::to_string(cfg.n_head) + " / " + std::to_string(cfg.n_kv_head) + ")");

        const bool has_output = (hdr_.flags & NCTR_FLAG_HAS_OUTPUT_WEIGHT) != 0;
        const bool swiglu     = (hdr_.flags & NCTR_FLAG_USE_SWIGLU) != 0;
        // Per layer: attn_norm, attn_q, attn_k, attn_v, attn_output,
        // ffn_norm, ffn_up, ffn_down = 8 base tensors, +1 for ffn_gate
        // when SwiGLU is on. (Verified against build_tensor_table()'s
        // actual slot list by test_nctr_loader.cpp, not just re-derived —
        // an earlier version of this had that at 7 and it was wrong.)
        size_t expected = 2 /*token_embd, output_norm*/
                        + (has_output ? 1 : 0)
                        + (size_t)cfg.n_layer * (8 + (swiglu ? 1 : 0));
        if (tensors.size() != expected)
            throw std::runtime_error("NCTR: tensor table has " + std::to_string(tensors.size()) +
                " entries, header (n_layer=" + std::to_string(cfg.n_layer) +
                ", flags) implies " + std::to_string(expected));

        if (cfg.kv_window == 0)
            cfg.kv_window = std::min(cfg.ctx_len, 4096u);

        return true;
    }

    bool has_optimizer_state() const { return (hdr_.flags & NCTR_FLAG_HAS_OPT_STATE) != 0; }

    static size_t bytes_for_type(GGMLType t, size_t n) {
        return GGUFLoader::bytes_for_type(t, n);   // same GGML quant types, same byte math — no reason to duplicate the table
    }

private:
    int fd_ = -1; size_t file_size_ = 0; void* mapped_ = MAP_FAILED;
    NCTRHeader hdr_{};

    void parse_tokenizer_section(const uint8_t* base) {
        std::string blob(reinterpret_cast<const char*>(base + hdr_.tokenizer_offset),
                          hdr_.tokenizer_length);
        // Stored as plain JSON, not a bespoke binary layout — same reasoning
        // as the manifest: this section is KB-scale, so there's no
        // performance case for a custom encoding, and JSON keeps it
        // greppable/diffable. Shape: {"tokens":[...], "merges":[...],
        // "bos_id":N, "eos_id":N, "unk_id":N} — deliberately mirrors what
        // GGUFLoader pulls from tokenizer.ggml.* today.
        auto v = json::parse(blob);
        const auto& toks = v["tokens"];
        tok_meta.tokens.reserve(toks.size());
        for (size_t i = 0; i < toks.size(); ++i) tok_meta.tokens.push_back(toks[i].get_str());
        const auto& merges = v["merges"];
        tok_meta.merges.reserve(merges.size());
        for (size_t i = 0; i < merges.size(); ++i) tok_meta.merges.push_back(merges[i].get_str());
        tok_meta.bos_id = v["bos_id"].get_int(1);
        tok_meta.eos_id = v["eos_id"].get_int(2);
        tok_meta.unk_id = v["unk_id"].get_int(0);
    }

    template <typename BoundsCheck>
    void build_tensor_table(const uint8_t* base, BoundsCheck&& in_bounds) {
        const bool swiglu     = (hdr_.flags & NCTR_FLAG_USE_SWIGLU) != 0;
        const bool has_output = (hdr_.flags & NCTR_FLAG_HAS_OUTPUT_WEIGHT) != 0;
        const uint32_t q_dim  = hdr_.n_head    * hdr_.head_dim;
        const uint32_t kv_dim = hdr_.n_kv_head * hdr_.head_dim;

        struct Slot { std::string name; std::vector<uint64_t> shape; };
        std::vector<Slot> slots;
        slots.push_back({"token_embd.weight",  {hdr_.n_embd, hdr_.n_vocab}});
        slots.push_back({"output_norm.weight", {hdr_.n_embd}});
        if (has_output) slots.push_back({"output.weight", {hdr_.n_embd, hdr_.n_vocab}});
        for (uint32_t i = 0; i < hdr_.n_layer; ++i) {
            std::string p = "blk." + std::to_string(i) + ".";
            slots.push_back({p+"attn_norm.weight",   {hdr_.n_embd}});
            slots.push_back({p+"attn_q.weight",      {hdr_.n_embd, q_dim}});
            slots.push_back({p+"attn_k.weight",      {hdr_.n_embd, kv_dim}});
            slots.push_back({p+"attn_v.weight",      {hdr_.n_embd, kv_dim}});
            slots.push_back({p+"attn_output.weight", {q_dim, hdr_.n_embd}});
            slots.push_back({p+"ffn_norm.weight",    {hdr_.n_embd}});
            if (swiglu) slots.push_back({p+"ffn_gate.weight", {hdr_.n_embd, hdr_.n_ff}});
            slots.push_back({p+"ffn_up.weight",      {hdr_.n_embd, hdr_.n_ff}});
            slots.push_back({p+"ffn_down.weight",    {hdr_.n_ff, hdr_.n_embd}});
        }

        in_bounds(hdr_.tensor_table_offset, slots.size() * kNctrTensorSlotBytes);
        const uint8_t* table = base + hdr_.tensor_table_offset;

        tensors.reserve(slots.size());
        for (size_t i = 0; i < slots.size(); ++i) {
            uint32_t quant_type; uint64_t data_offset, nbytes;
            std::memcpy(&quant_type,  table + i*kNctrTensorSlotBytes,      4);
            std::memcpy(&data_offset, table + i*kNctrTensorSlotBytes + 4,  8);
            std::memcpy(&nbytes,      table + i*kNctrTensorSlotBytes + 12, 8);

            uint64_t abs_offset = hdr_.data_section_offset + data_offset;
            in_bounds(abs_offset, nbytes);

            TensorInfo t;
            t.name        = slots[i].name;
            t.shape       = slots[i].shape;
            t.type        = static_cast<GGMLType>(quant_type);
            t.file_offset = abs_offset;
            t.nbytes      = nbytes;
            t.data_ptr    = base + abs_offset;
            tensors.push_back(std::move(t));
        }
    }
};

} // namespace loader

// =============================================================================
// Integration notes — not done yet, tracked here so the next pass has a
// checklist instead of a fresh read-through:
//
// 1. rawllm_forward.hpp: ModelWeights::find_exact()/build() (~L836-841) and
//    transformer_forward() (~L974) take `const loader::GGUFLoader&`
//    concretely. Template them on `typename Loader` — the only member
//    accesses are .tensors, .tok_meta, and the static bytes_for_type(), all
//    of which NCTRLoader now exposes with identical signatures.
//
// 2. NEON.cpp: run_generation() (~L657), generate() (~L745),
//    run_interactive() (~L782) take `const loader::GGUFLoader&` concretely
//    for the same reason. Same fix: template on Loader.
//
// 3. main(): wrap the load→validate→probe→tokenizer→generate sequence
//    (~L911-988) in `template<typename Loader> int run_model(...)`, sniff
//    the file's first 4 bytes ("GGUF" vs "NCTR"), and dispatch to
//    run_model<loader::GGUFLoader> or run_model<loader::NCTRLoader>.
//
// None of this touches the NANITY compute graph or quantization handling —
// only which bytes get read off disk into the TensorInfo/Config/
// TokenizerMeta structures the forward pass already consumes.
// =============================================================================
