# NANITY Architecture Specification — v1

**Status:** stable. **Spec version:** `1`. **Runtime:** NEON.

## 0. What this is, and why it exists

NEON used to try to run *any* GGUF model, of *any* architecture, by guessing
its shape from tensor names and dimensions. That worked until it didn't:
every new model family (fused QKV, fused gate/up, a different RoPE
convention, a different bias convention) needed a new heuristic bolted onto
`rawllm_loader.hpp`, and a model that didn't match any known heuristic
failed with "could not detect architecture" — true, but useless.

NANITY inverts that. It is **one specific, fixed transformer architecture**,
published here in enough detail that anyone can:

1. **Train** a model that conforms to it (any framework — a reference
   PyTorch implementation is provided in `modeling_nanity.py`, but the spec
   itself is framework-independent), and
2. **Export** it to a file NEON will load — either a GGUF file or a `.nctr`
   file (§2) — using the exact tensor names and metadata keys below
   (`train_nanity_fixed.py`'s `export_gguf()` / `export_nctr()` do this
   mechanically from a `modeling_nanity.py` checkpoint).

A file that doesn't declare `general.architecture = "nanity"` isn't
rejected by some compatibility check — it simply isn't a NANITY model, in
the same sense that a `.docx` isn't a `.pdf`. This is also why a NANITY
GGUF won't load in llama.cpp or other GGUF runtimes: they have no case for
`general.architecture = "nanity"` in their loaders, just as NEON has no
case for `"llama"` or `"phi3"` anymore. That's not an anti-compatibility
trick — it's just what "a different, openly-specified architecture" means.

## 1. Architecture overview

NANITY is a decoder-only, causal transformer:

- **Normalization:** RMSNorm, pre-norm placement (norm → sublayer → residual
  add), for both the attention and FFN sublayers, plus one final RMSNorm
  before the output projection.
- **Attention:** Grouped-Query Attention (GQA) with **separate** Q, K, V
  projections (no fused QKV tensor — ever). Standard multi-head attention
  (`n_kv_head == n_head`) and multi-query attention (`n_kv_head == 1`) are
  both just special cases of GQA and need no special handling.
- **Position encoding:** Rotary Position Embeddings (RoPE), one fixed
  rotation convention (§6) applied to Q and K before attention.
- **FFN:** SwiGLU, with **separate** gate, up, and down projections (no
  fused gate_up tensor — ever).
- **Bias:** none. No linear layer in a NANITY model has a bias term.
- **Embeddings:** input embedding is always present; the output (LM head)
  projection is **optionally tied** to the input embedding — if a model
  omits `output.weight`, the runtime reuses `token_embd.weight` for the
  output projection.
- **Mixture-of-experts:** out of scope for spec v1. Every block is dense.

This is deliberately a small, conventional design (close to what Llama 2 /
Mistral / Qwen2 use internally) — the novelty isn't exotic math, it's that
the *contract* is explicit and total: there is exactly one shape a
conforming model can take, with no fallback paths, so a NANITY-loading
runtime never has to guess.

## 2. Container format

NANITY models are distributed in one of two binary containers — **GGUF**
or **`.nctr`** — chosen by the publisher, not mandated by the runtime.
NEON determines which one it's looking at by sniffing the file's first 4
bytes (`"GGUF"` or `"NCTR"`), never by trusting the file extension, and
dispatches to the matching loader (`GGUFLoader` in `rawllm_loader.hpp`, or
`NCTRLoader` in `rawllm_nctr_loader.hpp`). Both loaders populate the same
internal tensor/config structures, so everything from §3 onward —
required metadata, required tensors, the computation graph, RoPE,
tokenizer handling, quantization — applies identically regardless of
which container a given file uses. Which container to publish in is a
distribution decision, not an architecture decision, and doesn't touch
`nanity.spec_version` at all (see the versioning note at the end of §2.2).

### 2.1 GGUF

The same binary container format used elsewhere in the GGUF ecosystem
(magic `"GGUF"`, little-endian, mmap-friendly, generic tagged
key-value metadata + named tensor table). NANITY does not redefine the
container; it defines what must be inside one (§3, §4).

Two facts about GGUF's tensor-shape convention matter for §4 below:

- `general.alignment` (optional `uint32`, default `32`) controls the
  byte-alignment of the tensor data section.
- A 2-D tensor's shape array `ne = [ne0, ne1]` lists the **fastest-varying
  (contiguous) dimension first**. A `nn.Linear(in_features, out_features,
  bias=False)` weight, which PyTorch stores row-major with shape
  `(out_features, in_features)`, therefore has `ne = [in_features,
  out_features]` — i.e. `ne` is the *reverse* of the PyTorch shape tuple,
  and the raw bytes are identical either way (just relabeled axes). Every
  shape in §4 is given in this `ne` form. `export_gguf()` handles this
  conversion automatically; you only need to think about it if you're
  writing your own exporter.

GGUF is the right choice for a publisher who wants to share a runnable
model without also publishing how it was trained — the file carries
weights and architecture metadata only. Nothing about the container
requires or implies anything about training provenance.

### 2.2 `.nctr`

A fixed-header container purpose-built for NANITY, used when a publisher
wants the release to be **fully reproducible** — not just "here are the
weights" (which GGUF already covers), but a verifiable account of how
they were produced: what data, what tokenizer, what optimizer settings,
how many steps.

Where GGUF's generic key-value stream exists because GGUF has to support
an open-ended set of architectures, `.nctr` doesn't need it: NANITY is one
fixed architecture, so the header is a **positional struct**, not tagged
KV pairs — read in a single fixed-size copy, no key lookups, no per-value
type dispatch. Tensor identity is likewise positional rather than
name-string based: given `n_layer` and two flag bits (`use_swiglu`,
`has_output_weight`), the full ordered tensor list (§4's table, in order)
is already fully determined, so nothing on disk needs to spell out
`"blk.3.attn_q.weight"` — table position 3 always means the same tensor
§4 says it must be.

**Layout** (see `rawllm_nctr_loader.hpp::NCTRHeader` for the authoritative
byte-for-byte struct):

| Section | Contents |
|---|---|
| Header (116 bytes, fixed offset 0) | Magic `"NCTR"`, a **container `format_version`** (independent of `nanity.spec_version` — see below), every §3 metadata value as a typed field (no key strings), a `flags` word (SwiGLU / tied-output-weight / optimizer-state-present bits), and byte offsets+lengths for the three sections below. |
| Manifest | UTF-8 JSON. Training provenance: dataset recipe (source, revision, content hash — a reference, not the raw data itself), tokenizer/base-weight references (HF repo + revision + content hash), optimizer hyperparameters, step count, a convergence snapshot (final loss, grad norm), and the reproducibility environment (seed, precision, framework version). This is what makes a `.nctr` release reproducible rather than just downloadable. |
| Tokenizer | UTF-8 JSON: `{"tokens": [...], "merges": [...], "bos_id", "eos_id", "unk_id"}` — the same information §7 describes for GGUF's `tokenizer.ggml.*` keys, just JSON instead of GGUF key-value pairs. |
| Tensor table + data | One `{quant_type: u32, data_offset: u64, nbytes: u64}` triple per tensor, in the fixed positional order above, followed by the raw tensor bytes. |

Optionally, a **second** table in the same `{quant_type, offset, nbytes}`
shape carries real AdamW optimizer moment tensors when
`flags.has_optimizer_state` is set. This is what lets someone fork a
model and resume training from an exact point, rather than only being
able to re-run the recipe from scratch with the manifest's hyperparameters
alone. It's opt-in per publish — it roughly triples the file's size — so
most `.nctr` releases will leave it unset.

`.nctr` is the right choice for a publisher who wants a release that's
reproducible end-to-end: someone else should be able to re-derive an
equivalent model from the manifest alone, and, if optimizer state was
included, resume training from exactly where it left off.

**Container version vs. spec version.** `.nctr`'s header `format_version`
and `nanity.spec_version` (§3) are deliberately two different numbers on
two different axes. `format_version` versions the *container layout* —
the byte offsets and section shapes described above. `nanity.spec_version`
versions the *architecture* (§4–§6), identically to what it already means
for a GGUF file. Bumping one never implies bumping the other: a
`format_version` change means "the bytes are laid out differently on
disk," not "the model computes something different," and vice versa.

## 3. Required metadata keys

| Key | Type | Meaning |
|---|---|---|
| `general.architecture` | string | **Must be exactly `"nanity"`.** |
| `nanity.spec_version` | uint32 | **Must be `1`** for this document. |
| `nanity.embedding_length` | uint32 | `n_embd` — model hidden size. |
| `nanity.block_count` | uint32 | `n_layer` — number of transformer blocks. |
| `nanity.attention.head_count` | uint32 | `n_head` — query heads. |
| `nanity.attention.head_count_kv` | uint32 | `n_kv_head` — KV heads. Must divide `n_head` evenly. |
| `nanity.attention.key_length` | uint32 | `head_dim` — per-head dimension (used for both K and V). |
| `nanity.feed_forward_length` | uint32 | `n_ff` — SwiGLU intermediate size. |
| `nanity.context_length` | uint32 | Max training/inference context length. |

### Optional metadata keys (sane defaults if absent)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `nanity.attention.layer_norm_rms_epsilon` | float32 | `1e-5` | RMSNorm epsilon. |
| `nanity.rope.freq_base` | float32 | `10000.0` | RoPE base (θ). |
| `nanity.rope.scale_linear` | float32 | `1.0` | Linear RoPE position scale. |
| `nanity.vocab_size` | uint32 | *(derived)* | Cross-checked against `token_embd.weight`'s shape if present; not required since vocab size is already implied by that tensor. |
| `general.alignment` | uint32 | `32` | GGUF data-section alignment. |
| `general.name` | string | — | Display name, cosmetic only. |

There is intentionally **no** `nanity.use_fused_qkv` or similar toggle —
the architecture doesn't have a fused-tensor mode to toggle.

## 4. Required tensors

`n_vocab` below is *derived* from `token_embd.weight`'s shape, not declared
separately. `q_dim = n_head * head_dim`, `kv_dim = n_kv_head * head_dim`.

| Tensor | Shape (`ne`) | Notes |
|---|---|---|
| `token_embd.weight` | `[n_embd, n_vocab]` | Input embedding table. |
| `output_norm.weight` | `[n_embd]` | Final RMSNorm, applied once before the output projection. |
| `output.weight` | `[n_embd, n_vocab]` | **Optional.** Absent ⇒ tied embeddings (output reuses `token_embd.weight`). |
| `blk.{i}.attn_norm.weight` | `[n_embd]` | Pre-attention RMSNorm, layer `i`. |
| `blk.{i}.attn_q.weight` | `[n_embd, q_dim]` | Query projection. |
| `blk.{i}.attn_k.weight` | `[n_embd, kv_dim]` | Key projection. |
| `blk.{i}.attn_v.weight` | `[n_embd, kv_dim]` | Value projection. |
| `blk.{i}.attn_output.weight` | `[q_dim, n_embd]` | Attention output projection. |
| `blk.{i}.ffn_norm.weight` | `[n_embd]` | Pre-FFN RMSNorm, layer `i`. |
| `blk.{i}.ffn_gate.weight` | `[n_embd, n_ff]` | SwiGLU gate projection. |
| `blk.{i}.ffn_up.weight` | `[n_embd, n_ff]` | SwiGLU up projection. |
| `blk.{i}.ffn_down.weight` | `[n_ff, n_embd]` | SwiGLU down projection. |

`{i}` ranges over `0 .. block_count-1`. Every tensor above except
`output.weight` is **required** for every layer; a model missing any one of
them is not a conforming NANITY model, and NEON will refuse to load it with
a message naming the exact tensor and the shape it expected instead.

No other tensor names are recognized. No synonyms (`embed_tokens.weight`,
`lm_head.weight`, `model.norm.weight`, …) are accepted — pick one canonical
name and the ambiguity disappears.

## 5. Computation graph

Per forward pass, given token ids `ctx[0..seq)`:

```
h = embed(ctx)                                      # [seq, n_embd]

for layer i in 0 .. n_layer-1:
    x      = RMSNorm(h, attn_norm[i], eps)
    q      = x @ attn_q[i]^T                         # [seq, q_dim]
    k      = x @ attn_k[i]^T                         # [seq, kv_dim]
    v      = x @ attn_v[i]^T                         # [seq, kv_dim]
    q, k   = RoPE(q), RoPE(k)                        # see §6
    attn   = causal_GQA_attention(q, k, v)           # group = n_head / n_kv_head
    h      = h + attn @ attn_output[i]^T

    x      = RMSNorm(h, ffn_norm[i], eps)
    gate   = silu(x @ ffn_gate[i]^T)
    up     = x @ ffn_up[i]^T
    h      = h + (gate * up) @ ffn_down[i]^T

logits = RMSNorm(h[-1], output_norm, eps) @ (output ?? token_embd)^T
```

`causal_GQA_attention`: standard scaled dot-product attention,
`softmax(QK^T / sqrt(head_dim) + causal_mask) V`, where each of the
`n_kv_head` KV heads is shared by `n_head / n_kv_head` query heads
(consecutive query heads `[g*group, (g+1)*group)` attach to KV head `g`).

## 6. RoPE convention (fixed — not configurable)

NANITY uses **adjacent-pair rotation**. For a head vector `v` of length
`head_dim`, at position `pos`, for `i` in `[0, head_dim/2)`:

```
freq  = rope_freq_base ^ (-2*i / head_dim)
theta = pos * freq / rope_scale_linear
c, s  = cos(theta), sin(theta)

v[2i]   ' = v[2i]*c - v[2i+1]*s
v[2i+1] ' = v[2i]*s + v[2i+1]*c
```

This is applied independently to every head of Q and every head of K. There
is no second convention (no "neox-style" split-half rotation) — `rawllm_forward.hpp`'s
`rope_apply()` and `modeling_nanity.py`'s `rotate_pairs()` both implement
exactly this formula, and must keep matching each other if either changes.

## 7. Tokenizer

This section describes GGUF's realization of tokenizer data; §2.2 covers
`.nctr`'s (a JSON section carrying the same information under different
field names) — both are container-level conventions, not NANITY-specific
ones, so nothing here affects §4–§6.

Standard GGUF tokenizer metadata is used as-is (this is container-level, not
NANITY-specific): `tokenizer.ggml.tokens` (string array), plus
`tokenizer.ggml.bos_token_id` / `eos_token_id` / `unknown_token_id`
(uint32). NEON's bundled tokenizer does greedy longest-match over this
vocab with byte-fallback (`<0xXX>` tokens) — train whatever tokenizer you
like, as long as the exported vocab follows this convention. Merge rules
(`tokenizer.ggml.merges`) are stored but not currently consumed by the
greedy matcher.

## 8. Quantization

`rawllm_forward.hpp` currently dequantizes on the fly and supports `F32`,
`F16`, `BF16`, `Q8_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q4_K`, `Q5_K`,
`Q6_K`. `Q2_K`, `Q3_K`, `Q8_1`, and `IQ4_NL` are recognized by the GGUF
reader but not yet wired into the dequantizer and will throw a clear error
if used. `export_gguf()` currently writes `F32`/`F16` (§2.1); `export_nctr()`
currently writes `F32` only (§2.2) — quantized export for either container
is a useful follow-up but isn't required to get a model running.

## 9. Reference implementation

| File | Role |
|---|---|
| `modeling_nanity.py` | PyTorch reference implementation — train a model with this. |
| `train_nanity_fixed.py` | Training loop; `export_gguf()` / `export_nctr()` export a checkpoint to either container (§2.1/§2.2). |
| `rawllm_loader.hpp` | `GGUFLoader::validate_config()` — checks a GGUF file against §3/§4. |
| `rawllm_nctr_loader.hpp` | `NCTRLoader::validate_config()` — the same checks, for a `.nctr` file (§2.2). |
| `rawllm_forward.hpp` | CPU reference forward pass implementing §5/§6, shared by both loaders. |

These are independent implementations of the same spec on both sides
(Python training/export, C++ loading/inference) — agreement between them,
not a code dependency, is what makes a model load correctly, regardless
of which container it's stored in.

## 10. Versioning policy

`nanity.spec_version` exists so this can change without silently breaking
old or new models against the wrong runtime — regardless of which
container (§2.1 GGUF or §2.2 `.nctr`) a given model is stored in; the two
containers version independently of each other (§2.2's closing note):

- **Non-breaking additions** (a new *optional* metadata key with a
  documented default) may ship without bumping the version.
- **Anything that changes a required tensor's name, shape, or the
  computation graph** (§4–§6) is a breaking change and **must** bump
  `nanity.spec_version`. A runtime built against version `N` should refuse
  (not guess at) a file declaring version `≠ N` — `validate_config()`
  already does this.

## 11. FAQ

**Why doesn't `general.architecture` just say `"llama"` so other tools can
load it?** Because it isn't a Llama model — even though the math is
similar, the tensor names, the optional-tied-embedding convention, and the
RoPE convention are NANITY's own, and claiming `"llama"` would make
*other* tools load it incorrectly (wrong shapes, wrong RoPE) rather than
refuse it cleanly the way NEON now does for non-NANITY files.

**Can I add MoE / a different attention variant / bias terms?** Not under
spec v1. That would need a new `nanity.spec_version` and corresponding
loader/forward changes — by design, this keeps "what NEON can run" a small,
fully-enumerated set rather than an open-ended guessing game again.

**What if my trained model doesn't fit these shapes?** Pad/project to fit,
or wait for a future spec version. The whole point of fixing the
architecture is that "doesn't fit" is now a question with a precise
answer, not a runtime mystery.
