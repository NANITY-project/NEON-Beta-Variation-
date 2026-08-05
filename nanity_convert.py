#!/usr/bin/env python3
"""
nanity_convert.py — detect, convert, and quantize ANY reasonably-standard
HF-style GGUF (Llama / Mistral / Qwen2 / Phi-3 / Phi-4-mini / merges and
hand-rolled exports of the same family) into a NANITY v1-conformant GGUF,
in one pass.

This is a merge of three earlier scripts into one tool:
  - nanity_detect.py            (tensor-name/shape fingerprinting, generic
                                  across architectures, not Phi-specific)
  - nanity_convert_quantize.py  (the actual split/permute/quantize/write/
                                  verify pipeline, with explicit opt-in
                                  flags for every lossy operation)
  - nanity_compilier.py         (an earlier, less careful one-shot version;
                                  superseded by this script — it guessed
                                  "llama"/"qwen" in a tensor name implies
                                  split-half RoPE with no opt-out and no
                                  bias/QK-norm checks at all. Don't use it.)

DESIGN PRINCIPLE (unchanged from the two scripts above): every operation
that changes the model's actual numerical behavior requires an explicit
flag. Nothing lossy happens silently. Mechanical re-namings and provably
lossless permutations (see permute_qk_rows_per_head below) happen
automatically.

----------------------------------------------------------------------------
ABOUT PARTIAL ROTARY EMBEDDINGS (rope_dim < head_dim — e.g. Phi-4-mini's
partial_rotary_factor=0.75) — READ THIS BEFORE CONVERTING A PHI-4-MINI
DERIVATIVE:

NANITY v1 (NANITY_ARCHITECTURE_SPEC.md §1, §6) defines exactly ONE rotation
convention: adjacent-pair rotation applied to the FULL head_dim. There is no
optional partial-rotary key in §3's metadata table, and §11 is explicit that
anything outside the enumerated computation graph is an out-of-spec change
requiring a new spec_version. A previous version of this conversion
pipeline wrote a `nanity.rope.dimension_count` key for partial-rotary
source models — that key is NOT part of spec v1, your `rawllm_loader.hpp`/
`rawllm_forward.hpp` have no code path for it, and writing it either (a)
gets the file rejected by a strict validator, or worse (b) gets silently
ignored, in which case the runtime rotates the full head_dim anyway with
NO warning — i.e. exactly the corruption below, just unacknowledged.

So this script does NOT write that key, and does NOT silently force full
rotation either. For a partial-rotary source model it REFUSES to convert
unless you pass --force-full-rotary, which:
  - rotates the full head_dim (matching what your runtime actually does),
  - prints an explicit warning naming the affected layers and the number
    of corrupted tail channels,
  - is an accuracy-losing approximation, not a clean conversion. Validate
    output quality (perplexity, spot-check generations) before trusting it.

The only fully correct fix is extending rawllm_forward.hpp to implement
partial rotation and bumping nanity.spec_version to 2 — that's a runtime
change, not something a converter script can paper over.
----------------------------------------------------------------------------

Usage:
    pip install gguf --break-system-packages   # if not already installed

    # Inspect only, no conversion:
    python3 nanity_convert.py source.gguf --detect-only
    python3 nanity_convert.py source.gguf --detect-only --dump-tensors
    python3 nanity_convert.py source.gguf --detect-only --json report.json

    # Convert + quantize:
    python3 nanity_convert.py source.gguf out.gguf --quant Q4_0

    # A partial-rotary source (e.g. Phi-4-mini) will refuse by default:
    python3 nanity_convert.py source.gguf out.gguf --quant Q4_0 \\
        --force-full-rotary     # accuracy loss, see warning above

    # Override anything that couldn't be inferred from metadata:
    python3 nanity_convert.py source.gguf out.gguf \\
        --n-head 24 --n-kv-head 8 --head-dim 128 --n-ff 8192 --quant Q4_K
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import gguf
    from gguf.constants import GGMLQuantizationType, GGUFValueType, GGML_QUANT_SIZES
except ImportError:
    sys.exit("Missing dependency 'gguf'. Install it with:\n    pip install gguf --break-system-packages\n")


# ============================================================================
# PART 1 — DETECTION (tensor name/shape fingerprinting, architecture-agnostic)
# ============================================================================

def read_all_metadata(reader: "gguf.GGUFReader") -> "OrderedDict[str, Any]":
    out: "OrderedDict[str, Any]" = OrderedDict()
    for key, rfield in reader.fields.items():
        try:
            out[key] = rfield.contents()
        except Exception as e:  # pragma: no cover - defensive
            out[key] = f"<unreadable: {e}>"
    return out


def meta_get(meta: dict, *candidates, default=None):
    for c in candidates:
        if c in meta:
            return meta[c]
    return default


LAYER_ROLE_PATTERNS: list[tuple[str, list[str]]] = [
    ("attn_norm",   [r"blk\.(\d+)\.attn_norm\.weight$",
                      r"model\.layers\.(\d+)\.input_layernorm\.weight$"]),
    ("attn_q",      [r"blk\.(\d+)\.attn_q\.weight$",
                      r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight$"]),
    ("attn_k",      [r"blk\.(\d+)\.attn_k\.weight$",
                      r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight$"]),
    ("attn_v",      [r"blk\.(\d+)\.attn_v\.weight$",
                      r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight$"]),
    ("attn_qkv",    [r"blk\.(\d+)\.attn_qkv\.weight$",
                      r"blk\.(\d+)\.attn\.wqkv\.weight$",
                      r"model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight$"]),
    ("attn_output", [r"blk\.(\d+)\.attn_output\.weight$",
                      r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight$"]),
    ("attn_q_norm", [r"blk\.(\d+)\.attn_q_norm\.weight$",
                      r"model\.layers\.(\d+)\.self_attn\.q_norm\.weight$"]),
    ("attn_k_norm", [r"blk\.(\d+)\.attn_k_norm\.weight$",
                      r"model\.layers\.(\d+)\.self_attn\.k_norm\.weight$"]),
    ("ffn_norm",    [r"blk\.(\d+)\.ffn_norm\.weight$",
                      r"model\.layers\.(\d+)\.post_attention_layernorm\.weight$"]),
    ("ffn_gate",    [r"blk\.(\d+)\.ffn_gate\.weight$",
                      r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight$"]),
    ("ffn_up",      [r"blk\.(\d+)\.ffn_up\.weight$",
                      r"model\.layers\.(\d+)\.mlp\.up_proj\.weight$"]),
    ("ffn_gate_up", [r"blk\.(\d+)\.ffn_gate_up\.weight$",
                      r"model\.layers\.(\d+)\.mlp\.gate_up_proj\.weight$"]),
    ("ffn_down",    [r"blk\.(\d+)\.ffn_down\.weight$",
                      r"model\.layers\.(\d+)\.mlp\.down_proj\.weight$"]),
    ("attn_q_bias",      [r"blk\.(\d+)\.attn_q\.bias$", r"model\.layers\.(\d+)\.self_attn\.q_proj\.bias$"]),
    ("attn_k_bias",      [r"blk\.(\d+)\.attn_k\.bias$", r"model\.layers\.(\d+)\.self_attn\.k_proj\.bias$"]),
    ("attn_v_bias",      [r"blk\.(\d+)\.attn_v\.bias$", r"model\.layers\.(\d+)\.self_attn\.v_proj\.bias$"]),
    ("attn_qkv_bias",    [r"blk\.(\d+)\.attn_qkv\.bias$", r"model\.layers\.(\d+)\.self_attn\.qkv_proj\.bias$"]),
    ("attn_output_bias", [r"blk\.(\d+)\.attn_output\.bias$", r"model\.layers\.(\d+)\.self_attn\.o_proj\.bias$"]),
    ("ffn_gate_bias",    [r"blk\.(\d+)\.ffn_gate\.bias$", r"model\.layers\.(\d+)\.mlp\.gate_proj\.bias$"]),
    ("ffn_up_bias",      [r"blk\.(\d+)\.ffn_up\.bias$", r"model\.layers\.(\d+)\.mlp\.up_proj\.bias$"]),
    ("ffn_down_bias",    [r"blk\.(\d+)\.ffn_down\.bias$", r"model\.layers\.(\d+)\.mlp\.down_proj\.bias$"]),
]

GLOBAL_ROLE_PATTERNS: list[tuple[str, list[str]]] = [
    ("token_embd",  [r"token_embd\.weight$", r"model\.embed_tokens\.weight$"]),
    ("output_norm", [r"output_norm\.weight$", r"model\.norm\.weight$"]),
    ("output",      [r"output\.weight$", r"lm_head\.weight$"]),
]


@dataclass
class LayerTensors:
    idx: int
    roles: dict = field(default_factory=dict)  # role -> (name, shape, dtype)


def classify_tensors(tensors) -> tuple[dict[int, LayerTensors], dict[str, Any], list[str]]:
    layers: dict[int, LayerTensors] = {}
    globals_: dict[str, Any] = {}
    unmatched: list[str] = []

    for t in tensors:
        name = t.name
        matched = False

        for role, patterns in LAYER_ROLE_PATTERNS:
            for pat in patterns:
                m = re.match(pat, name)
                if m:
                    i = int(m.group(1))
                    layers.setdefault(i, LayerTensors(idx=i))
                    layers[i].roles[role] = (name, tuple(int(x) for x in t.shape), t.tensor_type.name)
                    matched = True
                    break
            if matched:
                break
        if matched:
            continue

        for role, patterns in GLOBAL_ROLE_PATTERNS:
            for pat in patterns:
                if re.match(pat, name):
                    globals_[role] = (name, tuple(int(x) for x in t.shape), t.tensor_type.name)
                    matched = True
                    break
            if matched:
                break

        if not matched:
            unmatched.append(f"{name}  shape={tuple(int(x) for x in t.shape)}  dtype={t.tensor_type.name}")

    return layers, globals_, unmatched


@dataclass
class InferredConfig:
    n_layer: int = 0
    n_embd: int = 0
    n_vocab: int = 0
    q_dim: int = 0
    kv_dim: int = 0
    n_ff: int = 0
    qkv_fused: bool = False
    gate_up_fused: bool = False
    has_qk_norm: bool = False
    has_any_bias: bool = False
    tied_embeddings: bool = False
    bias_tensors_found: list = field(default_factory=list)
    inconsistent_layers: list = field(default_factory=list)
    n_head: Optional[int] = None
    n_kv_head: Optional[int] = None
    head_dim: Optional[int] = None
    rope_freq_base: Optional[float] = None
    rope_dim_count: Optional[int] = None   # partial-rotary indicator (diagnostic only — see header)
    rms_eps: Optional[float] = None
    context_length: Optional[int] = None
    declared_architecture: Optional[str] = None


def infer_config(layers: dict[int, LayerTensors], globals_: dict, meta: dict) -> InferredConfig:
    cfg = InferredConfig()
    cfg.n_layer = (max(layers.keys()) + 1) if layers else 0
    cfg.declared_architecture = meta_get(meta, "general.architecture")

    if "token_embd" in globals_:
        _, shape, _ = globals_["token_embd"]
        cfg.n_embd, cfg.n_vocab = shape[0], shape[1]

    cfg.tied_embeddings = "output" not in globals_

    n_ff_votes, q_dim_votes, kv_dim_votes = defaultdict(int), defaultdict(int), defaultdict(int)

    for i, lt in sorted(layers.items()):
        r = lt.roles
        if "attn_qkv" in r:
            cfg.qkv_fused = True
        elif {"attn_q", "attn_k", "attn_v"}.issubset(r.keys()):
            _, qshape, _ = r["attn_q"]
            _, kshape, _ = r["attn_k"]
            q_dim_votes[qshape[1]] += 1
            kv_dim_votes[kshape[1]] += 1
        else:
            cfg.inconsistent_layers.append((i, "missing attn_q/k/v and no fused attn_qkv"))

        if "ffn_gate_up" in r:
            cfg.gate_up_fused = True
            _, gushape, _ = r["ffn_gate_up"]
            n_ff_votes[gushape[1] // 2] += 1
        elif {"ffn_gate", "ffn_up"}.issubset(r.keys()):
            _, gshape, _ = r["ffn_gate"]
            n_ff_votes[gshape[1]] += 1
        else:
            cfg.inconsistent_layers.append((i, "missing ffn_gate/up and no fused ffn_gate_up"))

        if "attn_q_norm" in r or "attn_k_norm" in r:
            cfg.has_qk_norm = True

        for role_name, val in r.items():
            if role_name.endswith("_bias"):
                cfg.has_any_bias = True
                cfg.bias_tensors_found.append(val[0])

    if q_dim_votes:
        cfg.q_dim = max(q_dim_votes, key=q_dim_votes.get)
    if kv_dim_votes:
        cfg.kv_dim = max(kv_dim_votes, key=kv_dim_votes.get)
    if n_ff_votes:
        cfg.n_ff = max(n_ff_votes, key=n_ff_votes.get)

    namespaces = ["llama", "phi3", "phi", "qwen2", "gemma", "nanity", "gpt2", "falcon"]

    def find(*suffixes):
        for ns in namespaces:
            for suf in suffixes:
                k = f"{ns}.{suf}"
                if k in meta:
                    return meta[k]
        return None

    cfg.n_head = find("attention.head_count")
    cfg.n_kv_head = find("attention.head_count_kv")
    cfg.head_dim = find("attention.key_length")
    cfg.rope_freq_base = find("rope.freq_base")
    cfg.rope_dim_count = find("rope.dimension_count")
    cfg.rms_eps = find("attention.layer_norm_rms_epsilon")
    cfg.context_length = find("context_length")

    if cfg.head_dim is None and cfg.n_head and cfg.q_dim:
        cfg.head_dim = cfg.q_dim // cfg.n_head

    if cfg.qkv_fused and cfg.n_head and cfg.n_kv_head and cfg.head_dim:
        cfg.q_dim = cfg.n_head * cfg.head_dim
        cfg.kv_dim = cfg.n_kv_head * cfg.head_dim

    return cfg


def nanity_compat_report(cfg: InferredConfig) -> dict:
    """Mechanical (safe, automatic) vs blocking (needs an explicit flag) issues."""
    mechanical = []
    blocking = []

    if cfg.qkv_fused:
        mechanical.append("Fused attn_qkv.weight will be split into attn_q/attn_k/attn_v "
                           "(rows [0:q_dim]=Q, [q_dim:q_dim+kv_dim]=K, [..:+kv_dim]=V).")
    if cfg.gate_up_fused:
        mechanical.append("Fused ffn_gate_up.weight will be split into ffn_gate/ffn_up "
                           "(row order controlled by --gate-up-order, default gate_first).")
    if cfg.tied_embeddings:
        mechanical.append("No separate output.weight — fine, NANITY supports tied embeddings natively.")
    mechanical.append("RoPE convention: split-half ('rotate_half'/NeoX) rows will be permuted to "
                       "NANITY's adjacent-pair convention (§6) — exact, lossless relabeling, not an "
                       "approximation (see permute_qk_rows_per_head doc comment).")

    if cfg.rope_dim_count and cfg.head_dim and cfg.rope_dim_count < cfg.head_dim:
        blocking.append(
            f"PARTIAL ROTARY EMBEDDINGS: only {cfg.rope_dim_count} of {cfg.head_dim} head_dim channels "
            f"are rotated in the source model. NANITY v1 has exactly one rotation convention — full "
            f"head_dim, adjacent-pair — with no partial-rotary key in the spec. This script will REFUSE "
            f"to convert this model unless you pass --force-full-rotary, which rotates the full head_dim "
            f"and corrupts the {cfg.head_dim - cfg.rope_dim_count} tail channels the model never trained "
            f"to be position-dependent. This is an accuracy-losing approximation, not a clean fix — the "
            f"only clean fix is extending the runtime to support partial rotation under a new spec_version.")

    if cfg.has_any_bias:
        blocking.append(
            f"BIAS TENSORS FOUND ({len(cfg.bias_tensors_found)}): NANITY has no bias terms anywhere. "
            f"Zero-valued biases are dropped for free (lossless). Non-zero biases will cause this script "
            f"to refuse unless you pass --drop-nonzero-bias-anyway.")

    if cfg.has_qk_norm:
        blocking.append(
            "QK-NORM TENSORS FOUND (attn_q_norm / attn_k_norm): a real op NANITY v1 has no hook for. "
            "Refuses unless --drop-qk-norm-anyway is passed.")

    if cfg.inconsistent_layers:
        blocking.append(
            f"{len(cfg.inconsistent_layers)} layer(s) didn't match any known tensor-naming pattern: "
            f"{cfg.inconsistent_layers[:5]}{' ...' if len(cfg.inconsistent_layers) > 5 else ''}. "
            f"Conversion cannot proceed for those layers as-is.")

    return {"mechanical_fixes": mechanical, "blocking_issues": blocking}


def render_detect_report(path: str, meta: dict, cfg: InferredConfig, compat: dict,
                          unmatched: list[str], n_tensors: int, dump_tensors: bool, layers: dict) -> str:
    lines = []
    w = lines.append
    w(f"{'='*78}")
    w(f"NANITY DETECTION REPORT — {path}")
    w(f"{'='*78}")
    w("\n--- declared metadata (do not trust blindly) ---")
    w(f"  general.architecture = {meta_get(meta, 'general.architecture')!r}")
    w(f"  general.name         = {meta_get(meta, 'general.name')!r}")
    w(f"  total metadata keys  = {len(meta)} | total tensors = {n_tensors}")

    w("\n--- inferred config (from tensor shapes + whatever metadata exists) ---")
    for fname in ("n_layer", "n_embd", "n_vocab", "n_head", "n_kv_head", "head_dim",
                  "q_dim", "kv_dim", "n_ff", "rope_freq_base", "rope_dim_count",
                  "rms_eps", "context_length", "qkv_fused", "gate_up_fused",
                  "has_qk_norm", "tied_embeddings", "has_any_bias"):
        w(f"  {fname:20s} = {getattr(cfg, fname)}")
    if cfg.inconsistent_layers:
        w(f"  ! inconsistent_layers = {cfg.inconsistent_layers}")

    w("\n--- NANITY v1 compatibility ---")
    w("  mechanical fixes (safe, automatic):")
    for m in compat["mechanical_fixes"]:
        w(f"    - {m}")
    if compat["blocking_issues"]:
        w("\n  BLOCKING issues (need an explicit flag):")
        for b in compat["blocking_issues"]:
            w(f"    ! {b}")
    else:
        w("\n  No blocking issues. Mechanical conversion should be sufficient.")

    if unmatched:
        w(f"\n--- unmatched tensors ({len(unmatched)}) ---")
        for u in unmatched[:40]:
            w(f"    {u}")
        if len(unmatched) > 40:
            w(f"    ... and {len(unmatched) - 40} more")

    if dump_tensors:
        w(f"\n--- full tensor table ({n_tensors} tensors) ---")
        for i in sorted(layers):
            w(f"  layer {i}:")
            for role, (name, shape, dtype) in sorted(layers[i].roles.items()):
                w(f"    {role:16s} {name:50s} shape={shape} dtype={dtype}")

    w(f"\n{'='*78}")
    return "\n".join(lines)


# ============================================================================
# PART 2 — CONVERSION (split fused tensors, fix RoPE convention, quantize)
# ============================================================================

QUANT_NAME_TO_TYPE = {
    "F32": GGMLQuantizationType.F32,
    "F16": GGMLQuantizationType.F16,
    "Q8_0": GGMLQuantizationType.Q8_0,
    "Q4_0": GGMLQuantizationType.Q4_0,
    "Q4_1": GGMLQuantizationType.Q4_1,
    "Q5_0": GGMLQuantizationType.Q5_0,
    "Q5_1": GGMLQuantizationType.Q5_1,
    "Q4_K": GGMLQuantizationType.Q4_K,
    "Q5_K": GGMLQuantizationType.Q5_K,
    "Q6_K": GGMLQuantizationType.Q6_K,
}

BIAS_ROLES = [
    "attn_q_bias", "attn_k_bias", "attn_v_bias", "attn_qkv_bias",
    "attn_output_bias", "ffn_gate_bias", "ffn_up_bias", "ffn_down_bias",
]


def dequant(reader_tensor) -> np.ndarray:
    return np.asarray(gguf.quants.dequantize(reader_tensor.data, reader_tensor.tensor_type), dtype=np.float32)


def split_rows(mat: np.ndarray, sizes: list[int]) -> list[np.ndarray]:
    out, start = [], 0
    for s in sizes:
        out.append(mat[start:start + s])
        start += s
    assert start == mat.shape[0], f"split_rows: sizes {sizes} sum to {start}, matrix has {mat.shape[0]} rows"
    return out


def rope_permutation(head_dim: int, rope_dim: int) -> np.ndarray:
    """
    perm such that block[perm] re-keys a head's rows from HF split-half RoPE
    (pair i is (channel i, channel i+half)) to NANITY adjacent-pair (pair i
    is (channel 2i, channel 2i+1)), same angle theta_i either way. Q.K^T is
    invariant under any permutation applied identically to both operands, so
    this is an exact relabeling — attention output is bit-for-bit unchanged
    for the channels covered by rope_dim. Channels >= rope_dim are identity
    (untouched), which is correct for a FULL rotation (rope_dim == head_dim);
    see header docstring for why a true rope_dim < head_dim source requires
    --force-full-rotary rather than a silent partial mapping.
    """
    assert rope_dim % 2 == 0, f"rope_dim must be even, got {rope_dim}"
    half = rope_dim // 2
    perm = np.arange(head_dim)
    for i in range(half):
        perm[2 * i] = i
        perm[2 * i + 1] = i + half
    return perm


def permute_qk_rows_per_head(mat: np.ndarray, n_heads: int, head_dim: int, rope_dim: int) -> np.ndarray:
    assert mat.shape[0] == n_heads * head_dim, (mat.shape, n_heads, head_dim)
    perm = rope_permutation(head_dim, rope_dim)
    blocks = mat.reshape(n_heads, head_dim, mat.shape[1])
    blocks = blocks[:, perm, :]
    return blocks.reshape(mat.shape[0], mat.shape[1])


def quantize_tensor(arr: np.ndarray, qtype, name: str):
    if qtype == GGMLQuantizationType.F32:
        return arr.astype(np.float32), GGMLQuantizationType.F32
    if qtype == GGMLQuantizationType.F16:
        return arr.astype(np.float16), GGMLQuantizationType.F16
    block_size, _ = GGML_QUANT_SIZES[qtype]
    if arr.shape[-1] % block_size != 0:
        print(f"  ! {name}: in_features={arr.shape[-1]} not divisible by "
              f"{qtype.name} block size {block_size} — falling back to F16 for this tensor", file=sys.stderr)
        return arr.astype(np.float16), GGMLQuantizationType.F16
    q = gguf.quants.quantize(arr.astype(np.float32), qtype)
    return q, qtype


def copy_metadata_kv(writer: "gguf.GGUFWriter", key: str, rfield) -> None:
    main_type = rfield.types[0]
    val = rfield.contents()
    sub_type = rfield.types[-1] if main_type == GGUFValueType.ARRAY else None
    if main_type == GGUFValueType.STRING and not val:
        return
    writer.add_key_value(key, val, main_type, sub_type=sub_type)


def resolve_config(cfg: InferredConfig, args) -> InferredConfig:
    for field_name in ("n_head", "n_kv_head", "head_dim", "n_ff",
                        "rope_freq_base", "rms_eps", "context_length"):
        override = getattr(args, field_name, None)
        if override is not None:
            setattr(cfg, field_name, override)

    if cfg.head_dim is None and cfg.n_head and cfg.q_dim:
        cfg.head_dim = cfg.q_dim // cfg.n_head
    if cfg.q_dim == 0 and cfg.n_head and cfg.head_dim:
        cfg.q_dim = cfg.n_head * cfg.head_dim
    if cfg.kv_dim == 0 and cfg.n_kv_head and cfg.head_dim:
        cfg.kv_dim = cfg.n_kv_head * cfg.head_dim

    missing = [f for f in ("n_layer", "n_embd", "n_head", "n_kv_head", "head_dim", "n_ff")
               if not getattr(cfg, f)]
    if missing:
        sys.exit(
            f"Can't determine required field(s) {missing} from the file's metadata.\n"
            f"Supply them explicitly, e.g. --n-head 24 --n-kv-head 8 --head-dim 128 --n-ff 8192\n"
            f"(run with --detect-only first if you need help figuring out the right numbers)."
        )

    if cfg.rms_eps is None:
        cfg.rms_eps = 1e-5
    if cfg.rope_freq_base is None:
        cfg.rope_freq_base = 10000.0
    if cfg.context_length is None:
        sys.exit("context_length not found in metadata; supply --context-length explicitly.")

    # --- partial rotary: refuse by default, see header docstring ---
    rope_dim = cfg.rope_dim_count if cfg.rope_dim_count else cfg.head_dim
    if rope_dim < cfg.head_dim:
        if not args.force_full_rotary:
            sys.exit(
                f"Source model has PARTIAL ROTARY EMBEDDINGS: only {rope_dim} of {cfg.head_dim} head_dim "
                f"channels are rotated (e.g. Phi-4-mini partial_rotary_factor). NANITY v1 has exactly one "
                f"rotation convention — full head_dim, adjacent-pair — with no partial-rotary key in the "
                f"spec, and your runtime has no code path for one even if this script invented one. "
                f"Converting this model means either:\n"
                f"  (a) re-run with --force-full-rotary to accept rotating the full head_dim — this WILL "
                f"corrupt the {cfg.head_dim - rope_dim} tail channels the model never trained to be "
                f"position-dependent, and is an explicit accuracy loss you must validate afterward, or\n"
                f"  (b) don't convert this model until rawllm_forward.hpp implements partial rotation "
                f"under a new nanity.spec_version.\n"
                f"Refusing to proceed silently either way."
            )
        print(f"  ! --force-full-rotary set: rotating all {cfg.head_dim} channels instead of the "
              f"{rope_dim} this model actually trained with. This corrupts the "
              f"{cfg.head_dim - rope_dim} untouched tail channels at every position and is very likely "
              f"to produce degraded or incoherent output. Validate before shipping.", file=sys.stderr)
        cfg.rope_dim_count = cfg.head_dim  # full rotation applied below — explicit, accepted approximation
    else:
        cfg.rope_dim_count = cfg.head_dim

    return cfg


def convert_layer(i: int, roles: dict, tensor_by_name: dict, cfg: InferredConfig, args) -> dict:
    def get(role):
        name, _shape, _dtype = roles[role]
        return dequant(tensor_by_name[name])

    if "attn_qkv" in roles:
        qkv = get("attn_qkv")
        q, k, v = split_rows(qkv, [cfg.q_dim, cfg.kv_dim, cfg.kv_dim])
    else:
        q, k, v = get("attn_q"), get("attn_k"), get("attn_v")

    if "ffn_gate_up" in roles:
        gu = get("ffn_gate_up")
        if args.gate_up_order == "gate_first":
            gate, up = split_rows(gu, [cfg.n_ff, cfg.n_ff])
        else:
            up, gate = split_rows(gu, [cfg.n_ff, cfg.n_ff])
    else:
        gate, up = get("ffn_gate"), get("ffn_up")

    attn_norm = get("attn_norm")
    attn_out = get("attn_output")
    ffn_norm = get("ffn_norm")
    ffn_down = get("ffn_down")

    # --- bias check: NANITY has no bias path at all ---
    for bias_role in BIAS_ROLES:
        if bias_role in roles:
            b = get(bias_role)
            max_abs = float(np.abs(b).max())
            if max_abs > 1e-6:
                if not args.drop_nonzero_bias_anyway:
                    sys.exit(
                        f"layer {i}: {bias_role} is NON-ZERO (max abs={max_abs:.3g}). NANITY has no bias "
                        f"term anywhere — dropping this changes the model's outputs, it is not a safe "
                        f"relabeling. Re-run with --drop-nonzero-bias-anyway if that's acceptable."
                    )
                print(f"  ! layer {i}: dropping NON-ZERO {bias_role} (max abs={max_abs:.3g}) "
                      f"— accuracy will be affected", file=sys.stderr)

    # --- RoPE convention fix: full head_dim, adjacent-pair (cfg.rope_dim_count
    # was already resolved to cfg.head_dim above, with --force-full-rotary
    # gating that decision when the source was genuinely partial-rotary) ---
    q = permute_qk_rows_per_head(q, cfg.n_head, cfg.head_dim, cfg.rope_dim_count)
    k = permute_qk_rows_per_head(k, cfg.n_kv_head, cfg.head_dim, cfg.rope_dim_count)

    return dict(attn_norm=attn_norm, q=q, k=k, v=v, attn_out=attn_out,
                ffn_norm=ffn_norm, gate=gate, up=up, ffn_down=ffn_down)


def verify_output(path: str, expect: InferredConfig) -> bool:
    reader = gguf.GGUFReader(path)
    meta = read_all_metadata(reader)
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            print(f"  FAIL: {msg}")
            ok = False

    check(meta.get("general.architecture") == "nanity", "general.architecture != 'nanity'")
    check(meta.get("nanity.spec_version") == 1, "nanity.spec_version != 1")
    check(meta.get("nanity.embedding_length") == expect.n_embd, "embedding_length mismatch")
    check(meta.get("nanity.block_count") == expect.n_layer, "block_count mismatch")
    check("nanity.rope.dimension_count" not in meta,
          "found nanity.rope.dimension_count — this key is NOT part of spec v1 and your runtime has no "
          "code path for it; it should never be written")

    names = {t.name for t in reader.tensors}
    for i in range(expect.n_layer):
        for suffix in ["attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                       "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight",
                       "ffn_up.weight", "ffn_down.weight"]:
            check(f"blk.{i}.{suffix}" in names, f"missing blk.{i}.{suffix}")
    check("token_embd.weight" in names, "missing token_embd.weight")
    check("output_norm.weight" in names, "missing output_norm.weight")
    check(not any(n.endswith(".bias") for n in names), "found a .bias tensor (NANITY forbids these)")

    return ok


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source_gguf")
    ap.add_argument("output_gguf", nargs="?", default=None,
                     help="omit together with --detect-only to just inspect the source file")
    ap.add_argument("--detect-only", action="store_true",
                     help="print the detection/compatibility report and exit without converting")
    ap.add_argument("--json", metavar="PATH", help="(with --detect-only) write a machine-readable config card")
    ap.add_argument("--dump-tensors", action="store_true", help="print every tensor, grouped by layer/role")

    ap.add_argument("--quant", default="Q4_0", choices=sorted(QUANT_NAME_TO_TYPE),
                     help="quantization format for 2-D weight tensors (default: Q4_0)")
    ap.add_argument("--gate-up-order", choices=["gate_first", "up_first"], default="gate_first",
                     help="row order in fused gate_up_proj (default matches HF Phi3/Phi4: gate_first)")
    ap.add_argument("--force-full-rotary", action="store_true",
                     help="REQUIRED to convert a partial-rotary source model (e.g. Phi-4-mini). Rotates "
                          "the full head_dim, which NANITY v1 always does — but corrupts the tail channels "
                          "the source model never trained as position-dependent. Accuracy-losing; validate "
                          "output afterward. See header docstring.")
    ap.add_argument("--drop-nonzero-bias-anyway", action="store_true",
                     help="accept the accuracy loss of dropping a non-zero bias tensor")
    ap.add_argument("--drop-qk-norm-anyway", action="store_true",
                     help="accept the accuracy loss of dropping attn_q_norm/attn_k_norm (Qwen3/Gemma2-style)")
    ap.add_argument("--n-head", type=int, default=None)
    ap.add_argument("--n-kv-head", type=int, default=None)
    ap.add_argument("--head-dim", type=int, default=None)
    ap.add_argument("--n-ff", type=int, default=None)
    ap.add_argument("--rope-freq-base", type=float, default=None)
    ap.add_argument("--rope-scale-linear", type=float, default=1.0)
    ap.add_argument("--rms-eps", type=float, default=None)
    ap.add_argument("--context-length", type=int, default=None)
    ap.add_argument("--name", default=None, help="general.name for the output file")
    args = ap.parse_args()

    if not args.detect_only and not args.output_gguf:
        sys.exit("output_gguf is required unless --detect-only is set.")

    print(f"[1/5] reading {args.source_gguf} ...")
    reader = gguf.GGUFReader(args.source_gguf)
    meta = read_all_metadata(reader)
    layers, globals_, unmatched = classify_tensors(reader.tensors)
    cfg = infer_config(layers, globals_, meta)
    compat = nanity_compat_report(cfg)

    if args.detect_only:
        report = render_detect_report(args.source_gguf, meta, cfg, compat,
                                       unmatched, len(reader.tensors), args.dump_tensors, layers)
        print(report)
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"inferred_config": asdict(cfg), "nanity_compatibility": compat}, f, indent=2)
            print(f"\n[wrote config card to {args.json}]")
        return

    cfg = resolve_config(cfg, args)  # this is where --force-full-rotary gets enforced/checked

    print(f"[2/5] config: n_layer={cfg.n_layer} n_embd={cfg.n_embd} n_head={cfg.n_head} "
          f"n_kv_head={cfg.n_kv_head} head_dim={cfg.head_dim} n_ff={cfg.n_ff} vocab={cfg.n_vocab}")
    if unmatched:
        print(f"  ! {len(unmatched)} unmatched tensor(s) in source — they will NOT be carried over. "
              f"Run with --detect-only --dump-tensors if any of these mattered.", file=sys.stderr)

    for i, lt in sorted(layers.items()):
        missing_roles = []
        if "attn_qkv" not in lt.roles and not {"attn_q", "attn_k", "attn_v"}.issubset(lt.roles):
            missing_roles.append("attn q/k/v (or fused qkv)")
        if "ffn_gate_up" not in lt.roles and not {"ffn_gate", "ffn_up"}.issubset(lt.roles):
            missing_roles.append("ffn gate/up (or fused gate_up)")
        if missing_roles:
            sys.exit(f"layer {i} is missing required tensors: {missing_roles}. Aborting.")

        qk_norm_roles = [r for r in ("attn_q_norm", "attn_k_norm") if r in lt.roles]
        if qk_norm_roles:
            if not args.drop_qk_norm_anyway:
                sys.exit(
                    f"layer {i} has {qk_norm_roles} (QK-RMSNorm). NANITY v1 has no computational hook for "
                    f"normalizing Q/K before the attention dot product. Re-run with --drop-qk-norm-anyway "
                    f"only after accepting that accuracy loss."
                )
            print(f"  ! layer {i}: dropping {qk_norm_roles} (--drop-qk-norm-anyway set)", file=sys.stderr)

    tensor_by_name = {t.name: t for t in reader.tensors}

    if "token_embd" not in globals_:
        sys.exit("source file has no token_embd.weight — can't proceed.")
    if "output_norm" not in globals_:
        sys.exit("source file has no output_norm.weight — can't proceed (NANITY requires it, §4).")

    qtype = QUANT_NAME_TO_TYPE[args.quant]

    print(f"[3/5] quantizing to {args.quant} and writing {args.output_gguf} ...")
    writer = gguf.GGUFWriter(args.output_gguf, arch="nanity", use_temp_file=True)
    writer.add_uint32("nanity.spec_version", 1)
    writer.add_uint32("nanity.embedding_length", cfg.n_embd)
    writer.add_uint32("nanity.block_count", cfg.n_layer)
    writer.add_uint32("nanity.attention.head_count", cfg.n_head)
    writer.add_uint32("nanity.attention.head_count_kv", cfg.n_kv_head)
    writer.add_uint32("nanity.attention.key_length", cfg.head_dim)
    writer.add_uint32("nanity.feed_forward_length", cfg.n_ff)
    writer.add_uint32("nanity.context_length", cfg.context_length)
    writer.add_float32("nanity.attention.layer_norm_rms_epsilon", cfg.rms_eps)
    writer.add_float32("nanity.rope.freq_base", cfg.rope_freq_base)
    writer.add_float32("nanity.rope.scale_linear", args.rope_scale_linear)
    writer.add_uint32("nanity.vocab_size", cfg.n_vocab)
    if args.name:
        writer.add_string("general.name", args.name)
    elif "general.name" in meta:
        writer.add_string("general.name", str(meta["general.name"]))
    # NOTE: deliberately NOT writing nanity.rope.dimension_count anywhere —
    # see header docstring. Spec v1 has no such key.

    n_tok_keys = 0
    for key in meta:
        if key.startswith("tokenizer."):
            copy_metadata_kv(writer, key, reader.fields[key])
            n_tok_keys += 1
    print(f"  carried over {n_tok_keys} tokenizer.* metadata key(s)")

    quant_report = []

    def add_weight(name, arr):
        q, used_qtype = quantize_tensor(arr, qtype, name)
        writer.add_tensor(name, q, raw_dtype=used_qtype)
        quant_report.append((name, used_qtype.name, arr.shape))

    def add_norm(name, arr):
        writer.add_tensor(name, arr.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
        quant_report.append((name, "F32", arr.shape))

    print("  embeddings ...")
    token_embd = dequant(tensor_by_name[globals_["token_embd"][0]])
    add_weight("token_embd.weight", token_embd)
    del token_embd

    output_norm = dequant(tensor_by_name[globals_["output_norm"][0]])
    add_norm("output_norm.weight", output_norm)
    del output_norm

    if "output" in globals_:
        output_w = dequant(tensor_by_name[globals_["output"][0]])
        add_weight("output.weight", output_w)
        del output_w

    print(f"[4/5] converting + writing {cfg.n_layer} layers "
          f"(split fused tensors, fix RoPE convention, drop/validate bias) ...")
    for i in sorted(layers.keys()):
        layer_data = convert_layer(i, layers[i].roles, tensor_by_name, cfg, args)

        add_norm(f"blk.{i}.attn_norm.weight", layer_data["attn_norm"])
        add_weight(f"blk.{i}.attn_q.weight", layer_data["q"])
        add_weight(f"blk.{i}.attn_k.weight", layer_data["k"])
        add_weight(f"blk.{i}.attn_v.weight", layer_data["v"])
        add_weight(f"blk.{i}.attn_output.weight", layer_data["attn_out"])
        add_norm(f"blk.{i}.ffn_norm.weight", layer_data["ffn_norm"])
        add_weight(f"blk.{i}.ffn_gate.weight", layer_data["gate"])
        add_weight(f"blk.{i}.ffn_up.weight", layer_data["up"])
        add_weight(f"blk.{i}.ffn_down.weight", layer_data["ffn_down"])

        del layer_data
        print(f"  layer {i + 1}/{cfg.n_layer} done", end="\r", flush=True)
    print()

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    out_size = Path(args.output_gguf).stat().st_size
    print(f"  wrote {len(quant_report)} tensors, {out_size / 1e6:.1f} MB total")

    print(f"[5/5] verifying {args.output_gguf} against NANITY §3/§4 ...")
    if verify_output(args.output_gguf, cfg):
        print("  PASS — file is NANITY v1 conformant.")
    else:
        print("  FAIL — see above. Do not ship this file.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
