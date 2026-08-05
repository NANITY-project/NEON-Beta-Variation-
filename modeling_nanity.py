"""
modeling_nanity.py — reference PyTorch implementation of NANITY spec v1.

This is the training-side counterpart described in NANITY_ARCHITECTURE_SPEC.md
section 9. It implements exactly the architecture in that spec (decoder-only,
RMSNorm pre-norm, GQA with separate Q/K/V, adjacent-pair RoPE, SwiGLU with
separate gate/up/down, no biases anywhere, optionally-tied embeddings) and
nothing else — no fused QKV, no fused gate_up, no neox-style RoPE.

Module attribute names deliberately mirror the GGUF tensor names from spec
section 4 (token_embd, output_norm, output, attn_norm, attn_q/k/v,
attn_output, ffn_norm, ffn_gate/up/down) so that a state_dict -> GGUF
exporter is a near-trivial name-and-shape mapping. Linear layer weight
shapes follow nn.Linear(in, out).weight == (out, in), which is already the
PyTorch-shape-tuple form the spec's ne = [in, out] convention reverses into
— no transposes needed when exporting.

Intended use: instantiate NanityForCausalLM with a NanityConfig, leave the
weights at random init, and run a logit-distillation training loop against
a teacher model (e.g. KL divergence between student and teacher logits, or
hidden-state alignment). This file only defines the architecture; it does
not implement the training loop itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class NanityConfig:
    # --- required NANITY metadata (spec section 3) ---
    vocab_size: int = 200064
    embedding_length: int = 3072          # n_embd
    block_count: int = 36                  # n_layer
    attention_head_count: int = 24         # n_head
    attention_head_count_kv: int = 8       # n_kv_head
    attention_key_length: int = 128        # head_dim
    feed_forward_length: int = 9216        # n_ff
    context_length: int = 8192

    # --- optional NANITY metadata (spec section 3, with spec defaults) ---
    rms_norm_eps: float = 1e-5
    rope_freq_base: float = 10000.0
    rope_scale_linear: float = 1.0

    # --- forward-compatible, additive-only field (spec section 10: a new
    # optional metadata key with a documented default is non-breaking).
    # Mirrors the in-progress `nanity.rope.dimension_count` key: number of
    # head_dim channels that get rotated. None / == head_dim reproduces
    # spec v1's full-head rotation exactly. ---
    rope_dimension_count: Optional[int] = None

    # --- not part of the GGUF metadata; export-time choice only ---
    tie_embeddings: bool = True

    architecture: str = "nanity"
    spec_version: int = 1

    def __post_init__(self):
        if self.attention_head_count % self.attention_head_count_kv != 0:
            raise ValueError(
                f"attention_head_count ({self.attention_head_count}) must be "
                f"divisible by attention_head_count_kv ({self.attention_head_count_kv})"
            )
        if self.rope_dimension_count is None:
            self.rope_dimension_count = self.attention_key_length
        if self.rope_dimension_count % 2 != 0:
            raise ValueError("rope_dimension_count must be even (adjacent-pair rotation)")
        if self.rope_dimension_count > self.attention_key_length:
            raise ValueError("rope_dimension_count cannot exceed attention_key_length (head_dim)")

    @property
    def n_embd(self) -> int:
        return self.embedding_length

    @property
    def n_layer(self) -> int:
        return self.block_count

    @property
    def n_head(self) -> int:
        return self.attention_head_count

    @property
    def n_kv_head(self) -> int:
        return self.attention_head_count_kv

    @property
    def head_dim(self) -> int:
        return self.attention_key_length

    @property
    def n_ff(self) -> int:
        return self.feed_forward_length

    @property
    def q_dim(self) -> int:
        return self.n_head * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_head * self.head_dim

    @property
    def n_group(self) -> int:
        return self.n_head // self.n_kv_head

    @classmethod
    def nanity_4b(cls) -> "NanityConfig":
        """~4.0B parameter preset (tied embeddings). See param count at the
        bottom of this file for the exact figure for a given vocab size."""
        return cls(
            vocab_size=200064,
            embedding_length=3072,
            block_count=36,
            attention_head_count=24,
            attention_head_count_kv=8,
            attention_key_length=128,
            feed_forward_length=9216,
            context_length=8192,
            tie_embeddings=True,
        )

    @classmethod
    def nanity_1_5b(cls) -> "NanityConfig":
        """~1.43B parameter preset (tied embeddings), 48k vocab, 30 layers.
        Sized for from-scratch pretraining within a realistic token budget
        rather than for maximum breadth -- narrower vocab (reasoning/code/
        tool-use focused, not broad multilingual/trivia coverage) trades
        embedding-table params for more depth at the same total size."""
        return cls(
            vocab_size=50257,
            embedding_length=2048,
            block_count=30,
            attention_head_count=16,
            attention_head_count_kv=4,
            attention_key_length=128,
            feed_forward_length=5504,
            context_length=8192,
            tie_embeddings=True,
        )

    @classmethod
    def nanity_tiny(cls) -> "NanityConfig":
        """~13M parameter preset — NOT a real training config, exists purely
        so train_nanity.py's data/train/checkpoint/resume MECHANICS can be
        exercised end-to-end on a CPU with a few GB of RAM before trusting
        the same script on rented GPU time. Keeps the real 200064-token
        vocab (unlike nanity_1_5b's narrower 50257) so it loads with the
        same default tokenizer as nanity_4b — only depth/width shrink, not
        the vocab, since the embedding table's cost is linear in vocab size
        (~12.8M of this preset's ~13M IS the tied embedding table) rather
        than the quadratic-ish cost depth/width add, so there's no RAM
        reason to shrink it too.
        Rough RAM budget in FP32 (no BF16 path on CPU): weights + grad + 2
        AdamW moment buffers ~= 4x params ~= 4 * 13M * 4 bytes =~ 200MB,
        comfortably clear of an 8GB machine even with data/tokenizer/OS
        overhead alongside it. Do not use this preset's OUTPUT for anything
        — 2 layers / n_embd=64 has no capacity to learn anything real; it
        exists to prove the pipeline runs, not that the model is good."""
        return cls(
            vocab_size=200064,
            embedding_length=64,
            block_count=2,
            attention_head_count=4,
            attention_head_count_kv=2,
            attention_key_length=16,
            feed_forward_length=128,
            context_length=256,
            tie_embeddings=True,
        )

    def to_gguf_metadata(self) -> dict:
        """Spec section 3 keys, ready to hand to a GGUF writer."""
        return {
            "general.architecture": self.architecture,
            "nanity.spec_version": self.spec_version,
            "nanity.embedding_length": self.embedding_length,
            "nanity.block_count": self.block_count,
            "nanity.attention.head_count": self.attention_head_count,
            "nanity.attention.head_count_kv": self.attention_head_count_kv,
            "nanity.attention.key_length": self.attention_key_length,
            "nanity.feed_forward_length": self.feed_forward_length,
            "nanity.context_length": self.context_length,
            "nanity.attention.layer_norm_rms_epsilon": self.rms_norm_eps,
            "nanity.rope.freq_base": self.rope_freq_base,
            "nanity.rope.scale_linear": self.rope_scale_linear,
            "nanity.vocab_size": self.vocab_size,
            # additive/optional — only emit if it differs from "full head",
            # to stay byte-identical with spec v1 files when unused.
            **(
                {"nanity.rope.dimension_count": self.rope_dimension_count}
                if self.rope_dimension_count != self.attention_key_length
                else {}
            ),
        }


# --------------------------------------------------------------------------- #
# RMSNorm (spec section 1: pre-norm placement, both sublayers + final norm)
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


# --------------------------------------------------------------------------- #
# RoPE — adjacent-pair convention (spec section 6). NOT the neox split-half
# convention. v[2i], v[2i+1] are rotated as a pair, for i in [0, rot_dim/2).
# --------------------------------------------------------------------------- #

def build_rope_cache(
    max_pos: int, rot_dim: int, freq_base: float, scale_linear: float,
    device=None, dtype=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    i = torch.arange(0, rot_dim, 2, device=device, dtype=torch.float32)
    freq = freq_base ** (-i / rot_dim)
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    theta = torch.outer(pos, freq) / scale_linear  # [max_pos, rot_dim/2]
    return theta.cos().to(dtype), theta.sin().to(dtype)


def rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot_dim: int) -> torch.Tensor:
    """x: [..., seq, head_dim]. cos/sin: [seq, rot_dim/2].
    Only the leading rot_dim channels are rotated; any remainder (when
    rope_dimension_count < head_dim) passes through unchanged."""
    head_dim = x.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]

    x1 = x_rot[..., 0::2]  # v[2i]
    x2 = x_rot[..., 1::2]  # v[2i+1]

    while cos.dim() < x1.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    out_rot = torch.stack((out1, out2), dim=-1).flatten(-2)  # interleave back
    if rot_dim == head_dim:
        return out_rot
    return torch.cat([out_rot, x_pass], dim=-1)


# --------------------------------------------------------------------------- #
# Attention — separate Q/K/V (spec: "no fused QKV — ever"), GQA, no bias.
# --------------------------------------------------------------------------- #

class NanityAttention(nn.Module):
    def __init__(self, cfg: NanityConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.n_group = cfg.n_group
        self.rot_dim = cfg.rope_dimension_count

        self.attn_q = nn.Linear(cfg.n_embd, cfg.q_dim, bias=False)
        self.attn_k = nn.Linear(cfg.n_embd, cfg.kv_dim, bias=False)
        self.attn_v = nn.Linear(cfg.n_embd, cfg.kv_dim, bias=False)
        self.attn_output = nn.Linear(cfg.q_dim, cfg.n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        b, t, _ = x.shape

        q = self.attn_q(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.attn_k(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.attn_v(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = rotate_pairs(q, cos, sin, self.rot_dim)
        k = rotate_pairs(k, cos, sin, self.rot_dim)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present_kv = (k, v) if use_cache else None

        # GQA: expand KV heads to match Q heads (group = n_head / n_kv_head)
        if self.n_group > 1:
            k = k.repeat_interleave(self.n_group, dim=1)
            v = v.repeat_interleave(self.n_group, dim=1)

        is_causal = attn_mask is None and past_kv is None and t > 1
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_head * self.head_dim)
        out = self.attn_output(out)
        return out, present_kv


# --------------------------------------------------------------------------- #
# FFN — SwiGLU, separate gate/up/down (spec: "no fused gate_up — ever").
# --------------------------------------------------------------------------- #

class NanityFFN(nn.Module):
    def __init__(self, cfg: NanityConfig):
        super().__init__()
        self.ffn_gate = nn.Linear(cfg.n_embd, cfg.n_ff, bias=False)
        self.ffn_up = nn.Linear(cfg.n_embd, cfg.n_ff, bias=False)
        self.ffn_down = nn.Linear(cfg.n_ff, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))


# --------------------------------------------------------------------------- #
# Decoder block (spec section 5)
# --------------------------------------------------------------------------- #

class NanityBlock(nn.Module):
    def __init__(self, cfg: NanityConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.attn = NanityAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.ffn = NanityFFN(cfg)

    def forward(self, x, cos, sin, attn_mask, past_kv=None, use_cache=False):
        attn_out, present_kv = self.attn(
            self.attn_norm(x), cos, sin, attn_mask, past_kv, use_cache
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, present_kv


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #

class NanityForCausalLM(nn.Module):
    def __init__(self, cfg: NanityConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embd = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blk = nn.ModuleList([NanityBlock(cfg) for _ in range(cfg.n_layer)])
        self.output_norm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)

        if cfg.tie_embeddings:
            self.output = None
        else:
            self.output = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        cos, sin = build_rope_cache(
            cfg.context_length, cfg.rope_dimension_count,
            cfg.rope_freq_base, cfg.rope_scale_linear,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        self._apply_depth_scaling()

    def _init_weights(self, module: nn.Module):
        # Llama/nanoGPT-style small init; tuned for stable random-init
        # distillation training, not meant to be load-bearing on its own.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_depth_scaling(self):
        # Scale residual-stream-writing projections (attn_output, ffn_down)
        # by 1/sqrt(2*n_layer) so residual variance doesn't blow up with depth.
        scale = 1.0 / math.sqrt(2 * self.cfg.n_layer)
        for block in self.blk:
            nn.init.normal_(block.attn.attn_output.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(block.ffn.ffn_down.weight, mean=0.0, std=0.02 * scale)

    def output_weight(self) -> torch.Tensor:
        return self.token_embd.weight if self.output is None else self.output.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        return_hidden: bool = False,
    ):
        b, t = input_ids.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        h = self.token_embd(input_ids)
        cos = self.rope_cos[past_len:past_len + t].to(h.device)
        sin = self.rope_sin[past_len:past_len + t].to(h.device)

        # additive attention bias mask (for left-padding); SDPA's is_causal
        # fast path is used instead whenever there's no padding/cache.
        attn_mask = None
        if attention_mask is not None or past_key_values is not None:
            kv_len = past_len + t
            causal = torch.full((t, kv_len), float("-inf"), device=h.device).triu(1 + past_len)
            attn_mask = causal[None, None, :, :]
            if attention_mask is not None:
                pad = (1.0 - attention_mask[:, None, None, :kv_len].float()) * float("-inf")
                pad = torch.nan_to_num(pad, nan=0.0)
                attn_mask = attn_mask + pad

        presents = [] if use_cache else None
        for i, block in enumerate(self.blk):
            past_kv = past_key_values[i] if past_key_values is not None else None
            h, present_kv = block(h, cos, sin, attn_mask, past_kv, use_cache)
            if use_cache:
                presents.append(present_kv)

        h = self.output_norm(h)
        logits = F.linear(h, self.output_weight())

        if return_hidden:
            return logits, presents, h
        return logits, presents

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32, temperature: float = 1.0):
        past_key_values = None
        generated = input_ids
        cur_input = input_ids
        for _ in range(max_new_tokens):
            logits, past_key_values = self.forward(cur_input, past_key_values=past_key_values, use_cache=True)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            cur_input = next_token
        return generated


# --------------------------------------------------------------------------- #
# Param counting / sanity check / smoke test
# --------------------------------------------------------------------------- #

def count_params(model: nn.Module) -> int:
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue  # don't double-count tied weights
        seen.add(id(p))
        total += p.numel()
    return total


if __name__ == "__main__":
    cfg = NanityConfig.nanity_4b()
    print(cfg)
    model = NanityForCausalLM(cfg)
    n_params = count_params(model)
    print(f"\nTotal params: {n_params:,} ({n_params / 1e9:.2f}B)")

    # smoke test: forward pass, training-step shapes, KV-cache generation
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    logits, _ = model(x)
    assert logits.shape == (1, 16, cfg.vocab_size), logits.shape
    print(f"Forward pass OK, logits shape: {tuple(logits.shape)}")

    out = model.generate(x, max_new_tokens=4, temperature=0.0)
    assert out.shape == (1, 20)
    print(f"Greedy generation OK, output shape: {tuple(out.shape)}")

    print("\nGGUF metadata preview:")
    for k, v in cfg.to_gguf_metadata().items():
        print(f"  {k} = {v}")