#!/usr/bin/env python3
"""
train_nanity.py — full training pipeline for NANITY 4B on AMD MI300X.

Three sequential phases, each optional and independently resumable:
  Phase 1 [WARMUP]   — short general-text pre-training to stabilize the
                        embedding space before distillation. Skip if you
                        want pure distillation from random init (riskier
                        but faster to start).
  Phase 2 [DISTILL]  — KL-divergence distillation from a teacher model
                        (e.g. DeepSeek-R1-70B loaded on the same MI300X
                        in 4-bit, or from pre-generated offline logits).
  Phase 3 [FINETUNE] — standard CE fine-tuning on your own curated data
                        (identity, tool use, CoT format, human-like tone).

MI300X-specific setup (run ONCE before training):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
  pip install transformers accelerate gguf sentencepiece

Usage:
  # Phase 2 distillation (most common starting point):
  python3 train_nanity.py \
      --phase distill \
      --data    data/distill.jsonl \
      --teacher microsoft/Phi-4-mini-instruct \   # or path to DeepSeek-R1-70B
      --out     checkpoints/nanity_4b \
      --steps   50000 \
      --batch   4 \
      --grad-accum 8

  # Phase 3 fine-tune from a distillation checkpoint:
  python3 train_nanity.py \
      --phase   finetune \
      --data    data/finetune.jsonl \
      --resume  checkpoints/nanity_4b/ckpt_050000.pt \
      --out     checkpoints/nanity_4b_ft \
      --steps   5000

  # Export final checkpoint to GGUF:
  python3 train_nanity.py \
      --export  checkpoints/nanity_4b_ft/ckpt_005000.pt \
      --gguf    nanity_4b_trained.gguf

Data format (JSONL, one conversation per line):
  {"messages": [
      {"role": "system",    "content": "You are NANITY..."},
      {"role": "user",      "content": "Hello"},
      {"role": "assistant", "content": "Hi! How can I help?"}
  ]}

  The loss is computed ONLY on assistant turns (response tokens), never
  on the prompt. System/user tokens are included in the context but masked
  out of the loss, matching standard instruction-tuning practice.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os

# Fast (Rust-backed) tokenizers use their own internal thread pool per call.
# This is now the only source of tokenization speedup (no Python-level
# multiprocessing -- see ConversationDataset for why that was reverted on
# resource-constrained cloud containers), so it stays enabled.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Inline modeling_nanity -- import from the same directory, falling back to
# a sibling path that matches the NANITY project layout.
# ---------------------------------------------------------------------------
_here = Path(__file__).parent
for _candidate in [_here, _here.parent, _here / "NEON R2"]:
    if (_candidate / "modeling_nanity.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from modeling_nanity import NanityConfig, NanityForCausalLM  # noqa: E402


# ---------------------------------------------------------------------------
# LoRA — low-rank adapters for persona fine-tuning (Phase 3).
#
# Design intent: the CAPABILITY of a NANITY model (coding, Linux, reasoning)
# lives in the base weights, trained once via Phase 1/2. A PERSONA (this
# character's voice, identity, tone) is trained as a small low-rank delta on
# top -- swappable, forkable, and cheap to retrain against a different base
# later without repeating distillation. This mirrors NeoCortex's fork/publish
# model: persona adapters become small, shareable artifacts distinct from
# the base model they sit on.
#
# NEON.cpp / rawllm_loader.hpp do NOT support loading a base + adapter pair
# at inference time -- validate_config() expects one complete set of
# blk.{i}.* tensors, full stop. So the adapter is trained separately but
# MERGED into the base weights (W' = W + alpha/rank * B @ A) before GGUF
# export. The exported file is an ordinary, complete NANITY GGUF -- no
# runtime changes needed. True hot-swappable LoRA at inference time would
# require new C++ work in the loader/forward pass; this is out of scope
# here and can be added later without touching this training script.
# ---------------------------------------------------------------------------

LORA_TARGET_DEFAULT = ["attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up", "ffn_down"]


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear, adding a trainable low-rank delta.
    forward(x) = base(x) + scaling * (x @ A^T @ B^T)
    Only A and B receive gradients; base.weight is frozen."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        self.rank = rank
        self.scaling = alpha / rank
        # A: (rank, in_f) initialized Kaiming (as in the LoRA paper) so the
        # adapter starts with a nonzero-but-small random projection.
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        # B: (out_f, rank) initialized to ZERO so the adapter is a true
        # no-op at step 0 -- the merged model is byte-identical to the base
        # until training actually moves B away from zero.
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        # cast delta compute to the base weight's dtype (handles BF16 models)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + delta

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """Returns base.weight + the folded-in LoRA delta, same shape/dtype
        as the original nn.Linear.weight. Used at export time."""
        delta = (self.lora_B @ self.lora_A) * self.scaling   # [out_f, in_f]
        return self.base.weight + delta.to(self.base.weight.dtype)


def inject_lora(model: nn.Module, rank: int, alpha: float,
                 targets: list[str] = None) -> list[nn.Parameter]:
    """Freeze every parameter in `model`, then replace each nn.Linear whose
    attribute name is in `targets` with a LoRALinear wrapper. Returns the
    list of newly-created trainable LoRA parameters (for the optimizer).

    Only touches per-block attention/FFN projections (matches
    LORA_TARGET_DEFAULT) -- norms, token_embd, and output stay frozen and
    shared, which is what keeps a persona adapter small and keeps the
    underlying capability model's behavior stable."""
    targets = targets or LORA_TARGET_DEFAULT
    for p in model.parameters():
        p.requires_grad = False

    lora_params: list[nn.Parameter] = []
    blocks = model.blk if hasattr(model, "blk") else model._orig_mod.blk
    for block in blocks:
        for sub_name in ("attn", "ffn"):
            sub = getattr(block, sub_name)
            for name in targets:
                if not hasattr(sub, name):
                    continue
                orig = getattr(sub, name)
                if isinstance(orig, LoRALinear):
                    continue  # already wrapped (e.g. re-injecting on resume)
                wrapped = LoRALinear(orig, rank=rank, alpha=alpha).to(
                    next(orig.parameters()).device, next(orig.parameters()).dtype
                )
                setattr(sub, name, wrapped)
                lora_params.extend([wrapped.lora_A, wrapped.lora_B])

    n_trainable = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[lora] injected rank={rank} alpha={alpha} into {targets}")
    print(f"[lora] trainable params: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.3f}% of model)")
    return lora_params


def save_lora_checkpoint(model: nn.Module, step: int, out_dir: Path):
    """Saves ONLY the LoRA A/B matrices, not the base weights -- this is
    what makes a persona adapter small (tens of MB, not GB) and what makes
    it a distinct, forkable artifact from the base capability model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lora_{step:07d}.pt"
    state = {}
    blocks = model.blk if hasattr(model, "blk") else model._orig_mod.blk
    for i, block in enumerate(blocks):
        for sub_name in ("attn", "ffn"):
            sub = getattr(block, sub_name)
            for name in LORA_TARGET_DEFAULT:
                mod = getattr(sub, name, None)
                if isinstance(mod, LoRALinear):
                    key = f"blk.{i}.{sub_name}.{name}"
                    state[f"{key}.lora_A"] = mod.lora_A.detach().cpu()
                    state[f"{key}.lora_B"] = mod.lora_B.detach().cpu()
                    state[f"{key}.scaling"] = mod.scaling
    torch.save({"step": step, "lora_state": state}, path)
    print(f"[lora] adapter saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def merge_lora_for_export(model: nn.Module):
    """In-place: replace every LoRALinear with a plain nn.Linear holding the
    merged (base + delta) weight. Call this right before building the
    state_dict for GGUF export -- after this, model.state_dict() has the
    exact same keys/shapes as a non-LoRA model, so export_gguf() needs no
    special-casing."""
    blocks = model.blk if hasattr(model, "blk") else model._orig_mod.blk
    for block in blocks:
        for sub_name in ("attn", "ffn"):
            sub = getattr(block, sub_name)
            for name in LORA_TARGET_DEFAULT:
                mod = getattr(sub, name, None)
                if isinstance(mod, LoRALinear):
                    merged = nn.Linear(mod.base.in_features, mod.base.out_features, bias=False)
                    merged.weight = nn.Parameter(mod.merged_weight())
                    setattr(sub, name, merged)
    print("[lora] merged all adapters into base weights for export")

# ---------------------------------------------------------------------------
# Tokenizer helper — wraps HuggingFace AutoTokenizer with the NANITY chat
# template. We load the tokenizer from the HF repo of whichever base vocab
# you're using (Phi-4-mini-instruct for the 200064-token vocab). This runs
# only on the training machine; the GGUF export later carries the raw vocab
# array that NEON.cpp reads at inference time.
# ---------------------------------------------------------------------------

def load_tokenizer(name_or_path: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    return tok


def apply_chat_template(messages: list[dict], tokenizer,
                         think_end_markers: list[str] = None
                         ) -> tuple[list[int], list[bool], list[bool]]:
    """Encode a conversation into token ids, a loss mask, and an
    answer/think split.

    Returns:
        ids:         full token sequence (list[int])
        mask:        True where the loss should be computed (assistant turns only)
        is_answer:   True for tokens that are the FINAL ANSWER portion of an
                     assistant turn (only meaningful where mask is True).
                     False for reasoning/<think> tokens and for all non-
                     assistant tokens.

    Reasoning datasets (OpenThoughts, Bespoke-Stratos-style, or anything
    using <think>...</think>) put a long chain-of-thought BEFORE the actual
    answer inside the assistant turn. Both regions get loss=True today, but
    they are not equally important, and lumping them into one number hides
    exactly the failure mode where a model nails easy, formulaic reasoning
    boilerplate while barely improving on the much harder, much rarer answer
    tokens. Splitting them lets train() log them separately and optionally
    weight them differently.
    """
    if think_end_markers is None:
        # Default covers the OpenThoughts / Bespoke-Stratos / Sky-T1 family
        # convention (<|begin_of_thought|>...<|end_of_thought|>\n\n
        # <|begin_of_solution|>...<|end_of_solution|>) as well as the plain
        # <think>...</think> convention. If your data uses something else,
        # pass --think-end-markers explicitly -- otherwise everything is
        # silently counted as "answer" (see below), which is the safe
        # failure mode (no accidental down-weighting of real answer tokens).
        think_end_markers = ["<|begin_of_solution|>", "</think>", "<|end_of_thought|>"]

    ids: list[int] = []
    mask: list[bool] = []
    is_answer: list[bool] = []

    def enc(text: str) -> list[int]:
        # encode without adding special tokens -- we insert our own control
        # tokens explicitly so the model learns them as real structure.
        return tokenizer.encode(text, add_special_tokens=False)

    def add(token_ids: list[int], is_loss: bool, is_ans: bool = False):
        ids.extend(token_ids)
        mask.extend([is_loss] * len(token_ids))
        is_answer.extend([is_ans] * len(token_ids))

    def split_think_answer(content: str) -> tuple[str, str]:
        """Split at the first (earliest-appearing) end-of-thinking marker.
        Everything up to and including the marker is 'think'; everything
        after is 'answer'. If no marker is found, the WHOLE content counts
        as answer -- we never guess that untagged content is reasoning."""
        earliest = None
        for m in think_end_markers:
            idx = content.find(m)
            if idx != -1 and (earliest is None or idx < earliest[0]):
                earliest = (idx, m)
        if earliest is None:
            return "", content
        idx, m = earliest
        split_at = idx + len(m)
        return content[:split_at], content[split_at:]

    # Map role name -> control token string (NANITY format, confirmed by
    # check_tokenizer.py: these exist in the Phi-4-mini 200064-token vocab).
    role_token = {
        "system":    "<|system|>",
        "user":      "<|user|>",
        "assistant": "<|assistant|>",
    }
    end_token = "<|end|>"

    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        is_asst = role == "assistant"

        # role control token (NOT in the loss regardless of role)
        add(enc(role_token.get(role, f"<|{role}|>")), is_loss=False)

        if is_asst:
            think_text, answer_text = split_think_answer(content)
            if think_text:
                add(enc(think_text), is_loss=True, is_ans=False)
            add(enc(answer_text), is_loss=True, is_ans=True)
        else:
            add(enc(content), is_loss=False)

        # end token — count as part of assistant turn so the model learns to
        # emit it; treated as "answer" (it's the stop signal, small effect
        # either way). Masked out for non-assistant turns.
        add(enc(end_token), is_loss=is_asst, is_ans=is_asst)

    return ids, mask, is_answer


def crop_preserving_answer(ids: list[int], mask: list[bool], is_answer: list[bool],
                            max_len: int) -> tuple[list, list, list, bool]:
    """If the sequence is over max_len, drop tokens from the THINK region
    only (never the answer, never the system/user prompt) until it fits.
    Drops the EARLIEST think tokens first, keeping the reasoning steps
    closest to the answer intact (usually the most load-bearing part of a
    chain-of-thought) along with the full answer.

    If there aren't enough think tokens to drop to make room (i.e. the
    prompt + answer alone already exceed max_len), returns ok=False --
    there's no safe way to crop further, so the caller should skip the
    example rather than silently truncating off the answer, which is
    exactly the failure mode this function exists to prevent.
    """
    if len(ids) <= max_len:
        return ids, mask, is_answer, True
    excess = len(ids) - max_len
    think_idx = [i for i in range(len(ids)) if mask[i] and not is_answer[i]]
    if len(think_idx) < excess:
        return None, None, None, False
    drop = set(think_idx[:excess])
    new_ids       = [t for i, t in enumerate(ids)       if i not in drop]
    new_mask      = [m for i, m in enumerate(mask)      if i not in drop]
    new_is_answer = [a for i, a in enumerate(is_answer) if i not in drop]
    return new_ids, new_mask, new_is_answer, True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class Example:
    ids:       list[int]
    mask:      list[bool]   # True = compute loss here
    is_answer: list[bool]   # True = this loss token is final-answer, not think


class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 4096,
                 think_end_markers: list[str] = None, tokenizer_source: str = None,
                 num_workers: int = None):
        # NOTE: multiprocessing.Pool-based parallel tokenization was tried
        # here and reverted -- it relies on POSIX semaphores backed by
        # /dev/shm, which many rented cloud containers restrict, causing
        # BrokenPipeError / leaked-semaphore crashes that are effectively
        # undebuggable from inside the script. Back to single-process, but
        # TOKENIZERS_PARALLELISM is left enabled (see near the top of this
        # file) so the Rust-backed fast tokenizer can still use its own
        # internal thread pool per call -- no multiprocessing, no /dev/shm
        # dependency, most of the speed without the fragility.
        self.examples: list[Example] = []
        skipped = 0
        skipped_overflow = 0   # prompt+answer alone exceeded max_len
        cropped = 0            # fit only after trimming think tokens
        with open(path) as f:
            lines = f.readlines()
        print(f"[dataset] tokenizing {len(lines)} lines (single-process, "
              f"tokenizer-internal threading only) ...")
        for i, line in enumerate(lines):
            if i and i % 100_000 == 0:
                print(f"  ...{i}/{len(lines)} lines processed, "
                      f"{len(self.examples)} kept so far")
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages") or obj.get("conversations") or obj
            if not isinstance(msgs, list):
                skipped += 1
                continue
            # normalise "from"/"value" (ShareGPT style) -> "role"/"content"
            normed = []
            for m in msgs:
                role = m.get("role") or m.get("from", "")
                role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)
                content = m.get("content") or m.get("value", "")
                if role and content:
                    normed.append({"role": role, "content": content})
            if not normed:
                skipped += 1
                continue
            ids, mask, is_answer = apply_chat_template(
                normed, tokenizer, think_end_markers=think_end_markers)
            if len(ids) > max_len:
                ids, mask, is_answer, ok = crop_preserving_answer(ids, mask, is_answer, max_len)
                if not ok:
                    skipped += 1
                    skipped_overflow += 1
                    continue
                cropped += 1
            if not any(mask):   # no assistant tokens → useless example
                skipped += 1
                continue
            self.examples.append(Example(ids, mask, is_answer))
        print(f"[dataset] loaded {len(self.examples)} examples, skipped {skipped} from {path}")
        if cropped or skipped_overflow:
            print(f"[dataset] {cropped} example(s) exceeded max_len={max_len} and had "
                  f"think-tokens trimmed to preserve the full answer; "
                  f"{skipped_overflow} example(s) had prompt+answer alone "
                  f"exceeding max_len and were skipped entirely (increase "
                  f"--max-seq-len / context_length if this number is large).")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex.ids, ex.mask, ex.is_answer


def split_train_val(ds: "ConversationDataset", val_split: float, seed: int = 1234):
    """Deterministically carve off val_split fraction of ds as a held-out
    validation set, EXCLUDED from training. Same seed every run -> the same
    examples are held out across resumes, so val loss stays comparable
    across the whole run instead of drifting because the split changed.

    This is what actually lets you tell "the model is learning" apart from
    "the model is memorizing": train loss goes down either way, but val
    loss only goes down in the first case. If val loss flattens or rises
    while train loss keeps falling, that gap IS the overfitting signal.
    """
    n = len(ds)
    n_val = max(1, int(n * val_split)) if val_split > 0 else 0
    if n_val == 0 or n_val >= n:
        return ds, None
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = set(perm[:n_val])
    train_examples = [ex for i, ex in enumerate(ds.examples) if i not in val_idx]
    val_examples = [ex for i, ex in enumerate(ds.examples) if i in val_idx]

    train_ds = ConversationDataset.__new__(ConversationDataset)
    train_ds.examples = train_examples
    val_ds = ConversationDataset.__new__(ConversationDataset)
    val_ds.examples = val_examples
    print(f"[val] held out {len(val_examples)}/{n} examples for validation "
          f"({val_split*100:.1f}%), {len(train_examples)} remain for training")
    return train_ds, val_ds


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int) -> dict:
    """Average CE loss over up to max_batches of val_loader, broken out by
    think vs. answer tokens. Deliberately plain CE (no label smoothing, no
    KL term, no reweighting) regardless of training phase, so the numbers
    are comparable across phases and directly interpretable as 'how
    surprised is the model by real held-out tokens.'

    Returns a dict with 'loss' (overall), 'think_loss', 'answer_loss'.
    answer_loss is the one that actually tells you if the model is getting
    better at producing correct final answers, as opposed to just fluent
    reasoning-style text."""
    was_training = model.training
    model.eval()
    total_losses, think_losses, answer_losses = [], [], []
    for i, (input_ids, labels, attn_mask, answer_mask) in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids   = input_ids.to(device)
        labels      = labels.to(device)
        attn_mask   = attn_mask.to(device)
        answer_mask = answer_mask.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits, _ = model(input_ids, attention_mask=attn_mask)
            # Same fix as the training loop: select valid (assistant-turn)
            # positions BEFORE casting to fp32 over the vocab dimension,
            # instead of casting the whole (batch*seq, vocab) tensor and
            # relying on ignore_index to zero out the rest. This function
            # runs every --val-every steps, so left unfixed it was a second,
            # periodic OOM trigger independent of the one in the main loop.
            V = logits.shape[-1]
            valid = (labels.view(-1) != -100)
            valid_idx = valid.nonzero(as_tuple=True)[0]
            logits_valid = logits.view(-1, V).index_select(0, valid_idx).float()
            labels_valid = labels.view(-1).index_select(0, valid_idx)
            answer_sub = answer_mask.view(-1).index_select(0, valid_idx)
            think_sub  = ~answer_sub
            per_tok = F.cross_entropy(
                logits_valid, labels_valid, reduction="none",
            )
        if valid.any():
            total_losses.append(per_tok.mean().item())
        if think_sub.any():
            think_losses.append(per_tok[think_sub].mean().item())
        if answer_sub.any():
            answer_losses.append(per_tok[answer_sub].mean().item())
    if was_training:
        model.train()
    return {
        "loss":        sum(total_losses) / max(len(total_losses), 1),
        "think_loss":  sum(think_losses) / max(len(think_losses), 1) if think_losses else float("nan"),
        "answer_loss": sum(answer_losses) / max(len(answer_losses), 1) if answer_losses else float("nan"),
    }


def compute_used_vocab_mask(datasets: list, vocab_size: int, tokenizer=None) -> torch.Tensor:
    """Boolean mask, True for every vocab id that appears anywhere (as an
    input token, not just a loss target) across the given datasets. Also
    always marks the tokenizer's special/control tokens as used, regardless
    of frequency, since those need to stay trainable no matter what.

    Used by --freeze-unseen-vocab to pin every OTHER row of the (tied)
    embedding/LM-head at its random init, instead of letting cross-entropy's
    softmax normalization slowly suppress them just for never being the
    target on English-only data.
    """
    used = torch.zeros(vocab_size, dtype=torch.bool)
    for ds in datasets:
        if ds is None:
            continue
        for ex in ds.examples:
            idx = torch.tensor(ex.ids, dtype=torch.long)
            used[idx] = True
    if tokenizer is not None:
        for tid in tokenizer.all_special_ids:
            if 0 <= tid < vocab_size:
                used[tid] = True
    return used


def collate_fn(batch, pad_id: int = 0):
    """Right-pad to the longest sequence in the batch."""
    ids_list, mask_list, is_answer_list = zip(*batch)
    max_len = max(len(x) for x in ids_list)
    input_ids    = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
    labels       = torch.full((len(ids_list), max_len), -100,   dtype=torch.long)
    attn_mask    = torch.zeros(len(ids_list), max_len,           dtype=torch.bool)
    answer_mask  = torch.zeros(len(ids_list), max_len,           dtype=torch.bool)
    for i, (ids, msk, is_ans) in enumerate(zip(ids_list, mask_list, is_answer_list)):
        n = len(ids)
        input_ids[i, :n]  = torch.tensor(ids, dtype=torch.long)
        attn_mask[i, :n]  = True
        for j, is_loss in enumerate(msk):
            if is_loss:
                labels[i, j] = ids[j]
                answer_mask[i, j] = is_ans[j]
    return input_ids, labels, attn_mask, answer_mask


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay → small constant floor
# ---------------------------------------------------------------------------

def cosine_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    if step >= total:
        return lr_min
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Teacher wrapper (optional, for online distillation)
# ---------------------------------------------------------------------------

class Teacher:
    """Wraps a HuggingFace causal LM loaded in 4-bit for online distillation.
    Loads ONLY for --phase distill when --teacher is provided.
    MI300X with 192GB HBM can hold DeepSeek-R1-70B in 4-bit (~35GB) +
    the 4B student in BF16 (~8GB) + optimizer states (~16GB) comfortably."""

    def __init__(self, name_or_path: str, device: str):
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError:
            sys.exit("pip install transformers bitsandbytes for online distillation")

        print(f"[teacher] loading {name_or_path} in 4-bit on {device} ...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path, quantization_config=bnb, device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[teacher] loaded.")

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits.float()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step, loss, out_dir: Path, cfg: NanityConfig):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ckpt_{step:07d}.pt"
    torch.save({
        "step":          step,
        "loss":          loss,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "nanity_config": cfg.__dict__,
    }, path)
    # keep only the 3 most recent checkpoints to save disk
    ckpts = sorted(out_dir.glob("ckpt_*.pt"))
    for old in ckpts[:-3]:
        old.unlink()
    print(f"[ckpt] saved {path}")
    return path


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt.get("step", 0)
    print(f"[ckpt] resumed from {path} at step {step}")
    return step


# ---------------------------------------------------------------------------
# Build tokenizer.* GGUF keys directly from an HF tokenizer -- no donor GGUF
# required. Mirrors the standard llama.cpp GPT-2-style (byte-level BPE)
# vocab conversion: vocab list + token types from AutoTokenizer, merges +
# special-token ids from the tokenizer's own files via gguf.SpecialVocab.
# ---------------------------------------------------------------------------

def write_tokenizer_keys_from_hf(writer, tokenizer_name_or_path: str):
    import tempfile
    try:
        import gguf
        from gguf.constants import TokenType
    except ImportError:
        sys.exit("pip install gguf")
    from transformers import AutoTokenizer

    print(f"[export] loading HF tokenizer {tokenizer_name_or_path} ...")
    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)

    # gguf.SpecialVocab reads tokenizer.json / tokenizer_config.json /
    # special_tokens_map.json off disk (for merges + special-token ids), so
    # save a local snapshot even if tokenizer_name_or_path is a hub id.
    with tempfile.TemporaryDirectory() as tmp:
        tok.save_pretrained(tmp)

        vocab = tok.get_vocab()  # token string -> id
        vocab_size = max(vocab.values()) + 1
        reverse_vocab = {v: k for k, v in vocab.items()}
        added = set(tok.get_added_vocab().values())
        special_strs = set(tok.all_special_tokens)

        tokens, toktypes = [], []
        for i in range(vocab_size):
            if i not in reverse_vocab:
                # Some HF vocabs have gaps; GGUF needs a dense id->token array.
                tokens.append(f"[UNUSED{i}]")
                toktypes.append(TokenType.UNUSED)
                continue
            token = reverse_vocab[i]
            tokens.append(token)
            if i in added and token in special_strs:
                toktypes.append(TokenType.CONTROL)
            elif i in added:
                toktypes.append(TokenType.USER_DEFINED)
            else:
                toktypes.append(TokenType.NORMAL)

        writer.add_tokenizer_model("gpt2")  # byte-level BPE, matches NEON.cpp's codec
        writer.add_token_list(tokens)
        writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(tmp, load_merges=True)
        special_vocab.add_to_gguf(writer)

    print(f"[export] wrote {len(tokens):,} tokenizer entries from {tokenizer_name_or_path}")


# ---------------------------------------------------------------------------
# GGUF export — reads the Phi-4-mini tokenizer GGUF (or any NANITY GGUF that
# carries the tokenizer keys) and writes a new GGUF with the trained weights.
# ---------------------------------------------------------------------------

def export_gguf(
    checkpoint_path: str,
    output_path: str,
    tokenizer_source_gguf: Optional[str] = None,
    quant: str = "F16",
    lora_checkpoint: Optional[str] = None,
    tokenizer_hf: Optional[str] = None,
):
    """Export a training checkpoint to a NANITY GGUF file.

    tokenizer_source_gguf: path to any existing NANITY GGUF that carries the
    tokenizer.ggml.* keys (e.g. an old out4.gguf). If given, those keys are
    copied verbatim into the new file rather than regenerating them.

    tokenizer_hf: HF tokenizer id or local path (e.g.
    "microsoft/Phi-4-mini-instruct"). If tokenizer_source_gguf is NOT given,
    this is used instead -- the tokenizer.* keys are built fresh from the HF
    tokenizer (vocab + merges + special tokens), so you never need a
    pre-existing donor GGUF just to export a checkpoint. This is the normal
    path the first time you export a model.

    Exactly one of tokenizer_source_gguf / tokenizer_hf must be given.

    lora_checkpoint: optional path to a persona adapter saved by
    save_lora_checkpoint(). If given, checkpoint_path MUST be the base
    capability checkpoint that adapter was trained against -- the adapter
    deltas get folded into the base weights before writing, so the output
    is one complete, ordinary GGUF (no runtime changes needed to load it).
    """
    if not tokenizer_source_gguf and not tokenizer_hf:
        sys.exit("export_gguf needs either tokenizer_source_gguf (an existing "
                  "NANITY GGUF) or tokenizer_hf (an HF tokenizer id/path). "
                  "If you don't have a donor GGUF yet, pass --tokenizer "
                  "(e.g. microsoft/Phi-4-mini-instruct) instead of --tokenizer-gguf.")
    try:
        import gguf
        from gguf.constants import GGMLQuantizationType
    except ImportError:
        sys.exit("pip install gguf")
    import numpy as np

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("nanity_config", {})
    cfg = NanityConfig(**cfg_dict)
    model = NanityForCausalLM(cfg)
    model.load_state_dict(ckpt["model"])

    if lora_checkpoint:
        print(f"[export] merging LoRA adapter from {lora_checkpoint} ...")
        lora_ckpt = torch.load(lora_checkpoint, map_location="cpu", weights_only=False)
        lora_state = lora_ckpt["lora_state"]
        n_merged = 0
        for i, block in enumerate(model.blk):
            for sub_name in ("attn", "ffn"):
                sub = getattr(block, sub_name)
                for name in LORA_TARGET_DEFAULT:
                    key = f"blk.{i}.{sub_name}.{name}"
                    if f"{key}.lora_A" not in lora_state:
                        continue
                    orig = getattr(sub, name)
                    A = lora_state[f"{key}.lora_A"]
                    B = lora_state[f"{key}.lora_B"]
                    scaling = lora_state[f"{key}.scaling"]
                    delta = (B @ A) * scaling
                    orig.weight = nn.Parameter(orig.weight + delta.to(orig.weight.dtype))
                    n_merged += 1
        print(f"[export] merged {n_merged} adapted projections into base weights")

    model.eval()

    quant_type = {
        "F32": GGMLQuantizationType.F32,
        "F16": GGMLQuantizationType.F16,
        "Q4_0": GGMLQuantizationType.Q4_0,
    }.get(quant.upper(), GGMLQuantizationType.F16)

    def to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().float().numpy()

    def write_tensor(writer, name: str, weight: torch.Tensor):
        arr = to_np(weight)
        if quant_type == GGMLQuantizationType.F32:
            data = arr.astype(np.float32)
        elif quant_type == GGMLQuantizationType.F16:
            data = arr.astype(np.float16)
        else:
            data = gguf.quants.quantize(arr.astype(np.float32), quant_type)
        writer.add_tensor(name, data, raw_dtype=quant_type)

    # --- tokenizer keys: either copied from an existing NANITY GGUF, or
    # built fresh from an HF tokenizer if no donor GGUF is available ---
    src_reader = None
    if tokenizer_source_gguf:
        print(f"[export] copying tokenizer keys from {tokenizer_source_gguf} ...")
        src_reader = gguf.GGUFReader(tokenizer_source_gguf)

    print(f"[export] writing {output_path} ...")
    writer = gguf.GGUFWriter(output_path, arch="nanity", use_temp_file=True)

    # architecture metadata (spec section 3)
    meta = cfg.to_gguf_metadata()
    for k, v in meta.items():
        vtype = type(v)
        if isinstance(v, str):
            writer.add_string(k, v)
        elif isinstance(v, float):
            writer.add_float32(k, v)
        else:
            writer.add_uint32(k, int(v))

    # general.name
    writer.add_string("general.name", "NANITY 4B")

    if src_reader is not None:
        # tokenizer keys copied verbatim from the donor GGUF
        from gguf.constants import GGUFValueType
        skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
                "general.architecture", "general.name"} | set(meta.keys())
        for key, field in src_reader.fields.items():
            if key in skip or key.startswith("nanity.") or key.startswith("GGUF."):
                continue
            main_type = field.types[0]
            val = field.contents()
            sub_type = field.types[-1] if main_type == GGUFValueType.ARRAY else None
            try:
                writer.add_key_value(key, val, main_type, sub_type=sub_type)
            except Exception:
                pass
    else:
        # tokenizer keys built fresh from an HF tokenizer -- no donor GGUF
        # needed. This is the byte-level BPE (GPT-2/tiktoken-family) path,
        # matching NEON.cpp's codec.
        write_tokenizer_keys_from_hf(writer, tokenizer_hf)

    # tensors — same name convention as modeling_nanity.py's state_dict,
    # which mirrors the GGUF tensor names from the spec exactly.
    sd = model.state_dict()

    write_tensor(writer, "token_embd.weight",  sd["token_embd.weight"])
    write_tensor(writer, "output_norm.weight", sd["output_norm.weight"])
    if "output.weight" in sd:
        write_tensor(writer, "output.weight", sd["output.weight"])
    # tied embeddings: no output.weight → NEON reuses token_embd

    for i in range(cfg.n_layer):
        prefix = f"blk.{i}"
        sd_prefix = f"blk.{i}"
        write_tensor(writer, f"{prefix}.attn_norm.weight",   sd[f"{sd_prefix}.attn_norm.weight"])
        write_tensor(writer, f"{prefix}.attn_q.weight",      sd[f"{sd_prefix}.attn.attn_q.weight"])
        write_tensor(writer, f"{prefix}.attn_k.weight",      sd[f"{sd_prefix}.attn.attn_k.weight"])
        write_tensor(writer, f"{prefix}.attn_v.weight",      sd[f"{sd_prefix}.attn.attn_v.weight"])
        write_tensor(writer, f"{prefix}.attn_output.weight", sd[f"{sd_prefix}.attn.attn_output.weight"])
        write_tensor(writer, f"{prefix}.ffn_norm.weight",    sd[f"{sd_prefix}.ffn_norm.weight"])
        write_tensor(writer, f"{prefix}.ffn_gate.weight",    sd[f"{sd_prefix}.ffn.ffn_gate.weight"])
        write_tensor(writer, f"{prefix}.ffn_up.weight",      sd[f"{sd_prefix}.ffn.ffn_up.weight"])
        write_tensor(writer, f"{prefix}.ffn_down.weight",    sd[f"{sd_prefix}.ffn.ffn_down.weight"])
        print(f"  layer {i+1}/{cfg.n_layer} written", end="\r", flush=True)

    print()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    size_gb = Path(output_path).stat().st_size / 1e9
    print(f"[export] done: {size_gb:.2f} GB written to {output_path}")


# ---------------------------------------------------------------------------
# NCTR export — the .nctr counterpart to export_gguf() above. Reads the
# exact same checkpoint format (same torch.load, same NanityConfig, same
# state_dict key convention) so a checkpoint can be exported to EITHER
# container from the same training run, for the GGUF-vs-NCTR bit-identical
# output comparison. Layout matches rawllm_nctr_loader.hpp's NCTRHeader and
# build_tensor_table() exactly -- if either changes, update both sides.
# ---------------------------------------------------------------------------

def export_nctr(
    checkpoint_path: str,
    output_path: str,
    tokenizer_hf: str,
    lora_checkpoint: Optional[str] = None,
):
    """Export a training checkpoint to a NANITY .nctr file (F32 only for
    now -- quantized NCTR export, like quantized GGUF export, is a useful
    follow-up but not required to get a model loading)."""
    import struct
    import numpy as np
    from transformers import AutoTokenizer

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("nanity_config", {})
    cfg = NanityConfig(**cfg_dict)
    model = NanityForCausalLM(cfg)
    model.load_state_dict(ckpt["model"])

    if lora_checkpoint:
        print(f"[export] merging LoRA adapter from {lora_checkpoint} ...")
        lora_ckpt = torch.load(lora_checkpoint, map_location="cpu", weights_only=False)
        lora_state = lora_ckpt["lora_state"]
        for i, block in enumerate(model.blk):
            for sub_name in ("attn", "ffn"):
                sub = getattr(block, sub_name)
                for name in LORA_TARGET_DEFAULT:
                    key = f"blk.{i}.{sub_name}.{name}"
                    if f"{key}.lora_A" not in lora_state:
                        continue
                    orig = getattr(sub, name)
                    A = lora_state[f"{key}.lora_A"]
                    B = lora_state[f"{key}.lora_B"]
                    scaling = lora_state[f"{key}.scaling"]
                    delta = (B @ A) * scaling
                    orig.weight = nn.Parameter(orig.weight + delta.to(orig.weight.dtype))

    model.eval()
    sd = model.state_dict()

    def f32_bytes(t: torch.Tensor) -> bytes:
        return t.detach().float().numpy().astype(np.float32).tobytes()

    # ── manifest (see the earlier .nctr manifest schema discussion) ────────
    manifest = {
        "nctr_manifest_version": 1,
        "architecture": {"name": "nanity", "spec_version": cfg.spec_version},
        "model": {
            "n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_kv_head": cfg.n_kv_head,
            "head_dim": cfg.head_dim, "n_ff": cfg.n_ff, "ctx_len": cfg.context_length,
        },
        "base_weights": {"type": "from_scratch"},
        "tokenizer": {"source": "huggingface", "repo": tokenizer_hf},
        "training_trajectory": {"steps_completed": ckpt.get("step", 0)},
        "convergence_snapshot": {"final_training_loss": ckpt.get("loss")},
        "reproducibility_env": {"precision": "fp32"},
    }
    manifest_bytes = json.dumps(manifest).encode("utf-8")

    # ── tokenizer section (plain JSON, same shape NCTRLoader expects) ──────
    print(f"[export] loading HF tokenizer {tokenizer_hf} ...")
    tok = AutoTokenizer.from_pretrained(tokenizer_hf, trust_remote_code=True)
    vocab = tok.get_vocab()
    vocab_size = max(vocab.values()) + 1
    reverse_vocab = {v: k for k, v in vocab.items()}
    tokens = [reverse_vocab.get(i, f"[UNUSED{i}]") for i in range(vocab_size)]
    tokenizer_json = {
        "tokens": tokens,
        "merges": [],   # NEON's greedy matcher doesn't consume merges (see spec §7); omitted, not lost
        "bos_id": tok.bos_token_id if tok.bos_token_id is not None else 1,
        "eos_id": tok.eos_token_id if tok.eos_token_id is not None else 2,
        "unk_id": tok.unk_token_id if tok.unk_token_id is not None else 0,
    }
    tokenizer_bytes = json.dumps(tokenizer_json).encode("utf-8")

    if vocab_size != cfg.vocab_size:
        print(f"[export] WARNING: tokenizer vocab ({vocab_size}) != "
              f"cfg.vocab_size ({cfg.vocab_size}) -- token_embd.weight's "
              f"row count won't match the tokenizer NEON loads. This is "
              f"almost certainly wrong; double check --tokenizer matches "
              f"what this checkpoint was actually trained against.")

    # ── tensor table, in the EXACT order build_tensor_table() expects ──────
    order = ["token_embd.weight", "output_norm.weight"]
    if "output.weight" in sd:
        order.append("output.weight")
    for i in range(cfg.n_layer):
        p = f"blk.{i}"
        order += [
            f"{p}.attn_norm.weight",
            (f"{p}.attn_q.weight",      f"{p}.attn.attn_q.weight"),
            (f"{p}.attn_k.weight",      f"{p}.attn.attn_k.weight"),
            (f"{p}.attn_v.weight",      f"{p}.attn.attn_v.weight"),
            (f"{p}.attn_output.weight", f"{p}.attn.attn_output.weight"),
            f"{p}.ffn_norm.weight",
            (f"{p}.ffn_gate.weight",    f"{p}.ffn.ffn_gate.weight"),
            (f"{p}.ffn_up.weight",      f"{p}.ffn.ffn_up.weight"),
            (f"{p}.ffn_down.weight",    f"{p}.ffn.ffn_down.weight"),
        ]

    data_blobs, table_entries = [], []
    cursor = 0
    for entry in order:
        sd_key = entry[1] if isinstance(entry, tuple) else entry
        blob = f32_bytes(sd[sd_key])
        table_entries.append((0, cursor, len(blob)))   # quant_type=0 (F32)
        data_blobs.append(blob)
        cursor += len(blob)
    data_section = b"".join(data_blobs)
    tensor_table = b"".join(struct.pack("<IQQ", t, o, n) for t, o, n in table_entries)

    HEADER_SIZE = 116
    manifest_off  = HEADER_SIZE
    tokenizer_off = manifest_off + len(manifest_bytes)
    table_off     = tokenizer_off + len(tokenizer_bytes)
    data_off      = table_off + len(tensor_table)

    use_swiglu     = "output.weight" not in sd or True  # NANITY spec v1 has no non-SwiGLU export path here
    has_output     = "output.weight" in sd
    flags = (1 if use_swiglu else 0) | ((1 << 1) if has_output else 0)

    header = struct.pack(
        "<4s" + "I"*10 + "ff" + "I" + "f" + "II" + "Q"*6,
        b"NCTR", 1, cfg.spec_version,
        cfg.vocab_size, cfg.n_embd, cfg.n_layer,
        cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.n_ff, cfg.context_length,
        cfg.rope_freq_base, cfg.rope_scale_linear, cfg.rope_dimension_count,
        cfg.rms_norm_eps, flags, 32,
        manifest_off, len(manifest_bytes),
        tokenizer_off, len(tokenizer_bytes),
        table_off, data_off,
    )
    assert len(header) == HEADER_SIZE, f"header packed to {len(header)} bytes, expected {HEADER_SIZE}"

    with open(output_path, "wb") as f:
        f.write(header + manifest_bytes + tokenizer_bytes + tensor_table + data_section)

    size_gb = Path(output_path).stat().st_size / 1e9
    print(f"[export] done: {size_gb:.2f} GB written to {output_path} "
          f"({len(order)} tensors, F32)")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # ── device ──────────────────────────────────────────────────────────────
    # BUG: this used to just check torch.cuda.is_available() and, if False,
    # print one easy-to-miss WARNING line and silently continue on CPU --
    # which on a 4.5B model looks indistinguishable from "hanging" (a step
    # that takes seconds on MI300X takes many minutes on CPU, with the same
    # sparse print cadence). On a ROCm box "no GPU found" is almost always
    # one of a small number of causes, so surface them instead of guessing:
    #   1. torch was installed from the default PyPI index (CPU/CUDA wheel)
    #      instead of the ROCm wheel (--index-url .../whl/rocm6.2) -- the
    #      single most common cause. Check: python3 -c "import torch;
    #      print(torch.version.hip)" -- if that prints None, this is it.
    #   2. Container/session was started without GPU device passthrough
    #      (needs --device=/dev/kfd --device=/dev/dri and the render group).
    #   3. ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES is set to empty or
    #      excludes the card (check `echo $ROCR_VISIBLE_DEVICES`).
    #   4. rocminfo doesn't see the card at all (driver/host issue, not a
    #      Python/torch issue).
    hip_version = getattr(torch.version, "hip", None)
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[hw] GPU: {gpu_name}  (torch={torch.__version__}, "
              f"rocm/hip={hip_version or 'n/a'})")
        is_mi300x = "MI300X" in gpu_name or "MI300" in gpu_name
        if os.environ.get("PYTORCH_HIP_ALLOC_CONF") is None and \
           os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None:
            print("[hw] tip: if you see OOM after many steps despite a stable "
                  "per-step memory footprint, that's usually allocator "
                  "fragmentation, not a real leak. Try setting "
                  "PYTORCH_HIP_ALLOC_CONF=expandable_segments:True (or "
                  "PYTORCH_CUDA_ALLOC_CONF on a CUDA build) before launching.")
    else:
        device = "cpu"
        is_mi300x = False
        print("[hw] ERROR: no GPU visible to torch -- refusing to silently "
              f"train a 4-5B model on CPU. Diagnostics: torch={torch.__version__}, "
              f"torch.version.hip={hip_version!r}, "
              f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')!r}, "
              f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')!r}.")
        if hip_version is None:
            print("[hw]   -> torch.version.hip is None: this is very likely a "
                  "CPU-only or CUDA (not ROCm) torch build. Reinstall with:\n"
                  "         pip install torch torchvision torchaudio "
                  "--index-url https://download.pytorch.org/whl/rocm6.2")
        else:
            print("[hw]   -> torch IS a ROCm build but still sees no device: "
                  "check the container was launched with GPU passthrough "
                  "(--device=/dev/kfd --device=/dev/dri) and that "
                  "`rocminfo` / `rocm-smi` see the card from this shell.")
        if not args.allow_cpu:
            sys.exit("[hw] aborting (pass --allow-cpu to force CPU training "
                      "anyway, e.g. for a quick smoke test).")
        print("[hw] --allow-cpu set: continuing on CPU anyway (will be very slow).")

    # ── config ───────────────────────────────────────────────────────────────
    # BUG FIX: args.resume used to be loaded with torch.load() up to THREE
    # separate times (config here, weights below, plus the never-called
    # load_checkpoint() helper duplicating both). Each load pulls the WHOLE
    # checkpoint -- model weights AND optimizer moments, which for a 4.5B
    # model + AdamW is several times the bare model size -- into host RAM.
    # Holding 2-3 copies alive at once during resume was real memory
    # pressure and part of what was pushing this into OOM. Load once, reuse.
    ckpt_data = None
    if args.resume:
        ckpt_data = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = NanityConfig(**ckpt_data["nanity_config"])
        print(f"[config] loaded from checkpoint: {cfg}")
    else:
        cfg = {"4b": NanityConfig.nanity_4b, "1_5b": NanityConfig.nanity_1_5b,
               "tiny": NanityConfig.nanity_tiny}[args.preset]()
        print(f"[config] fresh {args.preset} config: {cfg}")

    # ── model ────────────────────────────────────────────────────────────────
    model = NanityForCausalLM(cfg).to(device)

    # BF16 conversion — MI300X has native BF16 HGEMM (2x FP32 throughput).
    # Do this BEFORE torch.compile so the compiler sees BF16 ops.
    if device == "cuda":
        model = model.to(torch.bfloat16)
        print("[precision] BF16 enabled")

    # ── load base weights + optionally inject LoRA ─────────────────────────
    # For LoRA runs, --resume must point at a BASE (capability) checkpoint,
    # not a previous LoRA adapter -- this loads that base, freezes it, and
    # injects fresh trainable low-rank adapters on top. Full fine-tune runs
    # (--lora-rank 0, the default) resume normally with everything trainable.
    start_step = 0
    if args.lora_rank > 0:
        if not args.resume:
            sys.exit("[error] --lora-rank requires --resume pointing at a trained "
                      "base capability checkpoint (LoRA adapts an existing base, "
                      "it doesn't train one from scratch).")
        model.load_state_dict(ckpt_data["model"])
        print(f"[lora] loaded base weights from {args.resume}")
        lora_params = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha,
                                   targets=args.lora_target)
        # start_step stays 0 -- this is a fresh adapter training run, distinct
        # from the base checkpoint's own step count.
    elif args.resume:
        model.load_state_dict(ckpt_data["model"])
        start_step = ckpt_data.get("step", 0)
        print(f"[resume] loaded full checkpoint from {args.resume} at step {start_step}")

    # torch.compile — works on ROCm via inductor backend; ~20-40% throughput
    # improvement on MI300X with almost zero code change. Done AFTER LoRA
    # injection so the compiled graph includes the adapter modules.
    if is_mi300x and not args.no_compile:
        print("[compile] torch.compile(inductor) ...")
        model = torch.compile(model, backend="inductor")
        print("[compile] done")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params / 1e9:.3f}B trainable parameters")

    # ── tokenizer ────────────────────────────────────────────────────────────
    tok_source = args.tokenizer or "microsoft/Phi-4-mini-instruct"
    print(f"[tokenizer] loading from {tok_source} ...")
    tokenizer = load_tokenizer(tok_source)
    pad_id = tokenizer.pad_token_id or 0

    # ── dataset ──────────────────────────────────────────────────────────────
    max_len = args.max_seq_len or cfg.context_length
    if args.max_seq_len and args.max_seq_len > cfg.context_length:
        print(f"[warn] --max-seq-len {args.max_seq_len} > cfg.context_length "
              f"{cfg.context_length}. Training will proceed at this length, "
              f"but remember to update context_length in modeling_nanity.py "
              f"and the KV-cache allocation in NEON.cpp before this "
              f"checkpoint is exported/served, or inference will be capped "
              f"back down to {cfg.context_length}.")
    think_end_markers = (args.think_end_markers.split(",") if args.think_end_markers
                         else None)
    ds = ConversationDataset(args.data, tokenizer, max_len=max_len,
                             think_end_markers=think_end_markers,
                             tokenizer_source=args.tokenizer)
    if len(ds) == 0:
        sys.exit("[error] empty dataset — check your JSONL format")

    # held-out validation set: either an explicit --val-data file, or carved
    # out of --data automatically. Either way it never appears in `loader`.
    if args.val_data:
        train_ds = ds
        val_ds = ConversationDataset(args.val_data, tokenizer, max_len=max_len,
                                     think_end_markers=think_end_markers,
                                     tokenizer_source=args.tokenizer)
        print(f"[val] using separate validation file: {args.val_data} "
              f"({len(val_ds)} examples)")
    else:
        train_ds, val_ds = split_train_val(ds, args.val_split)

    # BUG FIX (Python 3.14 / any 'forkserver' or 'spawn' start method):
    # `collate_fn=lambda b: collate_fn(b, pad_id=pad_id)` looks harmless but
    # a lambda closing over a local variable is NOT picklable, and
    # DataLoader(num_workers>0) has to pickle collate_fn to hand it to each
    # worker process -- UNLESS the start method is 'fork', which doesn't
    # pickle at all, it just copies the parent's memory (closures included)
    # via the OS fork() call. That's exactly why this worked silently on
    # setups using 'fork' (the POSIX default on older Python) and crashed
    # the moment it ran under 'forkserver' or 'spawn' (Windows/macOS
    # always, and POSIX Python 3.14+ by default):
    # _pickle.PicklingError: Can't pickle local object
    # 'train.<locals>.<lambda>'. functools.partial over the existing
    # MODULE-level collate_fn (defined above, not a closure) IS picklable --
    # same behavior, works under all three start methods, so it doesn't
    # matter what Python version or OS this runs under.
    collate = functools.partial(collate_fn, pad_id=pad_id)

    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=2,
            pin_memory=(device == "cuda"),
            collate_fn=collate,
            drop_last=False,
        )
    else:
        print("[val] WARNING: no validation set available -- early stopping "
              "and overfit detection are DISABLED. You are back to eyeballing "
              "train loss and guessing when to abort. Pass --val-data or "
              "leave --val-split at its default to avoid this.")

    # ── vocab freezing ───────────────────────────────────────────────────────
    used_vocab_mask = None
    if args.freeze_unseen_vocab:
        if args.lora_rank > 0:
            print("[vocab] --freeze-unseen-vocab has no effect under "
                  "--lora-rank > 0 -- LoRA already leaves the embedding "
                  "entirely frozen.")
        else:
            used_vocab_mask = compute_used_vocab_mask(
                [train_ds, val_ds], cfg.vocab_size, tokenizer=tokenizer
            ).to(device)
            n_used = int(used_vocab_mask.sum().item())
            print(f"[vocab] {n_used:,} / {cfg.vocab_size:,} vocab rows appear "
                  f"in the data ({100 * n_used / cfg.vocab_size:.1f}%). "
                  f"The remaining {cfg.vocab_size - n_used:,} rows will have "
                  f"their gradient zeroed every step (still random-init, "
                  f"ready for a future language) instead of being trained.")

    # ── epoch-aware step sizing ──────────────────────────────────────────────
    # Train loss alone doesn't tell you "too many steps"; but epoch count
    # over a KNOWN dataset size is still a useful sanity cap, because it's
    # the number of times the model has literally seen each example.
    #
    # BUG FIX: `step` in the training loop below increments once per
    # MICRO-batch (once per item pulled from `loader`, i.e. every args.batch
    # examples) -- an optimizer.step() only happens every args.grad_accum
    # micro-batches. --steps is compared directly against that micro-batch
    # counter (`if step >= start_step + args.steps: break`), so --steps is
    # in micro-batch units. This block used to compute
    # `steps_per_epoch = len(train_ds) // (batch*grad_accum)` -- i.e.
    # OPTIMIZER-step units -- and then divided the micro-batch-unit
    # `args.steps` by it. That mismatch inflated implied_epochs by exactly
    # grad_accum (e.g. 8x with the default --grad-accum 8), so the
    # --max-epochs cap fired far earlier than intended and silently
    # truncated `args.steps` (also assigned back in the wrong, optimizer-step
    # unit) to a small fraction of what was actually requested/needed.
    effective_batch = args.batch * args.grad_accum
    steps_per_epoch_optim = max(1, len(train_ds) // effective_batch)      # optimizer-step units, for display
    steps_per_epoch_micro = max(1, len(train_ds) // args.batch)           # micro-batch units, matches args.steps
    implied_epochs = args.steps / steps_per_epoch_micro
    print(f"[epochs] {len(train_ds)} train examples, effective batch "
          f"{effective_batch} -> {steps_per_epoch_optim} optimizer steps/epoch "
          f"({steps_per_epoch_micro} micro-batches/epoch). "
          f"--steps={args.steps} implies ~{implied_epochs:.1f} epochs.")
    if args.max_epochs and args.max_epochs > 0 and implied_epochs > args.max_epochs:
        clipped_steps = int(steps_per_epoch_micro * args.max_epochs)
        print(f"[epochs] WARNING: {implied_epochs:.1f} epochs over a "
              f"{len(train_ds)}-example dataset is very likely to memorize "
              f"a {cfg.n_layer}-layer / multi-billion-param model. Clipping "
              f"--steps {args.steps} -> {clipped_steps} (--max-epochs "
              f"{args.max_epochs}). Pass --max-epochs 0 to disable this cap "
              f"if you're confident you want the full run.")
        args.steps = clipped_steps

    # ── optimizer ─────────────────────────────────────────────────────────────
    # LoRA runs: optimizer only ever sees the small adapter params. Full
    # fine-tune runs: split into two param groups so embeddings/norms get
    # weight_decay=0 -- standard practice regardless, but ALSO fixes a real
    # gap in --freeze-unseen-vocab: zeroing the gradient on unused embedding
    # rows doesn't stop AdamW's decoupled weight decay from still nudging
    # those rows toward zero every step (decay applies directly to the
    # parameter value, independent of the gradient). No decay on this
    # tensor means "frozen" rows are actually fully frozen, not just
    # slow-drifting.
    if args.lora_rank > 0:
        param_groups = lora_params
    else:
        decay_params, no_decay_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "token_embd" in name or "output" in name or "norm" in name:
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        param_groups = [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        print(f"[optimizer] {sum(p.numel() for p in decay_params):,} params with weight_decay, "
              f"{sum(p.numel() for p in no_decay_params):,} without (embeddings/norms)")
    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-6,
            fused=True,
        )
        print("[optimizer] fused AdamW")
    except TypeError:
        optimizer = torch.optim.AdamW(
            param_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-6,
        )
        print("[optimizer] standard AdamW (fused not available)")

    # BUG FIX: optimizer state (Adam's per-parameter running mean/variance)
    # was never restored on resume -- model.load_state_dict() ran above, but
    # nothing ever called optimizer.load_state_dict(ckpt_data["optimizer"]).
    # A load_checkpoint() helper earlier in this file does this correctly
    # but was dead code, never called anywhere. Net effect: every resume
    # silently restarted Adam's moments from zero while the model weights
    # jumped back in from a fully-warmed-up state. The mismatch between
    # "well-trained weights" and "zeroed optimizer state" produces a burst
    # of oversized effective step sizes right after resume -- exactly the
    # kind of transient that spikes activations/gradients and can tip a
    # borderline-fitting run into OOM, on top of silently corrupting the
    # optimization trajectory every time you resumed. Only restore for a
    # full-fidelity resume (not LoRA, which builds a fresh adapter optimizer
    # every time by design, and not export-only invocations).
    if args.resume and args.lora_rank == 0 and ckpt_data is not None and "optimizer" in ckpt_data:
        try:
            optimizer.load_state_dict(ckpt_data["optimizer"])
            print("[resume] restored optimizer state (Adam moments) from checkpoint")
        except (ValueError, KeyError) as e:
            print(f"[resume] WARNING: could not restore optimizer state "
                  f"({e}) -- continuing with freshly-initialized optimizer "
                  f"state. This usually means param_groups shape changed "
                  f"(e.g. --freeze-unseen-vocab toggled between runs).")

    # free the raw checkpoint dict now that weights + optimizer state have
    # been consumed -- no reason to keep a second full copy of the model's
    # state_dict (tensors) alive in host RAM for the rest of the run.
    del ckpt_data

    # ── teacher (online distillation only) ───────────────────────────────────
    teacher = None
    if args.phase == "distill" and args.teacher:
        teacher = Teacher(args.teacher, device)

    # ── gradient checkpointing — saves memory at the cost of ~30% recompute.
    # With 192GB HBM this is optional for a 4B model, but enables larger
    # batch sizes (better gradient estimates) for the same memory budget.
    if args.grad_ckpt:
        from torch.utils.checkpoint import checkpoint as grad_checkpoint
        # patch each block's forward to use gradient checkpointing
        for block in (model.blk if hasattr(model, "blk") else model._orig_mod.blk):
            orig_fwd = block.forward
            def make_ckpt_fwd(f):
                def ckpt_fwd(*a, **kw):
                    return grad_checkpoint(f, *a, use_reentrant=False, **kw)
                return ckpt_fwd
            block.forward = make_ckpt_fwd(orig_fwd)
        print("[memory] gradient checkpointing enabled (~30% recompute overhead)")

    # ── training loop ─────────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler = None   # no GradScaler needed for BF16 on MI300X (BF16 doesn't overflow)

    step       = start_step
    total_loss = 0.0
    avg_loss   = 0.0
    total_think_loss  = 0.0
    think_loss_count  = 0
    total_answer_loss = 0.0
    answer_loss_count = 0
    tokens_since_log  = 0   # BUG FIX: see tok/s computation below
    log_every  = 50
    save_every = args.save_every
    t0         = time.time()

    best_val_loss     = float("inf")
    patience_counter  = 0
    best_ckpt_step    = start_step
    lr                = args.lr

    optimizer.zero_grad()

    def data_iter() -> Iterator:
        while True:
            yield from loader

    print(f"[train] starting phase={args.phase}, steps={args.steps}, "
          f"batch={args.batch}, grad_accum={args.grad_accum}, "
          f"lr={args.lr:.2e}")

    for input_ids, labels, attn_mask, answer_mask in data_iter():
        if step >= start_step + args.steps:
            break

        input_ids   = input_ids.to(device)
        labels      = labels.to(device)
        attn_mask   = attn_mask.to(device)
        answer_mask = answer_mask.to(device)

        # BUG FIX: forward+backward had no OOM handling at all -- any
        # transient memory spike (a batch with unusually long sequences
        # before padding, a stray memory fragment, etc.) raised an
        # unhandled RuntimeError/OutOfMemoryError straight out of the loop
        # and killed the whole run, discarding all progress since the last
        # checkpoint. Catch it, clear the (poisoned) gradient state and the
        # allocator cache, log GPU memory stats so this is diagnosable
        # instead of a silent kill, skip this one batch, and keep training.
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits, _ = model(input_ids, attention_mask=attn_mask)

                valid_flat  = (labels.view(-1) != -100)
                answer_flat_full = answer_mask.view(-1) & valid_flat

                # BUG FIX (major OOM cause): the previous version did
                # `logits.view(-1, V).float()` over EVERY position in the
                # batch -- including every system/user/prompt token, which
                # the loss mask excludes anyway. For vocab_size ~200k, that
                # fp32 tensor is (batch * seq_len * 200064 * 4 bytes), e.g.
                # at batch=4, seq=4096 that's ~13GB for logits alone, on top
                # of the ~6.5GB bf16 tensor already alive, PLUS
                # cross_entropy's internal softmax buffers of the same size
                # again -- for a single loss computation, before backward
                # even starts. Since assistant-turn tokens (answer + think)
                # are typically a small fraction of a full context window,
                # selecting them BEFORE the fp32 cast cuts this
                # proportionally -- e.g. ~5-10x smaller if only ~15% of the
                # sequence is assistant tokens, which is common for
                # reasoning data with long user/system prompts.
                V = logits.shape[-1]
                logits_flat = logits.view(-1, V)
                valid_idx = valid_flat.nonzero(as_tuple=True)[0]
                logits_valid = logits_flat.index_select(0, valid_idx).float()
                labels_valid = labels.view(-1).index_select(0, valid_idx)
                answer_sub  = answer_flat_full.index_select(0, valid_idx)   # per valid tok
                think_sub   = ~answer_sub

                # ── Phase-dependent loss ─────────────────────────────────
                if teacher is not None and args.phase == "distill":
                    # KL divergence between teacher and student on response
                    # tokens. This is the core of distillation: the student
                    # learns to match the teacher's FULL distribution (not
                    # just the argmax), which transfers substantially more
                    # information per example.
                    with torch.no_grad():
                        t_logits = teacher.logits(input_ids, attn_mask)
                        # align vocab size if teacher and student differ
                        v_min = min(V, t_logits.shape[-1])
                        # same fix as above: select valid rows BEFORE the
                        # log_softmax/exp blowup over the vocab dimension.
                        t_logits_valid = t_logits[..., :v_min].view(-1, v_min).index_select(0, valid_idx)
                    s_logits_valid = logits_valid[..., :v_min]
                    t_log_probs = F.log_softmax(t_logits_valid / args.temp_distill, dim=-1)
                    s_log_probs = F.log_softmax(s_logits_valid / args.temp_distill, dim=-1)

                    # KL(teacher || student) per-token, over response tokens only
                    kl_per_tok = F.kl_div(s_log_probs, t_log_probs.exp(),
                                          reduction="none").sum(-1)   # [n_valid]

                    # CE per-token on ground-truth labels (anchors generation
                    # to the actual reference text, prevents mode drift away
                    # from the target format)
                    ce_per_tok = F.cross_entropy(
                        logits_valid, labels_valid,
                        label_smoothing=args.label_smoothing,
                        reduction="none",
                    )
                    per_tok = 0.7 * kl_per_tok + 0.3 * ce_per_tok

                else:
                    # Phase 1 (warmup) or Phase 3 (finetune): plain
                    # cross-entropy. label_smoothing caps how low the loss
                    # can go for a single memorized token, which is the main
                    # thing that lets a 4-5B model "solve" a small dataset
                    # by reciting it instead of generalizing from it.
                    per_tok = F.cross_entropy(
                        logits_valid, labels_valid,
                        label_smoothing=args.label_smoothing,
                        reduction="none",
                    )

                # Reasoning-trace data (e.g. OpenThoughts) puts a lot more
                # <think> tokens than final-answer tokens into each example,
                # and <think> tokens are typically far more
                # formulaic/low-entropy. A single scalar loss over all of
                # them is easy to fool: the model can crater the average by
                # nailing repetitive reasoning boilerplate while barely
                # moving on the answer. answer_weight (default 1.0 = no
                # reweighting) lets the ACTUAL optimization target emphasize
                # answer tokens; the separate think/answer numbers below let
                # you see the split regardless of the weight. (per_tok/
                # answer_sub/think_sub are already restricted to valid
                # tokens, so no need to multiply by valid_flat again here.)
                weights = torch.ones_like(per_tok)
                weights[answer_sub] = args.answer_loss_weight
                denom = weights.sum().clamp(min=1e-8)
                loss = (per_tok * weights).sum() / denom

                think_loss_step = per_tok[think_sub].mean().item() if think_sub.any() else float("nan")
                answer_loss_step = per_tok[answer_sub].mean().item() if answer_sub.any() else float("nan")

            (loss / args.grad_accum).backward()
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise   # not an OOM -- a real bug, don't paper over it
            optimizer.zero_grad(set_to_none=True)
            if device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved  = torch.cuda.memory_reserved() / 1e9
                torch.cuda.empty_cache()
                print(f"[oom] step {step}: CUDA/HIP OOM on a batch of shape "
                      f"{tuple(input_ids.shape)} (allocated={allocated:.1f}GB, "
                      f"reserved={reserved:.1f}GB before cache clear). Skipping "
                      f"this batch and clearing the allocator cache. If this "
                      f"repeats often, lower --batch, raise --grad-accum to "
                      f"compensate, or pass --grad-ckpt to trade compute for "
                      f"memory.")
            else:
                print(f"[oom] step {step}: out-of-memory on batch shape "
                      f"{tuple(input_ids.shape)}. Skipping this batch.")
            continue

        total_loss += loss.item()
        tokens_since_log += int(attn_mask.sum().item())
        if not math.isnan(think_loss_step):
            total_think_loss += think_loss_step
            think_loss_count += 1
        if not math.isnan(answer_loss_step):
            total_answer_loss += answer_loss_step
            answer_loss_count += 1

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if used_vocab_mask is not None:
                # Zero the gradient on every vocab row that never appeared
                # in the data -- applied AFTER clipping (so unused rows
                # don't influence the clip norm's scale) and before the
                # optimizer step (so decoupled weight decay never touches
                # them either, since AdamW applies decay to every param
                # every step regardless of whether its grad is zero... but
                # zero grad here still means zero *update* from the moment
                # -- see note below on weight_decay).
                base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                embd_grad = base_model.token_embd.weight.grad
                if embd_grad is not None:
                    embd_grad[~used_vocab_mask] = 0
                if base_model.output is not None and base_model.output.weight.grad is not None:
                    base_model.output.weight.grad[~used_vocab_mask] = 0

            # update LR
            lr = cosine_lr(step, args.warmup_steps, args.steps, args.lr, args.lr * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.step()
            optimizer.zero_grad()

        step += 1

        if step % log_every == 0:
            avg_loss = total_loss / log_every
            avg_think = total_think_loss / max(think_loss_count, 1)
            avg_answer = total_answer_loss / max(answer_loss_count, 1)
            dt = time.time() - t0
            # BUG FIX: this used to assume every sequence in every batch was
            # exactly cfg.context_length tokens (log_every * batch *
            # context_length), but collate_fn right-pads only to the
            # longest item IN EACH BATCH -- for chat/reasoning data that's
            # usually well under context_length, so tok/s was consistently
            # overstated (sometimes by a large factor for short examples).
            # Use the actual attn_mask token count accumulated above.
            tok_per_sec = tokens_since_log / dt
            print(f"[step {step:7d}] loss={avg_loss:.4f}  "
                  f"think={avg_think:.4f}  answer={avg_answer:.4f}  "
                  f"lr={lr:.2e}  tok/s={tok_per_sec:,.0f}  dt={dt:.1f}s")
            total_loss = 0.0
            total_think_loss = 0.0
            think_loss_count = 0
            total_answer_loss = 0.0
            answer_loss_count = 0
            tokens_since_log = 0
            t0 = time.time()

        if step % save_every == 0 or step == start_step + args.steps:
            if args.lora_rank > 0:
                save_lora_checkpoint(model, step, out_dir)
            else:
                save_checkpoint(model, optimizer, step, loss.item(), out_dir, cfg)

        # ── validation + early stopping ──────────────────────────────────────
        # This is the actual replacement for "watch the loss and abort by
        # hand": val loss is the only number in this loop that can tell
        # memorization apart from learning. Rising/flat val loss next to a
        # still-falling train loss means the gap between them IS overfitting,
        # regardless of how low train loss has gotten.
        #
        # For reasoning data specifically, early-stopping is tracked on
        # ANSWER loss, not overall loss -- overall loss is dominated by the
        # (usually much more numerous, much more formulaic) think tokens, so
        # it can keep falling from reasoning-style fluency gains long after
        # the model has stopped improving on the thing that actually
        # matters: getting the final answer right.
        if val_loader is not None and step % args.val_every == 0 and step > start_step:
            val_metrics = evaluate(model, val_loader, device, args.val_batches)
            val_loss, val_think, val_answer = (
                val_metrics["loss"], val_metrics["think_loss"], val_metrics["answer_loss"])
            track_metric = val_answer if not math.isnan(val_answer) else val_loss
            print(f"[val   {step:7d}] loss={val_loss:.4f}  think={val_think:.4f}  "
                  f"answer={val_answer:.4f}  (train~{avg_loss:.4f})")

            if track_metric < best_val_loss - args.early_stop_min_delta:
                best_val_loss = track_metric
                patience_counter = 0
                if args.lora_rank > 0:
                    saved_path = save_lora_checkpoint(model, step, out_dir)
                    best_path = out_dir / "lora_best.pt"
                else:
                    saved_path = save_checkpoint(model, optimizer, step, loss.item(), out_dir, cfg)
                    best_path = out_dir / "ckpt_best.pt"
                # copy (not just track the step number) so this survives the
                # keep-3-most-recent rotation in save_checkpoint() even if
                # training continues well past this point before stopping.
                import shutil
                shutil.copy2(saved_path, best_path)
                best_ckpt_step = step
                print(f"[val   {step:7d}] new best (answer_loss={best_val_loss:.4f}) "
                      f"-- checkpoint saved ({best_path})")
            else:
                patience_counter += 1
                print(f"[val   {step:7d}] no improvement "
                      f"({patience_counter}/{args.early_stop_patience} "
                      f"since best={best_val_loss:.4f})")
                if (args.early_stop_patience > 0
                        and patience_counter >= args.early_stop_patience):
                    print(f"[early-stop] answer loss hasn't improved in "
                          f"{args.early_stop_patience} checks -- from here "
                          f"on the model is very likely just getting more "
                          f"fluent at reasoning-style text without getting "
                          f"more correct. Stopping at step {step}. Best "
                          f"checkpoint was at step {best_ckpt_step} "
                          f"(answer_loss={best_val_loss:.4f}); that's the "
                          f"one to export.")
                    break

    print(f"[train] phase {args.phase} complete at step {step}.")

    # auto-export GGUF at end of training, unless --no-auto-export was passed.
    # Uses --tokenizer-gguf if given (copies keys from a donor GGUF),
    # otherwise builds tokenizer keys fresh from --tokenizer (HF id/path).
    if not args.no_auto_export:
        # If we had a validation set, export the BEST checkpoint (lowest val
        # loss), not necessarily the last one written -- if early stopping
        # fired, or val loss simply bottomed out before the final save, the
        # last step's checkpoint is the more-overfit one.
        use_best = val_loader is not None and best_ckpt_step > start_step
        if use_best:
            print(f"[export] using best checkpoint (step {best_ckpt_step}, "
                  f"val_loss={best_val_loss:.4f}) instead of final step {step}")
        out_gguf = str(out_dir / "nanity_4b_final.gguf")
        tok_gguf = args.tokenizer_gguf
        tok_hf = None if tok_gguf else args.tokenizer
        if args.lora_rank > 0:
            lora_ckpt = out_dir / "lora_best.pt" if use_best else out_dir / f"lora_{step:07d}.pt"
            export_gguf(args.resume, out_gguf,
                        tokenizer_source_gguf=tok_gguf, tokenizer_hf=tok_hf,
                        quant=args.export_quant, lora_checkpoint=str(lora_ckpt))
        else:
            final_ckpt = out_dir / "ckpt_best.pt" if use_best else out_dir / f"ckpt_{step:07d}.pt"
            export_gguf(str(final_ckpt), out_gguf,
                        tokenizer_source_gguf=tok_gguf, tokenizer_hf=tok_hf,
                        quant=args.export_quant)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode")

    # -- train mode (default if --phase is given) ----------------------------
    tr = ap
    tr.add_argument("--preset",      choices=["4b", "1_5b", "tiny"], default="4b",
                     help="fresh-init architecture preset when not resuming from a checkpoint "
                          "('tiny' is a ~13M-param CPU smoke-test preset, not a real training config)")
    tr.add_argument("--phase",       choices=["warmup", "distill", "finetune"],
                    default="distill")
    tr.add_argument("--data",        help="path to JSONL training data")
    tr.add_argument("--out",         default="checkpoints/nanity_4b",
                    help="checkpoint output directory")
    tr.add_argument("--resume",      help="path to .pt checkpoint to resume from")
    tr.add_argument("--teacher",     help="HF model id or path for online distillation "
                                          "(only used with --phase distill)")
    tr.add_argument("--tokenizer",   default="microsoft/Phi-4-mini-instruct",
                    help="HF tokenizer id or path (default: Phi-4-mini-instruct)")
    tr.add_argument("--tokenizer-gguf", dest="tokenizer_gguf",
                    help="existing NANITY GGUF to copy tokenizer keys from "
                         "(used for auto-export at end of training)")
    tr.add_argument("--steps",       type=int, default=50000)
    tr.add_argument("--batch",       type=int, default=4,
                    help="per-GPU batch size")
    tr.add_argument("--grad-accum",  dest="grad_accum", type=int, default=8,
                    help="gradient accumulation steps (effective batch = batch * grad_accum)")
    tr.add_argument("--lr",          type=float, default=3e-4)
    tr.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=500)
    tr.add_argument("--save-every",  dest="save_every", type=int, default=2000)
    tr.add_argument("--grad-ckpt",   dest="grad_ckpt", action="store_true",
                    help="enable gradient checkpointing (saves ~40%% VRAM, ~30%% slower)")
    tr.add_argument("--allow-cpu",   dest="allow_cpu", action="store_true",
                    help="allow training to proceed on CPU if no GPU is visible "
                         "to torch (default: abort with diagnostics, since CPU "
                         "training of a 4-5B model is normally a misconfigured "
                         "ROCm environment, not an intentional choice)")
    tr.add_argument("--lora-rank",   dest="lora_rank", type=int, default=0,
                    help="enable LoRA fine-tuning with this rank (0 = full fine-tune, "
                         "the default). Typical persona-adapter values: 16-32. "
                         "Only meaningful with --phase finetune, and requires --resume "
                         "to start from a trained capability base.")
    tr.add_argument("--lora-alpha",  dest="lora_alpha", type=float, default=32.0,
                    help="LoRA scaling alpha (scaling = alpha / rank)")
    tr.add_argument("--lora-target", dest="lora_target", nargs="+",
                    default=None,
                    help=f"which projections to adapt (default: {LORA_TARGET_DEFAULT})")
    tr.add_argument("--no-compile",  dest="no_compile", action="store_true",
                    help="disable torch.compile (useful for debugging)")
    tr.add_argument("--temp-distill", dest="temp_distill", type=float, default=1.0,
                    help="temperature for distillation KL (1.0 = no sharpening)")
    tr.add_argument("--export-quant", dest="export_quant", default="F16",
                    choices=["F32", "F16", "Q4_0"],
                    help="quantization for auto-export GGUF (default F16)")
    tr.add_argument("--no-auto-export", dest="no_auto_export", action="store_true",
                    help="skip automatic GGUF export at the end of training "
                         "(export manually later with --export/--gguf)")

    # -- overfitting guards ---------------------------------------------------
    # A 4-5B model on a dataset of a few thousand (or even tens of thousands)
    # examples WILL memorize if you run it for enough epochs -- train loss
    # will glide down toward zero and look "great" right up until the model
    # is just reciting training examples back. Train loss alone can't tell
    # you this is happening; you need a held-out set the model never trains
    # on. These flags make that automatic instead of relying on eyeballing
    # the log and Ctrl-C'ing when it "looks too low."
    tr.add_argument("--val-data", dest="val_data",
                    help="path to held-out validation JSONL. If omitted, "
                         "--val-split of --data is carved off automatically "
                         "(deterministic, seeded) and excluded from training.")
    tr.add_argument("--val-split", dest="val_split", type=float, default=0.05,
                    help="fraction of --data to hold out for validation when "
                         "--val-data is not given (default 0.05 = 5%%)")
    tr.add_argument("--val-every", dest="val_every", type=int, default=250,
                    help="run validation every N optimizer steps")
    tr.add_argument("--val-batches", dest="val_batches", type=int, default=30,
                    help="number of validation batches to average per eval")
    tr.add_argument("--early-stop-patience", dest="early_stop_patience", type=int,
                    default=6,
                    help="stop training automatically after this many "
                         "consecutive validations with no improvement in val "
                         "loss (0 = disable early stopping)")
    tr.add_argument("--early-stop-min-delta", dest="early_stop_min_delta",
                    type=float, default=0.01,
                    help="minimum decrease in val loss to count as an "
                         "improvement (guards against stopping on noise)")
    tr.add_argument("--label-smoothing", dest="label_smoothing", type=float,
                    default=0.05,
                    help="label smoothing for the CE loss (default 0.05). "
                         "This is the single biggest lever against the "
                         "'loss collapses to near-zero' failure mode: it "
                         "caps how confident the model is rewarded for "
                         "being on any single token, so it can't just "
                         "memorize exact training strings for free reward. "
                         "Set to 0 to disable.")
    tr.add_argument("--max-epochs", dest="max_epochs", type=float, default=4.0,
                    help="if --steps would run more than this many epochs "
                         "over --data, --steps is clipped down and a warning "
                         "is printed (0 = no cap). Small distillation/"
                         "self-instruct datasets overfit a 4-5B model fast; "
                         "3-4 epochs is already generous for that regime.")

    # -- reasoning-trace (think vs. answer) handling --------------------------
    tr.add_argument("--think-end-markers", dest="think_end_markers", default=None,
                    help="comma-separated list of substrings marking the end "
                         "of a chain-of-thought / start of the final answer "
                         "within an assistant turn (e.g. OpenThoughts uses "
                         "'<|begin_of_solution|>'). Default covers "
                         "'<|begin_of_solution|>', '</think>', "
                         "'<|end_of_thought|>'. Content with none of these "
                         "markers is entirely counted as 'answer' (never "
                         "guessed as reasoning).")
    tr.add_argument("--answer-loss-weight", dest="answer_loss_weight", type=float,
                    default=1.0,
                    help="multiply the loss on final-answer tokens by this "
                         "factor relative to think/reasoning tokens (default "
                         "1.0 = no reweighting). Reasoning datasets have far "
                         "more think tokens than answer tokens per example, "
                         "and think tokens tend to be far more formulaic, so "
                         "an unweighted average loss can look great from "
                         "reasoning-style fluency while answer accuracy "
                         "barely moves. Try 2.0-3.0 if train/val logs show "
                         "'answer' loss stalling while 'think' loss keeps "
                         "dropping.")
    tr.add_argument("--max-seq-len", dest="max_seq_len", type=int, default=None,
                    help="max training sequence length in tokens (default: "
                         "cfg.context_length from the model config). Set "
                         "this explicitly if your data has longer traces "
                         "than the model's configured context_length -- e.g. "
                         "OpenThoughts traces commonly run 6-8k tokens "
                         "against a 4096 context_length, which silently "
                         "truncates every long example. NOTE: this only "
                         "changes what the training script feeds the model; "
                         "if you go above context_length, plan to update "
                         "cfg.context_length in modeling_nanity.py and the "
                         "KV-cache sizing in NEON.cpp to match at export/"
                         "inference time, or the exported GGUF metadata "
                         "will undersell what the checkpoint can actually do.")

    # -- vocab freezing (large vocab_size, English-only training data) -------
    tr.add_argument("--freeze-unseen-vocab", dest="freeze_unseen_vocab",
                    action="store_true",
                    help="pin the embedding/LM-head rows (tied) for every "
                         "vocab id that never appears anywhere in --data to "
                         "their random init -- their gradient is zeroed "
                         "every step, before weight decay too. Without this, "
                         "cross-entropy's softmax normalization still pushes "
                         "every non-target row's logit down a little at "
                         "every position, so 'unused' multilingual rows in a "
                         "large vocab slowly get suppressed toward 'never "
                         "predict this' even on English-only data, instead "
                         "of staying neutral for a future language to claim. "
                         "This flag keeps them genuinely untouched. Only "
                         "affects token_embd (and output.weight if "
                         "tie_embeddings=False); has no effect under "
                         "--lora-rank > 0, since LoRA already freezes "
                         "embeddings entirely.")

    # -- export mode ---------------------------------------------------------
    tr.add_argument("--export",  help="checkpoint to export to GGUF (skip training)")
    tr.add_argument("--gguf",    help="output GGUF path (used with --export)")
    tr.add_argument("--lora-checkpoint", dest="lora_checkpoint",
                    help="LoRA adapter .pt to merge into --export's base checkpoint "
                         "before writing GGUF (requires --export to point at the base)")

    args = ap.parse_args()

    # export-only shortcut
    if args.export:
        if not args.gguf:
            ap.error("--export requires --gguf")
        # Prefer an existing donor GGUF if given; otherwise build the
        # tokenizer keys fresh from the HF tokenizer (default: Phi-4-mini-
        # instruct). You do NOT need an existing NANITY GGUF to export.
        export_gguf(args.export, args.gguf,
                    tokenizer_source_gguf=args.tokenizer_gguf,
                    tokenizer_hf=None if args.tokenizer_gguf else args.tokenizer,
                    quant=args.export_quant,
                    lora_checkpoint=args.lora_checkpoint)
        return

    if not args.data:
        ap.error("--data is required for training")

    train(args)


if __name__ == "__main__":
    main()