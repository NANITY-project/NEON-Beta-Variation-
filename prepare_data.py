#!/usr/bin/env python3
"""
prepare_data.py -- pulls and formats the NECTAR-1.5B pretrain + SFT mix into
JSONL files that ConversationDataset (train_nanity_fixed.py) can ingest
directly.

Pretrain (Phase 1 "warmup"), target ~12B tokens:
  FineWeb-Edu (sample-10BT config)         ~7B tokens
  Proof-Pile-2 (arxiv + open-web-math + algebraic-stack, interleaved) ~3B tokens
  StarCoderData (bigcode/starcoderdata, a few languages interleaved) ~2B tokens

SFT (Phase 3 "finetune"), reasoning mix:
  OpenThoughts3-1.2M   subsampled to 600k
  OpenCodeReasoning    subsampled to 200k
  Bespoke-Stratos-17k  full (17k) -- final polish pass, kept in its OWN file

IMPORTANT -- vocab/tokenizer mismatch:
  NanityConfig.nanity_1_5b() hardcodes vocab_size=50257 (plain GPT-2 BPE),
  NOT the 200064-vocab Phi-4-mini-instruct tokenizer that train_nanity.py
  defaults to. If you train the 1.5B preset with the default --tokenizer,
  the tokenizer will emit ids >= 50257 and you'll get an out-of-range
  embedding index (crash, possibly late into a run if the offending token
  is rare). Use --tokenizer gpt2 for BOTH this script and the training
  command. Role/control tokens ("<|system|>" etc.) are NOT in the plain
  GPT-2 vocab, so they'll BPE-tokenize as ordinary text (a few tokens
  instead of one) -- that's fine functionally, just slightly less clean
  than a true single-token control marker. This script always tokenizes
  with the SAME tokenizer you train with, so pass --tokenizer to override
  if you go a different route (e.g. a GPT-2 tokenizer extended with added
  special tokens -- but then you must also bump vocab_size in
  modeling_nanity.py's nanity_1_5b() to match, or you'll get the same
  out-of-range crash from the new token ids).

Requires: pip install datasets transformers tqdm --break-system-packages

Usage:
  python3 prepare_data.py --out-dir data/ --tokenizer gpt2
  # writes data/pretrain_mix.jsonl and data/sft_reasoning.jsonl (+ a
  # separate data/sft_polish.jsonl for the small curated final pass)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Pretrain: stream raw text, chunk into <=CHUNK_TOKENS pieces, wrap each
# chunk as a single assistant-turn "conversation" so ConversationDataset can
# ingest it unchanged. Chunking matters: crop_preserving_answer() in the
# training script only trims THINK-region tokens, and a plain-text pretrain
# doc has none (no think-end marker found -> the whole thing counts as
# "answer") -- so an over-length doc would be SKIPPED ENTIRELY rather than
# truncated. Pre-chunking here avoids silently losing data that way.
# ---------------------------------------------------------------------------

CHUNK_TOKENS = 7800   # headroom under context_length=8192 for role/end tokens


def chunk_and_write(fout, tokenizer, text: str, source_tag: str):
    """Tokenize text, split into <=CHUNK_TOKENS pieces on token boundaries,
    write each as one JSONL line. Returns the number of tokens written."""
    if not text or not text.strip():
        return 0
    ids = tokenizer.encode(text, add_special_tokens=False)
    written = 0
    for i in range(0, len(ids), CHUNK_TOKENS):
        piece_ids = ids[i:i + CHUNK_TOKENS]
        if len(piece_ids) < 64:   # drop tiny trailing scraps
            continue
        piece_text = tokenizer.decode(piece_ids)
        fout.write(json.dumps({
            "messages": [{"role": "assistant", "content": piece_text}],
            "_source": source_tag,
        }) + "\n")
        written += len(piece_ids)
    return written


def pull_pretrain(out_path: Path, tokenizer, budget: dict):
    """budget: {"fineweb_edu": 7_000_000_000, "proof_pile_2": 3_000_000_000,
                "stack": 2_000_000_000}"""
    total_written = {k: 0 for k in budget}
    with open(out_path, "w") as fout:

        # -- FineWeb-Edu --------------------------------------------------
        print(f"[fineweb-edu] target {budget['fineweb_edu']:,} tokens")
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                           split="train", streaming=True)
        pbar = tqdm(total=budget["fineweb_edu"], unit="tok", desc="fineweb-edu")
        for row in ds:
            if total_written["fineweb_edu"] >= budget["fineweb_edu"]:
                break
            n = chunk_and_write(fout, tokenizer, row["text"], "fineweb-edu")
            total_written["fineweb_edu"] += n
            pbar.update(n)
        pbar.close()

        # -- Proof-Pile-2: interleave arxiv / open-web-math / algebraic-stack
        print(f"[proof-pile-2] target {budget['proof_pile_2']:,} tokens")
        pp2_subsets = [
            load_dataset("EleutherAI/proof-pile-2", name, split="train",
                         streaming=True)
            for name in ("arxiv", "open-web-math", "algebraic-stack")
        ]
        ds = interleave_datasets(pp2_subsets)
        pbar = tqdm(total=budget["proof_pile_2"], unit="tok", desc="proof-pile-2")
        for row in ds:
            if total_written["proof_pile_2"] >= budget["proof_pile_2"]:
                break
            n = chunk_and_write(fout, tokenizer, row["text"], "proof-pile-2")
            total_written["proof_pile_2"] += n
            pbar.update(n)
        pbar.close()

        # -- StarCoderData: interleave a handful of languages --------------
        print(f"[starcoderdata] target {budget['stack']:,} tokens")
        code_langs = ["python", "javascript", "cpp", "java", "go"]
        code_subsets = [
            load_dataset("bigcode/starcoderdata", data_dir=lang,
                         split="train", streaming=True)
            for lang in code_langs
        ]
        ds = interleave_datasets(code_subsets)
        pbar = tqdm(total=budget["stack"], unit="tok", desc="starcoderdata")
        for row in ds:
            if total_written["stack"] >= budget["stack"]:
                break
            n = chunk_and_write(fout, tokenizer, row["content"], "starcoderdata")
            total_written["stack"] += n
            pbar.update(n)
        pbar.close()

    print(f"[pretrain] done -> {out_path}")
    for k, v in total_written.items():
        print(f"  {k}: {v:,} tokens")
    return total_written


# ---------------------------------------------------------------------------
# SFT: these arrive already conversational. ConversationDataset already
# normalizes ShareGPT-style {"from": "human"/"gpt", "value": ...} to
# {"role": "user"/"assistant", "content": ...} -- so for datasets in that
# format we just pass the "conversations" field straight through as
# "messages" and let the training script's own normalization handle it.
# ---------------------------------------------------------------------------

def pull_openthoughts3(out_path, n_target: int, seed: int):
    """850k math / 250k code / 100k science in the source. Subsample
    proportionally so the 600k target keeps the same domain mix rather than
    just taking the first 600k rows (which would be arbitrarily domain-
    skewed depending on file ordering)."""
    print(f"[openthoughts3] target {n_target:,} examples (domain-stratified)")
    ds = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
    by_domain: dict[str, list] = {}
    for row in ds:
        by_domain.setdefault(row["domain"], []).append(row)
    rng = random.Random(seed)
    total_src = sum(len(v) for v in by_domain.values())
    written = 0
    with open(out_path, "a") as fout:
        for domain, rows in by_domain.items():
            take = int(n_target * len(rows) / total_src)
            rng.shuffle(rows)
            for row in rows[:take]:
                fout.write(json.dumps({
                    "messages": row["conversations"],   # already from/value
                    "_source": f"openthoughts3-{domain}",
                }) + "\n")
                written += 1
    print(f"[openthoughts3] wrote {written:,} examples -> {out_path}")


def pull_opencodereasoning(out_path, n_target: int, seed: int):
    print(f"[opencodereasoning] target {n_target:,} examples")
    ds = load_dataset("nvidia/OpenCodeReasoning", "split_0", split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n_target]
    written = 0
    with open(out_path, "a") as fout:
        for i in idx:
            row = ds[i]
            # 'output' is R1's full response, reasoning trace included --
            # this already contains the model's own <think>-style markers
            # from R1's generation, which the default --think-end-markers
            # list should catch; if R1's raw output doesn't use one of
            # </think> / <|end_of_thought|> / <|begin_of_solution|>, the
            # whole response falls back to counting as "answer" (safe
            # default -- see apply_chat_template's split_think_answer).
            fout.write(json.dumps({
                "messages": [
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]},
                ],
                "_source": "opencodereasoning",
            }) + "\n")
            written += 1
    print(f"[opencodereasoning] wrote {written:,} examples -> {out_path}")


def pull_bespoke_stratos(out_path):
    print("[bespoke-stratos-17k] pulling full set (17k)")
    ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
    written = 0
    with open(out_path, "w") as fout:
        for row in ds:
            # NOTE: same lineage/tooling as OpenThoughts (Bespoke Curator),
            # expected to carry a "conversations" field in from/value form.
            # If this dataset's schema has since changed, this will KeyError
            # loudly rather than silently writing garbage -- check the
            # dataset viewer on HF if so and adjust the field name below.
            fout.write(json.dumps({
                "messages": row["conversations"],
                "_source": "bespoke-stratos-17k",
            }) + "\n")
            written += 1
    print(f"[bespoke-stratos-17k] wrote {written:,} examples -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--tokenizer", default="gpt2",
                    help="MUST match what you pass to train_nanity_fixed.py "
                         "--tokenizer. Default 'gpt2' matches the 1.5B "
                         "preset's vocab_size=50257 -- see the module "
                         "docstring for why this has to match exactly.")
    ap.add_argument("--pretrain-fineweb-edu", type=int, default=7_000_000_000)
    ap.add_argument("--pretrain-proof-pile-2", type=int, default=3_000_000_000)
    ap.add_argument("--pretrain-stack", type=int, default=2_000_000_000)
    ap.add_argument("--sft-openthoughts3", type=int, default=600_000)
    ap.add_argument("--sft-opencodereasoning", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-pretrain", action="store_true")
    ap.add_argument("--skip-sft", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[tokenizer] loading {args.tokenizer} (used only to measure/chunk "
          f"pretrain docs by token count -- must match train_nanity_fixed.py's "
          f"--tokenizer or downstream training will mis-tokenize this data)")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if not args.skip_pretrain:
        pull_pretrain(out_dir / "pretrain_mix.jsonl", tokenizer, {
            "fineweb_edu":   args.pretrain_fineweb_edu,
            "proof_pile_2":  args.pretrain_proof_pile_2,
            "stack":         args.pretrain_stack,
        })

    if not args.skip_sft:
        sft_path = out_dir / "sft_reasoning.jsonl"
        if sft_path.exists():
            sft_path.unlink()   # pull_openthoughts3/opencodereasoning append
        sft_path.touch()
        pull_openthoughts3(sft_path, args.sft_openthoughts3, args.seed)
        pull_opencodereasoning(sft_path, args.sft_opencodereasoning, args.seed)

        # Kept as its OWN file, not merged into sft_reasoning.jsonl -- this
        # is meant as a small, separate final-polish pass (low LR, 2-3
        # epochs) AFTER the main SFT run, not mixed into the bulk set. See
        # the earlier discussion: curated small sets like this are for
        # quality-over-quantity polish, not bulk training signal.
        pull_bespoke_stratos(out_dir / "sft_polish.jsonl")

    print("\n[done] files written to", out_dir)
    print("  pretrain_mix.jsonl  -> --phase warmup")
    print("  sft_reasoning.jsonl -> --phase finetune (main SFT pass)")
    print("  sft_polish.jsonl    -> --phase finetune, --resume the SFT "
          "checkpoint, low --lr, few --steps (final polish pass)")


if __name__ == "__main__":
    main()
