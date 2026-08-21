#!/usr/bin/env python3
"""
prepare_data.py -- pulls and formats the NECTAR-1.5B pretrain + SFT mix into
pre-tokenized .bin files that ConversationDataset (train_nanity_fixed.py)
can load directly, with NO tokenization step at train time.

WHY .bin INSTEAD OF JSONL:
  This script now runs the exact same tokenization pipeline the trainer used
  to run at train-start (apply_chat_template + crop_preserving_answer, both
  imported from nanity_data_format.py so the two scripts can never drift out
  of sync) and writes the resulting (ids, mask, is_answer) arrays straight
  to disk in nanity_data_format's .bin layout. train_nanity_fixed.py detects
  a .bin path and loads it with a single streaming pass -- no JSON parsing,
  no chat-template application, no cropping, no separate pickle cache (the
  .bin file already IS the cache). The tokenizer used to write a .bin file
  is recorded in its header, so passing the WRONG --tokenizer at train time
  is caught with a loud warning instead of silently corrupting token ids.
  JSONL is still accepted by the trainer for data that didn't go through
  this script, but everything pulled here now goes straight to .bin.

Pretrain (Phase 1 "warmup"), target ~8B tokens:
  FineWeb-Edu (sample-10BT config)         ~4.5B tokens
  Proof-Pile-2 (arxiv + open-web-math + algebraic-stack, interleaved) ~2B tokens
  StarCoderData (bigcode/starcoderdata, a few languages interleaved) ~1.5B tokens

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

TOKENIZER BACKEND (tiktoken vs HF):
  tiktoken's "gpt2" encoding is byte-for-byte identical (same merges, same
  ids) to HF's GPT2Tokenizer, so it's a safe drop-in ONLY for --tokenizer
  gpt2. Any other tokenizer (e.g. the 200064-vocab Phi-4-mini tokenizer
  used by the 4B preset) has no tiktoken equivalent -- there's no way to
  guarantee identical ids, so this script always uses HF for anything that
  isn't in TIKTOKEN_EQUIVALENTS below. When gpt2 IS requested, pass
  --tokenizer-backend auto (default) to benchmark both backends on a
  sample and pick whichever actually encodes faster on this machine,
  rather than assuming tiktoken wins (it almost always does -- it's a Rust
  extension -- but the benchmark is cheap and removes the assumption).

MULTI-CORE TOKENIZATION:
  Both HF's fast (Rust-backed) tokenizers and tiktoken release the GIL
  while actually encoding/decoding, so a plain ThreadPoolExecutor gets
  real parallelism across cores for the CPU-bound tokenize+chunk step,
  without the pickling overhead multiprocessing would add for streaming
  dataset iterators. Pulling rows off the network (streaming=True) stays
  single-threaded in the main thread -- that part is I/O bound, not CPU
  bound, so more threads there wouldn't help. Worker count defaults to
  the number of cores actually available to this process (via
  os.sched_getaffinity on Linux, which respects container/cgroup CPU
  limits -- more accurate than os.cpu_count() on a shared cloud instance),
  override with --workers.

Requires: pip install datasets transformers tqdm --break-system-packages
Optional (faster gpt2 tokenization): pip install tiktoken --break-system-packages

Usage:
  python3 prepare_data.py --out-dir data/ --tokenizer gpt2 --max-len 8192
  # writes data/pretrain_mix.bin and data/sft_reasoning.bin (+ a
  # separate data/sft_polish.bin for the small curated final pass).
  # --max-len MUST match --max-seq-len/context_length at train time -- see
  # the module docstring in train_nanity_fixed.py.
"""
from __future__ import annotations

import argparse
import array
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
from tqdm import tqdm

try:
    from huggingface_hub.errors import GatedRepoError
except ImportError:  # older huggingface_hub versions
    from huggingface_hub.utils._errors import GatedRepoError

# nanity_data_format.py lives next to this script (same directory as
# train_nanity_fixed.py) -- it's the SHARED source of truth for
# apply_chat_template/crop_preserving_answer/the .bin format, so a file
# written here always tokenizes identically to what the trainer expects.
sys.path.insert(0, str(Path(__file__).parent))
from nanity_data_format import (  # noqa: E402
    BinDatasetWriter,
    DEFAULT_THINK_END_MARKERS,
    END_TOKEN,
    Example,
    ROLE_TOKEN,
    normalize_messages,
    tokenize_conversation,
    typecode_for_vocab,
)


def _fail_gated(repo_id: str):
    print(f"""
[auth] '{repo_id}' is a GATED dataset on Hugging Face -- streaming it needs:
  1. Open https://huggingface.co/datasets/{repo_id} while logged in to HF
     and click "Agree and access repository" (bigcode/starcoderdata requires
     accepting its terms before any download, even for public read access).
  2. Authenticate this machine with a token generated AFTER step 1:
       huggingface-cli login
     or:
       export HF_TOKEN=hf_your_token_here
  3. Re-run this script.

To route around it entirely instead (skip code data for this run), pass
--pretrain-stack 0 -- with a zero budget this script won't touch
'{repo_id}' at all.
""")
    raise SystemExit(1)


def _content_only(ds):
    """Drop every column except 'content'. starcoderdata's per-language
    subsets don't share an identical schema (e.g. max_stars_count is
    float64 in some languages, int64 in others), which makes
    interleave_datasets refuse to align them. We only ever read row["content"]
    here, so dropping the metadata columns we don't use sidesteps the
    mismatch entirely instead of needing them to match."""
    try:
        return ds.select_columns(["content"])
    except Exception:
        col_names = getattr(ds, "column_names", None) or []
        drop_cols = [c for c in col_names if c != "content"]
        if drop_cols:
            try:
                return ds.remove_columns(drop_cols)
            except Exception:
                pass
        return ds


def safe_load_dataset(repo_id: str, *args, **kwargs):
    """load_dataset() wrapper that turns a gated-repo 401 into a clear,
    actionable message instead of a raw traceback -- this can happen well
    after 'Resolving data files' succeeds, since streaming parquet datasets
    fetch the schema (which needs auth) lazily on first access."""
    try:
        return load_dataset(repo_id, *args, **kwargs)
    except GatedRepoError:
        _fail_gated(repo_id)
    except Exception as e:
        if "401" in str(e) or "gated" in str(e).lower() or "restricted" in str(e).lower():
            _fail_gated(repo_id)
        raise


# ---------------------------------------------------------------------------
# Tokenizer backend: HF AutoTokenizer vs tiktoken, auto-benchmarked.
# ---------------------------------------------------------------------------

# Only tokenizers where tiktoken's encoding is a verified byte-identical
# match to the HF tokenizer of the same name. Do NOT add entries here on a
# guess -- a mismatch silently produces different token ids than training
# will see, which is worse than just being slow.
TIKTOKEN_EQUIVALENTS = {
    "gpt2": "gpt2",
}

_BENCH_SAMPLE = (
    "The quick brown fox jumps over the lazy dog. " * 400 +
    "def fibonacci(n):\n    if n < 2:\n        return n\n"
    "    return fibonacci(n - 1) + fibonacci(n - 2)\n" * 200 +
    "\\int_0^1 x^2 \\,dx = \\frac{1}{3} is a standard result in calculus. " * 200
)


class FastTokenizer:
    """Thin wrapper unifying the HF and tiktoken encode/decode interface
    used by chunk_and_write(). Both backends are safe to call concurrently
    from multiple threads (the underlying Rust extensions are stateless /
    release the GIL during the call), which is what lets the worker pool
    below share a single instance across threads."""

    def __init__(self, backend: str, hf_tok, tt_enc=None):
        assert backend in ("hf", "tiktoken")
        self.backend = backend
        # hf_tok is kept even on the tiktoken backend (encode/decode still go
        # through tiktoken for speed) -- it's the source of truth for vocab
        # size via __len__ below, so id_typecode / vocab_ceiling in the .bin
        # header always match what train_nanity_fixed.py computes as
        # len(tokenizer) on the same --tokenizer name, regardless of which
        # backend actually did the encoding.
        self._hf = hf_tok
        self._tt = tt_enc

    def encode(self, text: str, add_special_tokens: bool = False):
        # add_special_tokens accepted (and ignored on the tiktoken backend,
        # which never adds any) for interface-compatibility with the plain
        # HF AutoTokenizer object apply_chat_template() was written against
        # -- apply_chat_template() (imported from nanity_data_format, shared
        # with the trainer) always calls encode(text, add_special_tokens=False).
        if self.backend == "tiktoken":
            return self._tt.encode_ordinary(text)
        return self._hf.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int]) -> str:
        if self.backend == "tiktoken":
            return self._tt.decode(ids)
        return self._hf.decode(ids)

    def __len__(self) -> int:
        # Real ceiling on any id encode() can produce for THIS tokenizer
        # name, matching train_nanity_fixed.py's `len(tokenizer)` check.
        return len(self._hf)


def _bench_encode(fn, sample: str, reps: int = 3) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(sample)
        best = min(best, time.perf_counter() - t0)
    return best


def load_tokenizer(name: str, backend_pref: str = "auto") -> FastTokenizer:
    """backend_pref: 'auto' (benchmark and pick faster), 'hf', or 'tiktoken'
    (forced -- errors out if `name` has no verified tiktoken equivalent)."""
    hf_tok = AutoTokenizer.from_pretrained(name)
    tt_name = TIKTOKEN_EQUIVALENTS.get(name)

    if backend_pref == "hf":
        print(f"[tokenizer] using HF AutoTokenizer('{name}') (forced)")
        return FastTokenizer("hf", hf_tok=hf_tok)

    if tt_name is None:
        if backend_pref == "tiktoken":
            raise SystemExit(
                f"[tokenizer] --tokenizer-backend tiktoken requested but "
                f"'{name}' has no verified tiktoken-equivalent encoding in "
                f"TIKTOKEN_EQUIVALENTS. Only {list(TIKTOKEN_EQUIVALENTS)} "
                f"are supported -- use --tokenizer-backend hf instead."
            )
        print(f"[tokenizer] no verified tiktoken equivalent for '{name}' "
              f"-> using HF AutoTokenizer")
        return FastTokenizer("hf", hf_tok=hf_tok)

    try:
        import tiktoken
    except ImportError:
        if backend_pref == "tiktoken":
            raise SystemExit(
                "[tokenizer] --tokenizer-backend tiktoken requested but "
                "tiktoken isn't installed: pip install tiktoken --break-system-packages"
            )
        print("[tokenizer] tiktoken not installed -> using HF AutoTokenizer "
              "(pip install tiktoken --break-system-packages to enable the "
              "faster path)")
        return FastTokenizer("hf", hf_tok=hf_tok)

    tt_enc = tiktoken.get_encoding(tt_name)

    if backend_pref == "tiktoken":
        print(f"[tokenizer] using tiktoken('{tt_name}') (forced)")
        return FastTokenizer("tiktoken", hf_tok=hf_tok, tt_enc=tt_enc)

    # auto: benchmark both on a mixed prose/code/math sample and pick the
    # faster one. tiktoken almost always wins (Rust vs Python), but this
    # measures rather than assumes it, and falls back cleanly if not.
    hf_t = _bench_encode(lambda s: hf_tok.encode(s, add_special_tokens=False), _BENCH_SAMPLE)
    tt_t = _bench_encode(tt_enc.encode_ordinary, _BENCH_SAMPLE)
    print(f"[tokenizer] benchmark on {len(_BENCH_SAMPLE):,} chars: "
          f"HF={hf_t*1000:.1f}ms  tiktoken={tt_t*1000:.1f}ms")
    if tt_t < hf_t:
        print(f"[tokenizer] tiktoken is {hf_t/tt_t:.1f}x faster -> using tiktoken('{tt_name}')")
        return FastTokenizer("tiktoken", hf_tok=hf_tok, tt_enc=tt_enc)
    else:
        print(f"[tokenizer] HF is faster (or tiktoken not faster) -> using HF AutoTokenizer")
        return FastTokenizer("hf", hf_tok=hf_tok)


def default_workers() -> int:
    """Cores actually available to this process. Prefer sched_getaffinity
    (Linux) over cpu_count() -- cpu_count() reports the machine's total
    core count even inside a cgroup-limited container/VM slice, which can
    overcommit threads on shared cloud instances; affinity reports what
    you actually get scheduled onto."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


# ---------------------------------------------------------------------------
# Pretrain: stream raw text, chunk into <=CHUNK_TOKENS pieces, wrap each
# chunk as a single assistant-turn "conversation" -- the exact same shape
# apply_chat_template() would build from
# {"messages": [{"role": "assistant", "content": piece_text}]}. Chunking
# matters: crop_preserving_answer() only trims THINK-region tokens, and a
# plain-text pretrain doc has none (no think-end marker found -> the whole
# thing counts as "answer") -- so an over-length doc would be SKIPPED
# ENTIRELY rather than truncated. Pre-chunking here avoids silently losing
# data that way.
#
# Unlike the old JSONL path, chunks are built DIRECTLY as (ids, mask,
# is_answer) arrays here -- tokenizer.encode(text) already gave us
# piece_ids, so we splice in the (cached, constant) role/end control-token
# ids around it instead of decoding piece_ids back to text and letting the
# trainer re-encode it through apply_chat_template later. That's both
# faster (one encode pass instead of two) and exactly reproducible: gpt2
# BPE isn't guaranteed to round-trip encode(decode(ids)) == ids at an
# arbitrary chunk boundary, so splicing the token ids we already have is
# more correct, not just faster.
# ---------------------------------------------------------------------------

CHUNK_TOKENS = 7800   # headroom under context_length=8192 for role/end tokens


def tokenize_and_chunk(tokenizer: FastTokenizer, text: str,
                        role_ids: list[int], end_ids: list[int], id_typecode: str):
    """CPU-bound step, run inside the worker thread pool. Returns a list of
    (Example, n_tokens) -- writing to disk stays in the main thread. n_tokens
    counts only the content tokens (matches the pre-existing progress-bar/
    budget accounting, which was measuring source text volume, not the
    handful of extra role/end control tokens added per chunk)."""
    if not text or not text.strip():
        return []
    ids = tokenizer.encode(text)
    out = []
    for i in range(0, len(ids), CHUNK_TOKENS):
        piece_ids = ids[i:i + CHUNK_TOKENS]
        n = len(piece_ids)
        if n < 64:   # drop tiny trailing scraps
            continue
        full_ids  = array.array(id_typecode, role_ids)
        full_ids.extend(array.array(id_typecode, piece_ids))
        full_ids.extend(array.array(id_typecode, end_ids))
        n_role, n_end = len(role_ids), len(end_ids)
        mask      = array.array("B", [0] * n_role + [1] * (n + n_end))
        is_answer = array.array("B", [0] * n_role + [1] * (n + n_end))
        out.append((Example(full_ids, mask, is_answer), n))
    return out


def pull_pretrain(out_path: Path, tokenizer: FastTokenizer, budget: dict,
                   workers: int, seed: int = 1234, tokenizer_name: str = "gpt2"):
    """budget: {"fineweb_edu": 7_000_000_000, "proof_pile_2": 3_000_000_000,
                "stack": 2_000_000_000}

    Sources are drawn via weighted-random interleaving (weight = each
    source's remaining budget) so the mix stays representative throughout
    the pull rather than running fully through one source before starting
    the next. Rows are pulled from the (I/O-bound) streaming iterators one
    at a time on the main thread, batched, then tokenized+chunked
    (CPU-bound) across `workers` threads in parallel."""
    rng = random.Random(seed)
    id_typecode = typecode_for_vocab(len(tokenizer))
    # Role/end control-token ids are the same for every chunk -- encode them
    # once here instead of inside the per-chunk hot loop.
    role_ids = tokenizer.encode(ROLE_TOKEN["assistant"])
    end_ids = tokenizer.encode(END_TOKEN)
    total_written = {k: 0 for k in budget if budget[k] > 0}
    active = set(total_written)  # only sources with budget > 0

    # Only build/touch a source's dataset if it actually has budget -- this
    # is what lets --pretrain-stack 0 fully route around a gated or broken
    # source instead of still hitting it during setup.
    sources = {}

    if "fineweb_edu" in active:
        print(f"[fineweb-edu] target {budget['fineweb_edu']:,} tokens")
        fineweb_iter = iter(safe_load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                                               split="train", streaming=True))
        sources["fineweb_edu"] = (fineweb_iter, "fineweb-edu", "text")

    if "proof_pile_2" in active:
        print(f"[proof-pile-2] target {budget['proof_pile_2']:,} tokens")
        pp2_subsets = [
            safe_load_dataset("EleutherAI/proof-pile-2", name, split="train", streaming=True)
            for name in ("arxiv", "open-web-math", "algebraic-stack")
        ]
        pp2_iter = iter(interleave_datasets(pp2_subsets))
        sources["proof_pile_2"] = (pp2_iter, "proof-pile-2", "text")

    if "stack" in active:
        print(f"[starcoderdata] target {budget['stack']:,} tokens")
        code_langs = ["python", "javascript", "cpp", "java", "go"]
        code_subsets = [
            _content_only(safe_load_dataset("bigcode/starcoderdata", data_dir=lang,
                                             split="train", streaming=True))
            for lang in code_langs
        ]
        stack_iter = iter(interleave_datasets(code_subsets))
        sources["stack"] = (stack_iter, "starcoderdata", "content")

    total_budget = sum(budget[k] for k in active)
    pbar = tqdm(total=total_budget, unit="tok", desc="pretrain mix (all sources)")
    print(f"[pretrain] tokenizing with {workers} worker threads "
          f"({default_workers()} cores available)")

    batch_size = max(workers * 4, 1)

    writer = BinDatasetWriter(out_path, tokenizer_source=tokenizer_name,
                               max_len=CHUNK_TOKENS + len(role_ids) + len(end_ids),
                               vocab_ceiling=len(tokenizer))
    with writer, ThreadPoolExecutor(max_workers=workers) as ex:
        while active:
            batch = []  # (choice, text, source_tag)
            while len(batch) < batch_size and active:
                remaining = {k: max(budget[k] - total_written[k], 0) for k in active}
                names = list(active)
                weights = [remaining[n] for n in names]
                if sum(weights) == 0:
                    active.clear()
                    break
                choice = rng.choices(names, weights=weights, k=1)[0]
                it, source_tag, field = sources[choice]
                try:
                    row = next(it)
                except StopIteration:
                    print(f"[{source_tag}] source exhausted at "
                          f"{total_written[choice]:,} tokens (target was "
                          f"{budget[choice]:,}) -- dropping from the mix")
                    active.discard(choice)
                    continue
                except Exception as e:
                    # A dead streaming iterator (e.g. a corrupted/truncated
                    # shard on the HF side -- zstd/parquet errors do happen)
                    # can't be resumed: once its generator raises, further
                    # next() calls just raise StopIteration. So there's no
                    # retry that helps here -- drop this source and keep
                    # going with whatever's left, rather than crashing the
                    # whole multi-hour pull over one bad shard.
                    print(f"[{source_tag}] read error at "
                          f"{total_written[choice]:,} tokens (target was "
                          f"{budget[choice]:,}): {type(e).__name__}: {e} "
                          f"-- dropping this source from the mix, "
                          f"continuing with the rest")
                    active.discard(choice)
                    continue
                batch.append((choice, row[field], source_tag))
                if total_written[choice] >= budget[choice]:
                    # already at/over budget from a previous batch; stop
                    # drawing more from it once this row is accounted for
                    active.discard(choice)

            if not batch:
                break

            futures = [ex.submit(tokenize_and_chunk, tokenizer, text,
                                  role_ids, end_ids, id_typecode)
                       for (_choice, text, _tag) in batch]
            for (choice, _text, _tag), fut in zip(batch, futures):
                for example, n_tok in fut.result():
                    writer.add_example(example)
                    total_written[choice] += n_tok
                    pbar.update(n_tok)
                if total_written[choice] >= budget[choice]:
                    active.discard(choice)

    pbar.close()

    print(f"[pretrain] wrote {writer.n_written:,} examples -> {out_path}")
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

def _tokenize_and_write(writer: BinDatasetWriter, tokenizer: FastTokenizer,
                         max_len: int, id_typecode: str, think_end_markers,
                         messages: list[dict], tag: str, counters: dict):
    """Shared tail end of all three SFT pulls: normalize -> tokenize ->
    crop -> write. counters is a dict of tag -> [written, skipped] updated
    in place so callers can print a summary."""
    normed = normalize_messages(messages)
    if not normed:
        counters[tag][1] += 1
        return
    ex, status = tokenize_conversation(normed, tokenizer, max_len, id_typecode,
                                        think_end_markers)
    if status != "ok":
        counters[tag][1] += 1
        return
    writer.add_example(ex)
    counters[tag][0] += 1


def pull_openthoughts3(writer: BinDatasetWriter, tokenizer: FastTokenizer,
                        max_len: int, id_typecode: str, think_end_markers,
                        n_target: int, seed: int):
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
    counters = {"openthoughts3": [0, 0]}
    for domain, rows in by_domain.items():
        take = int(n_target * len(rows) / total_src)
        rng.shuffle(rows)
        for row in tqdm(rows[:take], desc=f"[openthoughts3:{domain}]", leave=False):
            _tokenize_and_write(writer, tokenizer, max_len, id_typecode,
                                 think_end_markers, row["conversations"],  # already from/value
                                 "openthoughts3", counters)
    written, skipped = counters["openthoughts3"]
    print(f"[openthoughts3] wrote {written:,} examples ({skipped:,} skipped/overflow) "
          f"-> {writer.path}")


def pull_opencodereasoning(writer: BinDatasetWriter, tokenizer: FastTokenizer,
                            max_len: int, id_typecode: str, think_end_markers,
                            n_target: int, seed: int):
    print(f"[opencodereasoning] target {n_target:,} examples")
    # NOTE: the "split_0" config's only available split is also literally
    # named "split_0" (not "train") -- load_dataset(..., split="train")
    # 404s with "Unknown split 'train'. Should be one of ['split_0']."
    ds = load_dataset("nvidia/OpenCodeReasoning", "split_0", split="split_0")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n_target]
    counters = {"opencodereasoning": [0, 0]}
    for i in tqdm(idx, desc="[opencodereasoning]"):
        row = ds[i]
        # 'output' is R1's full response, reasoning trace included -- this
        # already contains the model's own <think>-style markers from R1's
        # generation, which the default --think-end-markers list should
        # catch; if R1's raw output doesn't use one of </think> /
        # <|end_of_thought|> / <|begin_of_solution|>, the whole response
        # falls back to counting as "answer" (safe default -- see
        # apply_chat_template's split_think_answer).
        messages = [
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]
        _tokenize_and_write(writer, tokenizer, max_len, id_typecode,
                             think_end_markers, messages,
                             "opencodereasoning", counters)
    written, skipped = counters["opencodereasoning"]
    print(f"[opencodereasoning] wrote {written:,} examples ({skipped:,} skipped/overflow) "
          f"-> {writer.path}")


def pull_bespoke_stratos(writer: BinDatasetWriter, tokenizer: FastTokenizer,
                          max_len: int, id_typecode: str, think_end_markers):
    print("[bespoke-stratos-17k] pulling full set (17k)")
    ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
    counters = {"bespoke-stratos-17k": [0, 0]}
    for row in tqdm(ds, desc="[bespoke-stratos-17k]"):
        # NOTE: same lineage/tooling as OpenThoughts (Bespoke Curator),
        # expected to carry a "conversations" field in from/value form.
        # If this dataset's schema has since changed, this will KeyError
        # loudly rather than silently writing garbage -- check the dataset
        # viewer on HF if so and adjust the field name below.
        _tokenize_and_write(writer, tokenizer, max_len, id_typecode,
                             think_end_markers, row["conversations"],
                             "bespoke-stratos-17k", counters)
    written, skipped = counters["bespoke-stratos-17k"]
    print(f"[bespoke-stratos-17k] wrote {written:,} examples ({skipped:,} skipped/overflow) "
          f"-> {writer.path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--tokenizer", default="gpt2",
                    help="MUST match what you pass to train_nanity_fixed.py "
                         "--tokenizer. Default 'gpt2' matches the 1.5B "
                         "preset's vocab_size=50257 -- see the module "
                         "docstring for why this has to match exactly.")
    ap.add_argument("--tokenizer-backend", choices=["auto", "hf", "tiktoken"], default="auto",
                    help="'auto' (default) benchmarks HF vs tiktoken on a sample and "
                         "picks whichever is faster; only applies when --tokenizer has "
                         "a verified tiktoken equivalent (currently just 'gpt2') -- "
                         "otherwise it's HF regardless, since a mismatched tiktoken "
                         "encoding would silently diverge from the vocab the trainer uses.")
    ap.add_argument("--max-len", dest="max_len", type=int, default=8192,
                    help="MUST match --max-seq-len (or cfg.context_length if "
                         "--max-seq-len isn't passed) at train time. SFT examples "
                         "longer than this get their THINK-region tokens cropped "
                         "(same crop_preserving_answer() logic the trainer used to "
                         "run at load time); examples that still don't fit after "
                         "cropping are skipped. Pretrain chunks are already sized "
                         "under this via CHUNK_TOKENS so they're never cropped.")
    ap.add_argument("--think-end-markers", dest="think_end_markers", default=None,
                    help="Comma-separated end-of-thinking markers, passed straight "
                         "to apply_chat_template(). Default (None) uses "
                         "nanity_data_format.DEFAULT_THINK_END_MARKERS -- MUST match "
                         "whatever you'd otherwise pass to train_nanity_fixed.py's "
                         "--think-end-markers, since this script now does that "
                         "splitting instead of the trainer.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Thread pool size for tokenize+chunk. Defaults to all cores "
                         "available to this process (os.sched_getaffinity on Linux).")
    ap.add_argument("--hf-token", default=None,
                    help="Hugging Face access token, for gated datasets (e.g. "
                         "bigcode/starcoderdata requires accepting its terms on the "
                         "HF site AND a token with access). Equivalent to running "
                         "'huggingface-cli login' or setting HF_TOKEN first -- use "
                         "this flag when you'd rather not do that interactively.")
    ap.add_argument("--pretrain-fineweb-edu", type=int, default=4_500_000_000)
    ap.add_argument("--pretrain-proof-pile-2", type=int, default=2_000_000_000)
    ap.add_argument("--pretrain-stack", type=int, default=1_500_000_000)
    ap.add_argument("--sft-openthoughts3", type=int, default=600_000)
    ap.add_argument("--sft-opencodereasoning", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-sft-openthoughts3", action="store_true",
                    help="Skip the OpenThoughts3 pull -- use when sft_reasoning.bin "
                         "already has it from a previous run and you're only resuming "
                         "the later steps (e.g. after an opencodereasoning failure).")
    ap.add_argument("--skip-sft-opencodereasoning", action="store_true")
    ap.add_argument("--skip-sft-polish", action="store_true")
    ap.add_argument("--skip-pretrain", action="store_true")
    ap.add_argument("--skip-sft", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or default_workers()
    think_end_markers = (args.think_end_markers.split(",") if args.think_end_markers
                          else DEFAULT_THINK_END_MARKERS)

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token, add_to_git_credential=False)

    print(f"[tokenizer] loading '{args.tokenizer}' (used to tokenize ALL data up "
          f"front now -- must match train_nanity_fixed.py's --tokenizer exactly, "
          f"or the ids written here won't match that run's embedding vocab)")
    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_backend)
    id_typecode = typecode_for_vocab(len(tokenizer))
    print(f"[tokenizer] vocab size {len(tokenizer)} -> packing ids as "
          f"array.array('{id_typecode}')")

    if not args.skip_pretrain:
        pull_pretrain(out_dir / "pretrain_mix.bin", tokenizer, {
            "fineweb_edu":   args.pretrain_fineweb_edu,
            "proof_pile_2":  args.pretrain_proof_pile_2,
            "stack":         args.pretrain_stack,
        }, workers=workers, seed=args.seed, tokenizer_name=args.tokenizer)

    if not args.skip_sft:
        sft_path = out_dir / "sft_reasoning.bin"
        # append=True -- a previous run's data (e.g. OpenThoughts3, which is
        # the expensive/slow part) is kept so a retry after a failure
        # further down doesn't have to re-pull it, as long as this run's
        # tokenizer/max_len/think_end_markers match what's already in the
        # file's header. To fully restart SFT from scratch, delete
        # sft_reasoning.bin yourself first.
        sft_writer = BinDatasetWriter(sft_path, tokenizer_source=args.tokenizer,
                                       max_len=args.max_len, vocab_ceiling=len(tokenizer),
                                       think_end_markers=think_end_markers, append=True)
        with sft_writer:
            if not args.skip_sft_openthoughts3:
                pull_openthoughts3(sft_writer, tokenizer, args.max_len, id_typecode,
                                    think_end_markers, args.sft_openthoughts3, args.seed)
            if not args.skip_sft_opencodereasoning:
                pull_opencodereasoning(sft_writer, tokenizer, args.max_len, id_typecode,
                                        think_end_markers, args.sft_opencodereasoning, args.seed)

        # Kept as its OWN file, not merged into sft_reasoning.bin -- this is
        # meant as a small, separate final-polish pass (low LR, 2-3 epochs)
        # AFTER the main SFT run, not mixed into the bulk set. See the
        # earlier discussion: curated small sets like this are for
        # quality-over-quantity polish, not bulk training signal.
        if not args.skip_sft_polish:
            polish_writer = BinDatasetWriter(out_dir / "sft_polish.bin",
                                              tokenizer_source=args.tokenizer,
                                              max_len=args.max_len, vocab_ceiling=len(tokenizer),
                                              think_end_markers=think_end_markers)
            with polish_writer:
                pull_bespoke_stratos(polish_writer, tokenizer, args.max_len, id_typecode,
                                      think_end_markers)

    print("\n[done] files written to", out_dir)
    print("  pretrain_mix.bin  -> --phase warmup")
    print("  sft_reasoning.bin -> --phase finetune (main SFT pass)")
    print("  sft_polish.bin    -> --phase finetune, --resume the SFT "
          "checkpoint, low --lr, few --steps (final polish pass)")
    print(f"\n  All three are pre-tokenized with --tokenizer {args.tokenizer!r} and "
          f"--max-len {args.max_len} -- pass the SAME --tokenizer (and, if you "
          f"used a non-default --max-len here, the same --max-seq-len) to "
          f"train_nanity_fixed.py. It will load these directly with no "
          f"re-tokenization; a mismatch is reported as a warning, not silently "
          f"ignored.")


if __name__ == "__main__":
    main()
