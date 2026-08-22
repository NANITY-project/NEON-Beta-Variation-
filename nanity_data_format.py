#!/usr/bin/env python3
"""
nanity_data_format.py -- shared conversation-tokenization + binary dataset
format for NANITY, imported by BOTH prepare_data.py (writer) and
train_nanity_fixed.py (reader/trainer).

Why this file exists:
  prepare_data.py used to write plain-text JSONL, and train_nanity_fixed.py
  tokenized it (apply_chat_template + crop_preserving_answer) the first time
  it was ever trained on, then cached the result as a pickle keyed on
  (tokenizer, max_len, think_end_markers, file size/mtime). That meant the
  FIRST run against any new data file always paid the full tokenization
  cost, and prepare_data.py's own pretrain tokenization pass (done just to
  chunk long docs to CHUNK_TOKENS) was thrown away and redone from scratch
  by the trainer.

  Now prepare_data.py does the real tokenization ONCE, up front, using the
  exact same apply_chat_template()/crop_preserving_answer() the trainer
  uses (imported from here, not reimplemented), and writes the resulting
  (ids, mask, is_answer) arrays straight to a .bin file. train_nanity_fixed.py
  reads that .bin directly into Example objects -- no JSON parsing, no
  chat-template application, no cropping, no separate pickle cache (the
  .bin file already IS the cache).

  Keeping apply_chat_template/crop_preserving_answer/Example defined ONCE,
  here, instead of duplicated in both scripts, is what guarantees a .bin
  file written by prepare_data.py tokenizes identically to what the old
  JSONL-at-train-time path would have produced -- a silent drift between
  two copies of this logic would be far worse than either script being
  slow, since it would corrupt training data without any error.

.bin file layout:
  8s   magic       b"NANTYBN1"
  I    header_len  (uint32, little-endian)
  ...  header      header_len bytes of UTF-8 JSON:
         {
           "tokenizer_source": str,      # must match --tokenizer at train time
           "max_len": int,               # max_len examples were cropped to
           "think_end_markers": [str],   # markers used for the think/answer split
           "id_typecode": "H" | "I",     # array.array typecode for ids
           "vocab_ceiling": int,         # len(tokenizer) at write time
         }
  then records back-to-back until EOF, each:
    I              n_tokens (uint32 LE)
    n_tokens * w   ids       (w = 2 bytes if id_typecode=='H' else 4, LE)
    n_tokens       mask       (1 byte each, 0/1)
    n_tokens       is_answer  (1 byte each, 0/1)

No trailing count/footer -- readers stream until EOF. That means writing is
a single forward pass (works fine both for a live HF-streaming pull and for
a multi-GB pretrain corpus) and a crashed/truncated write just yields fewer
examples on read rather than a corrupt file structure.
"""
from __future__ import annotations

import array
import json
import mmap
import random
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

MAGIC = b"NANTYBN1"
_HEADER_LEN_FMT = "<I"
_RECORD_LEN_FMT = "<I"

# Default think/answer split markers -- covers the OpenThoughts /
# Bespoke-Stratos / Sky-T1 family convention as well as plain
# <think>...</think>. Kept here (not just in the trainer) so prepare_data.py
# uses the identical default when the caller doesn't override it.
DEFAULT_THINK_END_MARKERS = ["<|begin_of_solution|>", "</think>", "<|end_of_thought|>"]

ROLE_TOKEN = {
    "system":    "<|system|>",
    "user":      "<|user|>",
    "assistant": "<|assistant|>",
}
END_TOKEN = "<|end|>"


# ---------------------------------------------------------------------------
# In-memory example representation (array.array-packed -- see the docstring
# in the original trainer for why: plain list[int]/list[bool] cost ~28 bytes
# of Python object overhead PER TOKEN, which OOM'd a 5M-line tokenization
# pass; array.array stores a real packed C buffer instead).
# ---------------------------------------------------------------------------

@dataclass
class Example:
    ids:       "array.array"   # typecode 'H' (vocab <= 65536) or 'I'
    mask:      "array.array"   # typecode 'B' -- 1 where loss is computed
    is_answer: "array.array"   # typecode 'B' -- 1 for final-answer tokens


def typecode_for_vocab(vocab_ceiling: int) -> str:
    """'H' (unsigned short, 2 bytes) covers any vocab up to 65536; larger
    vocabs (e.g. a 200064-token tokenizer) need 'I' (4 bytes) or ids would
    silently wrap and corrupt token ids above 65535."""
    return "H" if vocab_ceiling <= 0xFFFF else "I"


# ---------------------------------------------------------------------------
# Conversation -> (ids, mask, is_answer). Canonical implementation -- both
# prepare_data.py and train_nanity_fixed.py call this, so a .bin file
# written by one always matches what the other would have produced.
# ---------------------------------------------------------------------------

def normalize_messages(msgs: list[dict]) -> list[dict]:
    """ShareGPT-style {"from": "human"/"gpt"/"bot", "value": ...} ->
    {"role": "user"/"assistant"/..., "content": ...}. Passes through
    messages that already use "role"/"content". Drops any message missing
    a role or content after normalization."""
    normed = []
    for m in msgs:
        role = m.get("role") or m.get("from", "")
        role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)
        content = m.get("content") or m.get("value", "")
        if role and content:
            normed.append({"role": role, "content": content})
    return normed


def apply_chat_template(messages: list[dict], tokenizer,
                         think_end_markers: Optional[list[str]] = None
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
    """
    if think_end_markers is None:
        think_end_markers = DEFAULT_THINK_END_MARKERS

    ids: list[int] = []
    mask: list[bool] = []
    is_answer: list[bool] = []

    def enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    def add(token_ids: list[int], is_loss: bool, is_ans: bool = False):
        ids.extend(token_ids)
        mask.extend([is_loss] * len(token_ids))
        is_answer.extend([is_ans] * len(token_ids))

    def split_think_answer(content: str) -> tuple[str, str]:
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

    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        is_asst = role == "assistant"

        add(enc(ROLE_TOKEN.get(role, f"<|{role}|>")), is_loss=False)

        if is_asst:
            think_text, answer_text = split_think_answer(content)
            if think_text:
                add(enc(think_text), is_loss=True, is_ans=False)
            add(enc(answer_text), is_loss=True, is_ans=True)
        else:
            add(enc(content), is_loss=False)

        add(enc(END_TOKEN), is_loss=is_asst, is_ans=is_asst)

    return ids, mask, is_answer


def crop_preserving_answer(ids: list[int], mask: list[bool], is_answer: list[bool],
                            max_len: int) -> tuple[list, list, list, bool]:
    """If the sequence is over max_len, drop tokens from the THINK region
    only (never the answer, never the system/user prompt) until it fits.
    Returns ok=False if there aren't enough think tokens to drop -- caller
    should skip the example rather than truncate off the answer."""
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


def tokenize_conversation(messages: list[dict], tokenizer, max_len: int,
                           id_typecode: str,
                           think_end_markers: Optional[list[str]] = None
                           ) -> tuple[Optional[Example], str]:
    """Full pipeline: apply_chat_template -> crop_preserving_answer -> pack
    into an Example. Returns (None, "skip"|"overflow") on failure, else
    (Example, "ok"). was-cropped info is discarded here (prepare_data.py
    logs aggregate counts itself)."""
    ids, mask, is_answer = apply_chat_template(messages, tokenizer, think_end_markers)
    if len(ids) > max_len:
        ids, mask, is_answer, ok = crop_preserving_answer(ids, mask, is_answer, max_len)
        if not ok:
            return None, "overflow"
    if not any(mask):
        return None, "skip"
    packed = Example(
        array.array(id_typecode, ids),
        array.array("B", mask),
        array.array("B", is_answer),
    )
    return packed, "ok"


# ---------------------------------------------------------------------------
# Sequence packing -- concatenate many variable-length examples into
# fixed-length max_len rows instead of leaving each example as its own
# separately-padded row. Used at TRAIN LOAD TIME (after read_bin_dataset),
# not at prepare_data.py write time, so max_len can change between runs
# without needing to regenerate the .bin file.
# ---------------------------------------------------------------------------

def pack_examples(examples: "list[Example]", max_len: int, pad_id: int = 0,
                   seed: Optional[int] = None) -> "list[Example]":
    """Greedily concatenate examples end-to-end into fixed-length max_len
    sequences ("sequence packing"), instead of leaving each example as its
    own separately-padded row.

    Why this matters: the trainer's collate_fn right-pads every BATCH to
    its longest example. For reasoning-trace SFT data (huge length
    variance -- some conversations are a couple hundred tokens, some run
    to 8k) or the trailing chunk of each pretrain document (as short as
    64 tokens, see CHUNK_TOKENS in prepare_data.py), that means most rows
    in most batches are mostly pad tokens burning GPU cycles for zero
    loss signal. Packing eliminates almost all of that: every output
    sequence is exactly max_len (except possibly the very last one in the
    whole list, which gets padded like before), built by concatenating
    whole examples back-to-back until the next one wouldn't fit, then
    starting a fresh sequence.

    CROSS-DOCUMENT ATTENTION TRADE-OFF: packed sequences are NOT isolated
    with a block-diagonal attention mask -- a token near the start of an
    example CAN attend to the tail end of the previous example packed
    into the same row (RoPE positions also just keep incrementing across
    the boundary, they're not reset to 0 per document). This is the
    standard trade-off essentially every from-scratch packed-pretraining
    setup makes, because a real block-diagonal mask requires passing an
    explicit attn_mask into the model -- which is exactly what disables
    the SDPA is_causal fused-kernel fast path this trainer depends on
    (see the PERF FIX comments around attn_mask in evaluate()/the train
    loop in train_nanity_fixed.py). Leaving attn_mask=None is what keeps
    packing compatible with that fast path with zero changes to
    modeling_nanity.py. If this measurably hurts SFT quality (unrelated
    conversations blending together near a packing boundary), the real
    fix is per-document position_id resets plus a varlen/block-causal
    attention path -- a model-level change, out of scope here.

    Order: examples are shuffled once (seeded, so reproducible across
    resumes) before packing, so which documents end up sharing a row
    isn't just an accident of file order (e.g. an entire source dataset
    landing in the same handful of packed blocks because it was
    contiguous in the .bin file). The packing COMPOSITION is then fixed
    -- DataLoader(shuffle=True) still reshuffles which packed rows land
    in which BATCH every epoch, but it can't un-pack and re-pack them.
    Call this function again (e.g. with a different seed) to get a fresh
    packing if that fixed composition becomes a concern on a very
    long run.
    """
    if not examples:
        return []

    id_typecode = examples[0].ids.typecode
    order = list(range(len(examples)))
    if seed is not None:
        random.Random(seed).shuffle(order)

    packed: list[Example] = []
    buf_ids  = array.array(id_typecode)
    buf_mask = array.array("B")
    buf_ans  = array.array("B")
    n_truncated = 0

    def flush():
        nonlocal buf_ids, buf_mask, buf_ans
        if len(buf_ids) == 0:
            return
        pad_n = max_len - len(buf_ids)
        if pad_n > 0:
            buf_ids.extend(array.array(id_typecode, [pad_id]) * pad_n)
            buf_mask.extend(bytes(pad_n))    # zero-filled
            buf_ans.extend(bytes(pad_n))
        packed.append(Example(buf_ids, buf_mask, buf_ans))
        buf_ids  = array.array(id_typecode)
        buf_mask = array.array("B")
        buf_ans  = array.array("B")

    for i in order:
        ex = examples[i]
        n = len(ex.ids)
        if n > max_len:
            # Shouldn't happen -- crop_preserving_answer()/CHUNK_TOKENS
            # upstream both enforce <= max_len -- but don't silently
            # corrupt a packed row if it ever does anyway; truncate
            # defensively and warn instead.
            n_truncated += 1
            ex = Example(ex.ids[:max_len], ex.mask[:max_len], ex.is_answer[:max_len])
            n = max_len
        if len(buf_ids) + n > max_len:
            flush()
        buf_ids.extend(ex.ids)
        buf_mask.extend(ex.mask)
        buf_ans.extend(ex.is_answer)
        if len(buf_ids) == max_len:
            flush()
    flush()

    if n_truncated:
        print(f"[pack] WARNING: {n_truncated} example(s) exceeded max_len="
              f"{max_len} and were truncated before packing -- this should "
              f"not happen if crop_preserving_answer()/CHUNK_TOKENS upstream "
              f"are working correctly; investigate before trusting this run.")

    orig_examples = len(examples)
    orig_tokens = sum(len(ex.ids) for ex in examples)
    packed_slots = len(packed) * max_len
    util = 100.0 * orig_tokens / packed_slots if packed_slots else 0.0
    print(f"[pack] packed {orig_examples:,} examples ({orig_tokens:,} real "
          f"tokens) into {len(packed):,} sequences of length {max_len} "
          f"({packed_slots:,} slots, {util:.1f}% token utilization, "
          f"{100 - util:.1f}% padding) -- was {orig_examples:,} separately-"
          f"padded rows before packing")
    return packed


# ---------------------------------------------------------------------------
# .bin writer / reader
# ---------------------------------------------------------------------------

class BinDatasetWriter:
    """Streaming writer -- call add_example()/add_ids() per tokenized
    conversation, then close() (or use as a context manager). Never holds
    more than one example in memory, so it's safe for multi-GB pretrain
    corpora with millions of examples."""

    def __init__(self, path, tokenizer_source: str, max_len: int,
                 vocab_ceiling: int, think_end_markers: Optional[list[str]] = None,
                 append: bool = False):
        """append=True: if `path` already exists and is a valid .bin file
        whose header (tokenizer/max_len/think_end_markers/vocab_ceiling)
        matches these arguments, open it in append mode and keep writing
        records after the existing ones -- mirrors the old JSONL "append
        instead of overwrite so a retry doesn't have to re-pull the
        expensive part" behavior. If the header doesn't match (or the file
        doesn't exist), falls back to creating a fresh file, same as
        append=False."""
        self.path = Path(path)
        self.id_typecode = typecode_for_vocab(vocab_ceiling)
        header = {
            "tokenizer_source": tokenizer_source,
            "max_len": max_len,
            "think_end_markers": list(think_end_markers or DEFAULT_THINK_END_MARKERS),
            "id_typecode": self.id_typecode,
            "vocab_ceiling": vocab_ceiling,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if append and self.path.exists():
            try:
                existing = read_bin_header(self.path)
            except Exception:
                existing = None
            if existing == header:
                self._f: BinaryIO = open(self.path, "ab")
                self.n_written = 0
                return
            elif existing is not None:
                print(f"[nanity_data_format] {self.path} exists but its header "
                      f"doesn't match this run's tokenizer/max_len/"
                      f"think_end_markers -- overwriting instead of appending "
                      f"(existing: {existing}, this run: {header})")

        header_bytes = json.dumps(header).encode("utf-8")
        self._f = open(self.path, "wb")
        self._f.write(MAGIC)
        self._f.write(struct.pack(_HEADER_LEN_FMT, len(header_bytes)))
        self._f.write(header_bytes)
        self.n_written = 0

    def add_example(self, ex: Example):
        self.add_arrays(ex.ids, ex.mask, ex.is_answer)

    def add_arrays(self, ids: "array.array", mask: "array.array", is_answer: "array.array"):
        n = len(ids)
        if n == 0:
            return
        if ids.typecode != self.id_typecode:
            ids = array.array(self.id_typecode, ids)
        if sys.byteorder != "little":
            ids = array.array(ids.typecode, ids)
            ids.byteswap()
        self._f.write(struct.pack(_RECORD_LEN_FMT, n))
        self._f.write(ids.tobytes())
        self._f.write(bytes(bytearray(mask)))
        self._f.write(bytes(bytearray(is_answer)))
        self.n_written += 1

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_bin_header(path) -> dict:
    """Just the header -- cheap, for validating a .bin against the
    tokenizer/max_len/think_end_markers a training run was invoked with,
    without loading the whole file."""
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"{path}: not a NANITY .bin file (bad magic)")
        (header_len,) = struct.unpack(_HEADER_LEN_FMT, f.read(4))
        return json.loads(f.read(header_len).decode("utf-8"))


def read_bin_dataset(path) -> tuple[list[Example], dict]:
    """Reads a .bin file fully into memory and returns (examples, header).
    This is the ONLY thing train_nanity_fixed.py needs to call to load a
    pre-tokenized dataset -- no tokenizer, no chat template, no cropping.

    BLOCK-READ REWRITE: the previous version called f.read(4) once per
    record just to get the length prefix, then a couple more small
    f.read(n*w)/f.read(n)/f.read(n) calls for the payload -- FOUR Python-
    level read() calls per example. For an 8B-token pretrain file with
    millions of packed examples, that's tens of millions of individual
    read() calls, each paying interpreter + (for anything not already in
    the OS page cache) syscall overhead, even though the underlying
    io.BufferedReader was already doing some internal buffering.

    This version mmaps the whole file once (a single block mapping, not a
    read -- pages get faulted in by the OS as they're actually touched,
    so this doesn't front-load the whole file into RAM any harder than
    reading it would) and walks it with plain offset arithmetic:
    struct.unpack_from() and array.frombytes() both read directly out of
    the mmap buffer via the buffer protocol, with zero Python-level I/O
    calls in the per-record loop at all -- the "read" work reduces to
    pointer arithmetic over memory the OS has already mapped in.

    Falls back to plain buffered reads (the old behavior, just with a much
    larger buffer) if mmap isn't available for this path (e.g. a 0-byte
    file, or a filesystem/OS combination that rejects mmap).
    """
    path = Path(path)
    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError(f"{path}: empty file -- was this actually produced "
                          f"by prepare_data.py?")

    with open(path, "rb") as f:
        try:
            buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):
            # mmap can refuse on some filesystems (network mounts, certain
            # container overlay setups) -- fall back to a big buffered read
            # instead of hard-failing the whole load.
            f.seek(0)
            buf = f.read()

        pos = 0
        magic = bytes(buf[pos:pos + len(MAGIC)]); pos += len(MAGIC)
        if magic != MAGIC:
            raise ValueError(f"{path}: not a NANITY .bin file (bad magic) -- "
                              f"was this actually produced by prepare_data.py?")
        (header_len,) = struct.unpack_from(_HEADER_LEN_FMT, buf, pos); pos += 4
        header = json.loads(bytes(buf[pos:pos + header_len]).decode("utf-8"))
        pos += header_len

        id_typecode = header["id_typecode"]
        itemsize = array.array(id_typecode).itemsize
        big_endian = sys.byteorder != "little"

        examples: list[Example] = []
        end = len(buf)
        while pos < end:
            if pos + 4 > end:
                break  # truncated trailing record (crashed mid-write) -- stop cleanly
            (n,) = struct.unpack_from(_RECORD_LEN_FMT, buf, pos); pos += 4

            ids_nbytes = n * itemsize
            record_end = pos + ids_nbytes + n + n
            if record_end > end:
                break  # same: truncated trailing record, stop cleanly

            ids = array.array(id_typecode)
            ids.frombytes(buf[pos:pos + ids_nbytes]); pos += ids_nbytes
            if big_endian:
                ids.byteswap()

            mask = array.array("B")
            mask.frombytes(buf[pos:pos + n]); pos += n

            is_answer = array.array("B")
            is_answer.frombytes(buf[pos:pos + n]); pos += n

            examples.append(Example(ids, mask, is_answer))

        if isinstance(buf, mmap.mmap):
            buf.close()

    return examples, header


def is_bin_file(path) -> bool:
    """Cheap sniff so callers can auto-detect .bin vs .jsonl without relying
    on the file extension alone (in case someone renames it)."""
    path = Path(path)
    if path.suffix == ".bin":
        return True
    try:
        with open(path, "rb") as f:
            return f.read(len(MAGIC)) == MAGIC
    except OSError:
        return False
