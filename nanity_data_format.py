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
    pre-tokenized dataset -- no tokenizer, no chat template, no cropping."""
    path = Path(path)
    examples: list[Example] = []
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"{path}: not a NANITY .bin file (bad magic) -- "
                              f"was this actually produced by prepare_data.py?")
        (header_len,) = struct.unpack(_HEADER_LEN_FMT, f.read(4))
        header = json.loads(f.read(header_len).decode("utf-8"))
        id_typecode = header["id_typecode"]
        itemsize = array.array(id_typecode).itemsize
        while True:
            len_bytes = f.read(4)
            if len(len_bytes) < 4:
                break
            (n,) = struct.unpack(_RECORD_LEN_FMT, len_bytes)
            ids = array.array(id_typecode)
            ids.frombytes(f.read(n * itemsize))
            if sys.byteorder != "little":
                ids.byteswap()
            mask = array.array("B")
            mask.frombytes(f.read(n))
            is_answer = array.array("B")
            is_answer.frombytes(f.read(n))
            examples.append(Example(ids, mask, is_answer))
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
