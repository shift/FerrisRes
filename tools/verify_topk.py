#!/usr/bin/env python3
"""Verify a top-k teacher dump against its corpus.

    python tools/verify_topk.py <topk-dir> <corpus-train-shard> [--k 32]

Checks the file shape, then the thing that actually matters: **alignment**. A
`prompt_logprobs` array is easy to read one position late. If the dump is aligned,
the teacher's own top-1 prediction at position *i* is the token that is really
there at *i* a large fraction of the time; one position off, the top-1 matches
*i+1* instead. The script measures both rates and fails when the wrong one wins.

Exit status is 0 only when every structural check passes and the alignment test
picks the first hypothesis.
"""
import argparse
import glob
import json
import struct
import sys
from pathlib import Path


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def read_u16(path):
    raw = Path(path).read_bytes()
    if len(raw) % 2:
        fail(f"{path}: odd byte count, not u16 ids")
    return list(struct.unpack("<%dH" % (len(raw) // 2), raw))


def byte_decode(ids):
    """SimpleTokenizer: byte b is id b+3; ids 0-2 are <unk>/<eos>/<pad>."""
    out = bytearray()
    for i in ids:
        if 3 <= i <= 258:
            out.append(i - 3)
        elif i == 1:
            out.extend(b"\n")
    return bytes(out).decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topk_dir")
    ap.add_argument("corpus_shard")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--vocab", type=int, default=None)
    args = ap.parse_args()

    import numpy as np

    d = Path(args.topk_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    k = manifest.get("k", args.k)
    dtype = np.dtype([("idx", "<i4", (k,)), ("val", "<f4", (k,)), ("off", "<i4")])

    parts = sorted(d.glob("part-*.bin"))
    if not parts:
        fail(f"no part-*.bin in {d}")

    total = 0
    first_rec = None
    for p in parts:
        rec = np.fromfile(p, dtype=dtype)
        if rec.size == 0:
            fail(f"{p}: empty")
        if first_rec is None:
            first_rec = rec
        if not np.all(rec["val"][:, :-1] >= rec["val"][:, 1:]):
            fail(f"{p}: top-k values are not sorted descending")
        if np.any(np.diff(rec["off"]) < 0):
            fail(f"{p}: byte offsets are not non-decreasing")
        if np.any(rec["idx"] < 0):
            fail(f"{p}: negative token id")
        total += rec.size

    vocab = args.vocab or manifest.get("vocab")
    if vocab and np.any(first_rec["idx"] >= vocab):
        fail(f"token id beyond vocab {vocab}")

    if manifest.get("tokens") not in (None, total):
        fail(f"manifest says {manifest['tokens']} tokens, files hold {total}")
    print(f"structure ok: {len(parts)} parts, {total} records, k={k}")

    # ---- alignment ----
    ids = read_u16(args.corpus_shard)
    text = byte_decode(ids)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    tids = enc["input_ids"]

    n = min(total, len(tids) - 1)
    if n < 1000:
        fail(f"only {n} records to compare; need >= 1000")

    hit_same = 0
    hit_next = 0
    for i in range(n):
        top1 = int(first_rec["idx"][i][0]) if i < first_rec.size else None
        if top1 is None:
            break
        if i < len(tids) and top1 == tids[i]:
            hit_same += 1
        if i + 1 < len(tids) and top1 == tids[i + 1]:
            hit_next += 1

    same = hit_same / n
    nxt = hit_next / n
    print(f"alignment: top-1 == token[i] {same:.1%}   top-1 == token[i+1] {nxt:.1%}")
    if nxt > same:
        fail("the dump is one position off: its top-1 predicts the NEXT token, "
             "so the consumer must shift by one or the dump must be re-read")
    if same < 0.15:
        fail(f"top-1 matches the aligned token only {same:.1%} of the time; "
             "either the tokenizer differs from the teacher's or the dump is wrong")
    print("PASS")


if __name__ == "__main__":
    main()
