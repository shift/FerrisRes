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
    ap.add_argument("--allow-partial", action="store_true",
                    help="a subset of the parts is present; the token-count check becomes a warning")
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
        msg = f"manifest says {manifest['tokens']} tokens, files hold {total}"
        if args.allow_partial:
            print(f"warning: {msg} (partial set, {len(parts)} of {len(manifest.get('parts', []))} parts)")
        else:
            fail(msg)
    print(f"structure ok: {len(parts)} parts, {total} records, k={k}")

    # ---- alignment ----
    # The dump starts at token 1 by design (`logits[i-1]` predicts token `i`), so
    # record i is NOT token i. Locate each record by the byte offset it stores —
    # that is why the offset is in the record — and measure which hypothesis the
    # top-1 supports.
    ids = read_u16(args.corpus_shard)[:200_000]
    text = byte_decode(ids)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok.model_max_length = 10**9
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    tids = enc["input_ids"]
    by_offset = {o[0]: k for k, o in enumerate(enc["offset_mapping"])}

    same = nxt = prev = 0
    n = 0
    for i in range(first_rec.size):
        k = by_offset.get(int(first_rec["off"][i]))
        if k is None:
            continue
        top1 = int(first_rec["idx"][i][0])
        n += 1
        if k < len(tids) and top1 == tids[k]:
            same += 1
        if k + 1 < len(tids) and top1 == tids[k + 1]:
            nxt += 1
        if k > 0 and top1 == tids[k - 1]:
            prev += 1
    if n < 500:
        fail(f"only {n} records could be located by offset; needs >= 500")

    same /= n
    nxt /= n
    prev /= n
    print(f"alignment over {n} records:")
    print(f"  top-1 == token[k]   {same:.1%}")
    print(f"  top-1 == token[k+1] {nxt:.1%}")
    print(f"  top-1 == token[k-1] {prev:.1%}")
    if nxt > same:
        fail("the dump predicts the NEXT token relative to the offsets it stores: "
             "either the offsets or the logits slice is off by one")
    if same < 0.20:
        fail(f"top-1 agrees with the token at its own offset only {same:.1%}; "
             "the teacher's tokenizer and the dump disagree")
    print("PASS")


if __name__ == "__main__":
    main()
