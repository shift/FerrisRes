# Top-k teacher logits: the route that works

Status: **smoke produced artefact, 2026-11-26.** Supersedes
`tpu-topk-teacher.md` (TPU, six failures) and the vLLM variant (one failure).

Notebook: `notebooks/kaggle/qwen3_topk_vllm/qwen3_topk_vllm.ipynb` on
`docs/kaggle-notebooks` (kernel `sectionme/ferrisres-topk-vllm`).
Corpus dataset: `sectionme/ferrisres-corpus-release-slice`.
Checker: `tools/verify_topk.py`.

## What works

Plain `transformers`, one forward pass per 1024-token window, on one Kaggle T4.

| fact | value |
|---|---|
| teacher in the smoke | `Qwen/Qwen3-4B`, fp16 (Turing has no bf16), `sdpa` |
| measured throughput | **927.9 teacher tokens/s** on one T4 (200,508 tokens in 216 s) |
| projection for 22.7M tokens | **≈6.8 h** — one 9 h session, effectively no margin |
| alignment | `logits[i-1]` predicts token `i`; the record at `i` carries it |
| corpus decode | byte *b* is **id b+3**; ids 0–2 are `<unk>/<eos>/<pad>` |

## Three failure modes that cost runs, recorded

1. **TPU**: `torch_xla`/`torch` ABI mismatches, then no wheel for torch 2.11, then
   `libtpu` absent so PJRT reported `CPU`. Six runs, zero compute. The engine has
   no XLA path and never will without a second implementation.
2. **vLLM 0.34**: the engine aborts in its own kernel warmup on a compute-7.5
   card — `vllm/models/minimax_m3/.../index_decode_score.py` imports `cutlass`,
   which is not installed. Unrelated to the model. A serving engine is the wrong
   tool for a bulk forward pass.
3. **Decode**: the corpus stores `SimpleTokenizer` ids, not bytes. `id & 0xFF`
   shifts every byte by three and breaks UTF-8 at the first non-ASCII character.
   The packed stream also holds exactly **one** truncated multi-byte character, at
   a 512-token window cut (offset 17,854,464, `mod 512 == 0`), so decoding uses
   `errors="replace"`: one U+FFFD in 22.1M.

## To reach the real artefact

The smoke proves format, not scale. Two levers, in this order:

1. **batch 4–8 windows per forward**: 3–5×, one parameter, one process → ~1.7 h;
2. **both T4s, data parallel** (4B per card, shard ranges split): ~2×.

8B fp16 is 16 GB and does not fit one T4; it needs both, and the post that
prompted this thread measured tensor-parallel over PCIe working.

## Gate before scaling

`tools/verify_topk.py <dir> <train-shard>` must pass, and its alignment line must
show `top-1 == token[i]` ahead of `token[i+1]`. A dump that is one position off is
worse than no dump: the student would learn a shifted distribution and every
training number afterwards would be suspect.
