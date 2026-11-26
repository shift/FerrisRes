# Top-k teacher logits on Kaggle TPU

Status: **ready, unexecuted.** Notebook:
`notebooks/kaggle/qwen3_topk_teacher.ipynb` on `docs/kaggle-notebooks`.

## Why

Distillation needs the teacher's distribution over the corpus. Running the
teacher beside the student requires both in one graph, which means porting the
engine to XLA. Capturing **top-k** logits breaks that dependency: the student
reads a static artefact, and the teacher never has to run again.

The TPU is the right device for this one job: a pure forward pass with no
training state. Kaggle provisions `machine_shape: TpuV5E8` for `enable_tpu: true`
(measured from the pushed kernel's metadata, 2026-11-26), not the v3-8 that the
docs elsewhere assume.

Detect it through XLA, never through `/dev`: a Kaggle TPU VM exposes no
`/dev/accel*` and sets no `TPU_NAME`, so `xm.xla_device_hw(xm.xla_device())`
returning `TPU` is the only check that works.

## The artefact

One record per **teacher** token, in teacher token space:

| field | type | bytes |
|---|---|---:|
| `topk_idx` | int32 × k | 4k |
| `topk_val` | bf16 × k | 2k |
| `byte_offset` | int32 | 4 |

`byte_offset` is the start of the teacher token in the decoded UTF-8 text. It is
not optional. The student has a 32k vocabulary and the teacher has 151k, so the
two tokenisations do not line up; the student side aligns by byte span and reads
the teacher distribution at the last teacher token inside each student span.
Without the offsets the dump is unusable and the discovery costs a full run.

Sizes, k=32:

| tokens | bytes |
|---:|---:|
| 1M (smoke) | 0.19 GB |
| 22.7M (the current release slice) | 4.4 GB |
| 100M | 19.2 GB |

Start at 22.7M. That is exactly the slice that exists, it fits one Kaggle
dataset, and it is enough to see whether KD moves the student at all.

## Input: the corpus is token ids, and that is fine

The builder writes `u16` ids with a byte-level tokenizer. That tokenizer is
lossless — no normaliser, no post-processor — so `decode(ids)` returns the exact
source bytes. The notebook decodes each shard to text, then tokenises with the
teacher's tokenizer. This must be asserted, not assumed: check that the decoded
text re-encodes to the same id sequence.

## The run

1. **Smoke**: Qwen3-0.6B, 1M tokens, k=8. Proves the pipeline and measures
   tokens/s. Stop if it does not produce a well-formed shard.
2. **Real**: Qwen3-1.7B, 22.7M tokens, k=32.
3. Record: XLA compile time separately from run time, tokens/s, output size,
   and a spot-check that `topk_val` for the argmax equals the teacher's own logit
   for that token.

## Gates

- the notebook refuses to run without a TPU (no CPU fallback);
- decoded text re-encodes to the original ids exactly;
- `topk_idx` sorted descending by `topk_val`, k distinct indices;
- every `byte_offset` non-decreasing and inside the text;
- output shard count and total tokens match the input manifest;
- a manifest records teacher id, revision, tokenizer hash, k, and the input
  shard hashes.

## Out of scope

Training the student, the KL term, the GPU regression, the corpus rebuild, and
any engine change. This task produces an artefact and a measurement.
