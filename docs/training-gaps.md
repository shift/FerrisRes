# What stands between here and a trained model

Status: **register, 2026-11-26.** Ordered by what blocks what. Each row names the
one thing that closes it, not a programme.

| # | gap | blocks | closes with | state |
|---|---|---|---|---|
| 1 | **T4 backward pass produces rising loss** | every GPU training run at any scale | a per-tensor CPU-vs-GPU gradient comparison on the T4, then the kernel fix | **in progress** |
| 2 | 32k corpus not rebuilt | the first student's tokenizer | `corpus build` with `--tokenizer` (landed, `781f28f`) run on the release plan | ready, CPU only |
| 3 | corpus is 22.7M tokens, not 300–500M | quality | FineWeb-Edu / FineWeb-2 adapters (parquet converter) | missing |
| 4 | no teacher distillation artefact | text-aligned KD | top-k dump: 8B bf16 on one T4 with vLLM `prompt_logprobs` | notebook not written |
| 5 | replay+KL term not in the trainer | non-collapse during SFT | `docs/tasks/replay-kl.md` | task only |
| 6 | no eval suite | any claim that training worked | harness integration + eval runner | missing |
| 7 | no persona/soul data | persona adherence | `soul-*` source rows | planned |
| 8 | **prefill uses one core** | prefill UX on every CPU host | call the rayon kernels the model path already has (`docs/tasks/cpu-prefill-threads.md`) | ready, same file as the AVX2 work |
| 9 | prefix KV cache not wired to `infer` | prefill UX | wiring task | library only |
| 10 | multi-GPU not wired | throughput | local-SGD coordination | CPU-only proof |

## The order that actually works

1. **Fix #1.** Nothing else produces a model. Two of the three failures so far
   were my own notebook bugs; the GPU defect itself is isolated to the backward
   pass or the optimiser update, because the forward pass agrees with CPU to
   1.1e-5 and the CPU learns on the same data.
2. **#2 while #1 is being fixed.** It needs no GPU, and it is the difference
   between training on 259 byte ids and training on the 32k vocabulary the
   student is specified with.
3. **#3 and #4** are the two inputs to a real run: more tokens, and a teacher
   signal. Neither needs the GPU defect fixed to be *built*.
4. **#5 and #6** before the first long run, not after. A run without an eval is
   a number, not a result.
5. **#7–#10** after the first checkpoint exists.

## What this register deliberately excludes

The harness (`agentic-loop`) merge and its release binary, the TPU path, the 9B
tier, multimodal, and the Armor gate. They matter, but none of them moves the
first student closer, and three of them have already cost runs without producing
a measurement.
