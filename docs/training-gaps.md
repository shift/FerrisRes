# What stands between here and a trained model

Status: **register, 2026-11-26.** Ordered by what blocks what. Each row names the
one thing that closes it, not a programme.

| # | gap | blocks | closes with | state |
|---|---|---|---|---|
| 1 | **batched step used the last window's gradient** | every GPU training run with `batch_size > 1` | `c5cd360`; kernel gradients were correct on the T4 all along | **closed, verified** |
| 2 | 32k corpus not rebuilt | the first student's tokenizer | `corpus build` with `--tokenizer` (landed, `781f28f`) run on the release plan | ready, CPU only |
| 3 | corpus is 22.7M tokens, not 300–500M | quality | FineWeb-Edu / FineWeb-2 adapters (parquet converter) | missing |
| 4 | no teacher distillation artefact | text-aligned KD | `docs/tasks/topk-teacher.md`: transformers on one T4, 927.9 tok/s measured | **smoke artefact produced** (200,508 tokens), verification pending |
| 5 | replay+KL term not in the trainer | non-collapse during SFT | `docs/tasks/replay-kl.md` | task only |
| 6 | no eval suite | any claim that training worked | harness integration + eval runner | missing |
| 7 | no persona/soul data | persona adherence | `soul-*` source rows | planned |
| 8 | **prefill uses one core** | prefill UX on every CPU host | `perf/parallel-prefill` `21e966f`: **2.84× at 512, 3.0× at 2048**, bit-identical logits, 1634 lib tests | **implemented, unmerged** |
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

## The CPU ceiling the prefill work exposed

`perf/parallel-prefill` measured the parallel ternary kernel at **0.88–0.96
GFLOP/s on 4 threads**, against **36 GFLOP/s** for `matrixmultiply` fp32 GEMM on
the same shape. Threads are now exhausted: the remaining ~40× is the branchy
`if w > 0` inner loop, not the number of cores. Three branches overlap there and
must be reconciled once, not merged blind:

| branch | what it adds | file |
|---|---|---|
| `feat/cpu-bmi2-ternary` `42cf762` | BMI2 `pext` + AVX2 masked adds for the packed path | `src/model/ternary.rs` |
| `perf/ternary-kernels` `9fe334f` | branchless scalar kernels, same dispatch | `src/model/ternary.rs` |
| `perf/parallel-prefill` `21e966f` | MoE expert grouping, attention scratch, batched projections | `cpu_moe.rs`, `cpu_block_attn_res.rs`, `cpu_linear.rs` |

## What this register deliberately excludes

The harness (`agentic-loop`) merge and its release binary, the TPU path, the 9B
tier, multimodal, and the Armor gate. They matter, but none of them moves the
first student closer, and three of them have already cost runs without producing
a measurement.
