# The T4 path diverges

Status: **open, measured on Kaggle run 7 (2026-11-26).** Blocking: no training
claim can be made on any GPU until this is explained.

## The observation

The probe config (`--hidden-dim 64 --num-blocks 2 --block-size 4 --batch-size 4
--seq-len 128 --learning-rate 3e-3 --seed 42`, synthetic corpus, 144 steps):

| host | backend | initial loss | final loss | step time |
|---|---|---:|---:|---:|
| Intel HD 530 | vulkan (wgpu) | 3.557 | **0.0026** | ~0.16 s |
| Tesla T4 | vulkan (wgpu) | 5.764 | **8.258** | ~1.6 s |
| (reference) CPU f32 | — | converges | — | — |

Same binary family, same config, same corpus, opposite outcome. The T4 path
therefore has a defect in either the forward pass, the backward pass, or the
optimiser update.

## What run 9 ruled out (2026-11-26)

Same host, same seed, same corpus, CPU reference:

| backend | steps | initial | final | throughput |
|---|---:|---:|---:|---:|
| Tesla T4 | 144 | 5.763604 | **8.258290** | 321 tok/s |
| CPU (4 vCPU) | 48 | 5.763615 | **0.396426** | 907 tok/s |

Step-1 delta is **1.1e-5**, so the **forward pass is correct**; the corpus, seed
and learning rate are correct. The defect is in the **backward pass or the
optimiser update**. Cooperative matrices are no longer the first suspect — a
wrong matmul would have moved the forward too.

## The first hypothesis to test now

A race in gradient accumulation that scales with adapter parallelism. The
evidence fits: forward bit-identical, loss climbing monotonically, and two
previous gradient-buffer defects in this engine were both adapter-sensitive
(a 16×16 shader driven by 32×32 workgroups; grad shaders writing only column 0
and overwriting input grads). A T4 has 40 SMs against the HD 530's 24 EUs; a
missing barrier or a non-atomic `+=` is invisible on the smaller adapter and
double-counts on the larger one, which produces exactly a rising loss.

The test: compare one gradient tensor for one step between CPU and GPU on the
same batch. A ratio near 2.0 (or a count of duplicated contributions) names the
race; a ratio of 1.0 with a rising loss points at the update itself.

## The test

One behaviour: **the same config converges on the T4 or it does not.**
Name it `t4_converges_on_the_probe_config` in
`tests/gpu_train_t4.rs` (ignored unless a hardware Vulkan adapter is present).

It must:

1. build the probe config and corpus in-process;
2. run 144 steps on the first non-software adapter;
3. assert `final_loss < initial_loss * 0.9`;
4. assert every gradient tensor is finite;
5. on failure, print the adapter name, backend, and whether cooperative matrix
   was reported.

## Then, in order

1. Re-run the probe with the cooperative-matrix path disabled, if a switch
   exists; compare losses. One lever, one run.
2. Run the **CPU f32 reference** for the same 144 steps on the same host. If CPU
   diverges too, the corpus or the learning rate is at fault and the GPU is
   innocent.
3. Compare the first step's loss on both backends. They must agree to ~1e-3; the
   KD parity gate already holds that standard (`73a6902`).
4. Only then look at speed. A step that is 10× slower than the iGPU on a 64-hidden
   model is a dispatch problem (one invocation per head per token is the known
   suspect), and it is a separate task from correctness.

## Out of scope

The tokenizer, the corpus, the model shape, multi-GPU, and the `.deb`. This is
one probe config on one adapter.

## VERIFY

```
cargo test -p ferrisres gpu_train_t4 -- --ignored
```

A pass with zero tests is a failure: the filter must select
`t4_converges_on_the_probe_config`. On a host with no hardware Vulkan adapter the
test is skipped and reports that, not a pass.
