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

## The first hypothesis to test

The T4 driver advertises cooperative-matrix support and wgpu logs
`Found 6 cooperative matrix configurations`. If a kernel path selected on that
capability is used with f32 data, or if a matmul is dispatched with a workgroup
shape tuned for another adapter, gradients can be wrong while still running.

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
