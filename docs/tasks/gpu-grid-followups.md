# GPU dispatch-limit follow-ups

Status: **implemented, not executed.** Found by static review of
`int/perf-cpu-grid2d` (cherry-picked `fa74d2f` + `3e03917` onto `8989ada`).
No GPU exists on the review host, so nothing in this document is a test result.

## Fixed here (commit `0bb012b`)

`BackwardPass::loss_grad` and `softmax_grad` still used
`@builtin(global_invocation_id)` with a flat `dispatch_workgroups(wg, 1, 1)`.
The 2D-grid commit converted the head, elementwise and optimizer kernels but not
these. At vocab 32,768 and seq 1,024 the loss tensor is 33,521,664 elements,
i.e. **130,944 workgroups in one dimension** against the spec cap of 65,535 —
the workload the grid commit was meant to enable. Both shaders now read
`num_workgroups` and flatten the grid themselves.

Verification when a GPU is available:

```bash
cargo test --test gpu_train_grid_limit
cargo test --lib a_32k_vocabulary_loss_gradient_fits_a_2d_grid
```

The arithmetic test runs anywhere; the GPU test prints `skipped: no adapter`
and returns on a host without one, so a green run there proves nothing unless
the skip line is absent.

## Also fixed (commit `7d8ede0`, and the attention change beside it)

- `gemma_mapper::dispatch_backward` dispatched `nq + bs` threads for a shader
  covering `(nq + bs) * hd` elements — **1/768 of every gradient row** was
  written. Now folded over a 2D grid.
- `optimizer_scale` dispatched 98,304 groups in one dimension at 32k × 768, over
  the cap. Both kernels already had `idx >= total` guards; host and index fold.
- `gpu_transformer` causal attention kept scores in `array<f32, 256>` while the
  model runs `max_seq = 1024`. Scores are now recomputed per pass; stateless and
  correct at any length, at the cost of two extra QK dot products per position.

None of these has been executed on a GPU host.

## Open, ranked

1. **Elementwise over-dispatch without a guard** (`src/compute/kernels/elementwise.rs`).
   `grid()` clamps `gx` to 65,535 and raises `gy`, so the head gradient at 32k
   dispatches ~33.5M threads for 25.2M elements. Vulkan robustness discards the
   out-of-range writes, but WGSL permits clamping (which would corrupt the last
   accumulator), and ~8.3M threads are wasted. Fix: pass `numel` and return when
   `idx >= numel`, or set `gx = ceil(groups / gy)`.
2. **Adam guards on a different bound than it dispatches**
   (`src/training/optimizer.rs`). The shader bounds by `arrayLength(&grad)`
   (whole allocation); the host dispatches `ceil(numel/256)`. Exact today because
   every call passes a dedicated `GpuBuffer::new`, wrong for any future
   `from_existing` view. Fix: pass `numel` in `AdamParams`.
3. **Tokenizer head can be smaller than the ids `encode()` emits**
   (`src/main.rs`). `vocab_size()` excludes added tokens and BOS may be added, so
   ids can exceed `config.vocab`; `--vocab` is silently ignored when
   `--tokenizer` is set, and its help text still describes the byte tokenizer.
4. **`TrainOps::dispatch` asserts dimensions but not coverage**
   (`src/autodiff/gpu_ops.rs`). Add `gx * gy * WG >= elements` and, where the
   shader takes `grid_x`, `gx == grid_x`.

## Measured on a Kaggle T4, 0.2.9 (2026-11-27)

The per-step timing build ran the real shape (hidden 768, 8 layers, heads 12,
head_dim 64, intermediate 3072, vocab 16384, seq 512, batch 1) on a hardware T4
(Vulkan adapter, not llvmpipe). One optimizer step:

| phase | ms |
|---|---|
| forward | ~10 000 |
| backward | ~4 850 |
| CPU cross-entropy | ~100 |
| logits readback | ~58 |
| optimizer | ~170 |
| checkpoint (every 50) | 3 900 |
| **total** | **~15 200** |

Conclusion: the readback and CPU loss are **not** the bottleneck (2 % combined).
Forward and backward are 98 %, so the next work is in the GPU graph/kernels —
dispatch count, per-token/per-layer submissions, encoder reuse — not in moving
cross-entropy to the GPU.

Two environment facts that cost runs:

- Kaggle kernels must have `enable_internet: true`. Without it, apt fails at
  DNS (`Temporary failure resolving archive.ubuntu.com`) and the Vulkan ICD has
  no `libGLX_nvidia`; the failure is silent if apt's exit is piped through
  `tail`. The 0.2.9 notebook prints the real exit code.
- The `.deb` ships `usr/share/vulkan/icd.d/nvidia_icd.x86_64.json`; with the GL
  userspace installed and `VK_DRIVER_FILES`/`VK_ICD_FILENAMES` set,
  `ferrisres devices --json` lists the T4 as hardware.

## The attention fix is now measured (0.2.10, 2026-11-27)

Resumed the step-700 `FRTRAIN1` checkpoint on a T4 and ran to the 1200 s budget:

| phase | 0.2.9 | 0.2.10 |
|---|---|---|
| forward | ~10 000 ms | **2 550 ms** |
| backward | ~4 850 ms | ~4 875 ms |
| total step | ~15 200 ms | **8 011 ms** |

148 steps logged (701 → 848), checkpoint written at **825**, `resumed_from=700`.
Backward is unchanged, as expected: it never had the redundant QK dot. The
training step is now roughly halved and backward is the largest phase.

0.2.11 selects the 32x32 double-buffered matmul for M,N >= 32; its equivalence is
tested at non-square and non-multiple K, but its speed gain is not yet measured.
