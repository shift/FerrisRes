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
