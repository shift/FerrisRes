# The iGPU loses the device at 768 wide, 256 tokens

Status: **reproducible, cause not yet isolated.** Blocks every real training run
on this host, so it precedes all speed work.

## The observation

`ferrisres train --device gpu --hidden-dim 768 --seq-len <S> --batch-size 1`

| seq | outcome | wall |
|---:|---|---:|
| 128 | runs, 18.9 s/step | 606 s for 32 steps |
| 256 | **device lost** | 38 s |
| 512 | **device lost** | 27 s |

Activation memory at 768×256 is under 1 MB, so this is not exhaustion. `wgpu`
reports `Validation Error / Parent device is lost` from `Device::poll`, which on
Intel Vulkan means the GPU stopped responding — the kernel watchdog (TDR) fired,
or a shader wrote out of bounds.

## What the log says, and what it does not

With `RUST_LOG=debug` the last records before the loss are:

```
graph: record matmul id=NodeId(586) M=256 K=3072 N=768
elementwise: dispatch_add numel=196608
graph: record add id=NodeId(589) numel=196608
matmul: MatMulOp dispatch: M=256 K=768 N=259 workgroups=(16,17,1)
graph: record matmul id=NodeId(593) M=256 K=768 N=259
graph: added input 'targets' id=NodeId(594) numel=255
graph: record loss id=NodeId(595) batch=255 vocab=259
thread 'main' panicked ... Error in Device::poll: Validation Error
```

Every op is small (50M MAC at most). Because wgpu reports the loss at the next
`poll`, **the log names the op after the hang, not the op that hung**. Two
patterns are suspects, both in this code:

1. `src/compute/kernels/prefill_attn.rs` — `@workgroup_size(16,16)` with a
   per-invocation `for k_pos in 0..=q_pos` loop. Untiled: work per invocation
   grows linearly with sequence, and nothing parallelises across keys.
2. `src/autodiff/gpu_ops.rs:585-601` — `@workgroup_size(1)` with a serial
   `for i in 0..numel` reduction, deliberately single-threaded so the summation
   order matches the CPU oracle bit for bit. One lane, `seq × hidden` elements,
   per layer.

## The test that isolates it

Instrument, do not reason. Add `TIMESTAMP_QUERY` (already available in wgpu 29)
and wrap **each recorded op** in its own query pair; then run the failing shape
and print the op whose duration exceeds 500 ms. A dispatch above ~2 s explains
the TDR; a dispatch that is fast but followed by a fault points at an
out-of-bounds write and needs validation layers instead.

```
cargo test -p ferrisres --test gpu_op_timings -- --ignored --nocapture
```

Name it `no_single_dispatch_exceeds_the_watchdog_at_768x256`. It must run the
failing shape with per-op timestamps and fail with the op's name and duration if
any single dispatch exceeds 500 ms.

## Then, in order

1. If a dispatch is long: **tile it**. For attention that is one workgroup per
   query row with the key loop distributed across lanes and a workgroup
   reduction — the same shape as `flash_decode` already in the tree.
2. The `@workgroup_size(1)` reduction has an independent reason to change: the
   bit-exact ordering can be preserved without one lane by fixing the reduction
   tree (pairwise in index order), which costs an afternoon and buys back a whole
   EU per layer.
3. Only then revisit the 0.8%-of-peak efficiency.

## Out of scope

The T4 and the CUDA question; the learner; the corpus.
