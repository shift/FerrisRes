# CUDA backend decision

Status: **open decision, 2026-11-26. No code, no branch.** Kaggle's T4 cannot run
the wgpu/Vulkan path (measured, `docs/kaggle-readiness.md`). The GPU host runbook
tests whether a rented host can (`docs/gpu-host-runbook.md`). This task exists
for the case where that test also fails.

## The one question

**Do we pay for a second compute path to reach GPUs that mount the compute
driver only, or do we require a graphics-capable host?**

Everything below serves that sentence. If the answer is "require the host", this
task closes without code.

## Why this is a decision and not a work item

The engine is a wgpu engine. The kernels are WGSL, the backward pass is WGSL,
and the `.deb` declares one graphics dependency. A CUDA backend is not a port
of the build system; it is a second implementation of the compute core, with
its own packaging and its own parity evidence.

## Facts in the tree (measured 2026-11-26)

| fact | value | how |
|---|---|---|
| compute entry points | **78** `@compute` functions | `grep -rho "@compute" src \| wc -l` |
| files that hold them | **30** | `grep -rl "@compute" src \| wc -l` |
| backward entry points | **8** in `src/autodiff/backward.rs` | grep |
| kernel scope | matmul, coop matmul, prefill attention, flash decode, paged attention, MoE, ternary matmul, sparse ternary, RMSNorm, RoPE, softmax, causal mask, elementwise, optimizer, inter-block, TurboQuant, FFT, conv3d, im2col, patch embed, ToMe merge | `ls src/compute/kernels` |
| adapter order | Vulkan, then Metal, then DX12, then GL, then software | `src/compute/adapter.rs` |
| `.deb` dependency | `Depends: libvulkan1 (>= 1.3)` only | `packaging/build-deb.sh` |
| CUDA in the tree | none; no CUDA crate, no `.cu` file | `grep -rn cuda Cargo.toml`; find |

## The options

| option | cost | time | main risk | status |
|---|---|---|---|---|
| **A. Require a graphics-capable host** | no code; choose the provider by section 1 of the runbook | hours | fewer providers and images qualify; some free tiers are excluded | **recommended first** |
| **B. Add a CUDA backend** | 78 entry points, the backward pass, the training loop, packaging, a second parity suite | **calculated**: weeks; never measured | two compute paths that drift | open decision |
| **C. CPU only** | none | none | no GPU training on any host; the target tier is unreachable | rejected for the stated goal |

A does not answer "can we use a free T4". It answers "can we use a GPU". B is
the only option that reaches a host built for CUDA only, such as Kaggle.

## Cost of option B

Scope figures are **calculated**, not measured. No CUDA code exists, so there is
no measurement to quote. Rows that are simple facts are marked as such.

| work | scope | label |
|---|---|---|
| kernels | 78 entry points across 30 files; every WGSL kernel needs a CUDA equivalent at the same numerics | calculated |
| autodiff | 8 backward entry points plus the host-side orchestration that issues them | calculated |
| training loop | forward, loss, backward, Adam, gradient accumulation, checkpoint read-back | calculated |
| parity gates | the CUDA path must meet the values the wgpu path already meets: logits max abs **1.892e-4**, gradients ~**1e-4**, post-step **5e-5**, loss **3.557 → 0.0026** (measured on the wgpu loop) | measured reference, new work to meet it |
| packaging | the `.deb` either depends on a CUDA runtime or loads it at run time; the second breaks the CPU-only install, the first grows the package and adds a driver matrix | calculated |
| test surface | every GPU test needs a CUDA host; no CUDA host is on record (the one discrete-GPU run was a Colab session). Unrun tests are not evidence | fact |
| CI | no CUDA device in CI, so the second path is checked only when someone rents a card | calculated |

## Risks

1. **A second compute path.** Every kernel fix lands twice. The defects already
   found were shader-specific: a 32×32 workgroup count against a 16×16 shader,
   and a gradient shader that wrote only column 0 (measured, commit `73a6902`).
   A CUDA port repeats that class of bug with a different memory model, and a
   green wgpu suite says nothing about it.
2. **No tensor cores today.** The kernels are scalar f32 and ternary with no
   TF32 or MMA path. A direct CUDA translation would also ignore tensor cores,
   so it is not automatically faster than the CPU at hidden 64. The T4 answer
   needs the probe's tokens/s as a baseline before B is argued on speed.
3. **The `.deb` gains a CUDA dependency.** `libcuda` belongs to the driver;
   `libcudart` and cuBLAS belong to a toolkit. Bundling them adds size and a
   licence question; requiring them breaks the install on a CPU-only machine;
   loading them dynamically adds a fallback branch that no test covers.
4. **The support surface doubles.** Two backends times three modes (inference,
   training, export), and two failure modes when a number is wrong. The evidence
   register in `docs/hardware-matrix.md` would need a column per backend.
5. **Effort is unbounded until the first parity run.** The 78-kernel count is
   measured; the time to port them is not. Do not put B on a release plan before
   one kernel set passes parity.

## What answers it

A recorded host result from `docs/gpu-host-runbook.md`, with `verdict` set to
`no-graphics` or `no-adapter`.

Start option B only when both hold:

- two different providers, each documented as graphics-enabled, fail section 1
  of the runbook; **or** a hard requirement names a compute-only host (a free
  tier) as the only acceptable target;
- the probe baseline exists, so the second path has a number to beat.

Until then, the correct action is to rent a graphics-capable host and run the
runbook. "Kaggle cannot do it" alone is evidence for one host, not for every
host.

## Out of scope

Kernel port, a CUDA crate, a CUDA feature flag, a CUDA image, CUDA in CI, a
change to `packaging/build-deb.sh`.

## Rules

1. No code in this task. A documentation question is not answered by adding a
   dependency.
2. Do not present a CUDA estimate as a measurement. Every figure above is
   labelled calculated or unknown.
3. B is decided once, in one place. It does not start as a side branch of a
   different task.
