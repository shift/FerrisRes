# Kaggle GPU .deb and probe

Status: **0.2.5 built and gated at `int/gpu-deb` `21aed9c`; the T4 probe is
the remaining step.** Base: `int/kaggle-deb` + `feat/gpu-train-loop`.

## Behaviour (one)

A `.deb` whose `train --device gpu` runs on the T4, plus a notebook cell that
**stops if wgpu sees no hardware T4** (no CPU fallback). The package also carries
`export` and `devices`.

## Gate (not a cargo test)

1. Build on ubuntu:22.04 with `--features vulkan`, shared build lock.
2. Clean container: install, `--version`, `devices`, `train --help`,
   `export --help`.
3. On the first Kaggle GPU session: `nvidia-smi -L`, `devices --json` shows a T4
   with `software: false`, then 20 steps with `--device gpu` and a falling loss.

If `devices` shows llvmpipe, **stop** and report the Vulkan loader/ICD state.

## Out of scope

Multi-GPU, a real corpus, GGUF.

## Rules

Do not claim GPU training from a CPU run. Push the branch; no merge.
