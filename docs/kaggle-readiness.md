# Kaggle readiness

Status: **checklist, 2026-11-26. GPU only.** A CPU run on Kaggle is **not an
acceptable milestone**: it burns quota without using the T4 and proves nothing
about the target. The smoke must run `--device gpu` and show the T4.

## The gate stack

| # | item | state | owner |
|---|---|---|---|
| 1 | **GPU training loop** — wgpu forward, backward, Adam, `train --device gpu`, FRTRAIN1 read-back | **critical path**, branch not pushed (`feat/gpu-train-loop`) | agent |
| 2 | parity and learning gates: logits/gradients vs f32, loss falls, parameters change | the agent's Phase B/C | agent |
| 3 | **Vulkan on the Kaggle T4** — `ferrisres devices` must list the T4, not llvmpipe | **unverified**; the `.deb` ships an NVIDIA ICD, the image must expose the driver | first session |
| 4 | `.deb` with the GPU loop, built on ubuntu:22.04 with `--features vulkan` | **0.2.4 exists** (`int/kaggle-deb` `efdde0b`, 5.7 MiB, sha `a87a81a9…`) with train + export + devices, but **no GPU loop**; it is the integration base, not the Kaggle artefact | agent |
| 5 | GPU smoke notebook: `nvidia-smi`, `devices` shows T4, `train --device gpu` 20 steps, loss falls, tokens/s recorded | updated, not runnable until 1 | — |
| 6 | checkpoint read-back, export, infer from the GPU-trained checkpoint | after 1 | agent |
| 7 | resume from a GPU checkpoint across a 12-hour session | after 1 | agent |
| 8 | upload the `.deb` as a private Kaggle dataset | operator | you |
| 9 | Kaggle account with GPU T4 + internet | operator | you |

**If item 3 fails, the plan fails.** wgpu needs a Vulkan device; if the Kaggle
image cannot expose one to the container, GPU training there is not possible and
we say so rather than falling back to CPU. The first session is therefore a
**probe**, not a training run: `nvidia-smi -L`, `ferrisres devices --json`, and the
Vulkan loader/ICD paths. Only after the T4 appears does a training step run.

## The first GPU session, in order

1. `nvidia-smi -L` — a T4 must be listed.
2. `ferrisres devices --json` — an adapter with `software: false` and a T4 name.
   If it is llvmpipe, stop and report the missing ICD/driver; do not train.
3. A 20-step run at a tiny config with `--device gpu`:
   - logits/gradient parity was proven in the branch, not in the session;
   - the session proves the device path, the data path, checkpoint and export;
   - record tokens/s and the adapter name.
4. Only then a real corpus run, with resume.

## A real training run

Still needed after the GPU smoke:

| item | state |
|---|---|
| corpus builder: source rows → licensed, deduped, tokenised shards | spec only (`corpus-builder.md`) |
| a licensed corpus (small is fine) | none fetched |
| the 32k tokenizer, or a recorded decision to use the byte tokenizer | spec only |
| resume at corpus scale with the data offset | tiny tests only |
| checkpoint save/restore across 12-hour sessions via a Kaggle dataset | missing |
| GGUF/ternary export for serving | open |
| multi-GPU local SGD on two T4s | coordination layer only |

## What is *not* blocking

- `compute_lora_grad` wiring: affects `distill`, not `train`.
- prefix KV cache and parallel prefill: inference.
- the harness recall branch: not training.
- PLE: a separate competition.

## Risks

| risk | check |
|---|---|
| wgpu sees no T4 | `devices`; NVIDIA driver ICD and `libvulkan1` in the image |
| wgpu runs but is slower than CPU at this size | measure; a 111M f32 model on a T4 should still win, but the kernel path matters |
| 12-hour session | checkpoint to `/kaggle/working` and a dataset round-trip |
| quota | a GPU hour is one quota hour; the smoke is minutes, a real run hours |
| export from GPU weights | read back to FRTRAIN1 before export; verify tensor equality |

## Order

1. **Land `feat/gpu-train-loop`** with its parity and learning gates. Nothing
   else on this list can start.
2. Build the GPU `.deb`; probe the T4 with `devices`.
3. GPU smoke (20 steps) only if the T4 is visible.
4. Corpus + tokenizer; then a real run; then multi-GPU.
