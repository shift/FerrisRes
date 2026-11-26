# Kaggle readiness

Status: **checklist, 2026-11-26. GPU only.** A CPU run on Kaggle is **not an
acceptable milestone**: it burns quota without using the T4 and proves nothing
about the target. The smoke must run `--device gpu` and show the T4.

## The gate stack

| # | item | state | owner |
|---|---|---|---|
| 1 | **GPU training loop** — wgpu forward, backward, Adam, `train --device gpu`, FRTRAIN1 read-back | **landed** at `73a6902`: parity, learning, CLI and resume; 1634 lib + 8 GPU tests | done |
| 2 | parity and learning gates: logits/gradients vs f32, loss falls, parameters change | **passed**: logits 1.892e-4, grads ~1e-4, post-step 5e-5, loss 3.557→0.0026 | done |
| 3 | **Vulkan on the Kaggle T4** — `ferrisres devices` must list the T4, not llvmpipe | **T4 present, ICD missing.** `nvidia-smi` sees 2× T4 (driver 580.178.04); the image has `libvulkan.so.1` and **no** `/usr/share/vulkan/icd.d/`, so the loader has no driver. The notebook now writes `nvidia_icd.json` and exports `VK_DRIVER_FILES`; `devices` under it is not yet observed | next run |
| 4 | `.deb` with the GPU loop, built on ubuntu:22.04 with `--features vulkan` | **done**: 0.2.5 at `int/gpu-deb` `21aed9c`, 5.7 MiB, sha `db443ac2…`; gate PASS (version, devices, `--device`, CPU smoke, export, infer, glibc 2.34) | agent |
| 5 | GPU smoke notebook: `nvidia-smi`, `devices` shows T4, `train --device gpu` 20 steps, loss falls, tokens/s recorded | **kernel `sectionme/ferrisres-gpu-probe`** pushed; v1 confirmed the T4 and the missing ICD, then stopped | agent |
| 6 | checkpoint read-back, export, infer from the GPU-trained checkpoint | **done in the loop branch**; the packaged path is after 0.2.5 | agent |
| 7 | resume from a GPU checkpoint across a 12-hour session | **resume tested** (`total_step=42`); the HF round-trip is in the notebook | operator |
| 8 | upload the `.deb` as a private Kaggle dataset | operator | you |
| 9 | first session: notebook, GPU T4×2, internet, `HF_TOKEN` secret, Run All | operator | you |
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

## Two facts about the Kaggle image (measured, 2026-11-26)

Both cost a run to learn. Do not re-discover them.

1. **There is no Vulkan ICD manifest.** `libvulkan.so.1` is present;
   `/usr/share/vulkan/icd.d/` does not exist and there is no `nvidia_icd.json`.
   A Vulkan loader with no ICD enumerates **zero** adapters, so `devices` would
   report nothing even with the driver loaded. The fix is local and does not
   need root pathology: write

   ```json
   {"file_format_version": "1.0.0",
    "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.260"}}
   ```

   to `/kaggle/working/nvidia_icd.json` and export `VK_DRIVER_FILES` **and**
   `VK_ICD_FILENAMES` before the first wgpu call. Every notebook that trains does
   this; it is not a packaging change to the `.deb`.

2. **The first run after attaching a dataset mounts nothing.** `kernel-metadata`
   is correct (`dataset_sources: ["sectionme/ferrisres"]`, confirmed by pulling
   the kernel metadata back) and the dataset is `ready`, yet the run starts with
   an empty `/kaggle/input`; a re-push of the identical body mounts it. Relying
   on a human re-push is not a fix — the notebook discovers the file with
   `Path('/kaggle/input').rglob(...)` and falls back to
   `kagglehub.dataset_download('sectionme/ferrisres')`.

The T4 answer is not in doubt any more: 2× Tesla T4, 15,360 MiB each, driver
580.178.04. What remains is whether `devices` sees them **through wgpu**, which
is the ICD row above.

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
