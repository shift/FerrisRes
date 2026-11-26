# Kaggle readiness

Status: **checklist, 2026-11-26. GPU only.** A CPU run on Kaggle is **not an
acceptable milestone**: it burns quota without using the T4 and proves nothing
about the target. The smoke must run `--device gpu` and show the T4.

## The gate stack

| # | item | state | owner |
|---|---|---|---|
| 1 | **GPU training loop** — wgpu forward, backward, Adam, `train --device gpu`, FRTRAIN1 read-back | **landed** at `73a6902`: parity, learning, CLI and resume; 1634 lib + 8 GPU tests | done |
| 2 | parity and learning gates: logits/gradients vs f32, loss falls, parameters change | **passed**: logits 1.892e-4, grads ~1e-4, post-step 5e-5, loss 3.557→0.0026 | done |
| 3 | **Vulkan on the Kaggle T4** — `ferrisres devices` must list the T4, not llvmpipe | **PASS on run 4**: two `Tesla T4`, `vulkan`, `Discrete`, `software: false`, after installing `libnvidia-gl-580-server` with `--force-overwrite` in-session. Install cost ~10 s per session | done |
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

1. **There is no Vulkan ICD manifest *and* no NVIDIA graphics userspace — both
   are installable in-session, and this is what unblocks the T4.** `libvulkan.so.1` is present;
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

   **The manifest alone is not enough.** The image runs with
   `NVIDIA_DRIVER_CAPABILITIES=compute,utility`, so the kernel driver and
   `nvidia-smi` are mounted but `/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so*`
   and `/usr/share/vulkan/icd.d/` are absent. A manifest pointing at a missing
   library is a loader with no device, which is why wgpu fell back to software
   GL.

   **Install the graphics userspace in the session.** Ubuntu publishes the exact
   matching build (`580.178.04-0ubuntu0.24.04.1` for driver `580.178.04`), so
   it works:

   ```bash
   apt-get update -qq
   DEBIAN_FRONTEND=noninteractive apt-get install -y \
     -o Dpkg::Options::="--force-overwrite" \
     -o Dpkg::Options::="--force-confold" \
     libnvidia-gl-580-server
   ```

   `--force-overwrite` is required: the package ships
   `/usr/share/egl/egl_external_platform.d/15_nvidia_gbm.json`, which
   `libnvidia-egl-gbm1` already owns, and without the flag dpkg aborts and
   leaves nothing behind. Two details that cost runs:

   - `libnvidia-gl-580` (without `-server`) installs and exits 0 but ships **no**
     `libGLX_nvidia.so` in this image; use `-server`.
   - verify with `ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0` and
     `/usr/share/vulkan/icd.d/nvidia_icd.json` before believing apt.

   After the install, run 4 measured: `devices --json` reports

   ```json
   [{"index":0,"name":"Tesla T4","kind":"Discrete","backend":"vulkan","software":false},
    {"index":1,"name":"Tesla T4","kind":"Discrete","backend":"vulkan","software":false},
    {"index":2,"name":"Tesla T4/PCIe/SSE2","kind":"Other","backend":"gl","software":false}]
   ```

   Index 0 and 1 are Vulkan; select one with `WGPU_ADAPTER_INDEX`. The install is
   per session — the container is fresh each run — so it belongs in every
   training notebook and costs about ten seconds.

2. **The mount blue-pages twice before it works, and the path is nested.**
   The first run after attaching a dataset starts with an empty `/kaggle/input`
   (see below), and when it does mount, the path is
   `/kaggle/input/datasets/<owner>/<slug>/`, not `/kaggle/input/<slug>/`. An
   `rglob` from `/kaggle/input` is the only discovery that survives both.

3. **The first run after attaching a dataset mounts nothing.** `kernel-metadata`
   is correct (`dataset_sources: ["sectionme/ferrisres"]`, confirmed by pulling
   the kernel metadata back) and the dataset is `ready`, yet the run starts with
   an empty `/kaggle/input`; a re-push of the identical body mounts it. Relying
   on a human re-push is not a fix — the notebook discovers the file with
   `Path('/kaggle/input').rglob(...)` and falls back to
   `kagglehub.dataset_download('sectionme/ferrisres')`.

### First T4 measurement (run 7, 2026-11-26)

The engine ran on a T4 for the first time. It is reachable and it executes, but
it does **not** learn:

```
selected adapter 0 Tesla T4
step 144/144 loss=8.258290
Training complete: steps=144 initial_loss=5.763604 final_loss=8.258290
```

- **144 steps executed** on adapter 0 (`Tesla T4`, vulkan, Discrete);
- **loss rose** 5.764 → 8.258;
- **the forward pass is correct.** Run 9 ran the same config on the same host on
  the CPU: step-1 loss `5.763615` against the GPU's `5.763604`, a delta of
  **1.1e-5**, and the CPU then fell to `0.396426` in 48 steps. Corpus, seed,
  learning rate and forward pass are all sound;
- **the CPU is 2.8× faster than the T4** on this config: 907 tok/s against 321.
  Whatever is wrong in the backward pass also made the GPU the slower device;
- **~1.6 s/step** derived from the log timestamps (~228 s for 144 steps of
  4×128 tokens, ≈320 tok/s). The HD 530 did the same config at ~0.16 s/step, so
  the T4 is **~10× slower per step** at this size: dispatch-bound, not
  bandwidth-bound, and the cooperative-matrix path the driver advertises
  ("Found 6 cooperative matrix configurations") is a prime suspect;
- the kernel is ERROR only because the probe asserts `final_loss <
  initial_loss`. That assertion is doing its job.

The remaining suspect is the **backward pass or the optimiser update**, and the
fingerprint is a race rather than a formula: forward identical, loss climbing
monotonically. This engine has already had two gradient-buffer defects (a 16×16
shader driven by 32×32 workgroups; grad shaders writing only column 0 while
overwriting input grads), both fixed and verified on a 24-EU iGPU. A T4 has 40
SMs, so a missing barrier or a non-atomic add in gradient accumulation is
invisible there and double-counts on the larger adapter.

Until that is fixed, "the GPU works" means *enumerates, executes and computes a
correct forward pass*, not *trains*. No tokens/s figure may be quoted from this run as a
performance expectation: the workload is a 64-hidden toy, and it is slower than
the iGPU.

The T4 is reachable. Run 4 measured `Tesla T4`, backend `vulkan`, kind
`Discrete`, `software: false`, twice — index 0 and index 1 — after the in-session
install above. What remains unmeasured is **speed**: no FerrisRes step has run on
the T4 yet, so there is no tokens/s figure for any tier on this hardware. That
number is the next run, and it is the only thing standing between this row and a
real training campaign.

A second lesson from the same run, in the notebook rather than the host:
`devices --json` writes tracing log lines to **stdout** before the JSON array, so
`json.loads(stdout)` fails with `Expecting value: line 1 column 1`. Parse from the
first `[` (`json.JSONDecoder().raw_decode(stdout, stdout.find('['))`). It cost run
4 its training stages after the GPU had already been found.

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
