# GPU host runbook

Status: **runbook, 2026-11-26. Prepared, not executed on a rented GPU.** The
Kaggle T4 is closed (`docs/kaggle-readiness.md`). This procedure tests one
rented host that is expected to mount the NVIDIA graphics stack, and records
the answer in under an hour.

The purpose is one question: **does the engine enumerate a hardware adapter on
this host, and can it train on it?** A "no" is an answer. Do not fall back to
CPU and call it a probe.

## 1. What the host must provide

Three conditions. The brand is not the test.

| # | condition | why it is required | state on Kaggle |
|---|---|---|---|
| 1 | `NVIDIA_DRIVER_CAPABILITIES` contains `graphics`, or is `all` | the container runtime mounts the NVIDIA graphics userspace only when it is requested | **measured**: `compute,utility` |
| 2 | `libGLX_nvidia.so.<major>` exists, where `<major>` is the driver major | this is the Vulkan ICD library; without it a manifest is a loader with no device | **measured**: absent |
| 3 | `/usr/share/vulkan/icd.d/nvidia_icd.json` exists | the Vulkan loader reads manifests from this directory | **measured**: directory absent |

An unset `NVIDIA_DRIVER_CAPABILITIES` is not a pass. The container default is
`compute,utility`, which is the Kaggle configuration.

### The three checks

Run each command on the GPU host, inside the container or job that will run the
engine. Run all three **before** the `.deb` install: the package adds its own
ICD file, so after install Check 3 can pass for the wrong reason.

**Check 1 — capability variable**

```
printenv NVIDIA_DRIVER_CAPABILITIES
```

Pass: `all`, or a comma-separated list that contains `graphics`, for example
`compute,utility,graphics`.

Fail: `compute,utility`, an empty line, or exit status 1.

**Check 2 — graphics library**

```
ls -1 /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.*
```

Expected output, for a host with driver 580.178.04:

```
/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.580
/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.580.178.04
```

Pass: at least one line that ends in `.so.<major>`, and the same library is
listed by `ldconfig -p | grep libGLX_nvidia`.

Fail: `No such file or directory`. A different directory is acceptable only if
the loader search path finds it; Check 2 is about the library, not the path.

`<major>` is the driver major. Read it with
`nvidia-smi --query-gpu=driver_version --format=csv,noheader`; 580.178.04 gives
580.

**Check 3 — ICD manifest**

```
cat /usr/share/vulkan/icd.d/nvidia_icd.json
```

Expected content, in substance:

```json
{"file_format_version": "1.0.0",
 "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.260"}}
```

Pass: valid JSON, `library_path` names `libGLX_nvidia.so.0` or
`libGLX_nvidia.so.<major>`, and that file exists (Check 2).

Fail: `No such file or directory`. Do not reject the host on the `api_version`
patch number; any 1.3.x is usable.

The 0.2.5 `.deb` ships its own
`/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json`, which names
`libGLX_nvidia.so.0` (measured from the package contents, 2026-11-26). That file
is **not evidence**. It installs on a host with no NVIDIA graphics library as
well; on Kaggle the package was installed, its ICD file was present, and
`devices` still reported llvmpipe because the named library was absent
(measured, run 2).

The three conditions are properties of the host image. They are not repaired
from inside the job. If any check fails, go to section 4.

## 2. The probe

The probe is minutes of GPU time. One GPU is enough.

### Step 1 — verify and install the artefact

```
sha256sum ferrisres_0.2.5_amd64.deb
stat -c '%s' ferrisres_0.2.5_amd64.deb
```

Expected from the two commands, for the `int/gpu-deb` build at `21aed9c`
(measured 2026-11-26 on the build output):

```
db443ac27a515e366cb86cf9e338455cb8882c2c6fa4f766fbfc94e4027b1eec
5954834
```

A different hash or size stops the probe. Upload the file you hashed; do not
rebuild it on the rented host.

```
sudo dpkg -i ferrisres_0.2.5_amd64.deb
ferrisres --version
```

Expected: `ferrisres 0.2.5` (measured in the packaging gate).

### Step 2 — repair a missing manifest, if Check 3 failed

Do this only when Check 1 and Check 2 pass and Check 3 fails. Write the
manifest in a working directory and export both variables:

```
cat > /tmp/nvidia_icd.json <<'JSON'
{"file_format_version": "1.0.0",
 "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.260"}}
JSON
export VK_DRIVER_FILES=/tmp/nvidia_icd.json
export VK_ICD_FILENAMES=/tmp/nvidia_icd.json
```

This repairs a missing manifest. It cannot repair a missing library. On Kaggle,
the manifest was written and `devices` still reported
`llvmpipe (LLVM 20.1.2)`, backend `gl`, `software: true` (measured, run 2).

### Step 3 — enumerate adapters

```
ferrisres devices --json
```

The command prints a JSON array, sorted by name (`src/compute/pipeline.rs` at
`origin/feat/gpu-selection`; read, not run on a GPU here). One entry has these
fields:

| field | expected on a T4 | label |
|---|---|---|
| `index` | an integer, 0 for the first card | measured from the listing |
| `name` | `Tesla T4` | expected; the exact string is **unknown** until the run |
| `kind` | `Discrete` | expected |
| `backend` | `vulkan` | expected |
| `software` | `false` | required |
| `max_buffer`, `max_immediate` | integer bytes and integer | **unknown** until the run |
| `dedicated_vram` | bytes, or `null` when sysfs has no `mem_info_vram_total` | **unknown** until the run |
| `shared_memory` | `null` for a discrete adapter | expected |

A listing that contains only `llvmpipe`, `software: true` or backend `gl` is a
failure. Stop; the card is not reachable by this engine.

Take the index from the listing. Selection skips software adapters before it
counts, so the card's position in `devices` and its `WGPU_ADAPTER_INDEX` can
differ on a host that also lists llvmpipe (source read,
`src/compute/pipeline.rs`). Select the card and check the result:

```
export WGPU_ADAPTER_INDEX=<index>
WGPU_ADAPTER_INDEX=<index> ferrisres info
```

`ferrisres info` must name the card. A refusal names the usable adapters and
their count; use the position among those. A software adapter is skipped unless
`FERRISRES_ALLOW_SOFTWARE=1` is set, so do not set it.

### Step 4 — train about 58 steps on the GPU

Copy the shape of `notebooks/kaggle/gpu_probe/gpu_probe.ipynb`, cells 3 and 4.
The kernel already refuses to fall back: it asserts `nvidia-smi` sees the card,
asserts a non-software adapter exists, and asserts that the final loss is below
the initial loss. It is written for Kaggle. On another card, replace the
`'T4' in name` assertion with `software: false` and record the actual name.

The notebook writes a small corpus and then runs:

```
ferrisres train --data /tmp/probe.txt --data-format text --chat-template gemma \
  --hidden-dim 64 --num-blocks 2 --block-size 4 \
  --epochs 3 --batch-size 4 --learning-rate 3e-3 \
  --seq-len 128 --vocab 259 --seed 42 \
  --checkpoint-every 1000 --output /tmp/probe.ckpt --log-every 5 \
  --device gpu
```

The notebook estimates about 58 steps. Record the `steps=` value that the
summary prints, not the estimate.

A run that reports a CPU device, or that ends without a summary line, is a
failure. Record it as such.

### Step 5 — record the result

| field | source | label |
|---|---|---|
| adapter index and name | `devices --json`, and the `WGPU_ADAPTER_INDEX` you passed | measured |
| steps | `steps=` in the training summary | measured |
| tokens/s | `steps × batch_size × seq_len ÷ wall seconds`; batch 4 and seq 128 are in the command, wall seconds from `time` | calculated from a measured wall clock |
| initial loss, final loss | `initial_loss=` and `final_loss=` in the summary | measured |

Keep the whole log. The notebook also writes `/kaggle/working/probe.json`; on
another host, write the same file next to the log.

The tokens/s figure is a process wall clock. It includes start-up, model
initialisation and checkpoint write. It is a lower bound for steady-state
throughput, and one run is not a benchmark.

Gate: the run passes only if the adapter is `software: false` **and** the final
loss is below the initial loss. A run that executes but does not learn is
recorded as failed, not as a slow success.

## 3. Cost and cleanliness

| item | rule | label |
|---|---|---|
| GPU time for the probe | minutes: install, 58 steps at hidden 64, checkpoint | calculated from the step count |
| wall clock | under one hour including image start and upload; the hour is a ceiling, not a target | plan |
| price | one GPU for minutes at the provider's hourly rate; this document states no rate | **unknown** |
| cards | one. Multi-GPU is a separate gate (`docs/tasks/multi-gpu.md`) | decision |
| after the probe | copy `probe.json` and the log off the host, then stop the instance | rule |

Do not leave a GPU idle while you read the log. Save the files, stop the
instance, then read. A stopped instance can still bill for disk; a running GPU
bills for the GPU.

Before you stop, confirm that nothing of ours remains:

```
nvidia-smi
```

No `ferrisres` process may appear. If one is present, stop it, then stop the
instance.

## 4. If the host also lacks graphics capabilities

Stop. Do not install driver packages inside the job. On Kaggle,
`libnvidia-gl-580` installed with exit 0 and left no library, and
`libnvidia-gl-580-server` conflicted with `libnvidia-egl-gbm1` (measured, run 3).
That is a host image decision, not a job step.

Keep this evidence:

| evidence | command |
|---|---|
| capability variable | `printenv NVIDIA_DRIVER_CAPABILITIES` |
| graphics library | `ls -1 /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.*` |
| ICD directories | `ls -la /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/` |
| adapter listing | `ferrisres devices --json` |
| driver and cards | `nvidia-smi -L`; `nvidia-smi --query-gpu=driver_version --format=csv,noheader` |
| host identity | provider, image name and tag, region, date |

Record the result as a row in `docs/hardware-matrix.md`, with the command, the
date and the measured value. A host stays "unverified" until that row exists.

The only remaining path is a **CUDA backend** in place of wgpu/Vulkan. That is a
separate decision with its own cost; see `docs/tasks/cuda-backend-decision.md`.
Do not start it in this session.

One retry is allowed: the same checks on an image that the provider documents as
graphics-enabled. That is a different host, not a repair of this one.

## 5. What is explicitly not claimed

| claim | state |
|---|---|
| a tokens/s figure for this engine on any GPU at this size | **does not exist.** Unknown. The probe number is for hidden 64, 2 blocks, seq 128, batch 4; it does not predict Nano 46M or any tier |
| GPU throughput on the HD 530 | measured: 8.1× decode FFN, ~7.7× end-to-end decode, 2.6× prefill. These are **speedups against earlier code**, not tokens/s, and they are inference on integrated graphics |
| tokens/s from the Colab T4 distill run | **unknown**; the run exists and no throughput was recorded |
| a passing probe means a real training run will fit | **not claimed**. The probe tests the device path, the data path and the checkpoint write; it is not a corpus run |
| a T4 will beat this CPU at the probe size | **unknown** until both are measured on the same command |
| exact T4 values in `devices --json` | **unknown** until the run (name string, buffer limits, VRAM from sysfs) |

No number in this document is a promise. A measured number names its command
and its date; anything else is labelled.

## 6. Recording template

Write one file per host and keep it with the log:

```json
{
  "host": "<provider/image:tag>",
  "date": "YYYY-MM-DD",
  "driver_version": "<nvidia-smi>",
  "capabilities": "<printenv>",
  "graphics_library": "<ls result>",
  "icd_manifest": "<path or MISSING>",
  "adapter_index": 0,
  "adapter_name": "<devices --json>",
  "software": false,
  "steps": 58,
  "seconds": 0.0,
  "tokens_per_s": 0.0,
  "initial_loss": 0.0,
  "final_loss": 0.0,
  "verdict": "pass | no-adapter | no-graphics | ran-not-learned"
}
```

`verdict` is one of four values. `pass` requires a non-software adapter and a
falling loss. Every other value keeps the evidence above and stops the probe.
