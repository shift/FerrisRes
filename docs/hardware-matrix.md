# Hardware matrix

Status: **verified/unverified register, 2026-11-26.** A device is "supported" only
after a recorded command and a date. Everything else is designed or blocked.

## The matrix

| device | CPU decode | GPU inference | CPU training | GPU training | multi-process | status |
|---|---|---|---|---|---|---|
| **i5-6500 + HD 530** (this host) | yes | **yes**, with the Vulkan env below | yes (f32, tiny configs) | **no** — no GPU loop | coordination only; two CPU ranks not yet wired | verified for inference; training partial |
| **CPU-only (any x86)** | yes | n/a | yes (f32) | n/a | wired next | verified correctness; throughput unmeasured |
| **2× T4** (Kaggle) | yes | expected | yes | **blocked** on the GPU training loop | launcher tested with fake adapters only | **unverified** |
| **Pi 5** | designed | n/a | designed | n/a | n/a | designed, unverified |
| **phone / NPU** | designed | designed | no | no | n/a | designed, unverified |
| **Strix Halo** | designed | designed | designed | designed | planned | designed, unverified |
| **DGX Spark** | designed | designed | designed | designed | planned | designed, unverified |

## The Vulkan environment on this host

The adapter is only visible when the loader and the Mesa ICD are named:

```
export LD_LIBRARY_PATH=<vulkan-loader>/lib:<mesa>/lib
export VK_ICD_FILENAMES=<mesa>/share/vulkan/icd.d/intel_icd.x86_64.json
export WGPU_ADAPTER_INDEX=0
ferrisres devices
```

Resolve the two store paths from the flake (`nix build nixpkgs#vulkan-loader`
and the Mesa package); do not hard-code a `/nix/store` hash in a script. A shell
without these variables reports **0 usable adapters**, which is a configuration
fact, not a broken GPU.

## Evidence

| claim | command | date |
|---|---|---|
| one Intel adapter enumerated, selection by index | `ferrisres devices`, `WGPU_ADAPTER_INDEX=0 ferrisres info` | 2026-11-26 |
| software adapter filtered and named on out-of-range | `ferrisres devices`, out-of-range test | 2026-11-26 |
| f32 training learns and resumes | `cargo test -p ferrisres --test train_learns` | 2026-11-26 |
| multi-GPU coordination | 32 tests with threads/fake adapters | 2026-11-26 |
| two physical T4s | — | **not run** |

## Known blockers

| blocker | effect | owner |
|---|---|---|
| no GPU training loop | a T4 accelerates inference and teacher forward only | `feat/gpu-train-loop` |
| no Vulkan ICD in a bare shell | `devices` reports 0 adapters | operator env |
| two T4s share the name `NVIDIA T4` | evidence needs the PCI bus id | phase-1 amendment |
| multi-GPU launcher not wired to training | only fake-adapter tests | `int/train-multi-gpu` |
| `.deb` built only on ubuntu-22.04 | Kaggle install path | `int/train-deb` |

Expected performance per device is in `performance-this-host.md`; this table is
only what was measured.

## Generating this table

The evidence section is generated, not hand-written:

```
scripts/hardware_matrix.py --selftest                     # checks the validator
scripts/hardware_matrix.py --from records/ --out table.md  # reads *.json records
```

A record must name the **command**, the **date** and the **device**; a record
without them is refused and listed under "Refused records". A calculated number
is not a measurement and belongs in the prose, not in the generated table.

## Rules

1. A row changes to "verified" only with a command, a date and the measured
   number; "it should work" is not a status.
2. A calculated tok/s is never copied here. Measured or blank.
3. Every unverified row names the blocker, so the gap is visible in one place.
4. When a device is retired, the row stays with a date and a reason.
