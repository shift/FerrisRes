# Multi-GPU

Status: **coordination layer at `68e1cc0`; CPU local SGD wired at `11fb681`;
the wgpu training loop and the real two-T4 gate are open.** Phase 1 (`4d6141e`) made two cards reachable;
this adds the launcher, deterministic sharding, init identity, `/dev/shm`
averaging, the timeout barrier, per-rank evidence and the resume rule. None of
it has run on two physical GPUs, and the training loop does not call it yet.

## Topology

Kaggle gives **2× T4, 16 GB each**, on one host, over PCIe, with no NVLink;
4 CPU cores, 29 GB RAM. One hour of quota covers both cards, so using both
doubles throughput per quota hour.

## The shape

**Process per GPU.** One process with `--gpu-index 0`, one with
`--gpu-index 1`. No device is shared across processes.

| workload | strategy | collectives |
|---|---|---|
| **inference / serve** | N replicas, one per GPU, behind the daemon's router | none |
| **teacher forward** (data generation) | shard the corpus by rank; each worker writes its targets | none |
| **student training** | shard the data by rank; average weights every k steps, or merge once at the end | `/dev/shm` for same-host, periodic (local SGD) |
| tensor / pipeline parallel | **not now** — a 0.6B fits one 16 GB card; the transfers would dominate | — |

A 0.6B teacher + student fits one card with room for the optimizer, so neither
model needs to be split. The second card exists to process a second shard.

## What landed at `68e1cc0`

| piece | behaviour | evidence |
|---|---|---|
| `ShardPlan` | step `k` owns one id from `k*W..k*W+W-1`; the seed rotates the slot per step | union covers every batch once, ranks disjoint, reproducible, resume at step 37 equals the uninterrupted tail |
| init identity | `(seed, FNV-1a parameter hash)`; a mismatch names the rank and field | same seed/hash, changed seed, changed weights, zero world |
| weight exchange | 40-byte header (`FRSWGT01`), temp + atomic rename, rank-0 average, every rank verifies count and checksum | 4-rank mean within 1e-6, missing rank times out naming it, corrupt checksum refused |
| barrier | poll every 2 ms against a deadline (default 60 s) | `BarrierTimeout { step, missing ranks }`; cannot hang |
| launcher | `ferrisres workers --world-size N -- <cmd>`; refuses before spawn when adapters < N | world 3 with 2 adapters refused; two fake adapters spawn two children with the env/flags; one failure kills the group in <5 s |
| resume rule | `RunParallelism { world_size, rank }` + `check_resume` | a changed world size is refused, naming both |

Verification: `cargo test -p ferrisres multi_gpu` → 32 passed, 0 failed, 1
ignored; `cargo test -p ferrisres --lib` → 1667 passed, 0 failed, 2 ignored. The
ignored host test was not run (no Vulkan ICD in that shell).

**Still open:** a real world-size-2 run on two T4s, the per-rank tokens/s from
hardware, `--rank`/`--world-size` parsing in `train` and the averaging call in
the training loop, and a PCI bus id in the phase-1 adapter listing — two T4s
share the name `NVIDIA T4`, so "two distinct names" is only checkable once the
bus id is in the evidence.

## Requirements

1. **Adapter selection.** `ferrisres devices` lists adapters with index, name,
   kind, backend and buffer limits. `--gpu-index N` (and
   `WGPU_ADAPTER_INDEX`) selects one; software adapters are filtered unless
   `--allow-software`.
2. **A launcher.** Enumerate, then spawn one worker per adapter with
   `--gpu-index i --rank i --world-size N`; fail loudly if fewer adapters than
   the requested world size. "Automatic" means the launcher discovers, not
   assumes.
3. **Deterministic sharding** by `(rank, global step)`, so a resumed run feeds
   the same data. The checkpoint records `world_size` and `rank`; resuming with
   a different world size is refused (or scales the LR, explicitly).
4. **Init identity.** Same seed plus a hash check, or rank 0 broadcasts. Two
   ranks must start from identical weights.
5. **A barrier with a timeout.** If one rank dies, the other aborts with a named
   error instead of hanging. No watchdog-free loops.
6. **Per-rank evidence.** Every run logs the adapter name and tok/s per rank;
   a run that claims two GPUs must show two distinct names. The gate is a short
   two-GPU run that writes a checkpoint which loads, with the loss reported
   next to a one-GPU run on the same slice.

## What not to do

- **Per-step TCP all-reduce.** Gradients for 111M at fp32 are ~444 MB; loopback
  is 0.1–0.4 s per exchange against a 0.27 s step. Use `/dev/shm` and average
  every k steps.
- **Tensor/pipeline parallelism at 0.6B.** The model fits; the transfers cost
  more than the parallelism saves.
- **NCCL or CUDA IPC.** wgpu exposes neither; do not design around them.
- **Shared-memory KV or weights between processes.** Each process owns its
  device and its buffers; sharing weights means copying.

## Kaggle

The `.deb` installs one binary. Launch two workers from the notebook:

```
ferrisres devices                                   # must list two T4s
ferrisres serve --gpu-index 0 ... &                 # inference replicas
ferrisres serve --gpu-index 1 ... &
```

For training, the launcher starts both ranks with the same corpus dataset and
different `--rank`; checkpoints go to `/kaggle/working` and are uploaded every
~20 minutes as before. The quota cost is one hour per clock hour, not two.

## Phases

1. `devices` + `--gpu-index` (blocking everything else).
2. Inference replicas behind the daemon router.
3. Sharded teacher forward (the corpus bottleneck).
4. Local-SGD training with `/dev/shm` averaging and the safety rules.
5. The two-GPU gate and per-rank evidence.
