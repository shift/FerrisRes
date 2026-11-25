# Kaggle runbook

Status: **operational plan.** No run has been executed on Kaggle yet. This is
the procedure that keeps a 12-hour session from costing a week.

## Facts

| | Kaggle |
|---|---|
| GPU | 2× T4, 16 GB each |
| quota | ~30 GPU-hours/week, one shared pool; 1 hour of quota per clock hour even with both cards |
| session | 12 hours, runs with the tab closed |
| RAM / disk | 29 GB / 20 GB `/kaggle/working` |
| CPU sessions | no weekly quota — use them for data preparation |
| internet | requires phone verification |

## Prerequisites

1. A **tagged release** with the `.deb` that contains the kernel fixes
   (`fix/immediates-limit` and later). The CI artifact expires in 30 days; a
   release asset does not.
2. The corpus as a **private Kaggle dataset** (read-only at `/kaggle/input`),
   tokenised to uint16 where possible.
3. A Kaggle API token stored in **Kaggle secrets**, used to version the
   checkpoint dataset.
4. A **resumable** training command. `distill` already has `--resume`; `train`
   does not until `tasks/train-and-export.md` is implemented.

## Session procedure

1. **Verify the GPU before anything else.**
   ```
   !nvidia-smi
   !dpkg -i /kaggle/input/ferrisres-deb/ferrisres_0.2.3_amd64.deb
   !ferrisres info
   ```
   If `info` does not show a T4, stop. Do not start a run on an unverified
   device.
2. **Smoke run**: 10 steps at `seq_len 64` on a slice of the corpus. A run that
   cannot do 10 steps will not do 10,000.
3. **Long run** with the checkpoint path on `/kaggle/working`, a small
   `--checkpoint-every`, and logs to a file:
   ```
   !ferrisres distill … --checkpoint-every 200 --resume <latest> | tee run.log
   ```
4. **Upload the checkpoint every ~20 minutes** to a private dataset:
   ```
   !kaggle datasets version -p /kaggle/working/ckpt -m "step $(date +%s)"
   ```
   `/kaggle/working` is only saved when the notebook commits; a crash at hour 11
   otherwise loses the session.
5. **Stop the session** when the run ends or the quota is close. An idle
   notebook burns quota.

## Resume

Use the same command with `--resume <checkpoint>`. The checkpoint must carry
optimizer state, step and data offset; a checkpoint that only carries weights
restarts the schedule and is not a resume.

## Quota arithmetic

| run | GPU-hours | weeks at 30 h |
|---|---:|---:|
| Small 111M × 2B tokens | 18 | 0.6 |
| Base 598M × 2B tokens | 100 | 3.3 |
| Large 1.0B × 2B tokens | 169 | 5.6 |

Add 20–30% for restarts, evaluation and failed sessions. A 12-hour session is
nine sessions for Base; assume one of them is lost.

## Failure modes

- **Session dies at hour N** → resume from the last uploaded checkpoint; the
  loss of one upload interval, not the run.
- **Quota exhausted** → the run waits for the weekly reset; do not create a
  second account.
- **OOM** → reduce `seq_len` or batch; the checkpoint is unaffected.
- **DNS/proxy hang** → the engine loads from `/kaggle/input`; no network is
  needed during training.
- **Model does not load** → the `.deb` and the checkpoint must be from the same
  release; the loader refuses a format version it does not know.

## After the run

`evaluate` and `benchmark` on a quiet machine, with the checkpoint hash, config
and commit recorded. A Kaggle loss curve is not an evaluation.
