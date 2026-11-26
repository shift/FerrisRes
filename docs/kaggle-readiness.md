# Kaggle readiness

Status: **checklist, 2026-11-26.** Two different goals: a **pipeline smoke** on
Kaggle (days) and a **real training run** (weeks). They need different things.

## A. The smallest "try it" — CPU smoke

| # | item | state | owner |
|---|---|---|---|
| 1 | `.deb` **0.2.4** with `train` + `export` + `devices` | building (`int/kaggle-deb`) | agent |
| 2 | clean-container gate: install → train 20 steps → export → infer | in the same agent | agent |
| 3 | upload the `.deb` as a private Kaggle dataset at `/kaggle/input/ferrisres-deb/` | **operator**; the token is yours and the `.deb` is not in git | you |
| 4 | `train_smoke.ipynb` attached to that dataset | done, `dc30aa0` | — |
| 5 | Kaggle account with **GPU T4** enabled and internet on | operator | you |
| 6 | run: install → train → resume → export → infer; record loss and tokens | operator | you |

What this proves: the packaged binary runs on Kaggle, the data path, the CPU f32
trainer, checkpoint/resume, export and infer. **It does not use the T4 for
training** — the trainer is CPU until the wgpu loop lands.

Expected: minutes of GPU quota for the smoke (it is CPU-bound), a small
checkpoint, and a clean pass/fail.

## B. A real training run

| # | item | state |
|---|---|---|
| 1 | corpus builder: source rows → licensed, deduped, tokenised shards | spec only (`corpus-builder.md`) |
| 2 | a licensed corpus (even a small one, e.g. Rust/Nix docs) | none fetched |
| 3 | the 32k tokenizer, or an explicit decision to use the byte tokenizer | spec only (`tokenizer.md`) |
| 4 | **GPU training loop** (`--device gpu`) | branch not pushed (`feat/gpu-train-loop`) |
| 5 | resume at corpus scale, with the data offset | only tiny tests |
| 6 | checkpoint save/restore across 12-hour sessions via a Kaggle dataset | notebook lacks the dataset round-trip |
| 7 | GGUF/ternary export (for serving the result) | open |
| 8 | multi-GPU local SGD on two T4s | coordination layer only |

Until items 1–4 exist, a "real" run is the CPU loop on a tiny corpus: useful as a
pipeline test, not as a model.

## C. What is *not* blocking, despite being open

- `compute_lora_grad` wiring: affects `distill`, not `train`.
- prefix KV cache and parallel prefill: inference speed.
- recall/notebook branch: harness, not training.
- PLE competition: a separate track.

## D. Risks to check on the first Kaggle session

| risk | check |
|---|---|
| the `.deb` glibc floor | install in the Kaggle image; 2.34 ≤ 2.35 |
| the GPU is visible to wgpu | `ferrisres devices` on the T4; Vulkan ICD is shipped in the `.deb`, but NVIDIA's driver must expose it |
| CPU training is slow | the smoke's tokens/s is the number; do not extrapolate from a laptop |
| 12-hour session | resume from the checkpoint; nothing in `/kaggle/working` survives without an output/dataset |
| quota | 30 h/week; a CPU run can consume it without using the T4 |

## The order

1. Wait for the 0.2.4 agent; read its gate output.
2. Upload the `.deb`; run the smoke.
3. In parallel, start the corpus builder and the tokenizer; those are the gates
   for a real run.
4. When `feat/gpu-train-loop` lands, rebuild the `.deb`, run the same smoke with
   `--device gpu`, and compare tokens/s with the CPU run.
5. Then the first real corpus run, then multi-GPU.
