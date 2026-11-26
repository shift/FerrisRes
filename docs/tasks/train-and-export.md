# `train` and export

Status: **learning implemented at `d81becf` on `feat/train-cli`; export open.**
`train` now calls `training::train_loop::run` on the f32 reference model:
forward → shifted cross-entropy → backward → Adam, gradient accumulation over
the batch, `FRTRAIN1` checkpoints and `--resume`. Verified by
`tests/train_learns.rs`: 300 steps on a fixed sequence, loss 3.652375 →
0.000944, `params_changed=true`; `cargo test -p ferrisres --lib` 1630 passed.
Remaining: export, the BPE/32k tokenizer, corpus-scale and 12-hour resume
behaviour, and the block+LoRA path (separate defect).

## Goal

`ferrisres train` trains a model from scratch on this machine, resumes after a
kill, exports to the inference checkpoint, and passes the local gate before any
GPU time is spent.

## CLI

```
ferrisres train \
  --tier small | --hidden-dim 768 --num-layers 8 --heads 16 --kv-heads 1 \
  --head-dim 48 --intermediate 2048 --vocab 32768 \
  --seq-len 512 --steps 2000 --batch-size 8 --learning-rate 3e-4 \
  --data corpus.jsonl --data-format jsonl --chat-template qwen \
  --require-provenance --log-every 10 --checkpoint-every 200 \
  --output runs/small --resume runs/small/step-1000.ckpt --seed 42
```

`--tier` selects a `TierSpec`; explicit dimensions override it. The data path
uses the streaming loader and the provenance gate from `dataset-formats`.

## Loop

1. Stream examples, render with the chat template, tokenise, pack into
   `seq_len` windows, skip short tails.
2. Forward the f32 model (causal Block AttnRes, `attn_res_norm`, loop core).
3. Cross-entropy over `seq_len - 1` positions, mean.
4. Backward (hand-written, finite-difference verified).
5. Adam with the same parameter layout; gradient clipping at 1.0.
6. Log loss, tokens/s, step, and the data offset every `--log-every`.
7. Checkpoint every `--checkpoint-every`.

The run record and the pinned hashes are in `reproducibility.md`; the
end-to-end procedure is `training-runbook.md`.

## Checkpoint

One directory per run:

```
runs/small/
  config.json          # F32Config + tier + tokenizer id + format version
  step-000200.ckpt     # model f32, Adam m/v, step, data offset, RNG state
  step-000200.meta.json# loss history, tokens seen, wall time
  latest -> step-000200.ckpt
```

`--resume` loads model, optimizer, step, data offset and RNG. It refuses a
checkpoint whose architecture or tokenizer differs, naming both. Resuming with a
different world size is refused in this milestone (single process).

## Export

After training (or at each checkpoint, behind `--export-every`):

1. Build a `CpuBlockAttnResModel` from the f32 weights via
   `CpuLinear::from_weight` (ternary) and `CpuRmsNorm::from_weight`.
2. Write it with `checkpoint::save_model` and declare the features/tier.
3. **Gate**: on a held-out probe set, the exported model's argmax next token
   must match the f32 model on ≥ 99% of positions. Logits will differ; the
   predictions must not.
4. Record the quantisation gap (perplexity f32 vs ternary) in the run metadata.

## Tests

| test | asserts |
|---|---|
| toy corpus, 200 steps | loss falls below a fixed threshold (the anti-flat-loss gate) |
| kill at step k, resume | the loss curve continues; the data offset matches |
| export argmax | ≥ 99% agreement with the f32 model on a probe set |
| determinism | same seed and data → identical loss for the first N steps |
| provenance | a line missing `licence` is refused with its line number |
| architecture change | resuming into a different config is refused, both named |

## Local gate before Kaggle

1. `train` on a toy corpus for 200 steps: loss falls, checkpoint written.
2. Kill and `--resume`: loss continues.
3. Export, load, generate: the toy pattern is reproduced.
4. `evaluate` and `benchmark` on the exported model, on a quiet machine.

Only then does a Kaggle run make sense.

## Non-goals

- MoE training in this milestone: dense layers first, experts after the loop is
  proven.
- Multi-GPU training: single process, one adapter.
- A GPU training path: the f32 trainer is CPU; a GPU path is a later port with
  the same tests as the oracle.

## What blocks the first smoke (2026-11-26)

Measured against `main` at `ff98a05`; the branches named are not merged.

| blocker | evidence | unblocks it |
|---|---|---|
| **`train` does not learn** | `cmd_train` allocates zeroed logits and targets, runs `CrossEntropyLoss::compute`, reads the value back, and never calls backward or `optimizer.step`; the LoRA merge callback returns `None` | wire `f32_trainer` as specified above |
| **The only real path, `distill`, is blocked in the target config** | Block AttnRes with `boundary_layers` non-empty + LoRA has a 4.89 relative gradient error; `fix/gpu-interblock-generator` turns it into a panic | the suffix-sum and Jacobian rewrite |
| **Vocabulary mismatch** | the teacher is Qwen3 at 151k; Nano/Small are specified at 32k; distillation maps logits into the student vocab | a 151k student for the smoke, or a projection layer, or SFT instead of distillation for the small tiers |
| **No data pipeline** | `train` reads one text file with a 259-token byte tokenizer; there is no tokenise/shard command, no packing, no data offset; every source-index row is still `planned` | a small licensed slice + the loader in `dataset-formats` |
| **No resume or export for `train`** | `distill` has `--resume`, `--checkpoint-every`, `--converge`; `train` has none, and there is no export command | the checkpoint section above |
| **The Kaggle notebook stops on the guard** | `distill_smoke.ipynb` reaches the blocked config | a non-blocked config, or the gradient fix |
| **Branch split** | Qwen teacher configs are on `feat/teacher-benchmark`; the guard is on `fix/gpu-interblock-generator`; GPU selection on `feat/gpu-selection` | one integration branch |

**Shortest route to a Kaggle smoke without the gradient rewrite:** use
`distill` (it has resume and checkpoints) with a config that has **no boundary
layers** — the guard's condition is blocks *and* LoRA — and a student whose vocab
matches the teacher, on a small licensed slice at 128–512 tokens. The assertion
is that loss falls and a held-out sample improves. That proves the pipeline and
the Kaggle path; it is not the tiered model, and the 32k-vocab Small still needs
the projection or the gradient fix.

Not blocking the smoke: multi-GPU phase 2 (one T4 is enough at this size), Armor
(a separate model), the `.deb`, and host load (Kaggle is another machine).
