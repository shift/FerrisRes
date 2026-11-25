# `train` and export

Status: **task specification.** `train` is currently a stub; the f32 reference
model with a verified backward exists in `src/training/f32_trainer.rs` and is
not wired to the CLI.

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
