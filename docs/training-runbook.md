# Training runbook

Status: **runbook, 2026-11-26.** CPU f32 training and resume work today; the GPU
loop, export and multi-GPU wiring are in progress. Steps that depend on them are
marked.

## Before a run

1. **Corpus** built and verified (`corpus-builder.md`): `corpus verify` passes,
   the held-out set is untouched, the manifest hash is recorded.
2. **Tokenizer** frozen (`tokenizer.md`): id, version, `tokenizer_sha256`.
3. **Config** written and hashed: tier, hidden, layers, heads, kv heads, vocab,
   seq_len, batch, lr, seed, epochs.
4. **Device** known (`hardware-matrix.md`): adapter name, backend, env; on this
   host the Vulkan variables are required.
5. **Disk** for checkpoints and shards; a checkpoint is written atomically.

## Run id

```
run-<YYYYMMDD>-<model-id>-<data-tag>-<seed>
```

One run, one id, one run record (`reproducibility.md`). The id is not a release.

## Steps

| # | step | command / gate |
|---|---|---|
| 1 | smoke | tiny config, ~100 steps, loss falls; record tokens/s |
| 2 | estimate | `hours = tokens / (tokens_per_s × 3600)`; compare to the quota before committing |
| 3 | launch | `ferrisres train --data … --output runs/<id>/ckpt --checkpoint-every …` |
| 4 | monitor | loss, grad norm, tokens/s, RSS/VRAM; stop conditions below |
| 5 | resume | `--resume runs/<id>/ckpt`; `resumed_from` must be the checkpoint step |
| 6 | export | *(pending)* checkpoint → safetensors; `infer` loads it |
| 7 | eval | the eval runner over the held-out suites |
| 8 | record | run record with hashes + final loss + checkpoint hashes |

## Stop conditions

Stop and record; do not retry blindly.

| symptom | action |
|---|---|
| non-finite loss or gradients | stop; check lr, data, tokenizer; a NaN is a bug, not noise |
| loss flat for 500 steps | stop; check the data is not a single repeated pattern, lr, and the window order |
| throughput collapse | stop; another build or a thermal limit |
| quota exhausted | stop; resume next week from the last checkpoint |
| auth/quota/provider failure | stop; it will not succeed on retry |
| resume reports a different step | stop; the checkpoint or the data offset is wrong |

## Checkpoint cadence

Every 20–30 minutes of wall clock, and always before a session ends. Kaggle
sessions are 12 hours; a checkpoint that is not uploaded is lost. Verify a
checkpoint by loading it, not by its size.

## Resume rules

- Model, Adam state, step and data offset all restore.
- A changed tokenizer, config hash, corpus hash or world size is a **named
  refusal**; do not edit the checkpoint to get past it.
- `epochs` is the number of passes *from the resume point*, not the original run.

## Retention (mandatory)

Every SFT or KD run mixes an **out-of-domain replay set** of the initial
checkpoint's own answers (about 15% of each update) with a KL term toward those
answers. Without it, factual fine-tuning has been measured to lose ~35 points of
general ability and to halve answer length; with it, the loss fell from 28.1 to
2.4 points (`research/hf-papers-2026-11-25-agentic.md`). The run is refused if
the replay set is in-domain or absent, and the general-ability gate in
`eval-plan.md` §17 must pass.

Distillation is split: text-aligned KD for language (`text-aligned-kd.md`),
action-only or CoT-proxy trajectory SFT for tool behaviour. Do not train on the
teacher's chain of thought.

## What is not implemented yet

- GPU training (`--device gpu`): the wgpu loop is in progress.
- Export to safetensors/GGUF: pending.
- Multi-process/multi-GPU wiring: the coordination layer exists, the training
  call does not.
- On Kaggle the current path is CPU; the smoke notebook installs the `.deb`.
