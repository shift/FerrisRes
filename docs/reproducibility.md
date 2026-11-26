# Reproducibility

Status: **policy, 2026-11-26.** A run is reproducible when its inputs are pinned
by hash and its nondeterminism is stated, not hidden.

## What is pinned

| artefact | recorded as | gate |
|---|---|---|
| code | engine commit | the run refuses a dirty tree unless `--allow-dirty` |
| corpus | manifest hash + shard hashes | a shard mismatch refuses the run |
| tokenizer | id, version, `tokenizer_sha256` | a mismatch with the checkpoint refuses resume |
| chat template | file hash | rendered samples in the run record |
| config | JSON hash | printed at step 0 |
| seed | integer | printed; resume keeps it |
| checkpoint | `weights_sha256` + format version | resume validates config, world size, tokenizer |
| teacher | id, revision, licence | recorded when distillation is used |
| device | adapter name, backend, driver | recorded per rank |

## Determinism levels

| level | guarantee | where |
|---|---|---|
| **bitwise CPU** | same corpus, seed, code, config → same loss curve and same final weights | f32 reference trainer |
| **bitwise CPU resume** | resume at step k equals an uninterrupted run from step k, for the same window order | `train_learns` resume test |
| **GPU tolerance** | same inputs produce the same result within a stated tolerance, not bitwise | not yet claimed |
| **multi-process** | local-SGD averaging is exact; the *trajectory* depends on `exchange_every` and is recorded | coordination tests |

GPU reduction order is not deterministic across drivers and workgroup schedules.
A GPU run may therefore differ in the last bits; the tolerance belongs in the run
record, and "bitwise" must not be claimed for it.

## Rules for a run

1. The data order is a deterministic function of the sorted corpus and the seed.
   No filesystem order, no clock, no thread race.
2. Shuffling is seeded and its seed is in the record; a run without shuffling says
   so.
3. `batch_size` means windows accumulated into one optimizer step; the order of
   accumulation is fixed.
4. A resume must restore model, Adam state, step and data offset. A resume with a
   different tokenizer, config hash, world size or corpus hash is a named
   refusal.
5. A run writes a **run record** before the first step and updates it at the end;
   the record is the evidence, not the log file.

## Run record

```json
{
  "run": "run-20261126-frs-s-fineweb-42",
  "code_commit": "…",
  "config_sha256": "…",
  "corpus_manifest_sha256": "…",
  "tokenizer_sha256": "…",
  "chat_template_sha256": "…",
  "seed": 42,
  "device": {"name": "NVIDIA T4", "backend": "vulkan", "rank": 0, "world_size": 2},
  "tolerance": {"kind": "gpu", "loss_abs": 1e-4, "weights_rel": 1e-3},
  "steps": 2000,
  "initial_loss": 5.59,
  "final_loss": 4.12,
  "checkpoints": [{"step": 1000, "sha256": "…"}, {"step": 2000, "sha256": "…"}],
  "started": "…",
  "finished": "…"
}
```

A run without a record is an anecdote. A card that cites a run must cite the
record's hash, not a screenshot of a loss.

## Gates

| gate | pass |
|---|---|
| rebuild the corpus from its plan | identical shard hashes |
| rebuild the tokenizer from its manifest | identical `tokenizer.json` hash |
| CPU re-run from the same record | same final loss and weights hash |
| resume at step k | same trajectory as the uninterrupted run (CPU) |
| a changed tokenizer/config/world size on resume | refused, naming the field |
| a dirty tree | refused unless explicitly allowed and recorded |

## What is not reproducible

- A run whose corpus was fetched from a moving branch or an unversioned URL.
- A teacher that was updated without a revision bump.
- A GPU run claiming bitwise equality.
- A result quoted without a run record.
