# Release runbook

Status: **runbook, 2026-11-26.** No weights have been released. This is the order
that makes the first one auditable.

## Preconditions — all blocking

| # | condition | evidence |
|---|---|---|
| 1 | every release-blocking eval suite passes | eval record (`eval-runner.md`) |
| 2 | the provenance register is complete for every used source | `provenance-register.md` |
| 3 | NOTICE generated from the register, not from memory | file + hash |
| 4 | the model card has no `TBD` and cites run records | `model-card-template.md` |
| 5 | weights, tokenizer, corpus manifest and code are pinned by hash | `reproducibility.md` |
| 6 | the licence audit is recorded, including any open legal question | `model-licensing.md` |
| 7 | a clean install was tested | container log |
| 8 | `engine_min` is set and a model older than it refuses to load | config |

## Artefact set

```
frs-s-0.1.0-beta/
  model.bf16.safetensors
  model.q4_k_m.gguf            (optional, per format)
  config.json
  tokenizer.json
  chat-template.jinja
  model-card.md
  NOTICE
  LICENSE
  eval-record.json
  run-records.jsonl
  SHA256SUMS
```

The `.deb` for the engine is a separate artefact with its own version; it is not
part of the model directory.

## Steps

1. **Freeze.** Tag the engine commit `frs-s-v0.1.0-beta`; build from the tag, not
   from a working tree.
2. **Build weights.** Export, quantise, and re-run the eval on the **exported**
   artefact, not on the training checkpoint.
3. **Generate NOTICE** from the provenance register: teachers, corpora,
   modifications, clean-room notes.
4. **Write the card** from the eval record; no number without a command.
5. **Hash.** `SHA256SUMS` covers every file; the card's hashes must match.
6. **Verify.** In a clean container: install, load, run one prompt, run the fast
   eval suite; the model must run with **no licence file present**.
7. **Publish.** Weights and card together; the `.deb` alongside; never a weights
   file without its card.
8. **Point.** Move the `latest` pointer only after the upload verifies.
9. **Announce** with the card link and the eval record, not a benchmark screenshot.

## Deprecation and rollback

- Artefacts are **immutable**. A fix is a new patch version; an old file is never
  replaced.
- **Deprecation** is a metadata field plus a note in the card; files stay
  downloadable for the support window.
- **Rollback** is moving the `latest` pointer back and publishing a notice; the
  bad version stays available for forensics.
- **Security fixes** apply to the current major and the previous minor; a broken
  tokenizer is a major break, not a patch.

## Sign-off

| role | checks |
|---|---|
| model owner | card completeness, eval evidence |
| data owner | provenance and licence audit |
| security | safety suite, Armor version, incident history |
| release | tag, hashes, clean install |

A release with a missing sign-off does not publish. One person may hold two
roles; no one holds all four for the first public release.

## Never

- Overwrite a version or re-upload different bytes under the same name.
- Publish a weight whose card cites a run record that does not exist.
- Publish a card that claims "verified" without the command.
- Ship a `.deb` with a glibc newer than the documented floor.
