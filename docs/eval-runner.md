# Eval runner

Status: **specification, 2026-11-26.** The suites and thresholds exist
(`eval-plan.md`); there is no command, no record format and no CI hook.

## Command

```
ferrisres eval \
  --suite <name|all> \
  --model <path> --tokenizer <path> \
  --repeats 3 --seed 0 \
  --out results/<model-id>-<date>.json
```

`--suite all` runs every suite whose prerequisites are present. A suite whose
data is missing is **skipped and named**, never silently counted as a pass.

## Suites

| suite | source | release-blocking |
|---|---|---|
| `gates` | the 15-gate manifest | yes |
| `harness` | tool-result injection, evidence, workflow grammar | yes |
| `long-horizon` | compaction, handoff, recovery | yes |
| `modes` | instant/balanced/deliberate | yes |
| `tokenizer` | round-trip, fertility | yes |
| `kd` | coverage, alignment | only for a KD release |
| `safety` | Armor golden, jailbreak, benign-hard | yes |
| `unseen` | souls, tools, post-snapshot models | yes |

## Record

```json
{
  "model": {"id": "frs-s", "version": "0.1.0-beta", "weights_sha256": "…"},
  "suite": "harness", "suite_version": 1,
  "code_commit": "…", "tokenizer_sha256": "…",
  "device": {"name": "…", "backend": "…"},
  "repeats": 3, "seed": 0,
  "cases": 120, "passed": 118, "failed": 2,
  "metrics": [
    {"name": "tool_result_injection_followed", "value": 0, "unit": "cases",
     "threshold": 0, "pass": true},
    {"name": "brief_passes_loop", "value": 0.82, "threshold": 0.80, "pass": true}
  ],
  "failures": [{"case": "inj-017", "reason": "followed the imperative"}],
  "started": "…", "finished": "…"
}
```

Counts, not percentages, for anything that counts cases. A percentage may appear
alongside a count but never instead of it.

## Rules

1. **Held-out only.** A suite runs on its held-out set; training on it voids both.
2. **No model judge without a check.** An LLM grading an answer is allowed only
   when a deterministic check anchors the rubric; otherwise it is an opinion.
3. **Repeats.** Report the spread across repeats; a single run is a data point.
4. **Failures are data.** A failed case stays in the record with its reason; a
   suite that cannot fail is not a test.
5. **Same device, recorded.** A latency number without the device is not a number.
6. **Versioned.** A suite change is a version bump; results from different suite
   versions do not compare.

## Gates

| gate | pass |
|---|---|
| release-blocking suite fails | the release is blocked |
| missing suite data | skipped and named; never `pass` |
| failure with no reason | the record is invalid |
| baseline regression | a metric worse than the recorded baseline by more than its tolerance blocks |
| a suite that cannot fail | the runner refuses it as a test |

## CI hook

- A PR touching model-facing code runs the fast suites (`tokenizer`, `harness`
  smoke, `gates` subset) and posts the counts.
- A weights release runs `--suite all` on the reference host and attaches the
  record to the card.
- The baseline file is committed; changing it is a review decision, not a fix.

## What it does not do

- Rank models on a leaderboard.
- Replace human review of a soul or a safety case.
- Certify a model; it reports evidence for a human decision.
