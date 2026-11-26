# Content export for self-generated data

Status: **ready to delegate.** Repository: **agentic-loop**, base `origin/main`.

## Behaviour (one)

`agentic-loop dataset export --session <id> --content --out <path.jsonl>` writes
one redacted row per step with tool calls and results, and **refuses** a session
that has no verifier/evidence record. The metadata-only exporter stays unchanged.

## Files

- `crates/agentic-core/src/dataset_export.rs` (new or extend).
- `crates/agentic-loop-bin/src/main.rs` — one flag on the existing `dataset
  export`.
- `crates/agentic-core/tests/dataset_content_export.rs` (new).

## Test (name it)

`content_export_refuses_without_evidence`: a session with no `evidence_query`
record fails by name; a session with one writes rows and every credential-shaped
value is redacted.

## Verify

```
VERIFY: cargo test -p agentic-core dataset_content_export
```

## Out of scope

The selfplay workflows, dedupe, contamination, the allowlist. This is the export
only. No raw logs, no personal data.

## Rules

One row carries `source`, `generator`, `verifier`, `revision`, `licence`, `seed`,
`verified`, `redacted`. Push the branch; no merge.
