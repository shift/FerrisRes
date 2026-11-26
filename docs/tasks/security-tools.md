# Security scanner tool

Status: **ready to delegate.** Repository: **agentic-loop**, base `origin/main`.

## Behaviour (one)

A `sec_audit` tool runs `cargo audit --json` inside the workspace, normalises the
output to `{advisory, package, version, severity, patched}`, and is declared
read-only. A missing binary returns a named "not installed" error, never an empty
success.

## Files

- `crates/agentic-loop-tools-core/src/security_tools.rs` (new).
- `crates/agentic-loop-tools-core/src/lib.rs` — register it.
- `tests/security_tools.rs` (new).

## Test (name it)

`sec_audit_normalizes_a_fixture`: a fixture JSON on `PATH` (a tiny shell script or
a temp dir) produces the normalised fields; a missing binary is a named error.

## Verify

```
VERIFY: cargo test -p agentic-loop-tools-core sec_audit
```

## Out of scope

gitleaks, semgrep, fuzzing, network advisory updates. No scanning outside the
workspace.

## Rules

Risk class `Read`; add it to `declared_risk_class`. No new dependency unless the
JSON parsing needs one. Push the branch; no merge.
