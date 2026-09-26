# Critical audit fixes — 2026-09-16

> Subsequent update: A01 causal leakage is now repaired; see [2026-09-17 repair status](causal-fix-2026-09-17.md). The remaining-work list below records the state at this update's date.

This is a repair update to the [2026-09-15 audit](bug-audit-2026-09-15.md), not a replacement for its historical evidence.

## A04 — API Armor enforcement

Task: `e10937fe-2879-4075-89a7-3f9615561585`

- CLI `serve --armor` now installs an `ArmorLayer` on the API server instead of storing an unused flag.
- Both completion endpoints check input before invoking the model handler. Rejected requests receive generic HTTP 403 JSON without echoing sensitive content.
- Every generated choice is sanitized before JSON or SSE serialization. Security-lock failures fail closed.
- Filtering remains opt-in. This covers L0/L1 input filtering and L3 output redaction; the server interface does **not** expose hidden states for L2 probing.
- Current SSE remains buffered; the separate streaming/schema defect A09 is not fixed here.

Regression coverage: real CLI HTTP tests for sensitive/benign input, both routes, streaming flags and opt-out behavior; loopback HTTP tests with a counting handler verify rejected input never invokes generation and all output choices are redacted.

## A02 — GPU training parameter bindings

Task: `65ad0745-5902-4b43-bb29-b6a12096cd27`

- Adam, cross entropy and all parameterized `BackwardPass` shaders now read actual uniform bindings rather than zero-initialized private variables.
- Removed immediate-data requirements from these operations. A private shared helper allocates an initialized parameter buffer per dispatch, preventing multiple recorded operations from overwriting one another's parameters.
- Corrected the RMSNorm backward pipeline's read-only gradient-input binding declaration.
- Replaced unsafe optimizer parameter byte slicing with `bytemuck`-checked structs.
- Promoted the original GPU loss-dispatch audit reproduction to the default test suite.

Numerical GPU regressions run on a device explicitly created **without IMMEDIATES**: two independent cross-entropy dispatches in one encoder (log 2 / log 3), Adam first update, loss gradient, and softmax backward. These pass.

This fixes parameter plumbing, **not all training mathematics**. A18–A21 remain open (matmul gradients, graph accumulation, RMSNorm Jacobian and CLI surrogate gradients). Additional source observations concerning root-gradient seeding and embedding-gradient persistence are linked to A19 for investigation.

## Validation

All commands ran inside the repository Nix development shell:

```sh
nix develop --command bash -c 'cargo build -q --message-format short'
nix develop --command bash -c 'cargo test --lib --tests -- --test-threads=1'
nix develop --command bash -c 'cargo clippy --all-targets -- -D warnings'
```

Build, full suite and Clippy passed: **1,656 tests passed**, with eight still-unresolved audit probes ignored. Subsequently added one additional softmax-gradient regression; the focused four-test GPU parameter suite and Clippy passed again. Runtime code did not change between those runs.

`git diff --check` passed. No commits were created.

## Still critical / next

- **A01:** causal future-token leakage in CPU BlockAttnRes inference/training.
- **A03:** `AnyModel` file loaders discard trained tensor values.

Neither is repaired by this change. Other audit tasks remain pending; these two fixes are not a production-readiness certification.
