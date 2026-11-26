# Implementation handoff

Status: **handoff index, 2026-11-26.** Written so a smaller coding model can pick
up one task at a time without making design decisions. If a task needs a decision,
**stop and report**; do not guess.

## How to work one task

1. Read the task file and the doc it references.
2. Create a worktree from the named base:
   `git worktree add -q -b <branch> /tmp/opencode/wt-<task> <base>`
3. Change at most **two source files plus the test file**. `src/main.rs` gets one
   enum variant and one match arm, never a reformat.
4. Run the exact `VERIFY` with the shared build lock:
   `flock -w 7200 /tmp/opencode/build.lock env CARGO_TARGET_DIR=/tmp/opencode/target-<task> CARGO_BUILD_JOBS=4 <VERIFY>`
5. Commit in small steps, push the branch, **never merge**, never push `main`.
6. Report: branch, commit, exact command, pass/fail counts, and what you did not
   verify.

## Stop rules (a small model must not decide these)

- A change to the model maths (attention, backward, loss, quantisation).
- A licence question, a data mix, a vocabulary size.
- A change to the control-plane precedence or a permission default.
- A multi-crate refactor, or anything that edits more than two source files.
- A failing test whose cause is a design mismatch. Report, do not weaken the
  assertion.

## Ready now

| task | doc | base | size |
|---|---|---|---|
| 32k tokenizer | `tasks/tokenizer-32k.md` | `feat/train-cli` | S |
| GGUF export | `tasks/gguf-export.md` | `feat/train-export` | M |
| content export for self-generated data | `tasks/selfplay-export.md` | agentic-loop `origin/main` | M |
| `sec_audit` tool | `tasks/security-tools.md` | agentic-loop `origin/main` | S |
| minimal ReAct harness | `tasks/second-harness.md` | agentic-loop `origin/main` | S |
| corpus source adapters | `tasks/corpus-sources.md` | `feat/corpus-builder` | M |
| replay + KL in train | `tasks/replay-kl.md` | `feat/train-cli` | M |
| first eval-runner suite | `tasks/eval-runner.md` | `feat/train-cli` | M |
| compaction + commitment metrics | `tasks/compaction-commit-metrics.md` | agentic-loop `origin/main` | S |
| action-only / CoT-proxy targets | `tasks/cot-proxy-distill.md` | `feat/train-cli` | S |
| wire the fixed gradient into the CLI | `tasks/wire-distill-grad.md` | `fix/block-lora-gradient` | S |

## In flight — do not duplicate

| branch | work | state |
|---|---|---|
| `feat/corpus-builder` | corpus builder v1 | agent running |
| `int/kaggle-deb` | 0.2.4 `.deb` (train + export) | agent running |
| `feat/gpu-train-loop` | wgpu training loop | agent running, not pushed |
| `int/train-multi-gpu` | multi-process local SGD | agent running, not pushed |
| `feat/prefix-kv-cache` | prefix KV cache | agent running, not pushed |
| `perf/parallel-prefill` | parallel prefill | agent running, not pushed |

If one of these stalls, its task file is the specification: use the same one
behaviour/test/VERIFY rule.

## Blocked or not for a small model

| item | why |
|---|---|
| Kaggle GPU `.deb` (`tasks/kaggle-gpu-deb.md`) | blocked on the GPU loop; the gate is a GPU session, not a unit test |
| multi-GPU training | depends on the GPU loop and the integration branch |
| RECAP-style compaction training | RL design, a research task |
| the second-harness **eval** | depends on the harness and a model |
| PLE competition harness | a separate product/competition task |
| Legal information (`legal-information.md`) | needs a source registry, an index and a lawyer review per jurisdiction |
| Armor threshold/data work (`tasks/armor-gate.md`) | needs data curation and a threshold decision |

## Definition of done

- The exact `VERIFY` passes; the report has the counts, not a percentage.
- The commit is on its own branch, pushed; `main` is untouched.
- The test asserts the behaviour, and fails if the behaviour is removed.
- No secrets, no network in tests, no raw logs.
- If the task changed a documented behaviour, the owning doc is updated in the
  same branch; otherwise leave the docs alone.

## Where the specs live

`docs/tasks/` holds one file per task. The design documents they cite are the
source of truth for *why*; the task file is the source of truth for *what* and
*how to verify*.
