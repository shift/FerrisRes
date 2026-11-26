# Token economics

Status: **model with measured file ratios and estimated totals, 2026-11-26.** The
AST ratios below are measured on four real files with a line heuristic. Every
end-to-end number is an estimate until the benchmark in this document is run.

## The claim to test

Training on the AST tooling reduces tokens per **verified change**, not merely
tokens per tool call. The difference matters: a cheap read followed by three
failed edits costs more than one expensive read that succeeds.

## The four token terms

| term | what it contains | what moves it |
|---|---|---|
| input context | system, soul, tools, history, tool results | prefix KV cache, compaction, fewer retries, smaller reads |
| output | the model's own text: tool calls and new source | symbol-addressed edits, no restating, no whole-file rewrites |
| tool results | file reads, outlines, test output | skeleton-first reads, previews, YAML serialisation |
| turns | how many round trips until verified | procedure knowledge, first-try valid args, reparse-in-call |

Total cost is roughly `sum over turns of (context re-sent + output + result)`.
The AST tools attack terms 2–4 hard and term 1 indirectly.

## What the AST tools actually save

`ast_outline` returns `(kind, name, start, end, depth)` per symbol;
`ast_node_read` returns one node's text; `ast_edit` takes a range plus
replacement text, reparses, and returns previews and an error flag.

**Read side, measured with a line heuristic** (`chars / 3.7`, whole file vs one
line per item):

| file | bytes | file tokens | outline line tokens | ratio |
|---|---:|---:|---:|---:|
| `ast_tools.rs` | 61,819 | 16,708 | 996 | 16.8× |
| `tools-core/lib.rs` | 80,621 | 21,789 | 1,315 | 16.6× |
| `agentic-core/runner.rs` | 1,257,690 | 339,916 | 18,965 | 17.9× |
| `bin/main.rs` | 1,007,190 | 272,214 | 11,400 | 23.9× |

The heuristic counts names only, not the serialised `start`/`end` objects. With
per-entry position overhead the honest estimate is **8–12×**, still large. The
benchmark measures the real serialised number; this table is a sanity check, not
a result.

**Write side.** A whole-file rewrite of `main.rs` is ~272k output tokens. A
symbol body is a few hundred. Even a large function is under 2k. The ratio is two
orders of magnitude on the most expensive term.

**Retry side, likely the largest and least visible.** A text patch fails when
the context lines do not match, and the failure costs a whole turn: the context
was already sent, the error is another result, the retry resends everything.
Exact symbol addressing removes that class of failure. This is worth more than
the read saving on a long session and is the first thing to instrument.

## The caveat that keeps this honest

On a multi-turn task the dominant term is usually **history re-sent each turn**,
not the bytes of the file being edited. If the history is 30k tokens and the
file is 3k, halving the file saves little. The AST win compounds only with:

- **conversation prefix KV** — do not re-prefill the history each turn;
- **compaction** — summarise tool results once they are consumed, rather than
  carrying a 20k outline for the rest of the run;
- **result truncation** — previews are already capped at 200 chars in
  `ast_edit`; keep that discipline everywhere.

So "greatly improves token usage" is true per *edit*, and becomes true per
*task* when the cache and compaction work is in place. The benchmark must report
both, or the claim will look better than it is.

## Training targets specific to the AST surface

1. **Outline before read.** Never `file_read` a file over ~400 lines as the
   first action; outline, then read the symbol.
2. **Minimal node read.** Read the one node being changed, not its parent, not
   the file.
3. **Symbol replacement, not whole-file rewrite.** Emit the changed item only.
4. **Reparse is verification.** Treat `has_errors` as a failed edit and repair
   before proceeding; never report success on a broken tree.
5. **No echo.** Do not repeat unchanged code, do not summarise the diff back to
   the user unless asked.
6. **Fallback judgement.** For a 20-line config file, text is cheaper than
   parse-plus-coordinates; the model must know the threshold.
7. **Refuse non-parsed languages.** `.nix` has **no grammar in
   `agentic-loop-ast`** (33 grammars, none for Nix). Training must not teach a
   tool that cannot work: for Nix, the text tools are the only path.

## Defects found while writing this

| defect | where | why it defeats the saving |
|---|---|---|
| `ast_edit` writes the file **even when `has_errors` is true** | `ast_tools.rs` (`should_write` does not check the tree) | "reparse to verify" is a report, not a guard; a broken edit is written and costs a repair turn |
| `ast_edit` is addressed by `(row, column)` | same | coordinates must be read from an outline, go stale after the first edit, and cost tokens; a `symbol` address would be cheaper and stable |
| no Nix grammar | `grammars.rs` | the repo is Nix-heavy; the AST path silently does not exist there |
| YAML serialisation is env-gated | `AGENTIC_LOOP_YAML` | a ~30% saving on every structured result that is off by default |

The first is a correctness bug, not a token bug: an edit that breaks the tree is
written to disk. It should refuse the write and return the parse error, with a
dry-run available to inspect.

## Benchmark protocol

Same model, same tasks, two arms, measured per **verified** change:

| arm | tools |
|---|---|
| text | `file_read`, `file_edit`, `file_hash` |
| ast | `ast_outline`, `ast_node_read`, `ast_edit` |

Task set: 30 real changes in this workspace, each with a named verification
(small, medium, and >1,000-line files; single-symbol, multi-symbol, rename).

| metric | how |
|---|---|
| tokens per accepted change | input + output + results, summed over the task |
| tokens per **verified** change | includes every retry and re-read |
| turns to verify | round trips |
| failed edits | count and cause (context mismatch, invalid syntax, wrong node) |
| read amplification | bytes read / bytes changed |
| wall clock | at T4 and at CPU, since prefill dominates |

Gate: the AST arm must be at least 2× better on tokens per verified change on
files over 1,000 lines, and **no worse** on files under 200 lines. If it is worse
on small files, the trained fallback judgement is missing.

## What would falsify the claim

- tokens per verified change not better than text on >1,000-line files;
- more turns to verify under AST (coordinate churn);
- any accepted change with `has_errors` (a false success);
- the AST arm needing a re-read after an edit more often than the text arm.

## Prerequisite

The tool surface above exists and is tested, but no model has been trained to
use it, and the text-vs-AST benchmark is not written. Until both exist, this is
a design with measured inputs, not a measured outcome.
