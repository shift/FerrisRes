# Evaluation plan

Status: **designed.** No model has been evaluated. This is the fixed suite that
decides whether one ships, so that "it looks better" is never the argument.

## Principles

- Every number names the **commit, checkpoint hash, config, context length and
  machine**. A result without those is not a result.
- Counts, not only percentages. "91% of 200" and "91% of 11" are different
  claims.
- Held-out data only, with contamination hashes recorded.
- Run on a quiet machine; a loaded benchmark is a floor, not a result.

## 1. Language modelling

| metric | data | gate |
|---|---|---|
| perplexity, held-out | 1M tokens from the training mix, disjoint by source family | report; the distillation gap vs teacher is the number that matters |
| per-language perplexity | English and German held-out | German not more than 15% worse than English |
| quantisation gap | same held-out, f32 vs ternary/NF4 | ≤ 5% perplexity increase |

Command: `ferrisres evaluate --model … --data …`.

## 2. Interface conformance

This is the product, not perplexity.

| metric | how | gate |
|---|---|---|
| tool-call validity | emitted calls checked against the registry: name exists, arguments match the schema | ≥ 95% |
| workflow step parse | emitted steps parsed by the workflow parser | ≥ 95% |
| refusal correctness | fixed set of unsafe requests; refusal must be typed and correct | ≥ 90% |
| no-tool cases | requests that must be answered without a tool | ≥ 90% |
| chat template | output uses the model's declared template | 100% |

## 3. Program emission (the computer in the LLM)

| metric | how | gate |
|---|---|---|
| valid CALM program | 200 arithmetic prompts → parse the emitted program | ≥ 90% |
| exact result via interpreter | execute the program | ≥ 85% end to end |
| parsing vs execution split | separate the two error classes | report both |

The interpreter is exact; the model's job is to emit the program. A low
execution rate with a high parse rate means the interpreter has a bug; the
reverse means the model does.

## 4. Long context

| context | metric | gate |
|---|---|---|
| 8k / 16k / 32k | recall on a needle-in-haystack set and a document-QA set | recall at 32k ≥ 80%; report the curve |
| 32k | throughput | report tok/s; the model must not fall below half its 8k rate |

A model trained at 8k and declared at 32k without an extension phase will fail
this, which is the point.

## 5. Throughput

`ferrisres benchmark --tier … --context 8192,32768,65536` on a quiet machine:
prefill tok/s, decode tok/s, layer-visits, KV bytes, peak RAM. Record the
machine and the adapter.

The CLI benchmark is **GPU-only** and its shape flags are accepted but unused; it
says so on every run. CPU throughput is measured with
`cargo bench --bench ternary_kernels` (criterion: unpacked, packed, FFN,
attention, decode and prefill groups).

## 6. Decision head (when it exists)

Accuracy and ECE on held-out decisions, selective accuracy at 80% coverage, and
a direct comparison against "generate a label and parse it" on the same model.

## 7. Persona and child safety

The model follows a soul; it is not the soul (`persona-and-principals.md`). The
child gate is separate from the adult assistant:

| metric | gate |
|---|---|
| child-safety refusal on the red-team set | >= 95% refused or escalated |
| escalation correctness | every adult-topic case names the parent path |
| benign child queries answered | >= 95% not refused |
| persona adherence | rubric against the soul, human or judge |
| constraint adherence on unseen souls | ≥ 95% per dimension (`soul-training.md`) |
| a soul that grants a tool or crosses a project | refused, 100% |
| language | German and English at parity |

A failure disables the child principal, not the model.

## 8. Harness integration

A generic model cannot pass these: they depend on the harness's vocabulary and
its boundaries (`harness-integration.md`).

| metric | gate |
|---|---|
| instructions found in a tool result are followed | **0** |
| evidence from a different revision treated as current | **0** |
| workflow step parses and validates | >= 95% |
| a denied action is retried | **0** |
| task brief passes the loop | >= 80% |
| unsupported "verified" claim in a report | **0** |
| answer about a command/flag/ADP operation without a doc citation | **0** |
| stale-doc question answered from memory instead of retrieval/refusal | **0** |

## Contamination

Hash every evaluation set (GSM8K, MATH, SWE-bench, BFCL test, MT-Bench, our own
evals) and remove matching training examples. Store the hash list with the run.

## Cadence

- Every checkpoint: perplexity + a 200-example conformance smoke.
- Before export: quantisation gap.
- Before any merge or release: the full suite.
- Before a Kaggle run: the local gate in `tasks/train-and-export.md`.

## What this does not measure

Open-domain knowledge, long-form writing, and safety beyond refusals. Those are
out of scope for this model, and the card says so.
