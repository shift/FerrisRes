# Paper candidates

Status: **assessment, 2026-11-26.** A paper needs a claim, a baseline, an
ablation, reproducibility and a licence-clean artefact. This filters the project
by that standard, not by how interesting the engineering is.

## Ready now (workshop / short paper)

### A. Silently wrong gradients in blocked, looped training

**Claim.** A blocked/looped transformer trained with LoRA can produce gradients
that are wrong by multiples of the finite-difference value (measured: **4.89×**)
while every loss curve looks healthy. A finite-difference audit over the *same*
forward, plus a hard gate, catches it; without the gate the whole distillation
path was invalid.

**Evidence we have.** The exact defect and its named missing terms (suffix-sum,
final-norm Jacobian, attention Jacobian, per-layer `d_hidden`); the audit
(`max_rel_error_blocked_lora`), the guard, a before/after number, and a 1e-2
tolerance test. A real "we almost shipped wrong weights" story.

**Missing.** A second architecture or framework where the same bug class can be
shown (otherwise it is one bug, not a class). The audit harness's cost.

**Venue.** MLSys/NeurIPS reproducibility or ML-safety workshop; a short paper.

### B. Guardrail evaluation at the operating point

**Claim.** AUROC is the wrong headline for a security filter. A 2.2M injection
classifier reaches **AUROC 0.977** and still misses **47% of injections at a 1%
false-positive budget**, because the benign tail overlaps the injection mode.
Template-disjoint evaluation and per-family FNR change the conclusion.

**Evidence we have.** The score distributions, per-family FNR (37.8–83.9%), the
threshold, the full-corpus run, and a trainer fault (train/eval length mismatch)
that produced a *constant* scorer with perfect-looking FPR.

**Missing.** An independent benign set (template leakage inflates current
numbers); a second detector for comparison.

**Venue.** Security/ML workshop or a datasets-and-benchmarks track.

### C. Tokens per verified change for AST tools

**Claim.** Coding agents should be measured in **tokens per verified change**,
including retries; AST-addressed edits cut read cost 8–12× and whole-file
rewrites by two orders of magnitude, but retries, not reads, dominate the saving.

**Evidence we have.** Outline-vs-file ratios on four real files; the AST tools
and their tests; the defect that `ast_edit` wrote broken trees (now gated).

**Missing.** A 30-task benchmark with a text-tool arm, token accounting and
turns-to-verify. The protocol exists (`token-economics.md`); the run does not.

## Near-term (needs weeks and runs)

| candidate | claim | missing |
|---|---|---|
| **Cross-tokenizer distillation at 111M** | byte-aligned soft targets make a 32k student distil from a 151k teacher without a shared vocabulary | the implementation, the coverage numbers, a hard-CE baseline |
| **Does harness specialisation transfer?** | a small model trained on one harness transfers to a minimal ReAct scaffold only if the surface was randomised; preregistered falsification | the model, the second harness, the unseen-tool split |
| **Trained compaction at 111M** | behaviour-distortion training (RECAP) works at 111M/8k, with the typed notebook | training, RL budget, the divergence eval |
| **HAD residual for small agents** | distilling what the teacher adds *beyond* the harness beats imitating the full output | teacher trajectories, the harness-state contrast |

Each is a real question with a clean design, and a negative result is publishable
because the gates are preregistered.

## Long-term (systems paper)

**FerrisRes: one Rust/wgpu stack from ternary kernels to a local-SGD cluster.**
The claim would be a measured frontier: ternary Block AttnRes with top-1 MoE,
3-bit KV, bit-exact CPU↔GPU parity, 8.1× FFN and ~7.7× decode from kernel work,
and two-process local SGD on consumer GPUs. It needs what we do not have: an
end-to-end trained model, throughput and energy against llama.cpp/MLX on the same
hardware, and a quality baseline. Months, not weeks.

## Recommendation

Write **A** first. It is complete, short, reproducible, and honest; the numbers
exist and the artefact is a test. Then **B**, because a negative evaluation
result about guardrails is useful to everyone shipping one. **C** is a small
empirical study that needs a benchmark run, not new research.

Do not start the systems paper until there is a trained model and a baseline on
the same hardware; otherwise it is a description of an engine, not a result.

## Rules for any submission

- No raw sessions, transcripts or personal data; derived records only.
- Weights Apache-2.0, code AGPL-3.0-or-later; the paper's artefacts state both.
- MoLT is clean-room (no licence); cite, do not vendor.
- Report counts, seeds and variance; a single run is a data point.
- State what is calculated vs measured, as the docs already do.
