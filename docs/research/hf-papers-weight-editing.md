# HF papers — weight editing

Status: **scan, 2026-11-26.** The `?q=` filter is global, not date-scoped; the 26th
has no weight-editing paper of its own. The search returns the 2022–2026 line.
Five are relevant; the rest are named.

## 1. Editing does not remove knowledge reliably

**The Tokens Remember** (`2609.29045`): localized edits and unlearning can be
bypassed by **alternative tokenizations** of the same string. Across five LLMs,
six datasets and six techniques, **38.6% of alternative tokenizations recover the
pre-edit response**; a reference-free attack detects modified facts with F1 84.2%.
The conclusion for us is direct: **you cannot include NC/SA/gated data and later
"edit it out" for a release.** Even a perfect weight edit leaves the licence
obligation, and the edit is not robust. A release is retrained on the permissive
corpus. This confirms the private-test/release split in `first-student.md`.

## 2. Task arithmetic — cheap, legible variants

**Editing Models with Task Arithmetic** (`2212.04089`) and **Task Arithmetic in
the Tangent Space** (`2305.12827`): a task vector is the difference between a
fine-tuned checkpoint and its base; vectors can be negated, added and analogised.
For us this is the cheapest route to **domain variants** (`frs-s-code`,
`frs-s-de`) once two checkpoints exist on the same base, and a way to *remove* a
behaviour without a full retrain. It needs matched checkpoints and an eval per
variant; it is not a substitute for data.

## 3. Repetition loops are localised

**Can Editing 1 Neuron Fix Repetition Loops?** (`2606.13705`): Gemma 4 loops on
long enumerations at rates up to 95%, traced to a few MLP neurons or routed
experts; a single sign-inverted neuron fixes the E2B case. Editing removes the
loop but **cannot supply a missing fact** — a knowledge-precision failure remains.
For a 111M model, this is a diagnostic for a concrete failure, not a training
strategy.

## 4. Tensor surgery is a tool, not a method

**BrainSurgery** (`2606.09707`): declarative YAML plans for tensor
transformations, with shape/dtype/value assertions. The idea is worth borrowing
for our export/quantisation path (reproducible, validated tensor edits); the code
licence must be checked before any use.

## 5. Weight-space programming is real but uneven

**Exploring Weight-Space Programming** (`2611.09382`): editing 1,023 parameters
raised MATH-500 by 29.45 points, but outcomes vary and 6 of 16 trajectories kept
the original weights when validation failed. **IWIBench** (`2611.08647`) finds
capabilities diverge sharply across models and tasks. This is a research
direction for later, not a substitute for the first student.

## Safety, for an open-weight release

- **Abliteration** (`2608.18093`, `2609.16204`, `2609.05794`): refusal can be
  removed by projecting out a direction with a few contrastive prompts; defenses
  poison the estimator. Our safety card already says Armor is a filter and the
  runtime is the boundary; an open-weight model's alignment is **not** a
  guarantee, and an attacker may edit it.
- **Memorization vs reasoning** (`2510.24256`): editing out memorized data harms
  fact retrieval and arithmetic; memorization and reasoning share weight
  directions. Another reason a corpus is not a bag of facts to be scrubbed later.

## Not assessed

`ThinkEdit` (`2503.22048`), `Steer2Edit` (`2602.09870`), `LUNE` (`2512.07375`),
`Separable Expert Architecture` (`2604.21571`), `GUARD-IT` (`2605.12765`),
`HypoLoop` (`2611.23718`), `Can LLMs Write Backdoors?` (`2611.17395`),
`Edit-Ready Training` (`2611.26408`), video/image editing papers (not weights).

## Recommendation

No change before the first student. After it exists:

1. Try **task arithmetic** for the code/German variants if the eval supports it;
2. borrow the **assertion discipline** of tensor surgery for export;
3. treat **repetition-loop repair** as a diagnostic;
4. never treat weight editing as a licence-cleanup step.
