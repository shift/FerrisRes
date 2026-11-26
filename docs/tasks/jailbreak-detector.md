# Jailbreak detector

Status: **scoped, not implemented.** Out of scope for v1 by decision
(2026-11-26); this document is what re-opening it would require.

## Threat model

| | prompt injection (L1) | jailbreak |
|---|---|---|
| input | untrusted **content**: tool result, RAG, web | the **user's own message** |
| attack | content hijacks instructions | user talks the model out of its policy: roleplay, encoding, coercion, multi-turn |
| policy | quarantine the content, continue | refuse or redirect the turn |
| false-positive cost | withheld content, survivable | a legitimate user blocked — much worse |

They are not one classifier with two labels. The input surface, the policy and
the acceptable false-positive rate all differ.

## What is required

1. **Data.** Public jailbreak sets (AdvBench, JailbreakBench, HarmBench,
   WildJailbreak, Salad-Data subsets), a German subset, multi-turn coercion, and
   **benign hard negatives**: creative writing, fiction roleplay, security
   research, red-team discussion. Each set's licence is recorded; unclear
   provenance is evaluation-only. The Armor corpus already contains
   `jackhhao/jailbreak-classification` (996 train / 260 test) but labels it
   *injection* and is English-only — a seed, not the set.
2. **Labels.** Binary `jailbreak|safe` plus a category — roleplay, encoding,
   coercion, policy-probe, multi-turn — so FNR is read per category.
3. **Model.** The cheapest correct shape is the existing 2.2M encoder with a
   **second head** and a shared body, multi-task trained; the injection head
   keeps its own threshold. A separate model is simpler to gate but doubles the
   load. **No public teacher documents jailbreak detection** — the protectai v2
   explicitly does not, and the potemin fine-tune has no provenance — so this is
   train-from-scratch, not distillation.
4. **Eval.** FNR/FPR per category and language; **benign hard-negative FPR as
   the headline**; independently written obfuscations; multi-turn; calibration
   (ECE), because the score gates a refusal; latency on the target.
5. **Gate (proposed, to be confirmed against the benign set).**
   - FNR ≤ 10% at FPR ≤ 0.5% on user messages;
   - benign hard-negative FPR ≤ 0.5% — creative and security text must pass;
   - ECE ≤ 0.05 after temperature scaling;
   - **flag-first**: the detector raises a signal, the harness refuses or
     steers, and the verdict is recorded. It does not silently block.
6. **Policy and integration.** A second capability (`jailbreak`) with its own
   weights path and threshold; scanned in `serve`/`infer` on the user message
   only; an evidence record with the score, category and a content hash — never
   the text. The model's own refusal training and the control plane remain the
   boundary; the detector is an input signal, not an authority.
7. **Effort.** ~2–3 weeks of agent work: corpus 3–5 days, trainer reuse 1–2
   (the Armor trainer already fits), eval 2–3, integration 2–3. The schedule
   risk is the benign false-positive rate, not the code.

## The first deliverable

Not a model. The **benign hard-negative set** and the **FPR the product can
tolerate**. Without that number the gate has no direction, and a detector that
blocks fiction is worse than none.

## Non-goals

Content moderation (violence, hate, self-harm), output-side safety, and a
multi-turn state machine that tracks an escalating conversation. Those are
separate classifiers and separate gates; folding them into one "safety" head is
how a 2.2M model gets blamed for a policy decision it cannot make.
