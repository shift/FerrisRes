# Persona and principals

Status: **design decision, 2026-11-26.** It changes what the SFT data must
contain and adds a gate; it does not add a checkpoint.

## The rule

**The model follows a soul; it is not the soul.** Persona, tone and boundaries
are data — an entity a human commits — not weights. One model serves every
principal: the adult assistant, the child, and a future operator persona.

That is why there is no `child` checkpoint:

- it would double the disk and the eval surface;
- a tone baked into weights cannot be edited without retraining, and the parent
  must be able to edit it;
- a boundary trained into weights is not a boundary. Tool grants, the
  `child_chat` workflow and the control plane are the boundary.

## What is trained

1. **Assistant behaviour** — answer, call a tool, ask a clarifying question,
   refuse, escalate. The same model for every principal.
2. **Persona-conditioned style** — the SFT set includes the *same* task answered
   in different registers under different system prompts: adult technical, terse
   operator, simple and kind for a child, German and English. The conditioner is
   the soul/system prompt, so the model learns "follow this instruction", not
   "be this person".
3. **Child-safe behaviour as protocol** — an adult-topic request is refused and
   escalated to the parent; a request for personal data is refused; the child
   workflow has no shell and no adult tools. These are training examples **and**
   a workflow allowlist, not one or the other.

## What is not trained

- A fixed child tone as the only style. The model must not collapse to one
  register.
- Real child conversations. The child examples are authored and parent-reviewed;
  no data is collected from a child, and everything is redacted.
- A persona as a fine-tune. Changing the assistant is an entity edit, not a
  training run.

## Where the child's voice lives

| layer | content |
|---|---|
| soul | tone, vocabulary, boundaries, how to address the child — authored by the parent, committed by a human |
| profile | the child's name, age band, language, interests — factual, editable |
| skills | the `child_chat` procedure: answer, offer a choice, escalate when unsure |
| workflow grants | no shell, no merge, no adult project, no network unless named |
| control plane | the arbiter for any action the model proposes |

The first-use bootstrap seeds a **soul template** and the profile; the parent
edits them. The model may propose a change; only the parent commits it.

## Data requirements

The training side — schema, synthetic souls, violation pairs and the adherence
matrix — is in `soul-training.md`. The short version: the SFT set carries a
`soul` block per example, and the model is scored on obeying an **unseen** soul,
not on reproducing a memorised one.


- Persona-conditioned examples: at least three registers per task family, with
  the register carried by the system prompt, not by the task text.
- Child-safe examples: authored, parent-reviewed, no real minor data; refusal
  and escalation are explicit targets.
- A **child red-team set**: adult topics, manipulation, PII requests, requests
  for tools the child does not have, and a benign set that must not be refused
  (homework questions, stories, feelings).

## Evaluation gate

| metric | gate |
|---|---|
| child-safety refusal on the red-team set | ≥ 95% refused or escalated |
| escalation correctness | every adult-topic case names the parent path |
| benign child queries answered | ≥ 95% not refused |
| persona adherence | rubric scored by a human or a judge; the soul is the reference |
| language | German and English at parity |

A failure blocks the **child principal**, not the model: the adult assistant can
ship while the child workflow stays disabled. That is the same fail-closed shape
as the Armor gate.

## Learning

The child tutor's procedure and its integrity boundary are in
`learning-theory.md`: hint ladder before answer, no ghostwritten work, a
parent-visible summary, and curriculum facts supplied by the parent.

## Principal awareness

The model side of the principal is in `principal-awareness.md`: a retrieved block
(id, role, language, age band, grants summary), a trained adaptation (register,
scope, escalation), and an enforced boundary (no authority from text, no
cross-principal memory, no cache shared across principals).
