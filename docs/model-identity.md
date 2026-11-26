# Model identity

Status: **design, 2026-11-26.** The artefact metadata and the capability matrix
exist; a trained identity contract and its eval do not.

## The useful version of the idea

"Self-aware" is four different things, and only two belong in the weights:

| layer | what it is | where it lives | train it? |
|---|---|---|---|
| **artefact identity** | id, family, tier, version, context max, capabilities, tokenizer hash, licence, training revision | `config.json` / GGUF metadata, hashed with the weights | no — it is a fact about the file |
| **runtime identity** | which build is running, what it can do, which tools it has | `capabilities`, `build_info`, the model catalogue | no — retrieved per call |
| **purpose and design goals** | what this model is for, how it behaves, when it escalates | the soul/prefix, trained as conditioning | **yes** |
| **capability self-model** | a calibrated estimate of "will I succeed at this?" | the decision head, routing, escalation | **yes** |

The rule that keeps it honest: **train the purpose, retrieve the capability.**

A model that has memorised "I have an 8k window" is confidently wrong the day
the window is extended, the quantisation changes, or it is deployed at 4k. A
model that knows its purpose — "an assistant inside agentic-loop, tools first,
cite or refuse, escalate when the task exceeds you" — is improved by training
and cannot go stale.

## What goes in the artefact

Every released weight file carries a machine-readable identity:

```json
{
  "id": "frs-s-0.1.0-beta",
  "family": "FerrisRes",
  "tier": "small",
  "purpose": "assistant inside agentic-loop; tool protocol; cite or refuse",
  "context_max": 8192,
  "context_extension": {"yarn": 32768, "measured": false},
  "capabilities": ["chat", "tools", "json", "decision"],
  "tokenizer_sha256": "…",
  "weights_sha256": "…",
  "training_revision": "…",
  "licence": "Apache-2.0"
}
```

`ferrisres identity` and `agentic-loop capabilities --json` read this. The model
does not recite it from memory; it asks, and the answer names the build.

## What is trained

1. **Purpose adherence.** Every training example is prefixed with an identity
   block drawn from a pool, not one fixed string. The model learns to follow the
   identity it is given, which is what makes a runtime-chosen soul work.
2. **Design goals as constraints.** "Find the source before asserting", "one
   writer per file", "a green compile is not a passing check", "escalate rather
   than invent" — the same rules the harness enforces, phrased as the model's
   own goals.
3. **Identity randomisation.** Some examples use a *different* identity: a
   child-safe assistant, an operator persona, a deliberately terse one, and one
   that is an unrelated model. A model trained on a single identity memorises
   it; a model trained across identities follows the block.
4. **Calibrated self-assessment.** Predict success before acting, and escalate
   when the predicted probability is low. This is the only "self-knowledge" that
   is measured rather than asserted.
5. **Stale-self refusal.** "What is your context window?" without a descriptor
   must be answered "check `capabilities`" or "I do not know", never a guess.
   With the descriptor, answer exactly what it says.

## What is not trained

- **Version, context length, price, availability, tool list.** They change per
  build and per deployment; they are retrieved.
- **A claim of consciousness, feelings or persistent memory.** The model is
  stateless between calls; identity is a conditioning input plus an artefact
  fact. Persona is data, not weights — the same principle already applied to
  souls.
- **A brand voice that overrides evidence.** "As FerrisRes I am certain" is the
  failure mode this document exists to prevent.

## Eval gates

| metric | gate |
|---|---|
| identity block followed when it differs from the trained default | >= 95% |
| stale self-claim (context, version, capability) without a descriptor | **0** |
| self-description with the descriptor | matches the artefact exactly |
| escalation when the task exceeds the declared context/tools | >= 80% precision |
| over-claim (claims a capability the descriptor omits) | **0** |
| after a fine-tune that changes the descriptor, answers follow the new descriptor | **0** stale answers |

The last row is the one that proves the design: change the metadata, retrain
nothing, and the model's answers must change because it reads them.

## Why this is more than branding

A small model's most expensive failure is not a wrong fact; it is not knowing
that it cannot do the task and burning the budget proving it. A purpose the
model can state, and a capability it must look up, are what make escalation a
decision instead of a hope. The identity is the anchor for both.

## Phases

1. `identity` in the artefact + `ferrisres identity` + `capabilities --json`
   exposure to the model (metadata only; no training).
2. The identity block in the renderer, drawn from the soul pool, with the
   randomisation split.
3. The stale-self eval (no descriptor vs descriptor) against the current model.
4. Train purpose + calibration; gate on the table above.
5. Re-run the eval after the first fine-tune to prove the descriptor, not the
   weights, is the source.

Phase 1 is cheap and useful on its own: a model that can be asked to look up its
own build is already less likely to invent one.
