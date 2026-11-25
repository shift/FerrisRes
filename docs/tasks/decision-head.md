# Decision head (Jev-style System One)

Status: **task specification, not implemented.**

A small head on the model's hidden state that answers typed questions with
calibrated probabilities in a single prefill, and generates no text. It is the
model-side counterpart of `agentic-loop-decision`: the head proposes, the
control plane disposes.

## 1. What it is

Three question primitives, after TypeSafe's Jev:

| primitive | input | output |
|---|---|---|
| **Choice** | up to 255 options, each a key and optional description | probability for every option, selected option, confidence |
| **Score** | 2–10 ordered level descriptions | probability per level, probability-weighted level |
| **Noul** | a yes/no statement | probability of yes |

A request is a `state` (text or JSON) plus a map of named questions. All
questions are evaluated against the same state in one forward pass; adding
questions barely changes latency. There is no generation, so there is no
parsing and no invented label.

## 2. Why it fits FerrisRes

- `agentic-loop-decision` already defines bounded control questions
  (escalate, authorize, continue, route). The head is the model that answers
  them; the crate already has calibration and shadow mode.
- The capability descriptor gains a `decision` feature, so a checkpoint can
  declare a head and the runtime can refuse `decide` without one.
- The model runtime acquires the head as a specialist artefact, like Armor L1.
- The control plane consumes probabilities, not prose: a threshold or an
  expected-value rule, recorded as typed evidence.

## 3. Architecture

- Readout on the final hidden state at the position before a `<|mark|>` token,
  `logits = W · h`, soft-capped like an LM head. `W` is `[hidden, labels]`.
- **Single-token labels.** Each option maps to one token id; the option's score
  is that token's logit. Options are defined at inference, but must map to a
  unique single token (or a short code), so the tokenizer is part of the head
  artefact.
- **Choice**: softmax over option tokens. **Score**: softmax over level tokens,
  expected level = Σ p_i · i. **Noul**: sigmoid over the yes/no pair.
- Temperature (one scalar, or vector scaling per label) is fitted after
  training on a held-out calibration set. The head ships with its temperature.

## 4. Input format

```
<state>…</state>
Question: Which kind of action is this?
Options: R = read-only; W = internal write; X = external transfer
Answer: <|mark|>
```

The option codes (R/W/X) are the single tokens that get scored. The question id
is not sent to the model; all meaning is in the descriptions, as in Jev.

## 5. Training

- Data: `(state, question, options, answer)` triples. Sources: the harness's
  own decisions (workflow choice, tool risk class, refusal, escalation), plus
  public classification sets rendered into the same format.
- Base model frozen; train the head only (cheap), then optionally LoRA on the
  top layers if the frozen hidden state is not separable enough.
- Loss: cross-entropy on the answer token; for Score, ordinal cross-entropy.
- Calibration: temperature or vector scaling on a held-out set, never on the
  training split. Report ECE before and after.
- Split by source family; contamination check against the evaluation sets.

## 6. Calibration and the gate

| metric | gate |
|---|---|
| accuracy on held-out | ≥ the generative path's accuracy at equal cost |
| ECE (expected calibration error) | ≤ 0.05 after temperature scaling |
| selective accuracy at 80% coverage | ≥ 95% |
| latency | one prefill, ≤ 1.5× the base model's prefill at the same context |
| option-count behaviour | accuracy within 2 points up to 25 options; report the break point |

The comparison that matters is against "generate a label and parse it" on the
same 0.6B, not against Jev. A head that is not better than the generative path
on this model is not worth shipping.

## 7. Engine exposure

- `ferrisres decide --model <path> --head <path> --request <json>` returns the
  typed answers as JSON, with probabilities and confidence.
- A `DecisionHead` artefact next to the model: `head.safetensors` +
  `decision_config.json` (labels, tokenizer id, temperature, soft-cap).
- The checkpoint header declares `decision`; the capability resolver refuses
  `Workload::Decision` without a head.
- The harness calls it through the model runtime, not by filename.

## 8. Harness integration

| question | primitive | consumer |
|---|---|---|
| which workflow for this message? | Choice over the allowlist | router |
| is this tool call read-only, internal write, external transfer? | Choice | control plane risk class |
| does this action violate policy? | Noul | control plane |
| how urgent is this message? | Score (4 levels) | notification/quiet-hours policy |
| should I answer, ask a clarifying question, or refuse? | Choice | chat workflow |

The head's output is a *proposal*: thresholds and the precedence ladder decide.
Every decision is shadow-recorded first, as the control plane already does.

## 9. Tasks

| # | task | acceptance |
|---|---|---|
| D1 | head module + single-token scoring + soft-cap | unit tests on synthetic logits; choice/score/noul shapes |
| D2 | training loop on frozen base + head | loss decreases; head params only |
| D3 | calibration (temperature/vector) + ECE report | ECE ≤ 0.05 on held-out |
| D4 | `decide` command + JSON schema | golden request/response; refusal without head |
| D5 | harness integration behind the decision crate | shadow mode; no action without the arbiter |
| D6 | comparison against the generative path | accuracy and latency table |

## 10. Non-goals and risks

- **Not a replacement for the generative model.** It answers bounded questions;
  it cannot explain or converse.
- **A 0.6B hidden state is weaker than a frontier decision model.** The gate is
  "better than generating a label on the same model", not "as good as Jev".
- **Single-token labels limit option count** and collide with the tokenizer;
  the head artefact pins the tokenizer.
- **Calibration is the product.** A confident wrong answer is worse than a
  generated sentence a human can judge; ECE is a hard gate.
- **No action authority.** The head proposes; the control plane decides.
