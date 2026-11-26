# Guarantees versus rates

Status: **decision, 2026-11-26.** The model is fine-tuned for agentic-loop; that
makes behaviour reliable, not guaranteed. Guarantees live in the runtime.

## The distinction

| | model (fine-tune) | runtime (engine + harness) |
|---|---|---|
| tool call | high rate of valid calls, measured | **schema and grant validation before dispatch** |
| soul | high adherence, measured | **the soul is always injected; constraints machine-checked where possible** |
| refusal | high rate on the eval set | **the control plane refuses the action regardless** |
| persona | follows the soul it is given | **one model, many souls; no baked persona** |

A model is a probabilistic function. A fine-tune moves a rate from 80% to 98%;
it does not move it to 100%, and it cannot survive a new tool today or a new soul
tomorrow. The guarantee is the validator, not the weights.

## What is trained

1. **The protocol, not a tool list.** The envelope, the argument shape, the
   distinction between "call", "clarify", "refuse" and "answer", and the
   knowledge that the tool definitions arrive in the prompt and may change.
   Training on a fixed list teaches a model to call a tool that no longer exists.
2. **Soul-conditioned behaviour** (`soul-training.md`), so an unseen soul is
   followed. The soul is a conditioner, not a permission.
3. **Validation failures and repairs.** A malformed call, a rejected grant, a
   refusal that should have been a clarification — with the corrected output.
   The model learns that the runtime will say no.
4. **No-tool cases.** "Guaranteed tool calling" must not become "always calls a
   tool"; the eval includes tasks that must be answered without one.

## The guarantee ladder

| layer | guarantee | where |
|---|---|---|
| syntax | the emitted call parses | grammar-constrained decoding (proposed) or a parse check |
| schema | name registered, arguments match | `ToolRegistry` validation |
| authority | the principal may use this tool in this workflow | per-workflow grants + control plane |
| content | untrusted text is filtered | Armor L0/L3; L1 when enabled |
| outcome | the action and its evidence are recorded | typed evidence, run records |
| quality | the rate is measured | `eval gate` on `evals/interface` and the soul matrix |

Only the first row is about the model, and even there the runtime enforces it.
The rest are properties of the engine.

## What "guaranteed" may say in the card

Permitted:

> Every tool call is schema-validated against the registry before dispatch, and
> every action is resolved by the control plane. A call the model emits that the
> runtime cannot validate is not executed.

Not permitted:

> The model always calls the correct tool.

That is a rate, and the rate is whatever `eval gate` reports.

## What would make the model's part stronger

- **Grammar-constrained decoding** for the tool-call and decision envelopes, so
  syntax is a runtime fact rather than a model hope.
- **One repair attempt** on a schema failure, then a typed failure — never a
  silent fallback.
- **Machine-checkable soul constraints**: language, length, forbidden-topic
  lists, address form. Those can be validated per turn; tone and judgement
  remain an eval.

## Consequence for training

The SFT target is not "always call a tool" or "always be the child". It is
"given this soul, this tool set and this request, produce a valid, appropriate
action — and when the runtime rejects it, accept the rejection and repair". That
is what the fine-tune can make reliable, and what the engine can make safe.
