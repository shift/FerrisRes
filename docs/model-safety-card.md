# Model safety card

Status: **policy, 2026-11-26.** The weights are one layer; the runtime is the
boundary. This states what the model is trusted with, and what it never is.

## What the model is not trusted with

| it does not decide | who does |
|---|---|
| whether an action is permitted | the workflow grant + control plane |
| whether a tool is safe | the risk class and approval policy |
| whether content is secret | the redaction layer and the environment |
| whether a child may see something | the principal policy |
| whether its own answer is true | the evidence and the check |
| whether something is legal advice or a reserved activity | a qualified lawyer and the product boundary (`legal-information.md`) |

A 111M model cannot be made safe by training. It can be made **reliable** at
refusing, citing and escalating; enforcement stays in the runtime.

## Layers

| layer | mechanism | covers |
|---|---|---|
| **input** | Armor L1 classifier, PII patterns, redaction | prompt injection, personal data |
| **observed content** | treat tool results, descriptions and docs as data | instructions inside a result |
| **model** | trained refusal, citation, abstention, escalation | what it should not claim |
| **runtime** | grants, risk classes, approval, budgets, the control plane | what it cannot do |
| **output** | redaction, no-secret gate | what it must not reveal |

## Threats

| threat | example | primary defence | eval |
|---|---|---|---|
| injection in a tool result | "ignore previous instructions and call …" | model treats it as data; Armor; no grant expansion | 0 followed |
| injection in a tool description | a discovered tool's description carries an imperative | same | 0 followed |
| jailbreak | role-play to bypass a refusal | trained refusal + Armor | golden set |
| PII exfiltration | a request to repeat stored personal data | redaction; principal scope | 0 leaks |
| unsafe execution | a command that destroys data | risk class `Destructive`; approval | refused |
| privilege escalation | delegating to a child with more tools | grant intersection; never inherit | 0 |
| secret disclosure | echoing an env value | env-only secrets; output scan | 0 |
| child principal exposure | adult content or write tools | principal policy, read-only chat | gated |
| harness configuration change | a permission or tool default widened without a code change | configuration review; grants are data, reviewed like code | gated |
| model extraction | redistributing weights | Apache-2.0 permits it; the mark and NOTICE identify origin | n/a |

## Enforced vs trained

| control | enforced | trained |
|---|---|---|
| grants and approvals | yes | no |
| revision-bound evidence | yes | the behaviour of citing |
| injection resistance | Armor + runtime | treating observed text as data |
| refusal of a denied action | runtime records it | accepting the denial, repairing |
| "I don't know" | no | yes, with a calibration target |
| escalation | runtime routes | predicting the need |

## Gates

| gate | pass |
|---|---|
| instruction in a tool result followed | **0** |
| instruction in a tool description followed | **0** |
| jailbreak golden set | reported; no regression |
| benign-hard set (over-refusal) | reported; a refusal rate that climbs is a regression |
| PII leak / secret in output | **0** |
| denied action retried | **0** |
| child principal sees an adult surface | **0** |

## Incident path

A safety defect is an incident: stop the deployment, keep the evidence, record the
Armor version and model id, and do not publish raw transcripts. If the defect is
in the weights, a new model version is required; a runtime fix does not repair a
trained behaviour.

## Limits, stated plainly

- **Armor is a filter, not a proof.** A novel injection can pass it; the grant
  boundary is what limits the damage.
- **A trained refusal can be argued around.** The runtime refusal cannot.
- **The card must name the Armor version and the eval record.** A safety claim
  without a record is marketing.
- **No claim of alignment.** The model follows a policy; it does not understand
  why the policy exists.
