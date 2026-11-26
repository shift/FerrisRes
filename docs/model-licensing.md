# Model licensing

Status: **decision, 2026-11-26.** The model is open source and the engine has no
commercial edition. This document maps every artefact to a licence, and states
the rules that keep the release clean — including the generated-data policy for
hosted providers.

## The decision

| artefact | licence | why |
|---|---|---|
| **weights** (`frs-s-0.1.0-beta`, …) | **Apache-2.0** | permissive, OSI-approved, patent grant, usable by both AGPL and proprietary downstreams |
| tokenizer, config, chat template | Apache-2.0 | shipped with the weights |
| model card | CC-BY-4.0 | documentation, attribution only |
| training and inference code (`ferrisres`) | **AGPL-3.0-or-later, no commercial option** | open source, copyleft; there is no commercial edition and no entitlement to buy |
| eval harness and eval data | Apache-2.0 / CC0 | reproducible scoring |
| dataset manifest and provenance | CC0 | facts about data, not the data |
| generated data | per generator, recorded per row | see below |

**Open source is not the same as permissive.** The engine under AGPL-3.0 is open
source. The weights under Apache-2.0 are permissive. Both statements are true at
once, and neither depends on the commercial agentic-loop product.

## "No commercial execution from ferrisres"

Concretely, this means:

- no licence JWT, key or activation is required to build, run or serve the
  model;
- no feature is gated by an entitlement in the inference or training path;
- no phone-home, telemetry or usage reporting;
- no "commercial edition" of the engine, and no dual-licence offer in the
  repository;
- the commercial surface is the separate agentic-loop product. It may use the
  model, but the model does not need it.

This has to be true in the repository, not only in this document: the `LICENSE`
header and the README currently offer a commercial licence. They are changed to
AGPL-3.0-or-later only, and existing grants (if any) are unaffected by the
change.

## The weights: Apache-2.0, with a NOTICE

Apache-2.0 requires attribution and preservation of notices. The NOTICE shipped
with the weights names:

- the teachers used for distillation (Qwen3-0.6B and Qwen3-1.7B, Apache-2.0;
  Gemma 4 E2B only if it was actually used, also Apache-2.0);
- the corpora, with the licence of each and the dataset revision;
- the modifications: architecture, tokenizer, training procedure, and the
  fact that the weights are distilled rather than the teacher's own.

A teacher whose licence is not compatible (a custom or non-commercial licence)
cannot be used for an open-weight release, whatever the quality gain. Gemma 1–3
fall in that class; Gemma 4 does not.

## Data rules for an open-weight release

The weights are the artefact that carries the data obligations, so the corpus
is stricter than "it is on the internet":

| licence class | decision |
|---|---|
| CC0, MIT, Apache-2.0, BSD | **allowed** |
| CC-BY (attribution) | **allowed**, with attribution in NOTICE |
| CC-BY-SA (share-alike) | **excluded** by default; a recorded decision is required, because share-alike reaches the derived weights |
| NC (non-commercial) | **excluded**; an open, commercially usable model cannot be built on it |
| gated / custom terms | **excluded** unless the terms explicitly permit training and redistribution of weights |
| unknown | **excluded**; a row without a licence does not train |

This changes two open questions in `source-index.md` into decisions: xLAM is
excluded if its NC report is confirmed, and Dolly is excluded unless the
share-alike decision is taken. CulturaX is excluded if its gated terms forbid
weight redistribution.

## Generated data from hosted providers

Free hosted access makes data generation cheap, and that is real. It does not
make the
outputs freely licensed: **free access is not a training licence.** The output
is governed by the terms of the model that produced it, and many hosted models
forbid using their outputs to train a competing model.

So a hosted provider is used under a per-generator allowlist, not as a default:

| requirement | rule |
|---|---|
| **generator allowlist** | only models whose terms permit training on outputs — in practice open-weight models under Apache-2.0/MIT with no output restriction (Qwen3, Gemma 4, and any local model we run). A model is added by name and version after its terms are recorded, never by provider default. |
| **provenance per row** | `generator: <provider>/<model>@<version>`, `licence`, `terms_snapshot: <date#hash>`, `prompt_id`, `seed`, `verified: true|false`. A row without these does not train. |
| **no private input** | prompts are authored or synthetic. Never send a soul, a child-principal message, a run record, a transcript, a credential or a customer document to a hosted API. Redaction happens before the request, not after. |
| **mechanical verification** | generated examples are accepted by a check, not by the model's confidence: the doc-QA answer matches the doc, the tool call validates against the schema, the code compiles, the arithmetic checks. Reject, don't repair silently. |
| **no eval contamination** | generated rows are checked against the held-out sets before they enter the corpus. |
| **rate and quota** | a budget per run; a quota or auth failure stops generation (it will not succeed on retry). |
| **no raw logs** | prompts and responses are kept as derived records with a hash, not as a transcript dump. |

The allowlist is the release gate. A single generator used outside it can make
the whole weight release undistributable, which is a much larger loss than the
tokens it saved.

## Release checklist

Before a weights release, all of these are true and recorded:

1. weights hash, parameter count, architecture, tokenizer hash;
2. `NOTICE` with teachers and corpora, each with its licence;
3. model card with intended use, limits, eval results and the snapshot date of
   the catalogue it was trained against;
4. dataset manifest: every source, licence, revision and content hash;
5. generated-data audit: every generator on the allowlist, every row carrying
   provenance;
6. no secret, credential or personal data in any artefact (scanner run, result
   recorded);
7. the model runs with no licence file present, from a clean checkout;
8. a legal review recorded for the AGPL-code-produces-weights position. The
   position we operate on is that training a model with AGPL software does not
   make the weights a derivative work of that software; this is a position, not
   a court ruling, and it is written down before the release rather than after.

## What this does not settle

- Whether a share-alike corpus can ever be used for a permissively licensed
  weight release. The conservative answer is no.
- The output terms of every hosted generator. Each is read and recorded; none is
  assumed.
- Whether the tokenizer inherits anything from the teacher. Qwen3's tokenizer is
  Apache-2.0; a tokenizer rebuilt from scratch would remove the question, a
  copied one carries the notice.
