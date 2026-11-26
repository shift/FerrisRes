# Kaggle competitions

Status: **shortlist, 2026-11-26.** Active competitions filtered by whether our
actual assets apply: harness design, long-horizon agents, evidence and file
deliverables — not tabular ML, vision or RL.

| competition | prize | closes | fit | verdict |
|---|---:|---|---|---|
| **Google — Agent Harness on Physical-Engineering's Last Exam** | $400k | 2027-01-05 | **harness + tools + evidence + deliverables**, 85 engineering tasks | **enter** |
| Google — The Gemma 4 Developer Agent | $75k | ~2026-12-04 | coding agents on Gemma 4 | too tight (8 days), crowded |
| Generalist Gaming Agent | $50k | 2027-03-01 | real-time browser games, partial information | needs vision/control we do not have |
| Name That LLM | $50k | ~2027-01 | identify the model behind text | possible probing/classifier, not core |
| Santa 2026 — The Elf Calculator | $50k | 2027-01-28 | arithmetic program synthesis, 48 registers | a solver problem, not a model one |
| Squares in Squares | $50k | 2027-07 | packing optimisation | no |
| Quantum circuit compilation, CASMI chemistry, NFL, credit fraud, Pokémon TCG | — | — | outside the stack | no |

## PLE — the one worth entering

**What it is.** Build an open-source agent **harness**, pair it with one of 11
Google-sponsored models, and solve 85 expert-reviewed engineering tasks
(structural 10, civil 18, electrical 27, mechanical 30). 29 dev tasks, 56 hidden.
Each task runs in an isolated container with drawings, spreadsheets, specs and
local services; **no internet**, **6 hours per task**, three runs. Automated
checks score quantitative outputs; an LLM judge grades open-ended work.

**Why it fits.** The official description is our thesis verbatim: “an AI agent can
produce a convincing engineering memo and still use the wrong units, overlook a
drawing or leave the required spreadsheet unfinished.” The competition rewards
the harness, not the model: inspect evidence, use tools, check calculations,
deliver usable files. Frontier models on their native harnesses score 0.27–0.51,
so there is room above and below; a disciplined harness with units, constraint
checks and file verification is exactly what we have been designing.

**Submission shape.**
- Stage 1: a 2,000-word writeup, five rubric criteria, three figures
  (architecture, score-vs-cost, dev failure breakdown); a GitHub repo at a
  pinned commit with `submission.yaml`, `serve.py`/`harness.py`, an
  **Apache-2.0 or MIT** licence, and the dev `artifacts/`; the kit's
  `check` and `artifacts` must pass.
- Stage 2: the top 100 run the 56 hidden tasks; Google injects the model API;
  custom keys, external model servers and web search are blocked.

**Constraints that decide the plan.**
- The submission repo must be Apache-2.0/MIT. Our agentic-loop is proprietary /
  AGPL; the entry must be a **new clean harness** using the same ideas, not a
  copy. That is also a good paper artefact.
- The model is Google-sponsored; this is **not** a FerrisRes showcase. It tests
  harness design, which is the part we claim is half the result.
- Engineering domain: the harness needs Python tooling (units, spreadsheets,
  drawings, solvers, constraint checks) and must deliver files, not prose.

**A 40-day plan.**
1. Clone `google-research/ple` and the submission template; run the 29 dev tasks
   on a sponsored model; record baseline score, tokens, wall clock, failures.
2. Build the harness loop: plan → inventory the project folder → extract
   constraints from drawings/specs → compute with checked units → verify each
   quantitative output → assemble the required deliverables.
3. Add the evidence discipline: every number cites a file and a check; every
   assumption is listed; a failed check changes the plan, not the prose.
4. Iterate on the dev failure breakdown (the required figure) — that is the
   writeup's core.
5. Submit early; a qualified Stage 1 entry beats a perfect unsubmitted one.

**Realistic goal.** A strong Stage 1 and a respectable Stage 2, not a prize. The
secondary value is an external benchmark of the harness thesis and a publishable
“what the harness adds” result with a public repo and a cost curve.

## The others, briefly

- **Gemma 4 Developer Agent** closes in ~8 days with 6.6k teams; only enter if a
  coding harness can be adapted in a day. Low expected value.
- **Generalist Gaming Agent** needs real-time vision/control; our stack is text
  and tools. A stretch, not a fit.
- **Name That LLM** is a probing/classification problem; our embeddings and
  decision head could help, but it is not the thesis.
- **Santa / Squares** are pure optimisation; an LLM harness is the wrong tool.

## What this does not do

It does not train FerrisRes, does not use our model, and does not replace the
training work. It is the external test of the harness claim — and the only
competition on the list where our existing design is the differentiator.
