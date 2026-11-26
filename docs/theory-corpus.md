# Theory corpus

Status: **inventory, 2026-11-26.** The rule is already set: **theory is trained,
facts are retrieved, effects are executed.** This lists the theories worth
training, in the order they earn their place, with the data shape and the gate
for each.

## What counts as theory

A theory is **durable, transferable and checkable**:

- durable: it does not change with a release (unlike an API or a price);
- transferable: it applies to code, designs and tasks the model has not seen;
- checkable: an example can be verified by a test, a measurement, a proof step or
  a tool.

A fact ("cargo 1.100 stabilised X") is retrieved. An effect ("run the test") is
executed. A **theory card** is:

```
principle | why it holds | when it applies | failure mode | a check or tool
```

## Tier 1 — used every day by this agent

| theory | what is trained | check |
|---|---|---|
| **Program semantics** | types as propositions, sum/product, ownership and borrowing, lifetimes, variance, generics vs trait objects, monomorphisation, coherence | the compiler; a type-level test |
| **Debugging method** | reproduce → isolate → hypothesise → instrument → falsify → fix; one variable at a time; the difference between correlation and cause | the failing test before and after |
| **Complexity and performance** | Big-O, amortisation, roofline, memory hierarchy, cache lines, false sharing, Amdahl, Little's law, tail latency | a measurement, not a guess |
| **Concurrency** | data races vs race conditions, memory ordering, locks vs channels, actor ownership, cancellation safety, deadlock/livelock, backpressure | loom/miri, a stress test |
| **Data and storage** | ACID, isolation levels, MVCC, WAL, indexes, query plans, caching and invalidation, schema evolution | a query plan or a transaction test |
| **Distributed systems** | partial failure, at-least/exactly-once, idempotency keys, leases and fencing, clock skew, retries with backoff, sagas/outbox | fault injection, a duplicate-delivery test |
| **API and interface design** | naming, compatibility, versioning, additive change, error contracts, idempotency, least surprise, boundary vs internal | a semver/compatibility check |
| **Requirements** | ambiguity, acceptance criteria, non-goals, constraints, the smallest behaviour, the test that proves it | a task that passes the loop |
| **Decision under uncertainty** | expected value, cost of a wrong answer, calibration, when to escalate, optimal stopping, evidence sufficiency | a labelled routing/escalation set |
| **Technical writing** | audience, structure, claim vs evidence, counts not percentages, "verified / implemented / designed", German register | a review rubric |

## Tier 2 — high leverage, adjacent

| theory | what is trained | check |
|---|---|---|
| **Cryptography use** | confidentiality/integrity/authenticity, AEAD, nonces, KDFs, signatures, key management, constant time, randomness; never implement crypto | a misuse checklist; the library's test |
| **ML theory** | bias/variance, regularisation, optimisation, loss choice, precision/recall/calibration, leakage and contamination, distribution shift, statistics and seeds | the eval record |
| **Transformer internals** | attention, KV cache, RoPE, quantisation, MoE, LoRA, distillation, sampling, perplexity/CE | a parity or ablation test |
| **Reliability** | SLO/SLI/error budgets, blast radius, graceful degradation, circuit breakers, observability (logs/metrics/traces), incident command, postmortem | an incident drill |
| **Numerical methods** | floating point, stability, conditioning, error propagation, units and dimensional analysis | a reference computation |
| **Statistics** | sampling, confidence intervals, significance vs effect size, multiple comparisons, causal vs correlational | a reproducible analysis |
| **Information literacy / source navigation** | which registry owns which object, identifier systems, access patterns, licence families, no invented endpoints | `data-source-navigation.md` |
| **Licensing and privacy principles** | copyright vs licence, permissive vs copyleft, patent grant, attribution/NOTICE, CC variants, GDPR principles (minimisation, purpose, retention) — identify when to ask a human | a licence check; the provenance register |
| **Control and supervision** | feedback loops, stability, deadbands, budgets, restart policy, one authority, precedence | the control-plane trace |

## Tier 3 — domain verticals

| theory | why | status |
|---|---|---|
| **Engineering fundamentals** (units, statics, circuits, thermodynamics, safety factors) | the PLE competition; a general engineering assistant | optional, after Tier 1–2 |
| **Education / learning science** | the education impact track; explaining to a person | optional |
| **Personal-assistant conventions** | calendars, email, privacy, German formal/informal, note-taking | small, product-specific |
| **Legal method (EU/DE/UK)** | IRAC, hierarchy, temporal application, citation, jurisdiction, advice boundary — the law itself is retrieved | `legal-information.md`; needs a lawyer review |
| **Business and unit economics** | token cost, build vs buy, licence cost | small |

## What is not theory

- **Facts that move:** API signatures, prices, CVEs, model lists, standards
  clauses. Retrieve and cite.
- **Professional advice:** medical, legal, financial. Explain the principle, then
  defer to a qualified human and a tool.
- **Offensive security:** exploit chains and malware. The theory of why a class
  is dangerous is enough.
- **Private data and transcripts.**
- **A memorised standard** (a building code, a tax table). Retrieve the current
  text, cite the version.

## Data shape

| shape | example | gate |
|---|---|---|
| theory card | principle + why + when + failure + check | a human review; no claim without a check |
| worked example | a small code or design case where the theory decides | the test/measurement |
| contrastive pair | the plausible-but-wrong application vs the correct one, with the reason | the check distinguishes them |
| counter-example | when the principle does **not** apply | a review label |
| tool-grounded case | the principle predicts; the tool confirms or refutes | the tool output at a revision |

Data sources follow `source-index.md`: our own authored cards (CC0), permissive
docs and books summarised in original words (MIT/Apache; never verbatim
share-alike), and permissive code for examples. The GoF book, OWASP texts and
similar are summarised, not copied.

## Gates

| gate | pass |
|---|---|
| claim without a check | **0** on the theory suite |
| outdated fact stated as theory | **0** (retrieve instead) |
| principle applied where its "when not" holds | **0** on the counter-example set |
| explain-why vs recite | the model states the mechanism, not only the name |
| transfer to an unseen domain | >= 70% on a held-out task family |
| professional-advice boundary | defers or refuses, 100% |

## Order

1. **Program semantics, debugging, complexity, concurrency** — the daily loop.
2. **API/requirements/writing** — the output is a patch, a review and a report.
3. **Data, distributed systems, reliability** — the harness and the product are
   distributed even when the model is local.
4. **Decision under uncertainty and control** — the escalation and budget
   behaviour.
5. **Crypto use, ML, numerical methods, statistics** — high leverage, narrower.
6. **Licensing/privacy principles** — needed before a release.
7. **Domain verticals** — only with a concrete product or competition.

A 111M model will hold the vocabulary and the procedure, not deep expertise. The
value is recognition, correct application and knowing when to call the tool; the
eval must measure that, not trivia.
