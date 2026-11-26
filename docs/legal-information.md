# Legal information (EU, Germany, UK)

Status: **design, 2026-11-26.** Scope is **legal information and drafting
support**, not legal advice and not a reserved legal service. The model must know
**where** the law is, cite it at a date, and stop when the question needs a
lawyer.

## The boundary, first

| the model may | the model may not |
|---|---|
| explain a concept and point to the provision | tell the user what to do in their case |
| summarise a retrieved statute or judgment | state a rule from memory |
| draft a checklist, a clause outline, a letter structure | draft a document for filing without a lawyer's review |
| spot issues and name the jurisdiction | decide a dispute or predict an outcome |
| compare EU / DE / UK positions, with citations | treat them as interchangeable |
| say "this is information, not advice; consult a qualified lawyer" | charge for legal advice |

Two statutory constraints matter for the product, not just the prompt:

- **Germany:** the Rechtsdienstleistungsgesetz (RDG) restricts legal services for
  remuneration; "information" and a lawyer-reviewed product are different from
  "Rechtsberatung".
- **UK:** the Legal Services Act 2007 reserves certain activities (conduct of
  litigation, advocacy, reserved instrument activities, probate) to authorised
  persons.

A lawyer partner and a jurisdiction-specific review are prerequisites, not
polish.

## The source registry (what "knows the sources" means)

| jurisdiction | primary sources | identifiers | update |
|---|---|---|---|
| **EU** | EUR-Lex: treaties, regulations, directives, decisions; consolidated versions; OJ | CELEX, ELI, OJ reference | consolidations |
| **EU case law** | CURIA | case number, ECLI | per judgment |
| **EU guidance** | EDPB, Commission guidance | document id | per publication |
| **Germany** | gesetze-im-internet.de (official consolidated), Bundesgesetzblatt | §, Gesetz abbreviation, BGBl reference | amendments |
| **DE case law** | BGH, BVerfG, EuGH via official sites | Az., ECLI | per decision |
| **DE guidance** | BfDI, ministry guidance | document id | per publication |
| **UK** | legislation.gov.uk, "as at" versions | statute, section, SI number | revised versions |
| **UK case law** | Find Case Law (The National Archives), UKSC | neutral citation | per judgment |
| **UK guidance** | ICO, GOV.UK | document id | per publication |
| **Devolved** | Scotland, Wales, Northern Ireland | separate legislation | separate |

Each registry entry records: jurisdiction, source, official URL, identifier
scheme, licence, update cadence, language(s), and a **retrieval date**. The
registry is data, versioned, and reviewed like code.

## What is trained, what is retrieved

**Trained (procedure):**

- legal method: issue spotting, IRAC (issue, rule, application, conclusion),
  hierarchy of norms, lex specialis, temporal application, proportionality;
- citation form per jurisdiction;
- jurisdiction awareness: EU vs member state vs UK; Germany federal vs Land; UK
  England/Wales vs Scotland vs NI; post-Brexit retained EU law and the Windsor
  Framework;
- terminology (German legal German; official EU language versions);
- the boundary: information vs advice, when to stop, when to refer;
- drafting structure for a checklist or a clause outline.

**Retrieved (never memorised):**

- the text of a provision at a date;
- a judgment's holding;
- amendments, transitional provisions, repeal;
- guidance documents.

A statute is a changing fact. Training it into the weights guarantees a
confident answer about a repealed rule.

## Tools

| tool | does | check |
|---|---|---|
| `legal_sources` | list the registry entries for a jurisdiction | every entry has a licence and an update date |
| `legal_search` | query official sources, filtered by jurisdiction and `as_of` date | returns identifiers, not prose only |
| `legal_read` | read a provision by CELEX/ELI/§/section at a date | the text is the official one at that date |
| `legal_check_citation` | does this citation exist at this date? | **the anti-hallucination gate** |
| `legal_diff` | what changed between two dates | amendments and transitional rules |
| `legal_define` | terminology in DE/EN | the source for the definition |

The tools return **identifiers and dates**; the model renders the explanation.
Index the texts (BM25 + embeddings) with a version; do not train them.

## Data and licences

| source | licence | rule |
|---|---|---|
| German official works (laws, ordinances, official decisions) | §5 UrhG: not protected | index; attribute the source |
| EUR-Lex | reuse under the Commission's reuse decision; verify | index; do not imply endorsement |
| UK legislation.gov.uk | Open Government Licence v3.0 | attribute; check the "as at" date |
| UK case law via Find Case Law | OGL (verify per document) | attribute |
| Commentaries, headnotes, paywalled databases | copyrighted | **do not use** |

No user matter, no client document, no legal advice transcript in the corpus.
Synthetic scenarios are authored; real matters are never training data.

## Evaluation gates

| gate | pass |
|---|---|
| citation that does not resolve at the stated date | **0** |
| provision invented (wrong article/paragraph) | **0** |
| wrong jurisdiction presented as the user's | **0** |
| stale law after an amendment | **0**; `legal_diff` must fire |
| answer to a "what should I do" question without the advice boundary | **0** |
| reserved legal activity performed without a lawyer | **0** |
| language mismatch (German term explained in English only) | reported |
| source missing a licence or date | **0** |

The first two are the whole game: a legal assistant that invents Art. 6(1)(f)
GDPR is worse than one that says "I cannot verify that".

## Honest limits

- **A 111M model cannot be a lawyer.** It can retrieve, explain, compare and
  draft an outline; the judgement, the strategy and the filing need a qualified
  person.
- **The law is multilingual and multi-version.** EU acts are authentic in every
  official language; Germany and the UK have their own consolidation practices.
- **The product is information.** The moment it is sold as legal advice, the RDG
  and the Legal Services Act apply, and the answer is a lawyer, not a model.

## What this would take

1. A source registry and an indexer per jurisdiction (a data project, not a
   training one).
2. `legal_*` tools with citation and date checks.
3. A synthetic legal-method corpus and an evaluation set with citations.
4. A lawyer review of the boundary, the registry and a sample of answers, per
   jurisdiction.
5. A product decision: information only, or a lawyer-supervised service.
