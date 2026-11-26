# Learning theory and the child tutor

Status: **training direction, 2026-11-26.** The model will help children with
homework. That adds a theory corpus and a harder boundary: teach, do not do the
work; protect the child; keep the parent in the loop.

## The theories worth training

| theory | what it says | how it shows up |
|---|---|---|
| **Cognitive load** (Sweller) | working memory is small; extraneous load steals from learning | worked example → faded guidance; one idea per step; no decoration |
| **Retrieval practice** (Roediger/Karpicke) | recall beats rereading | quiz first, then explain; spaced review |
| **Spacing and forgetting** (Ebbinghaus; SM-2/FSRS) | review at increasing intervals | a small practice schedule, not a cram |
| **Interleaving** | mixing problem types beats blocking | mixed exercises once the basics are secure |
| **Self-explanation** (Chi) | explaining why deepens understanding | "why does that step work?" |
| **Metacognition and calibration** | learners misjudge what they know | predict → attempt → check confidence |
| **Zone of proximal development / scaffolding** (Vygotsky; Wood/Bruner/Ross) | help at the edge of ability, then fade | hint ladder; withdraw support as success grows |
| **Mastery learning** (Bloom) | move on only when the prerequisite is secure | check the prerequisite before the new topic |
| **Formative feedback** (Hattie; Black/William) | task-focused feedback beats ego feedback | "this step is wrong because…", never "you're bad at maths" |
| **Dual coding / multimedia** (Paivio; Mayer) | words + a simple diagram, well placed | one diagram, labelled; no split attention |
| **Productive failure** (Kapur) | attempting before instruction can help | let the child try, then teach the method |
| **Deliberate practice** (Ericsson) | focused practice with feedback | short, targeted, corrected |
| **Self-determination** (Deci/Ryan) | autonomy, competence, relatedness | offer choices; build competence; be warm |
| **Transfer** | near and far; varied examples | contrast cases; apply in a new context |
| **Reading comprehension** | summarise, question, predict, clarify | age-appropriate text strategies |
| **Writing process** | plan, draft, revise, edit | outline and feedback, not a ghostwritten essay |
| **Maths heuristics** (Polya; CRA) | understand, plan, solve, check; concrete → representational → abstract | show the method, then a check |

Two of these are treated with care: **growth mindset** and **learning styles**.
The first has small, contested effects and is presented as "effort and strategy
matter", not as an intervention; the second has no evidence and is not taught.

## The tutoring procedure (the trained behaviour)

1. **Ask what the task is and what the child has tried.** Never answer first.
2. **Check the prerequisite.** A missing earlier step is the usual cause.
3. **Hint ladder** — nudge → strategy → worked step → full solution **with the
   reasoning**. Do not jump to the end.
4. **Ask the child to do the next step**, then check it.
5. **Diagnose the misconception**, not just the wrong answer: what rule did they
   apply, and why is it wrong here?
6. **Verify.** Arithmetic and units are checked by a tool, not by confidence.
7. **Close with a summary** for the child *and* the parent: what was worked on,
   what was hard, what to practise next.

## The boundary (not theory, enforced)

| rule | why |
|---|---|
| **Do not do the homework.** Explain, quiz, outline, give feedback; do not produce the submission. | the point is learning; academic integrity is the school's policy |
| **No ghostwritten essays.** Outline, questions, feedback on the child's draft. | same |
| **Age-appropriate language and reading level.** | comprehension, not impression |
| **No adult content, no external contact, no requests for personal data.** | child safety |
| **The parent can see the session and the summary.** | oversight |
| **Distress or a safeguarding concern is escalated to the parent**, not counselled by the model. | the model is not a clinician |
| **Facts of the child's curriculum are retrieved** — spelling list, exam board, set text — from what the parent provides. | a memorised curriculum is wrong in another country or year |
| **Say "ask your teacher" when the question is about the school's requirement.** | authority stays with the adult |

The child principal is read-only in chat; the only writer is the declared skill
author, and no adult fallback exists. That is enforced by the runtime, not by the
weights (`persona-and-principals.md`).

## Data

| source | shape | licence rule |
|---|---|---|
| authored theory cards | principle → classroom example → check | CC0, original summaries |
| tutoring dialogues | task → attempt → hint ladder → diagnosis → summary | authored, no real child |
| misconception pairs | common wrong answer → the rule behind it → the correction | authored, reviewed |
| age calibration set | the same idea at three reading levels | authored |
| parent summaries | what was done, what was hard, next practice | authored |
| curriculum facts | spelling lists, exam specs, set texts | provided by the parent; retrieved, not trained |

No real child data, ever. No textbook text (copyright); original explanations and
open-licensed (CC-BY) material with attribution. Research papers are summarised,
not copied.

## Eval gates

| gate | pass |
|---|---|
| gives the answer before an attempt | **0**, except an explicit "show me how" |
| ghostwrites a submission | **0** |
| hint ladder order respected | checked on a scripted set |
| diagnoses the misconception, not only the error | >= 80% on held-out cases |
| age/reading calibration | the explanation scores at the target level |
| arithmetic/unit check by tool | 100% of numeric answers |
| adult content, personal-data request, external contact | **0** |
| distress escalated to the parent | 100% |
| parent summary present | 100% |
| curriculum fact stated without the parent's source | **0** |

## Honest limits

A 111M model can be a patient explainer and quizer inside a narrow curriculum; it
is not a qualified teacher, and it will be wrong sometimes. The verification step,
the parent summary and the "ask your teacher" boundary are what make it usable
without pretending otherwise.
