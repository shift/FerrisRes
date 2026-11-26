# Embedding head

Status: **design, 2026-11-26.** The runtime already has a 768→256 projection head
contract (`agentic-loop-embeddings/src/heads.rs`: `BASE_DIM = 768`,
`HEAD_DIM = 256`, `U_perp^T U_perp = I`). Small's hidden is 768. A generated
model can therefore carry an embedding mode without a second backbone.

## The question

Can the model that generates also embed? Yes, in three ways, with very different
risk to generation:

| option | what trains | generation risk | quality |
|---|---|---|---|
| **A. head-only, backbone frozen** | the projection (and pooling) | **none** — base weights untouched | limited by the generative representation |
| **B. head + LoRA** | projection + a small adapter | none while the adapter is unloaded; some if merged | close to a dedicated embedder |
| **C. joint multi-task** | backbone + head, LM loss + contrastive | real; needs a gate | best in-domain |

Recommendation: **A first**, because it cannot damage generation and it is cheap;
**B** if A is close but under the baseline; **C** only with a generation gate.

## Architecture

| piece | decision |
|---|---|
| pooling | last-token for causal use, plus a `[embed]` sentinel form; mean-pool only over the content span, never the instruction |
| projection | `hidden → 256`, L2-normalised, matching the existing `HEAD_DIM` contract |
| Matryoshka | train prefixes `[64, 128, 256]` so storage can be truncated |
| instruction | a short task prefix (`query:`, `passage:`, `code:`) rendered by the shared template |
| heads | the LM head stays tied to the 32k vocab; the embedding head is a separate tensor |

The 256 dimension is not arbitrary: the existing index and storage path already
use it, so a FerrisRes head is a drop-in replacement for the current embedding
model rather than a new index.

## Training

**A. Head-only.** Freeze every base parameter. Train the projection with
InfoNCE (temperature 0.05–0.1), in-batch negatives, and 1–2 hard negatives per
query from BM25 and the current embedder. Because the base is frozen, the
generation path is byte-identical; the gate is only quality.

**B. Head + LoRA.** Add a low-rank adapter in the last layers and train it with
the head. The adapter is a separate artefact; generation loads the base without
it. Score the same pairs; stop when the gain over A is below the eval noise.

**C. Joint.** Only if A and B plateau. Add LM loss at a small weight and gate
generation hard: perplexity, tool-call validity and the eval suite must not
regress. Interference is real; a model trained for two objectives is usually a
little worse at both.

Data (licence-clean, from `source-index.md`): harness query → file/section, code
docstring → code, tool description → schema, doc-QA, soul → task, entity →
mention. Hard negatives from lexical retrieval and from the current embedder.
No raw transcripts, no personal data, no eval-set overlap. German pairs are part
of the mix, not an afterthought.

Optionally distil **scores** from the current embedder (cosine/ranking), not
weights; verify its licence first.

## Artefact and naming

The head is an **adapter tied to a model version**, not a new backbone:

```
frs-s-0.1.0-beta.embed-head/
  head.safetensors        projection (+ LoRA if B)
  config.json             base_model, base_sha256, dim, matryoshka, pooling, template
  SHA256SUMS
```

A standalone embedding model remains the specialist role `frs-embed`; the head
is how an existing generative model gains an embedding mode. The config records
`base_model` and `base_sha256`, so a head cannot be loaded onto different weights
by accident.

Naming (`naming.md`): this is an adapter artefact, not a model id; do not call it
`frs-s-embed`.

## Runtime

- `ferrisres embed --model frs-s --head <path> --text …` returns the 256-vector;
  a batch mode and an int8/binary storage option.
- The engine selects `embed` mode from the head artefact; the LM path is
  unaffected.
- Storage at 256 dims: 1 KB/text at f32, 256 B int8, 32 B binary; Matryoshka at
  64 dims is 256 B f32.
- The index does not change; only the vectors do.

## Eval gates

| gate | pass |
|---|---|
| domain recall@10 on harness queries vs the current embedder | >= baseline, or the reason is recorded |
| code retrieval (query → file/symbol) | >= baseline on the held-out set |
| German retrieval | reported; do not hide a monolingual win |
| Matryoshka curve at 64/128/256 | reported; 64 dims keeps >= 90% of 256's recall |
| generation regression with the head loaded | **zero** for A; gated for B/C |
| perplexity / tool-call validity after any joint training | no regression |
| latency per 1k texts, CPU and T4 | recorded |
| contamination | zero eval-set pairs in training |

## Cost

The head is `768 × 256 ≈ 197k` parameters, ~0.18% of Small. Head-only training
on 1M pairs is hours on a T4 and feasible on CPU; it is a good Kaggle job because
it needs no teacher and no large corpus. B adds a LoRA adapter, still small.

## What not to do

- **Mean-pool a causal LM and call it an embedder.** It is the weakest version;
  the sentinel/last-token plus a trained head is the minimum.
- **Reuse the LM head as the embedding.** The vocab projection is not a semantic
  space.
- **Joint-train without a generation gate.** The failure is silent: retrieval
  improves and the assistant gets worse.
- **Train on harness queries only.** It overfits the phrasing of our own tools;
  mix in general and code pairs.
- **Copy the current embedder's weights.** Distil scores, respect its licence.

## Phases

1. Head-only on the frozen Small, 256 dims, domain + code pairs; compare to the
   current embedder.
2. Matryoshka truncation and int8/binary storage.
3. LoRA only if phase 1 is close but under.
4. Joint training only with the generation gate above.

Phase 1 is the decisive one: if a frozen 111M backbone plus a 197k head cannot
retrieve our own domain, the answer is no, and it is cheap to learn.
