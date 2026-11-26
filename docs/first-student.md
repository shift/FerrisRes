# First student

Status: **decision, 2026-11-26.** The first student is the **shipping model**, not
a pipeline proof: `frs-s-0.1.0-beta`, 111M, 32k tokenizer, 8k context, ternary
experts, top-1 MoE, MQA, 3-bit KV — shaped for **this machine**.

## Why this shape

The i5-6500 + HD 530 profile is the one the Small tier was designed for:

| constraint | Small |
|---|---|
| loop core fits 6 MB L3 | 4.6 MB |
| disk | 36 MB |
| KV at 8k (3-bit) | 2.7 MB |
| decode target (6.3 GB/s, 2.5× gap) | ~65 tok/s |
| prefill | cache-dependent; the prefix cache is implemented |

Do not redesign before measuring. If the gates fail, change **one lever** (GQA
4:1, loop visits, quantisation) and re-measure; `hardware-shapes.md` has the
order.

## The training path

1. **Release corpus** (permissive) 300–500M tokens, 32k tokenizer.
2. Pretrain/SFT with the replay+KL term (`tasks/replay-kl.md`) so general ability
   does not collapse.
3. Optional **text-aligned KD** from Qwen3-1.7B after the 32k tokenizer exists;
   it is not the first dependency.
4. Export to safetensors/GGUF, load with `infer`, measure on this host.
5. Eval suite, model card, NOTICE, release checklist.

No blocks+LoRA until the CLI gradient wiring is done (`tasks/wire-distill-grad.md`);
the first student trains the plain config that is verified.

## The two corpora

| corpus | contents | use |
|---|---|---|
| **release** | permissive only (MIT/Apache/BSD/CC0/CC-BY/ODC-By) | ships as `frs-s-0.1.0-beta` |
| **research** | release + xLAM (NC), Dolly (SA), CulturaX (gated) | a comparison run only; **not deployed, not redistributed** |

The research corpus exists to answer "does the extra data improve quality?" The
shipping model cannot use it: NC and share-alike data do not become compatible
with an Apache-2.0 weight release because the run was internal, and a model
cannot unlearn them. If the research result is compelling, the release model is
**retrained** on the release corpus, not filtered.

## Gates for "shipping"

- eval suite, harness integration, retention, modes and security (the blocking
  suites);
- export loads in `infer` and in a clean `.deb`;
- measured decode and prefill on this host, recorded;
- model card, NOTICE, licence audit and release checklist;
- no NC/SA/gated source in the release corpus.

Until all of that, the run is a candidate, not a release.
