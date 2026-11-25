# Armor L1: train the scanner and enable it

Status: **task specification, not implemented.** L1 and L2 are currently
hash-initialised placeholders (`L1Scanner::new`, `ProbeHead::new`); only L0
(regex PII) and L3 (output redaction) are real. This document is the plan for
making L1 real and turning it on.

## 1. Scope

In scope: the L1 neural injection scanner — data, model, training, evaluation,
weight loading, runtime policy, evidence, tests.

Out of scope: L2 (hidden-state probe), languages beyond English and German,
and any claim of adversarial robustness. L2 stays disabled.

## 2. Current state

| item | fact |
|---|---|
| `ArmorGuardConfig` defaults | `vocab_size 1000`, `hidden_dim 256`, `num_layers 4`, `threshold 0.7` |
| weights | **none** — every tensor comes from a `sin`-based hash in `with_config` |
| tokenizer | `tokenize()` hashes into a 1000-entry vocabulary |
| CLI | `--armor` is opt-in; `armor.toml` enables all four layers |
| verdict | `verify_input` returns `Block` and stores the **full input text** in `ViolationRecord` |
| cost | ~5 M params, CPU forward, no GPU needed |

Two defects must be fixed as part of this work, independent of training:

1. Random initialisation must not be reachable from a production constructor.
   Rename it `new_random()` and gate it behind `#[cfg(test)]`.
2. `ViolationRecord.input` must never hold raw untrusted text. Store
   `content_sha256`, `byte_len`, `source`, `score`, `threshold`, `action`.

## 3. Data

### 3.1 Label schema

One JSONL line per example:

```json
{"text": "...", "label": "injection", "lang": "de", "source": "redteam:2026-11",
 "licence": "CC0-1.0", "redacted": true}
```

`label ∈ {safe, injection}`. `source` is a family, not a URL, so splits can be
made by family and leakage is detectable.

### 3.2 Sources

| family | use | licence rule |
|---|---|---|
| public injection corpora (e.g. `deepset/prompt-injections`, `jackhhao/jailbreak-classification`, `Lakera/gandalf_ignore_instructions`, `xTRam1/safe-guard-prompt-injection`) | training candidates | **verify each card**; Apache-2.0 / MIT / CC-BY / CC0 only |
| our own red-team strings (harness, tools, workflows) | training + evaluation | internal, provenance recorded |
| benign tool output (JSON, logs, diffs, shell output, file content) | negatives | derived from our own runs, redacted |
| benign German text and instructions | negatives | permissive corpora only |
| security documentation and red-team reports | **hard negatives** | permissive only |

Datasets whose licence cannot be established may be used for **evaluation
only**, never for training. This is the same rule as the model data: if it
cannot name its licence, it does not train a shipped artefact.

### 3.3 Composition

- 60 k train / 7.5 k dev / 7.5 k test, split by source family.
- ≥ 30 % German, ≥ 30 % hard negatives, 50/50 label balance after weighting.
- Include obfuscations in dev/test only (base64, zero-width characters,
  homoglyphs, translated injections) so the test measures something honest.
- German injections are both translated (permissive model, human-checked) and
  natively written; "Ignoriere alle vorherigen Anweisungen" must be in the
  set.
- Deduplicate by normalised text hash across all families.

### 3.4 Manifest

`assets/armor/l1-data.toml`: one entry per family with `name`, `licence`,
`revision` (commit or dataset hash), `n_examples`, `sha256` of the built
JSONL. The build step refuses an entry without a licence.

### 3.5 Teacher (optional, for distillation)

`protectai/deberta-v3-base-prompt-injection-v2` — Apache-2.0, 184M params,
English-only. Published on 20k unseen prompts: accuracy 95.25%, precision
91.59%, recall 99.74%, F1 95.49%. The v1 model is superseded; do not use it.

Two consequences:

- **Its operating point is high-recall, lower-precision.** The gate below
  (FNR ≤ 5% at FPR ≤ 1%) is stricter on false positives than the teacher's
  published numbers. If the student cannot reach it, either move the
  threshold and accept more false positives under quarantine, or report the
  failure and leave L1 off. Do not loosen the gate silently.
- **It is English-only and does not detect jailbreaks**, and it
  false-positives on system prompts. It is a label source for tool/RAG
  content, not a safety oracle, and it is not a source of German labels —
  those must be translated and human-checked as in §3.2.

## 4. Model

Keep the encoder shape, change the input representation.

| field | value | reason |
|---|---|---|
| layers | 3 | injection detection is shallow; 4 is not needed |
| hidden | 192 | ~2.5 M params, < 5 ms on the i5 |
| heads | 4 | head_dim 48 |
| ffn | 768 | 4× hidden |
| max_seq | 256 | tool results are truncated before scanning |
| vocab | byte-level 256 + 4 096 hashed word buckets | multilingual without a tokenizer asset |
| pooling | mean over non-pad positions | |
| head | linear → sigmoid | BCE |

The tokenizer is part of the asset: hash function, bucket count and
normalisation (lowercase, NFKC, whitespace collapse) are versioned in the file
header. A scanner whose tokenizer changed silently is a different model.

## 5. Training

New module `src/training/armor_trainer.rs`:

- bidirectional encoder (no causal mask), mean pool, sigmoid classifier;
- BCE with class weights, Adam, 3–5 epochs, early stop on dev FNR at
  FPR ≤ 1 %;
- a finite-difference gradient check on a tiny config (hidden 16, 1 layer,
  seq 8), every tensor, the same bound as `f32_trainer`;
- a loss-decrease test on a synthetic balanced batch;
- deterministic seed.

Compute: a T4 run is minutes; CPU is hours. Use a Kaggle GPU session for the
full run and CPU sessions (no GPU quota) for data preparation. One build at a
time.

## 6. Evaluation and the enable gate

Report per language and per source family:

| metric | gate |
|---|---|
| FNR at FPR ≤ 1 % | **≤ 5 %** |
| AUROC | ≥ 0.95 |
| calibration (ECE) | ≤ 0.05 |
| recall gain over L0 alone at the same FPR | ≥ 20 points |
| German FNR − English FNR | ≤ 5 points |

Threshold is chosen from the curve to satisfy the gate and written into the
config. **0.7 is not a threshold, it is a default.**

If the gate fails, L1 stays disabled and the numbers are published in the
model card. A scanner that is worse than regex must not be shipped as
security.

## 7. Asset format and loader

`assets/armor/l1.bin`:

```
magic      "FRAL" (4 bytes)
version    u16
tokenizer  { id: u16, buckets: u32, normalisation: u8 }
config     { layers: u16, hidden: u16, heads: u16, ffn: u16, max_seq: u16 }
tensors    f32 LE in a fixed order:
           token_embed [vocab, hidden]
           pos_embed   [max_seq, hidden]
           per layer:  q,k,v,o [hidden,hidden], ffn1 [hidden,ffn],
                       ffn2 [ffn,hidden], ln1,ln2 [hidden]
           classifier  [hidden], bias f32
sha256     appended, verified against the manifest
```

`L1Scanner::load(path) -> Result<Self>` verifies magic, version, shapes and
hash, and returns a named error on any mismatch. `ArmorLayer::with_config`
returns `Err(ArmorError::MissingWeights)` when `l1_enabled` and no path is
configured. No silent fallback.

## 8. Runtime enablement

```toml
[armor]
enabled     = true
mode        = "quarantine"          # quarantine | block | flag
sources     = ["tool", "rag", "web", "file"]
l0_enabled  = true
l1_enabled  = true
l1_weights  = "assets/armor/l1.bin"
l1_threshold = 0.62                 # from the measured curve
l2_enabled  = false
l3_enabled  = true
```

Policy:

- **Sources**: tool results, RAG documents, web fetches, untrusted file reads,
  Telegram media captions. The user's own message is not blocked; it may be
  scanned and recorded.
- **Quarantine, not drop**: the text is replaced by
  `[withheld: armor.l1 score=0.83 source=tool:web_fetch sha256=…]`, the model
  sees the notice, and the turn continues. `block` refuses the turn; `flag`
  passes with a marker.
- **Per principal**: the child principal's inbound is scanned; the adult's
  user messages are allowed but recorded.
- **Evidence**: one `armor.verdict` record per scan with `source`, `score`,
  `threshold`, `action`, `content_sha256`, `byte_len`. Never the text.
- **Failure**: scanner error on an untrusted source fails closed (quarantine);
  on a user message it fails open and logs.
- **Authority**: Armor is a detector. Tool allowlists, least privilege and the
  control plane remain the boundary.

## 9. Tests

| test | asserts |
|---|---|
| loader rejects | bad magic, version, shape, hash — each with a named error |
| no random production init | `L1Scanner::new_random` is `#[cfg(test)]`; a production build cannot call it |
| startup refusal | `l1_enabled` without weights fails to start |
| golden corpus | 200 known injections (EN/DE) caught at the configured threshold |
| benign corpus | 1 000 benign examples (EN/DE, tool output, security docs) quarantined no more than the stated FPR |
| integration | a malicious tool result is quarantined, the model sees the notice, and the raw text is absent from logs and evidence |
| threshold | changing `l1_threshold` changes the verdict at the boundary |
| latency | ≤ 5 ms per 256 tokens on the i5 (`#[ignore]`, run manually) |

## 10. Tasks

| # | task | acceptance |
|---|---|---|
| T1 | corpus build + manifest | 60 k/7.5 k/7.5 k, every entry licensed, German ≥ 30 %, deduped |
| T2 | `armor_trainer` + gradient check | finite-difference check passes on every tensor; loss decreases |
| T3 | train + threshold + eval report | gate table above, numbers published |
| T4 | loader + startup validation | refuses missing/invalid weights; `new_random` test-only |
| T5 | runtime hook + quarantine + evidence | integration test passes; no raw text stored |
| T6 | golden corpus + latency | tables above pass |

T1 and T2 can run in parallel. T5 depends on T4. T6 depends on T3 and T5.

## 11. Risks

- **False positives on German and technical text** — the hard negatives exist
  for this; measure per language.
- **Distribution shift** — tool output changes; re-evaluate quarterly.
- **Evasion** — a 2.5 M classifier will be evaded by determined attackers.
  This is a filter, not a proof.
- **Licence contamination** — evaluation-only datasets must never enter
  training; the manifest enforces it.
- **Latency on a Pi 5** — measure; if over budget, scan the first 256 tokens
  only and record the truncation.

## 12. Open questions

1. Hashed word buckets or a small BPE trained on the corpus? Hashing avoids a
   second asset; BPE is more accurate. Start hashed, measure, then decide.
2. Does L1 run before L3 on output, or only on input? Input only in v1.
3. Quarantine or block for the adult's tool results? Quarantine in v1, with
   the notice visible to the user.
