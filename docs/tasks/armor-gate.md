# Armor gate

Status: **diagnosed at `0a95bc6`; the gate fails.** The collapse was a trainer
length mismatch, not the corpus or the GEMM. Fixing it restores separation, but
the detector is not shippable at a 1% false-positive budget.

## The root cause

Training scanned a **sampled short prefix**; evaluation scans **128 tokens**. The
model learned a cue present in the first few tokens, then mean-pooled over a
sequence three times longer at evaluation, diluting it.

| trainer | train AUROC@128 | dev AUROC@128 | dev FNR @1% FPR |
|---|---:|---:|---:|
| sampled length | 0.3532 | 0.2871 | 100% (119/119) |
| fixed length 128 | **0.9987** | **0.9638** | 57.14% (68/119) |

On the full corpus with fixed length: dev AUROC **0.9772**, but dev FNR
**281/598 = 46.99%** at FPR **17/1733 = 0.98%**.

The fix is one flag, `ArmorTrainerConfig::fixed_length = true` — no new
parameters, no artefact change. It is committed as opt-in so old runs stay
reproducible; it should become the default.

## Why AUROC is misleading here

A security scanner runs at a **fixed false-positive budget**, not at a global
ranking. At 1% FPR the threshold must exceed the 99th percentile of benign
scores, and the benign tail (max `0.99669`) overlaps almost all injections
(p50 `0.99628`). AUROC 0.977 and FNR 47% are the same model.

Per-family dev FNR at the best 1%-FPR threshold: en-obfuscated 72.5%, de-obfuscated
68.0%, jackhhao 83.9%, Lakera 37.8%. Every family loses; obfuscations are worst.

## Correctness that is not in question

- Tiled vs scalar GEMM at real shapes (192×192, 768×192, 192×768, seq 128,
  4352 buckets): max absolute difference **0.0**.
- Real-shape finite-difference gradient check (hidden 192, ffn 768, seq 128):
  agrees to **7.8e-6**.
- The loop, backward and round-trip artefact are correct.

The collapse was one trainer/eval disagreement. Verify the definitive version
against these numbers before changing the maths again.

## Remaining trainer bug

Early stopping still selects on a **12-token dev scan**. In the 2k fixed run it
picked epoch 1 even though epochs 2–3 had a 25× lower loss. Early stopping must
scan at the evaluation length.

## Data problems

- **Template leakage:** dev and test share combinatorial structure with train.
  Real-world numbers would be worse than these.
- **Obfuscations** are dev/test-only and generated from the same templates; they
  are the worst families and cannot be fixed by more of the same.
- **German** (33.2% of the corpus) is not native-reviewed.
- **Balance:** 42/58 injection/safe, unweighted.

## Plan, in order

1. **Trainer:** make fixed-length the default (with a compatibility note); early
   stopping at the evaluation length. Re-run the full corpus.
2. **Threshold method:** the gate is FNR at a stated FPR. Report it with counts
   and a confidence interval, per family, on dev **and** an independent benign
   set. A single global threshold with a heavy benign tail is the failure; a
   per-family threshold or a calibrated score is the candidate fix.
3. **Score function:** if fixed length + threshold still fails, test **max-pool
   over windows** (the cue is local). It needs a `FRAL` format version bump and is
   currently untested.
4. **Data:** real independent hard negatives, more real injections, obfuscations
   derived independently of the templates, native German review, rebalance or
   class weights. More templates alone will not fix the tail.
5. **Gate:** dev must pass before a test run is spent. Test is scored once, at the
   dev threshold, and reported as counts.

## Target

Proposed: **FNR ≤ 10% at FPR ≤ 1%** on dev and on test, with no family above
20%, and the benign tail measured on data not used to choose the threshold.
Until then Armor L1 stays disabled and the runtime keeps the deterministic L0/L3
layers.

## Not verified

- A fixed-length multi-epoch full run.
- Any test number for the fixed-length configuration (dev does not clear the gate).
- Max-pool.
