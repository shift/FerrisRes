# Teacher benchmark

Status: **implemented at `9f83183`; measured on this host; the second runtime
(T4) is not measured.** `ferrisres benchmark teacher` runs the teacher-forcing
forward through the same loader as `evaluate`, with no student and no sampling.

## The right metric

A teacher is not sampled from; it is run **forward over the corpus** to produce
logits and hidden states. So the number that matters is **prefill tokens/s at
the training sequence length**, plus the cost of storing the targets:

```
distillation hours for 1B tokens ≈ 1e9 / (prefill_tokens_per_second × 3600)
target bytes per token ≈ vocab × 4 (fp32 logits) + layers × hidden × 2 (bf16
                          hidden states), before any compression
```

Decode throughput is reported for completeness but is not the selection metric.

## Protocol

For each candidate teacher, on each runtime (this i5 / HD 530, T4, Strix Halo,
DGX Spark):

1. `seq_len` ∈ {128, 512, 2048} — the training window, not the model's maximum.
2. 1 warmup pass, then 3 timed passes; report the median.
3. Teacher-forcing forward only: no sampling, no KV growth beyond the sequence.
4. Record: tokens/s, wall time, peak RSS (CPU) or VRAM (GPU), device and
   adapter, model format and quantisation, config.
5. Derive hours per 1B tokens and target bytes per token.
6. Quality is **not** measured here. A fast teacher that is worse is still
   worse; the quality comparison is `evaluate` against the same held-out text.

## Command (proposed)

```
ferrisres benchmark teacher \
  --model-path <teacher.gguf> --model-format gguf --config qwen3-0.6b \
  --tokenizer <tokenizer.json> --seq-len 512 --iterations 3 --warmup 1 --json
```

Output, as the harness gate expects:

```
metric=teacher_prefill_tps value=<f64>
```

plus a JSON record with the fields above and the derived hours/bytes.

## Measured on this host (CPU, HD 530 idle)

`warmup 1`, `iterations 2`, mapped `gemma_mapper` teacher forward, GGUF
dequantized to f32 on load. Host load was 5.6–10.8 from other work, so treat the
absolute numbers as a floor with noise and the **shape** as the result.

| model | quant | seq | tokens/s | hours / 1B tokens | peak RSS | median s/pass |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | Q8_0 | 128 | 7.73 | 35,924 | 5,938 MB | 16.55 |
| Qwen3-0.6B | Q8_0 | 512 | 1.90 | 146,406 | 5,938 MB | 269.85 |
| Qwen3-0.6B | Q4_K_M | 128 | 6.34 | 43,825 | 5,938 MB | 20.19 |
| Qwen3-0.6B | Q4_K_M | 512 | 2.62 | 106,199 | 5,937 MB | 195.75 |

Three conclusions, in order of importance:

1. **The cost is quadratic in sequence length, as expected for a full forward.**
   4x the tokens cost ~16x the time (16.55 s → 269.85 s). Extrapolating the
   512 number to a 2k window is ~16x again, and to 8k ~256x: one 8k sequence
   would take about 14 hours on this host. **Teacher labelling at the training
   window cannot run on this CPU.** It runs on a T4, or not at all.
2. **Quantisation saves nothing on this path.** Q4_K_M and Q8_0 both dequantize
   to f32 on load, so peak RSS is identical (5.9 GB) and the forward is the same
   f32 code. The Q4/Q8 spread in the table is host contention, not quantisation.
   This invalidates the assumption that a Q4 E4B fits a 16 GB card: the loader
   would still hold roughly 20 GB of f32 weights.
3. **5.9 GB for a 0.6B f32 model is 2.4x the weights alone.** The load path
   keeps more than one copy; reducing that is the cheapest way to make a larger
   teacher fit, and it should be fixed before E4B is attempted.

## Derived budget

| tokens | seq 128 | seq 512 | seq 2k (extrapolated) |
|---:|---:|---:|---:|
| 10M | 359 h | 1,464 h | ~23,000 h |
| 100M | 3,594 h | 14,641 h | — |

The 128-token number is the only one that is a measurement at the window; the
rest are the quadratic projection and must be confirmed on the T4.

## Candidates

| teacher | quant | notes |
|---|---|---|
| Qwen3-0.6B | Q8_0, Q4_K_M | the smoke teacher; fastest |
| Qwen3-1.7B | Q8_0, Q4_K_M | the Base teacher |
| Gemma 4 E2B | Q8_0 | the proven Colab path |
| Gemma 4 E4B | Q4_K_M | the largest that plausibly fits a 16 GB T4 |

## What it decides

The teacher for the first real run is the fastest one that fits the runtime and
passes the quality comparison — not the largest one that loads. On a 6.3 GB/s
host, a 1.7B teacher at Q4 is ~2.5× the bytes of 0.6B and roughly 2.5× slower
per token; that difference decides whether a corpus run takes a day or a week.

## Caveats

- Quantisation changes speed **and** quality; a Q4 teacher is not a Q8 teacher.
  Benchmark both, and record the quant with the throughput.
- The teacher forward in the current code maps through `gemma_mapper`; a native
  dense path would change the numbers. Report which path was measured.
- Storage of the targets can exceed the compute cost: fp32 logits at 151k vocab
  are ~600 KB per token; a 1B-token run needs compression or top-k logits.
- **E4B is unmeasured.** The ~5B-parameter teacher needs roughly 20 GB resident
  in f32; this host had ~17 GB free under load, so running it risked OOM. It
  remains the target for the T4 run, not for this host.
- The measured path is `gemma_mapper`, not the native ternary student and not a
  GPU path. A GPU number from a different path is not comparable.
