# Teacher benchmark

Status: **protocol, not implemented.** The existing `benchmark` measures GPU
kernels only; `infer` maps a teacher into an untrained student. Neither answers
"which teacher can I afford to distil from on this runtime?"

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
