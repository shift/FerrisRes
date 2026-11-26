# Prefill uses one core

Status: **ready, sequenced after `cpu-packed-parallel-layout.md`** — same file
(`src/model/ternary.rs`), and the parallel kernel it must call has a known layout
defect until that task lands.

## The observation

Measured on this host while watching `top`: prefill occupies **one** core.

The cause is a wiring gap, not a missing kernel:

| call site | kernel | threads |
|---|---|---|
| `cpu_linear.rs:100` | `ternary_matmul_parallel` | rayon |
| `cpu_moe.rs:47` | `ternary_matmul_parallel` | rayon |
| **`ternary_linear.rs:94`** `TernaryLinear::forward` | `ternary_matmul` | **serial** |
| **`ternary_linear.rs:120`** `forward_packed` | `ternary_matmul_packed` | **serial** |

`cpu_block_attn_res` — the model's CPU path — calls `TernaryLinear`, so every
prefill projection runs single-threaded while the parallel kernels sit unused in
the same module.

## The change (one file: `src/model/ternary.rs`, plus the call site)

1. In `TernaryLinear::forward` and `forward_packed`, use the parallel kernel when
   `seq >= THRESHOLD` (start at 8) and the serial kernel below it. Do not
   parallelise `seq == 1`: the dispatch cost is larger than the work.
2. Do not change any public signature, and do not spawn threads yourself — rayon
   is already a dependency and its pool is already sized by `RAYON_NUM_THREADS`.
3. Respect `RAYON_NUM_THREADS=1`: a user asking for one thread must get one.

## The test

Name it `prefill_parallel_matches_serial`. In the same `#[cfg(test)] mod`.

1. build an input of `seq=64`, compare the threaded and serial results at `1e-5`;
2. run the same call with `RAYON_NUM_THREADS=1` and assert it still matches — the
   serial fallback must not be reachable only through a feature flag;
3. print a timing line for `out_rows=768, in_cols=768, seq=512` at 1 thread and
   at 4 threads. **The ratio is the evidence** that more than one core is used;
   asserting `num_threads()` proves only that rayon exists.

## VERIFY

```
cargo test -p ferrisres prefill_parallel_matches_serial
```

A pass with zero tests is a failure: the filter must select that name.

## Measure

On the i5-6500 (4 cores) and the i5-8400T (6 cores), record prefill tok/s at
`seq=512` with 1, 2, 4 and 6 threads, and the 8k-prompt wall time at 4 threads.
Expected: 3–4× at 4 threads on the 6500. If the gain is below 2×, stop and report
the scaling curve rather than tuning further.

## Out of scope

The GPU path, the AVX2 work in the serial kernels (a separate task in the same
file), KV cache threading, and the prefix cache.
