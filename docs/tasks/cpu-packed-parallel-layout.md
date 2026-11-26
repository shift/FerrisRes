# `ternary_matmul_packed_parallel` reads the wrong layout

Status: **ready, sequenced after `cpu-avx2-decode.md`** — same file
(`src/model/ternary.rs`), one writer at a time.

## The defect

`pack_ternary` writes a **flat bit stream**: element `i` lives at bits `2i` and
`2i+1`, with no per-row padding. `ternary_matmul_packed` originally assumed that,
but read it as if rows were byte-aligned, and was wrong whenever
`in_cols % 4 != 0` (fixed in `42cf762`).

`ternary_matmul_packed_parallel` makes the *other* assumption:

```rust
let row = r * ((in_cols + 3) / 4);   // byte-padded rows
```

That is a different layout from the one the packer writes. The two agree only
when `in_cols % 4 == 0`, which is true for every width the model currently uses —
so no test catches it, and a shape change silently produces wrong numbers.

## Why it matters beyond tidiness

This is the **prefill** path. A wrong kernel here does not crash; it produces a
plausible but incorrect projection, which then reads as "the model is a bit
worse than expected". That is the most expensive kind of defect to find later.

## The change (one file: `src/model/ternary.rs`)

Make `ternary_matmul_packed_parallel` read the flat stream, exactly as
`ternary_matmul_packed` now does: `flat = r*in_cols + j`, `byte = flat >> 2`,
`shift = (flat & 3) * 2`. Keep the rayon structure. Reuse the kernel selection
from the serial path if the shapes allow; do not duplicate the extraction logic.

## The test

Name it `packed_parallel_matches_serial_at_every_width`. Same `#[cfg(test)] mod`.

1. `in_cols` over 1..=17 and `out_rows` over 1..=5;
2. compare against `unpack_ternary` + `ternary_matmul`, and against the serial
   packed path;
3. `1e-5` relative tolerance;
4. panic with `in_cols`, `out_rows` and the element index.

## VERIFY

```
cargo test -p ferrisres packed_parallel_matches_serial_at_every_width
```

A pass with zero tests is a failure: the filter must select that name.

## Out of scope

The packer, the checkpoint format, the GPU packed kernel, the AVX2 work in the
serial path.
