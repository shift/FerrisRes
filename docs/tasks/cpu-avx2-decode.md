# AVX2+FMA on the f32 ternary matvec

Status: **blocked on `cpu-bmi2-ternary.md`** — same file
(`src/model/ternary.rs`), so it cannot be written concurrently. One writer per
file.

## Why

Prefill is compute-bound and it is the UX problem on this host: a cold 8k prompt
is minutes at 20–40 tok/s. Decode is a DRAM wall and SIMD will not move it much;
prefill is a scalar loop and SIMD moves it 3–5×.

`src/model/ternary.rs:292` (`ternary_matmul`) and `:591`
(`ternary_matmul_decode`) are plain scalar loops over `i8` weights and `f32`
inputs.

## The change (one file: `src/model/ternary.rs`)

1. Widen the inner loop to eight `f32` lanes with `_mm256_fmadd_ps` when
   `is_x86_feature_detected!("avx2")` and `"fma"`.
2. Keep a scalar remainder loop for the tail; do not assume `in_cols % 8 == 0`.
3. Keep the scalar path for hosts without AVX2 — the `.deb` ships elsewhere.
4. **Never** add `-C target-cpu=native`. Use `#[target_feature]` on the kernel
   and runtime dispatch in the caller.
5. Do not change the function signatures.

## The test

Name it `avx2_matvec_agrees_with_scalar`. Same `#[cfg(test)] mod`, after the
BMI2 test.

It must:

1. compare SIMD and scalar results for `in_cols` over 1..=23;
2. assert `max |Δ| <= 1e-5` in absolute terms on values of order 1;
3. run the scalar path explicitly, so a host without AVX2 still exercises the
   comparison and reports "skipped: no avx2" rather than passing silently.

## VERIFY

```
cargo test -p ferrisres avx2_matvec_agrees_with_scalar
```

A pass with zero tests is a failure: the filter must select
`avx2_matvec_agrees_with_scalar`.

## Measure

On the i5-6500 and the i5-8400T, `out_rows=768, in_cols=768`:

- `seq=1` (decode): record tok/s before and after. Expect small.
- `seq=512` (prefill): record tok/s before and after. Expect 3–5×.

The two hosts differ in L3 (6 MB vs 9 MB). Report both; the difference is the
evidence that residency, not SIMD width, sets the decode ceiling.

## Out of scope

The packed 2-bit path (previous task), F16C, bf16, AVX-512, the GPU kernels.
