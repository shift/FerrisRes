# BMI2 packed ternary matvec

Status: **ready.** Base: `int/gpu-deb` (`21aed9c`) or `main`. One file.

## Why

`ternary_linear.rs` keeps two copies of every weight: an unpacked `i8` array for
`forward()` and a 2-bit packed copy for `forward_packed()`. The comment says
packed is "slightly slower ... due to bit-unpacking overhead", so the fast paths
use the unpacked copy. That copy is 4× larger, and it is why the loop core needs
4.6 MB where it could need 1.2 MB.

A 4× smaller core changes cache residency on every host we own. Small's core plus
a 32k KV would fit 6 MB L3 with room, instead of spilling. That is the reason
this task exists; the unpack speed is the means.

## The latent defect in the current unpack

`src/model/ternary.rs:401` (`ternary_matmul_packed`) reads

```rust
let byte_idx = (row_offset + q * 4) / 4;
let packed_byte = packed[byte_idx];
let code = ((packed_byte >> (bit_pos * 2)) & 0b11) as usize;
```

`row_offset = r * in_cols`. When `in_cols % 4 != 0`, a row does not start on a
byte boundary, so `packed_byte >> (bit_pos * 2)` reads the wrong pair of bits.
The function is therefore wrong for any width that is not a multiple of four.
Fix it or refuse it, but do not leave it silent.

## The change (one file: `src/model/ternary.rs`)

1. Extract 16 codes from four bytes with one `_pext_u64` when
   `is_x86_feature_detected!("bmi2")`, instead of four shifts per byte.
2. Accumulate with AVX2 masked adds when `is_x86_feature_detected!("avx2")`:
   a ternary value is -1, 0 or 1, so no multiply is needed. Build the mask from
   the code, then `_mm256_add_ps` / `_mm256_sub_ps` under it.
3. Keep the scalar loop as the fallback. Dispatch once per call, not per element.
4. Handle `in_cols % 4 != 0` correctly: either read the bit offset, or reject the
   layout at construction with a named error.
5. Do not change `unpack_ternary`, the checkpoint format, or the GPU kernels.

## The test

Name it `packed_ternary_matches_unpacked_at_every_width`. Place it in the
existing `#[cfg(test)] mod` in `src/model/ternary.rs`.

It must:

1. loop `in_cols` over 1..=17 and `out_rows` over 1..=5, so a non-multiple of
   four is covered;
2. compare `ternary_matmul_packed` against `unpack_ternary` + `ternary_matmul`;
3. assert every element within `1e-5` relative;
4. run twice: with the SIMD path forced on, and forced off, and assert both
   agree with the scalar reference;
5. panic with the offending `in_cols` in the message, not just a diff.

## Measure, do not assume

Record, on this host (i5-6500):

```
cargo test -p ferrisres packed_ternary_matches_unpacked_at_every_width
```

and one timing line per path (unpacked, packed scalar, packed SIMD) for
`out_rows=768, in_cols=768, seq=1`. The expected gain is 4–8× on the unpack and
a 4× smaller resident core; a claim without the timing line is not evidence.

## Out of scope

`ternary_matmul` and `ternary_matmul_decode` (the unpacked f32 paths) — they are
the next task, in the same file. f16, AVX-512, the GPU kernels, and the model
shape.
