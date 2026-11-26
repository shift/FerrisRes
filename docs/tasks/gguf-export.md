# GGUF export

Status: **ready after `feat/train-export`.** Base: `feat/train-export` @ `44a9409`.

## Behaviour (one)

`ferrisres export --checkpoint <FRTRAIN1> --output <prefix> --format gguf
--quant q8_0` writes a GGUF whose metadata matches the model config and whose
tensors round-trip; `infer --model-format gguf` loads it.

## Files

- `src/training/export_gguf.rs` (new) — reuse `src/model/gguf.rs` readers.
- `src/main.rs` — add `--format` and `--quant` to `export`.
- `tests/gguf_export.rs` (new).

## Test (name it)

`gguf_export_round_trips`: export, load, assert tensor names/shapes and f32
equality for `q8_0` within the documented tolerance; assert the config fields.

## Verify

```
VERIFY: cargo test -p ferrisres gguf_export
```

## Out of scope

Ternary export, a 32k tokenizer, serving. Do not change the FRTRAIN1 format.

## Rules

One quantisation (`q8_0`) before `q4_k_m`. Report the file size and the tolerance.
Push the branch; no merge.
