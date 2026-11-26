# 32k tokenizer

Status: **ready to delegate.** Base: `feat/train-cli` @ `d81becf`.

## Behaviour (one)

`ferrisres tokenizer train --corpus <dir-or-file> --vocab 32768 --out <dir>`
trains a BPE with byte fallback and writes `tokenizer.json`,
`special_tokens.json`, `SHA256SUMS`, and a manifest with the sample paths and
hashes. `tokenizer inspect` prints vocab size and hash; `tokenizer fertility
--text <file>` prints tokens/byte.

## Files

- `src/tokenizer/train.rs` (new) — use the `tokenizers` crate; no Python.
- `src/main.rs` — one `Tokenizer` command variant and match arm.
- `tests/tokenizer_32k.rs` (new).

## Test (name it)

`tokenizer_round_trips_and_is_deterministic`: train twice on the same fixture
and assert the two `tokenizer.json` hashes match; `decode(encode(x)) == x` for
UTF-8 text, tabs, trailing spaces and a string literal with `\n`.

## Verify

```
VERIFY: cargo test -p ferrisres tokenizer_32k
```

## Out of scope

The corpus mix, the chat template, integrating the tokenizer into `train`, and
the 32k vocabulary decision for Base. No network.

## Rules

Special tokens: `<pad> <bos> <eos> <system> </system> <user> </user> <assistant>
</assistant> <tool> </tool> <tool_call> </tool_call> <tool_result> </tool_result>
<fim_prefix> <fim_suffix> <fim_middle>`. No normalisation. Push the branch; no merge.
