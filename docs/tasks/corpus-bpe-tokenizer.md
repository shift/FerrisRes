# Corpus build with the 32k tokenizer

Status: **ready.** Base: `int/corpus-tokenizer` (`c2bc3ea`), which holds the
corpus builder, the source adapters and the trained 32k tokenizer.

## The defect

`ferrisres corpus build` has one tokenizer: `SimpleTokenizer`, 259 ids, one id
per byte (`src/corpus/build.rs:305`). The trained artefact
`artifacts/frs-tokenizer-32k/0.1.0/tokenizer.json` (vocab 32,768, sha
`98aa17a1…`) cannot be used. A 32k corpus is therefore impossible, and the first
student is specified with a 32k tokenizer.

## The change

Two files.

### `src/corpus/plan.rs`

The plan gains an optional tokenizer file. Keep the existing default; a plan
without it must build byte shards exactly as today.

```jsonc
"tokenizer": { "kind": "bpe", "path": "artifacts/frs-tokenizer-32k/0.1.0/tokenizer.json" }
```

- `kind` is `byte` (default) or `bpe`.
- A `bpe` spec without `path` is a validation error.
- A `bpe` path that does not exist is a validation error.
- `vocab_size` in the manifest record comes from the loaded tokenizer, not from
  a constant.

### `src/corpus/build.rs`

- Load the file with `crate::model::tokenizer::HfTokenizer::from_tokenizer_json`.
- Encode every document with it instead of `SimpleTokenizer`.
- Record the tokenizer `sha256` and `vocab_size` in the manifest. The file hash
  is read from the bytes, not from the path.
- Refuse a plan whose tokenizer `vocab_size` is greater than `65535`; shards are
  `u16` and a silent truncation is worse than a refusal.
- Refuse to append to an existing output directory whose `tokenizer_id` differs.

## The test

Name it `corpus_bpe_uses_the_trained_tokenizer`. Put it in a
`#[cfg(test)] mod tests` inside `src/corpus/build.rs`, so the change touches two
files.

It must:

1. build a tiny corpus with the committed 32k artefact;
2. assert `manifest.tokenizer.vocab_size == 32768`;
3. assert `manifest.tokenizer.sha256` equals the file hash;
4. assert the decode of the first window equals the source prefix;
5. build the same input twice and assert identical shard hashes;
6. assert a `byte` plan still yields 259 ids.

## VERIFY

```
cargo test -p ferrisres corpus_bpe
```

One crate, one filter. A pass with zero tests is a failure: the filter must
select `corpus_bpe_uses_the_trained_tokenizer`.

## Out of scope

Training a tokenizer, changing the model, changing the shard format, near-dedupe,
a tokenizer other than the committed artefact.
