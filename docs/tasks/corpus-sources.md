# Corpus source adapters

Status: **ready after `feat/corpus-builder` lands.** Base: `feat/corpus-builder`.

## Behaviour (one)

The plan's source `kind` accepts `local`, `git` and `hf` with a pinned
`revision`; a source without a revision is refused; an `hf` source reads only
the files named in a local manifest and verifies their sha256; `offline: true`
turns a network attempt into a named error.

## Files

- `src/corpus/sources.rs` (new) — a `Source` trait with three implementations.
- `tests/corpus_sources.rs` (new).

## Test (name it)

`unpinned_revision_is_refused` and `offline_refuses_network`: a `git` source
without a revision fails by name; an `hf` source with `offline: true` fails
without opening a socket. Use a `file://` fixture store; no real network.

## Verify

```
VERIFY: cargo test -p ferrisres corpus_sources
```

## Out of scope

Real HF API pagination, gated datasets, licence decisions, MinHash. A source
that needs a credential stops with a named error.

## Rules

Licence and revision remain mandatory on every source. Push the branch; no merge.
