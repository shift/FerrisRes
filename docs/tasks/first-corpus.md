# First corpus

Status: **ready after the source adapters and tokenizer land.** The release
corpus is permissive; the research corpus adds the overrides.

## Release corpus (~300–500M tokens)

| source | licence | use |
|---|---|---|
| `rust-lang/book`, rustlings, rust-analyzer docs | MIT/Apache-2.0 | Rust |
| nixpkgs docs/packages, nix.dev examples | MIT / verify CC | Nix |
| agentic-loop + ferrisres docs/code | ours | harness vocabulary |
| FineWeb-Edu sample | ODC-By | English |
| FineWeb-2 `deu` sample | ODC-By | German |
| public-domain long text | per source | 8k windows |

## Private test corpus (same + overrides)

| source | licence | condition |
|---|---|---|
| xLAM function calling | NC reported | **private test only**; non-commercial, not distributed |
| Dolly 15k | CC-BY-SA-3.0 | **private test only**; share-alike matters on distribution |
| CulturaX `de` | gated terms | **private test only**; if the card forbids training, exclude it |

The private test corpus is a **separate plan and output directory**. A row from
it never enters the release shards; the manifest records the split, and the build
is not distributed or served.

## Steps

1. Fetch each source at a pinned revision (source adapters), record licence and
   hash.
2. Build with `ferrisres corpus build` (release plan) → verify → stats.
3. Train the 32k tokenizer on the release sample; record its hash.
4. Rebuild the shards with the 32k tokenizer; re-verify.
5. Build the research corpus with the same steps and a `research/` out dir.
6. Upload the release shards as a Kaggle/HF dataset; the research shards stay
   local/private.

## Gates

- licence coverage 100%, no unknown;
- secret scan clean;
- heldout never in train;
- identical rebuild hashes;
- tokenizer hash recorded;
- research rows absent from the release manifest.

## Out of scope

MinHash near-dedupe, the full 2B mix, German native review. Those are later.
