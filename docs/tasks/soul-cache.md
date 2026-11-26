# Soul and prefix caching

Status: **task specification.** The engine persists session KV (`kv_persist`)
and has a student KV cache, but nothing keyed by a stable prefix. Every request
re-prefills the system template, the soul and the tool definitions.

## Why

The stable prefix is soul (100–300 tokens child, <1k operator) + tool
definitions (1–4k) + the system template + optionally a per-principal profile.
That is 1–5k tokens re-read and re-computed on **every** turn, and it dominates
time-to-first-token on a 6.3 GB/s host. It changes rarely.

## What to cache

| segment | reusable across | invalidated by |
|---|---|---|
| system template | every request | template change |
| **soul** | every turn, and every tool set | soul edit (version/hash), principal |
| **tools** | every turn with the same set | tool set change, soul length change (RoPE shift) |
| profile / memory prefix | turns of one principal | profile edit; **never shared across principals** |

Conversation is not cached; it changes every turn.

The soul and the tools are **separate entries**. The soul's KV does not depend
on the tool definitions that follow it, so editing a tool does not throw the
soul away; adding a turn does not throw either away. A segment entry records its
start position, because changing the soul's length shifts every later segment
and invalidates it.

## Key

```
sha256(model_checkpoint, tokenizer_id, template, segment_id, segment_version,
       segment_text_hash, kv_bits, sliding_window, start_position)
```

A hit requires all of it. A soul edit bumps `segment_version`; the harness's
`soul set` already versions the entity, so the version is available.

## Storage

- In memory, an LRU with a byte budget (`[soul] cache_max_mb`).
- Optional disk persistence for restarts (`kv_persist`), **encrypted at rest**
  when the segment contains a private soul or profile. A soul KV is a
  representation of the soul text, not a hash; treat it as the text.
- Per-principal scope. Two principals never share an entry, even with identical
  text; the isolation is the point.

## Correctness

| test | asserts |
|---|---|
| cache hit vs fresh prefill | logits after the prefix within 1e-4 |
| soul edit | version bump misses; stale entry unused |
| tool edit | soul segment hits; tools segment misses |
| soul length change | later segments miss (position shift) |
| sliding-window layers | the cached window is the last `window` tokens, not the whole prefix |
| global layers + block summaries | summaries at the prefix end are cached and reused |
| two principals | identical text, different principal → no hit |
| quantisation | `kv_bits` in the key; no 3-bit cache reused for a 4-bit run |

## API

- Engine: `PrefixCache { get(key) -> Option<PrefixKv>, put(key, kv) }` used by
  `TokenGenerator`/`two_phase`; a hit starts prefill at the segment end.
- Harness: computes the key from the soul entity, the tool set and the model
  descriptor; registers the segment after the first miss.
- Config: `[soul] cache = true`, `cache_max_mb`, `cache_dir`, `encrypt = true`.

## Metrics and gate

`TTFT` with and without the cache, hit rate, bytes held, and a logits-equality
test. The gate is correctness first: a cached prefix that changes an answer is
worse than a slow one. Only then report the TTFT gain.

## Phases

1. Full stable-prefix cache (system + soul + tools) as one entry.
2. Segmented cache (soul and tools independent).
3. Disk persistence with encryption and per-principal scope.

## Risks

- **Stale soul** — the operator edits the soul and the bot keeps the old voice
  until the version is bumped; the harness must bump on `soul set`.
- **Position shift** — any length change invalidates later segments; encode it
  in the key rather than discovering it in the output.
- **Privacy** — a KV cache is the soul in another form; scope and encrypt it.
- **Quantisation drift** — a cached 3-bit KV and a fresh 4-bit prefill are not
  interchangeable; keep the precision in the key.
- **Memory** — a 4k-token tools segment across a few principals is a few MB at
  3-bit KV; the LRU budget bounds it, not the model size.
