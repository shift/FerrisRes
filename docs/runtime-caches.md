# Runtime caches

Status: **inventory, 2026-11-26.** The soul/tool prefix cache is the first
entry. Every cache here needs a hit/miss test and an invalidation test; a cache
that changes an answer is worse than the work it saves.

## The principle

Cache **stable work**, key it by everything that changes the result, scope it to
the principal, and prove a hit equals a fresh computation. Never cache a user's
answer, a time-dependent decision, or a secret.

## Prefix KV cache (implemented)

`feat/prefix-kv-cache` @ `9960441`: SHA-256 key over model fingerprint, tokenizer
version, principal scope and the exact token ids; longest-prefix lookup; an LRU
bounded by real buffer bytes; a checksummed disk tier where corruption is a miss;
scope isolation. The equality gate is a next-token logit difference of
**1.19e-7** against a cold prefill. The wall win is smaller than the token win —
12.98 s → 6.46 s on this host — because the suffix currently goes through
`forward_decode` one token at a time; batched suffix prefill needs a `start_pos`
API. The cache is not wired to `infer` yet (`tasks/prefix-cache-wiring.md`).

## The inventory

| cache | what it avoids | expected | effort | risk |
|---|---|---|---|---|
| **Soul/tool/principal prefix KV** (`tasks/soul-cache.md`) | re-prefilling 1–5k stable tokens every turn | **implemented at `9960441`**: 2,000 → 500 prefilled on a 1,500 shared prefix; equality 1.19e-7; 12 tests | done (library); CLI wiring open | **privacy: the key includes the principal**; no cross-principal hit |
| **Conversation prefix KV** | re-prefilling the whole history each turn | **large on multi-turn** | days | window/summary staleness |
| **Unchanged suffix after an edit** | re-prefilling text a notebook edit did not change | up to 15% prefill (reported) | days | an edit must be an invalidation point, not a flush |
| **Rendered tool/skill text** | re-rendering the same tool block and skill text | small CPU, feeds the KV cache | hours | tool-set hash |
| **Token ids for stable text** | re-tokenising soul, tools, skills | small | hours | tokenizer change |
| **Embedding cache** | re-embedding the same memory/RAG/skill chunks | moderate on retrieval | hours | privacy, model change |
| **Compiled grammar/schema** | recompiling the tool grammar per request | moderate with constrained decoding | days | grammar version |
| **Evidence per revision** | re-running a build/test for the same commit | **large CPU/GPU saving** | already in `agentic-loop-evidence` | revision identity |
| **Workspace file reads** | re-reading large files each turn | moderate on big repos | hours | mtime/hash invalidation |
| **Workflow parse** | re-parsing an unchanged `.workflow` | small | hours | file hash |
| **Single-flight** | two identical model calls at once | moderate on the API | hours | none |
| **Weight pre-touch** | page faults on first token | TTFT/jitter | hours | RAM |

Not a cache, but the same class of saving: **speculative decoding** (Nano draft)
and **tool-block selection** (top-k instead of the full registry).

## Order

1. **Conversation prefix KV** — extends the soul cache and is the largest
   per-turn win for a chat assistant.
2. **Evidence per revision** — the agent loop re-runs the same test for the same
   commit; the evidence crate is already revision-bound, so this is a wiring
   job, not a design. It saves more CPU than any KV cache.
3. **Rendered text + token ids** — trivial, and they are the inputs to the KV
   cache.
4. **Embedding and grammar caches** — worth it once retrieval and constrained
   decoding are on the path.
5. **Workspace file reads / workflow parse** — cheap, useful on large repos.
6. **Single-flight, pre-touch** — API and TTFT polish.

## Gates

| test | asserts |
|---|---|
| hit equals fresh | the same logits/answer within the documented tolerance |
| invalidation | editing the source (soul, tool, file, revision) misses |
| scope | two principals never share an entry |
| budget | the LRU stays under its byte limit; eviction is logged |
| staleness | a summary or truncated window cannot serve a request past its boundary |

## What not to cache

- A user's answer, a model output for a task (idempotent-looking tasks are not).
- A decision with a time, policy or evidence dependency unless the key includes
  all three.
- Secrets, tokens, raw session transcripts.
