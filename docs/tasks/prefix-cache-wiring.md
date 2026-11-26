# Wire the prefix KV cache into the request path

Status: **ready to delegate.** Base: `feat/prefix-kv-cache` @ `9960441`.

## Behaviour (one)

`infer` (and the request path) uses `PrefixKvCache` with the model fingerprint,
tokenizer version and principal scope; a second call with a shared prefix
prefills **only the suffix** and produces the same next-token logits as a cold
call within 1e-5.

## Files

- `src/main.rs` — one flag set (`--prefix-cache <dir>`, `--cache-budget-bytes`)
  and the lookup/insert around the prefill call; this file is shared, so change
  only the infer path.
- `src/inference/prefix_kv_cache.rs` — no algorithm change; use the public API.
- `tests/prefix_cache_wiring.rs` (new).

## Test (name it)

`warm_prefix_prefills_only_the_suffix`: a fake or tiny real model, a 300-token
prompt with a 200-token shared prefix; assert the second call reports 100
prefilled tokens and the logits match the cold call within 1e-5. A second test:
a different principal scope is a miss.

## Verify

```
VERIFY: cargo test -p ferrisres prefix_cache_wiring
```

## Out of scope

Batched suffix prefill (a `start_pos` API in the model), per-scope disk quotas,
disk reclamation, exact-match logits reuse.

## Rules

The cache key includes the principal; never share a prefix across principals. A
cache miss is a full prefill, not a wrong answer. Push the branch; no merge.
