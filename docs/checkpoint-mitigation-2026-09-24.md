# Generic checkpoint loading: fail-closed mitigation

Parent audit task **A03** (`95b2091a-4955-466a-bca0-be613d300647`) is **not fully repaired**.
Safety mitigation: `2c16d6d4-ec3c-439d-b73d-35edb11a0ef7`. Decision: ADR-027.

## Behavior now

`AnyModel::from_safetensors`, `from_gguf`, and `from_path` always return an explicit `Unsupported` error. They do not read/validate the path or allocate a substitute model. Previously, these methods returned success after reading metadata while discarding checkpoint tensor values.

This intentionally disables a broken generic pretrained-loading path; it does **not** implement checkpoint import. Explicit model constructors, architecture detection, raw safetensors/GGUF readers, and separate model-specific CPU/Gemma loaders remain available and unchanged.

## Why full loading remains unfinished

The generic GPU model representation lacks complete learned normalization/embedding/head ownership and architecture settings needed for faithful checkpoint execution. Copying only projection matrices would still silently omit learned behavior.

Tracked prerequisites:

- `09197a72-e77e-43f3-abab-acb71503dd6c`: complete model weight/configuration representation. [Learned layer norms](learned-layer-norms-2026-09-25.md) and [final model norm plus the embedding kernel](output-normalization-embedding-2026-09-25.md) are implemented; embedding/head ownership and complete architecture settings remain unfinished.
- **A16**: validated attention metadata API added in the [2026-09-25 update](attention-metadata-2026-09-25.md), replacing removed width-guessing helpers. Integration into future import remains part of A03.
- **A17**: architecture-correct gated FFN — repaired in the [2026-09-25 update](standard-ffn-fix-2026-09-25.md). This does not complete checkpoint import.

Restore the loading APIs only after supported architectures have strict shape/name validation, complete tensor mapping, and numerical parity against a reference checkpoint. Missing or unsupported tensors must not fall back to untrained values.

## Tests

Two new tests in `tests/checkpoint_loading.rs` create valid, nonzero partial safetensors and GGUF fixtures and verify the raw readers can read the data. Both reproduced false successful loading before this mitigation; both now verify explicit rejection.

Nix build, full suite (**1,666 passed**, six unrelated audit reproductions ignored), all-target Clippy with warnings denied, and `git diff --check` passed. No commits were made. Earlier A01/A02/A04 fixes remain intact.
