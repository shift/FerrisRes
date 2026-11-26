# CLI surface

Status: **register, 2026-11-26.** What the `ferrisres` binary offers on the
development branches. It is **not** a release description: a command is only
released when it is in a tagged `.deb`. The Rust API reference is separate
(`api-reference.md`).

## Global

| command | state | notes |
|---|---|---|
| `ferrisres --version` | release | prints the crate version |
| `ferrisres info` | release | device/module summary |
| `ferrisres devices [--json]` | `feat/gpu-selection` | adapters, index, limits, software flag |
| `ferrisres grammars list/fetch` | release | tree-sitter wasm registry |

## Training

| command | state | notes |
|---|---|---|
| `ferrisres train` | `feat/train-cli` | f32 reference loop; real backward + Adam |
| `--data`, `--data-format`, `--chat-template` | `feat/train-cli` | text/JSONL, shared renderer |
| `--hidden-dim`, `--num-blocks`, `--block-size`, `--epochs`, `--batch-size`, `--learning-rate` | `feat/train-cli` | plain block config only |
| `--seq-len` (128), `--vocab` (259), `--seed` (42) | `feat/train-cli` | byte tokenizer until the 32k one lands |
| `--checkpoint-every`, `--output`, `--resume`, `--log-every` | `feat/train-cli` | `FRTRAIN1` checkpoints |
| `--lora-rank` | `feat/train-cli` | **rejected**: the block+LoRA path is the open defect |
| `--device gpu` | in progress | `feat/gpu-train-loop`; default is CPU |
| `--rank`, `--world-size`, `--exchange-dir`, `--exchange-every` | in progress | `int/train-multi-gpu` |

## Distillation

| command | state | notes |
|---|---|---|
| `ferrisres distill` | release (blocked path) | teacher forward + CPU student; the blocked config refuses |
| `--model-path`, `--model-format`, `--config`, `--tokenizer` | release | teacher load |
| `--seq-len`, `--steps`, `--learning-rate`, `--temperature` | release | |
| `--data`, `--data-format`, `--chat-template`, `--require-provenance` | release | |
| `--output`, `--checkpoint-every`, `--resume`, `--log-every` | release | |
| `--converge`, `--converge-patience` | release | stop on a plateau |

## Benchmarks

| command | state | notes |
|---|---|---|
| `ferrisres benchmark` | release | kernel benchmark |
| `ferrisres benchmark teacher` | `feat/teacher-benchmark` | `metric=teacher_prefill_tps`, JSON record |
| `--model-path`, `--model-format`, `--config`, `--tokenizer` | `feat/teacher-benchmark` | |
| `--seq-len`, `--iterations`, `--warmup`, `--json` | `feat/teacher-benchmark` | |

## Multi-process

| command | state | notes |
|---|---|---|
| `ferrisres workers --world-size N -- <cmd>` | `feat/multi-gpu` | enumerates, refuses if adapters < N, spawns one child per adapter |
| `--cpu` | in progress | `int/train-multi-gpu`; skips adapter enumeration |
| evidence `rank-*.json`, `launch.json` | `feat/multi-gpu` | adapter, rank, elapsed, tokens/s if written |

## Inference

| command | state | notes |
|---|---|---|
| `ferrisres infer` | release | prompt → tokens; teacher-mapped skeleton without a model path |
| `--model-path`, `--model-format`, `--tokenizer`, `--config` | release | |
| `--prompt`, `--max-tokens`, `--temperature`, `--template` | release | |
| `--yarn-scale`, `--armor`, `--cognitive` | release | |
| `--persist-kv`, `--kv-path` | release | prefix cache |

## Planned, not present

| command | owner |
|---|---|
| `ferrisres export --checkpoint …` | `feat/train-export` |
| `ferrisres corpus build/verify/stats/sample` | `corpus-builder.md` |
| `ferrisres eval --suite …` | `eval-runner.md` |
| `ferrisres tokenizer train/inspect/fertility` | `tokenizer.md` |
| `ferrisres identity` | `model-identity.md` |
| `ferrisres embed --head …` | `embedding-head.md` |

## Rules

1. A command is "release" only when a tagged `.deb` contains it; a branch is not
   a release.
2. Model commands need **no licence key**; a licence belongs to the harness, not
   the model.
3. `--help` is the contract; this register is the map. If they disagree, `--help`
   is right.
4. Every command that writes prints a path, and every command that can refuse
   names the reason and the field.
