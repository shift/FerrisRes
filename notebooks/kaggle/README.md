# Kaggle workbooks

Two notebooks, and only one is runnable today.

| notebook | state | what it does |
|---|---|---|
| `teacher_benchmark.ipynb` | **runnable now** | installs the `.deb`, lists adapters, measures teacher **prefill tokens/s** at 512/2048 tokens for Qwen3-0.6B Q8/Q4, prints `hours_per_10M` |
| `distill_smoke.ipynb` | **prepared, blocked** | installs, runs a 10-step distill smoke, and **stops loudly** on `lora_gradient_invalid_with_blocks`; the long run is behind `RUN_LONG = False` |

## Prerequisites

1. **A `.deb` with the current subcommands.** The GitHub release does not have
   `devices`, `benchmark teacher` or the `qwen3-0.6b` preset yet. Build from the
   branch (`int/ferris-main` + the teacher-bench and gpu-selection branches),
   run `packaging/build-deb.sh`, and attach the result as a private Kaggle
   dataset at `/kaggle/input/ferrisres-deb/`.
2. **GPU T4 ×2** and internet enabled.
3. **Secrets** — the Kaggle API token is a secret, never a notebook cell:
   use **Add-ons → Secrets** and read it with `kaggle_secrets.UserSecretsClient`.
4. **Datasets** — for the smoke, the corpus at
   `/kaggle/input/ferrisres-corpus/train.jsonl` with provenance fields; the
   teacher GGUF and `tokenizer.json` either downloaded in the notebook or
   attached.

## Order

1. `teacher_benchmark` — pick the teacher by `hours_per_10M` on this runtime.
2. Fix the LoRA gradient; rebuild the `.deb`.
3. `distill_smoke` — 10 steps, then flip `RUN_LONG`.
4. `evaluate` + `benchmark` on a quiet machine.

## Two GPUs

The second T4 is idle until `WGPU_ADAPTER_INDEX` exists in the `.deb`
(`docs/tasks/multi-gpu.md`). Then:

```
!ferrisres devices                    # must list two T4s
!WGPU_ADAPTER_INDEX=0 ferrisres serve ... &
!WGPU_ADAPTER_INDEX=1 ferrisres serve ... &
```

For training, two ranks must be launched with different `--rank` and a shared
corpus; that launcher is phase 2 of the multi-GPU task, not in these notebooks.

## Quota

One hour of quota covers both cards. The teacher benchmark is minutes; a
distillation run at 0.6B and 10M tokens is ~17 h of teacher forward on one T4
(`hours_per_10M` in the benchmark output), so start with the smoke and a small
slice, not the full corpus.

## Pushing from the CLI

`kernel-metadata.example.json` is the shape for `kaggle kernels push -p`.
Copy it, set your username, the code file and the dataset sources; never put a
key in it.
