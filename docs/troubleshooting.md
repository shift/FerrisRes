# Troubleshooting

Status: **symptom → cause → fix, 2026-11-26.** Every entry names where to look;
none is a guess presented as a diagnosis.

## Runtime

| symptom | likely cause | fix | look at |
|---|---|---|---|
| `devices` reports 0 adapters | Vulkan loader/ICD not in the environment | set `LD_LIBRARY_PATH` and `VK_ICD_FILENAMES` | `hardware-matrix.md` |
| `WGPU_ADAPTER_INDEX=N is out of range` | fewer usable adapters than N; software filtered | choose a listed index or `--allow-software` | `ferrisres devices` |
| "software adapter" selected | llvmpipe is the only adapter | install the GPU ICD, or accept CPU | `devices` |
| OOM / device lost | batch or sequence too large, or KV too big | halve batch first, then seq, then quantise | run record |
| first token very slow | prefill, no prefix cache, cold weights | warm the model, enable the cache, check prompt length | `runtime-caches.md` |
| sudden slowdown | another build on the machine, thermal limit | stop the other build; check load | `uptime` |

## Training

| symptom | likely cause | fix |
|---|---|---|
| loss flat from step 0 | empty/short data, wrong tokenizer, lr too low | check the token count and a decoded sample; the byte tokenizer is not the 32k one |
| non-finite loss | lr too high, bad data, overflow | lower lr; stop, do not retry blindly |
| `resume` refuses | tokenizer, config, corpus or world size changed | restore the recorded artefacts; never edit the checkpoint |
| `resumed_from` is not the checkpoint step | wrong file or stale checkpoint | point at the file the run record names |
| checkpoint loads but predictions are garbage | tokenizer mismatch or an unexported head | compare `tokenizer_sha256` and the config |
| "data has N tokens" less than expected | short tail skipped, or a single file was passed | check `--data-format` and the shard index |
| `lora_rank` refused | the block+LoRA path is the open gradient defect | use the plain config; it is not a flag to bypass |

## Kaggle

| symptom | cause | fix |
|---|---|---|
| no GPU visible | runtime not set to GPU T4 | set the accelerator before starting |
| `dpkg -i` fails | the `.deb` is built for a newer glibc | rebuild on ubuntu-22.04 |
| session killed at 12 h | quota or timeout | resume from the last uploaded checkpoint |
| dataset too large | 20 GB limit | ship shards, not the full corpus |
| quota exhausted | 30 h/week used | resume next week; check the run record |

## Harness / Telegram

| symptom | cause | fix |
|---|---|---|
| bot silent | token missing/rotated, polling error | token in the environment only; check the log |
| "license required" | no licence, expired, or wrong HOME | `ferrisres`/`agentic-loop license status`; the model itself needs none |
| revocation warning every start | the default revocation file is absent | expected on a normal install; an explicit env path that is missing is an error |
| tool refused, "not listed in this step's Tools" | the workflow grant is the authorization | add the tool to the step, or `tool_search` for a read-only one |
| child principal answers as an adult | wrong principal mapping | check the principal config; there is no adult fallback |

## Nothing here matches

Capture the **run record**, the exact command, the device line and the first
failing log line. A report without those is a story; with them it is a bug.
