# Dataset formats and chat templates

`ferrisres distill --data` accepts structured training data, not only a file of
raw text. Examples are streamed from disk and rendered with a chat template at
load time; the rendered text is then tokenised with the same
`HfTokenizer::encode_raw` call as before.

The implementation is `src/training/dataset.rs`.

## CLI

```bash
ferrisres distill \
  --model-path ./model.safetensors \
  --config e2b \
  --data chat.jsonl \
  --data-format auto \
  --chat-template gemma \
  --require-provenance
```

| Flag | Values | Default | Meaning |
|------|--------|---------|---------|
| `--data-format` | `auto`, `text`, `jsonl` | `auto` | `auto` chooses by extension; `text` treats the whole file as one example; `jsonl` reads one JSON object per line. |
| `--chat-template` | `gemma`, `qwen`, `lfm`, `plain` | `gemma` | Template used to render `messages` examples. `text` examples are never re-rendered. |
| `--require-provenance` | flag | off | Abort at the first example missing `source`, `licence` or `redacted`. Without it, missing fields are counted and reported once at the end. |

`.txt` files (and extension-less paths) still load as a single text example and
produce exactly the token sequence they produced before, regardless of
`--chat-template`.

## Formats

| Path extension | Format |
|----------------|--------|
| `.txt`, `.text`, no extension, anything else | whole file = one `Text` example |
| `.jsonl`, `.json` | one JSON object per line |
| `.zst`, `.zstd` | refused: decompress first with `zstd -dc` |

zstd is not a dependency. A compressed path is refused with a message such as:

```text
refusing zstd-compressed input `training_data.jsonl.zst`: zstd input is not supported.
Decompress it first, for example: zstd -dc `training_data.jsonl.zst` > `training_data.jsonl`
```

The zstd magic number (`28 B5 2F FD`) is also checked when the file is opened,
so a compressed file with the wrong extension is refused too.

## Accepted JSON shapes

Each non-blank line must contain exactly one of `text` or `messages`.

### Raw text

```json
{"text": "The quick brown fox.", "source": "book-1", "licence": "public-domain", "redacted": false}
```

### Chat

```json
{"messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "4."}], "source": "synthetic", "licence": "cc-by-4.0", "redacted": true}
```

### Tools and reasoning

`tools` holds the function definitions offered to the model. `reasoning` holds
the assistant's reasoning for the example; it is attached to the last assistant
turn.

```json
{"messages": [{"role": "user", "content": "What is the weather in Paris?"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]}, {"role": "tool", "content": "{\"temp_c\": 18}", "tool_call_id": "call_1"}, {"role": "assistant", "content": "It is 18 °C."}], "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}], "reasoning": "I need current weather data, so I call the tool first.", "source": "tool-demo", "licence": "cc0-1.0", "redacted": true}
```

Message fields:

| Field | Required | Notes |
|-------|----------|-------|
| `role` | yes | `system`, `user`, `assistant`, `tool`, or a provider role. Must not be empty. |
| `content` | no | String, `null` or absent. `null`/absent becomes `""` so assistant tool-call turns work. Arrays (multimodal parts) are refused. |
| `name` | no | Function name used by some datasets. |
| `tool_calls` | no | Array of `{"name", "arguments"}` or the OpenAI form `[{"id", "type": "function", "function": {"name", "arguments"}}]`. `arguments` may be an object or a JSON-encoded string. |
| `tool_call_id` | no | Identifier of the call a `tool` message answers. |

### Provenance

`source`, `licence` (alias `license`) and `redacted` are optional on every
example. `redacted: false` counts as present.

- With `--require-provenance`, the first example missing any field stops the run
  with, for example:
  `line 2: example is missing required provenance field `licence` (--require-provenance)`.
- Without it, the loader counts `missing_source`, `missing_licence` and
  `missing_redacted` in `DatasetStats` and the run logs one
  `dataset_missing_provenance` event after the dataset is exhausted.

## Templates

| Template | Turn tokens | Reasoning |
|----------|-------------|-----------|
| `gemma` | `<start_of_turn>{role}\n…<end_of_turn>` | dropped: Gemma 2/3 has no reasoning channel |
| `qwen` | `<|im_start|>{role}\n…<|im_end|>` | ` thinking…</think>` at the start of the last assistant turn |
| `lfm` | `<|startoftext|>` then ChatML `<|im_start|>{role}\n…<|im_end|>` | ` thinking…</think>` at the start of the last assistant turn |
| `plain` | `{role}: {content}` lines | `reasoning: …` line before the last assistant line |

`gemma` maps `assistant` to `model` and folds a leading `system` message into the
first user turn, because Gemma 2/3 has no system role. A `tool` message becomes a
user turn prefixed with `Tool response:`.

### Tool rendering

Tool definitions and calls are rendered in a stable form (JSON object keys are
sorted by `serde_json`):

- `gemma`: definitions are appended to the first user turn inside
  `<tools>…</tools>`; each call is `<tool_call>{json}</tool_call>` inside the
  model turn.
- `qwen`: definitions are appended to the system turn (a system turn is created
  if absent) with the standard `# Tools` / `<tools>…</tools>` block; each call is
  `<tool_call>{json}</tool_call>` inside the assistant turn.
- `lfm`: definitions are appended to the system turn as `List of tools: [ … ]`;
  calls are `<|tool_call_start|>[{json}, …]<|tool_call_end|>`.
- `plain`: `tools: {json}` and `tool_calls: {json}` lines.

The OpenAI `{"type": "function", "function": …}` envelope is unwrapped to
`{"name": …, "arguments": …}` so the same dataset renders consistently.

## Streaming

`.jsonl` input is read through a `BufReader` one line at a time; the iterator
owns one reusable line buffer and never reads the whole file. A 10 GB corpus is
therefore not loaded into RAM. `DatasetStats::bytes_read` exposes bytes consumed
so a caller can observe that the first example is yielded long before the file is
exhausted.

`.txt` input is one example by definition, so it is read in full once.

## Not supported

- zstd-compressed input. Decompress with `zstd -dc` first.
- Multimodal message content (arrays of image/audio parts). `content` must be a
  string or `null`.
- Loss masking. Rendering produces one flat token stream; the trainer cannot mask
  prompt tokens or train only on assistant turns.
- Per-message reasoning fields such as `reasoning_content`. Use the example-level
  `reasoning` field.
- Gemma 4's `<|turn>` / `<turn|>` template and its `<|channel>thought` reasoning
  channel. `gemma` is the Gemma 2/3 `<start_of_turn>` / `<end_of_turn>` format.
