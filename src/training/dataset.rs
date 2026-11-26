//! Streaming training-data loading and chat-template rendering.
//!
//! # Accepted input
//!
//! A dataset is one file. [`DataFormat::Auto`] picks the format from the file
//! extension:
//!
//! | Extension          | Format |
//! |--------------------|--------|
//! | `.txt`, `.text`, anything else | [`DataFormat::Text`] — the whole file is one example |
//! | `.jsonl`, `.json`  | [`DataFormat::Jsonl`] — one JSON object per line |
//! | `.zst`, `.zstd`    | refused: decompress first with `zstd -dc` |
//!
//! zstd is deliberately not a dependency: a compressed path is refused with a
//! message that names the command to run, and zstd-compressed bytes are also
//! detected by their magic number so a mislabelled file cannot slip through.
//!
//! # JSONL shapes
//!
//! Raw text, one document per line:
//!
//! ```json
//! {"text": "The quick brown fox.", "source": "book-1", "licence": "public-domain", "redacted": false}
//! ```
//!
//! A chat example:
//!
//! ```json
//! {"messages": [{"role": "system", "content": "Be terse."},
//!               {"role": "user", "content": "Hi"},
//!               {"role": "assistant", "content": "Hello."}],
//!  "tools": [{"type": "function", "function": {"name": "now", "parameters": {"type": "object"}}}],
//!  "reasoning": "The user greeted me, so I greet back.",
//!  "source": "synthetic", "licence": "cc-by-4.0", "redacted": true}
//! ```
//!
//! A line must carry exactly one of `text` or `messages`. `tools` and
//! `reasoning` are only valid with `messages`. Message objects accept
//! `role` (required), `content` (string or null), `name`, `tool_calls` and
//! `tool_call_id`; `tool_calls` is either an array of
//! `{"name", "arguments"}` objects or the OpenAI form
//! `[{"id", "type": "function", "function": {"name", "arguments"}}]`.
//! `arguments` may be a JSON object or a JSON-encoded string.
//!
//! `source`, `licence` (alias `license`) and `redacted` are optional on every
//! example. With [`DatasetOptions::require_provenance`] the first example that
//! is missing any of them aborts the run with the line number and the field
//! name. Without it the missing fields are counted in [`DatasetStats`] and the
//! caller reports them once at the end.
//!
//! # Rendering
//!
//! [`TrainingExample::render`] turns an example into the text that is tokenised
//! with `HfTokenizer::encode_raw`, exactly as the previous whole-file path did.
//! [`TrainingExample::Text`] is returned unchanged for every template, so a
//! `.txt` corpus tokenises byte-for-byte as before.
//!
//! [`ChatTemplate::Gemma`] uses the Gemma 2/3 turn format:
//!
//! ```text
//! <start_of_turn>user
//! ...<end_of_turn>
//! <start_of_turn>model
//! ...<end_of_turn>
//! ```
//!
//! A leading `system` message is folded into the first user turn (Gemma has no
//! system role). Tool definitions are appended to that same user turn inside
//! `<tools>...</tools>`. Tool calls are rendered inside the model turn as
//! `<tool_call>{...}</tool_call>`, and `tool` messages become user turns
//! prefixed with `Tool response:`. Gemma has no reasoning channel, so
//! `reasoning` is dropped for this template.
//!
//! [`ChatTemplate::Qwen`] uses ChatML:
//!
//! ```text
//! <|im_start|>system
//! ...<|im_end|>
//! <|im_start|>user
//! ...<|im_end|>
//! <|im_start|>assistant
//! ...<|im_end|>
//! ```
//!
//! Tool definitions go in the system turn between `<tools>` and `</tools>`;
//! tool calls go in the assistant turn as
//! `<tool_call>{"name": ..., "arguments": ...}</tool_call>`. `reasoning` is
//! rendered at the start of the last assistant turn as
//! ` thinking...</think>`.
//!
//! [`ChatTemplate::Lfm`] is the LFM2/LFM2.5 format: the same ChatML turns,
//! preceded by the `<|startoftext|>` BOS token, with tool definitions in the
//! system turn as `List of tools: [...]`, tool calls wrapped in
//! `<|tool_call_start|>[...]<|tool_call_end|>`, and reasoning in
//! ` thinking...</think>`. LFM carries reasoning, so it is rendered.
//!
//! [`ChatTemplate::Plain`] is a debugging format with one `role: content` line
//! per message, plus `tools: ...`, `tool_calls: ...` and `reasoning: ...`
//! lines. It renders every field.
//!
//! JSON is emitted by `serde_json`, whose object maps are sorted, so the tool
//! blocks are byte-stable across runs.
//!
//! # Streaming
//!
//! [`DatasetIter`] owns a [`BufReader`] and one reusable line buffer. It never
//! reads more than one line ahead, so a 10 GB `.jsonl` corpus is not loaded
//! into RAM. [`DatasetStats::bytes_read`] exposes the number of bytes consumed
//! so callers (and tests) can prove that the first example is produced long
//! before the file is exhausted. `.txt` input is inherently one example and is
//! read in full once.
//!
//! # Not supported
//!
//! - zstd input (decompress with `zstd -dc` first).
//! - Multimodal message content (arrays of image/audio parts); `content` must
//!   be a string or null.
//! - Loss masking: rendering produces one flat token stream, so the trainer
//!   cannot mask prompt tokens or train only on assistant turns.

use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize};

/// Magic number at the start of every zstd frame.
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

// ---------------------------------------------------------------------------
// Formats
// ---------------------------------------------------------------------------

/// How a dataset file is interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataFormat {
    /// Choose by file extension (see the module docs).
    #[default]
    Auto,
    /// The whole file is a single [`TrainingExample::Text`].
    Text,
    /// One JSON object per line.
    Jsonl,
}

impl DataFormat {
    /// Resolve [`DataFormat::Auto`] against a path.
    ///
    /// `.jsonl`/`.json` are JSONL, `.zst`/`.zstd` are refused, and everything
    /// else (including `.txt` and extension-less paths) is raw text so that
    /// existing `--data corpus.txt` invocations keep working.
    pub fn resolve(self, path: &Path) -> Result<DataFormat, DatasetError> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();

        // Compressed input is refused regardless of the requested format.
        if ext == "zst" || ext == "zstd" {
            return Err(DatasetError::CompressedInput { path: path.to_path_buf() });
        }

        match self {
            DataFormat::Auto => match ext.as_str() {
                "jsonl" | "json" => Ok(DataFormat::Jsonl),
                _ => Ok(DataFormat::Text),
            },
            other => Ok(other),
        }
    }

    /// Stable name used in logs and error messages.
    pub fn as_str(self) -> &'static str {
        match self {
            DataFormat::Auto => "auto",
            DataFormat::Text => "text",
            DataFormat::Jsonl => "jsonl",
        }
    }
}

impl FromStr for DataFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(DataFormat::Auto),
            "text" | "txt" | "raw" => Ok(DataFormat::Text),
            "jsonl" => Ok(DataFormat::Jsonl),
            other => {
                Err(format!("unknown data format `{}`; expected one of: auto, text, jsonl", other))
            }
        }
    }
}

/// Chat template used to render [`TrainingExample::Chat`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplate {
    /// Gemma 2/3 `<start_of_turn>` / `<end_of_turn>` turns.
    #[default]
    Gemma,
    /// Qwen ChatML: `<|im_start|>` / `<|im_end|>`.
    Qwen,
    /// LFM2/LFM2.5: `<|startoftext|>` plus ChatML, LFM tool tokens.
    Lfm,
    /// `role: content` lines, for inspection and raw-text corpora.
    Plain,
}

impl ChatTemplate {
    /// Stable name used in logs and error messages.
    pub fn as_str(self) -> &'static str {
        match self {
            ChatTemplate::Gemma => "gemma",
            ChatTemplate::Qwen => "qwen",
            ChatTemplate::Lfm => "lfm",
            ChatTemplate::Plain => "plain",
        }
    }

    /// Whether this template has a channel for [`TrainingExample::Chat`]'s
    /// `reasoning` field.
    ///
    /// Qwen and LFM render it as ` thinking...</think>`; Plain renders a
    /// `reasoning:` line. Gemma 2/3 has no reasoning channel and drops it.
    pub fn renders_reasoning(self) -> bool {
        !matches!(self, ChatTemplate::Gemma)
    }
}

impl FromStr for ChatTemplate {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gemma" => Ok(ChatTemplate::Gemma),
            "qwen" => Ok(ChatTemplate::Qwen),
            "lfm" => Ok(ChatTemplate::Lfm),
            "plain" => Ok(ChatTemplate::Plain),
            other => Err(format!(
                "unknown chat template `{}`; expected one of: gemma, qwen, lfm, plain",
                other
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Examples
// ---------------------------------------------------------------------------

/// One message of a chat example.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// `system`, `user`, `assistant`, `tool`, or a provider-specific role.
    pub role: String,
    /// Text content. JSON `null` and a missing field both become `""`.
    #[serde(default, deserialize_with = "string_or_null")]
    pub content: String,
    /// Optional function name, used by some tool-call datasets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool calls requested by an assistant message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    /// Identifier of the tool call this `tool` message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Convenience constructor for tests and callers.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// A single training example with optional provenance.
#[derive(Debug, Clone, PartialEq)]
pub enum TrainingExample {
    /// Raw text; rendered unchanged by every template.
    Text { text: String, source: Option<String>, licence: Option<String>, redacted: Option<bool> },
    /// A chat conversation, optional tool definitions and optional reasoning.
    Chat {
        messages: Vec<Message>,
        tools: Option<serde_json::Value>,
        reasoning: Option<String>,
        source: Option<String>,
        licence: Option<String>,
        redacted: Option<bool>,
    },
}

impl TrainingExample {
    /// Raw-text example with no provenance.
    pub fn text(text: impl Into<String>) -> Self {
        TrainingExample::Text { text: text.into(), source: None, licence: None, redacted: None }
    }

    /// Chat example with no tools, reasoning or provenance.
    pub fn chat(messages: Vec<Message>) -> Self {
        TrainingExample::Chat {
            messages,
            tools: None,
            reasoning: None,
            source: None,
            licence: None,
            redacted: None,
        }
    }

    /// Provenance fields that are absent, in the order `source`, `licence`,
    /// `redacted`.
    pub fn missing_provenance(&self) -> Vec<&'static str> {
        let (source, licence, redacted) = match self {
            TrainingExample::Text { source, licence, redacted, .. } => (source, licence, redacted),
            TrainingExample::Chat { source, licence, redacted, .. } => (source, licence, redacted),
        };

        let mut missing = Vec::new();
        if source.is_none() {
            missing.push("source");
        }
        if licence.is_none() {
            missing.push("licence");
        }
        if redacted.is_none() {
            missing.push("redacted");
        }
        missing
    }

    /// Render this example into the tokeniser's input text.
    ///
    /// [`TrainingExample::Text`] is returned unchanged for every template.
    pub fn render(&self, template: ChatTemplate) -> String {
        match self {
            TrainingExample::Text { text, .. } => text.clone(),
            TrainingExample::Chat { messages, tools, reasoning, .. } => {
                render_chat(template, messages, tools.as_ref(), reasoning.as_deref())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Options, stats, errors
// ---------------------------------------------------------------------------

/// Loader options.
#[derive(Debug, Clone, Copy, Default)]
pub struct DatasetOptions {
    /// Abort at the first example that lacks `source`, `licence` or `redacted`.
    pub require_provenance: bool,
}

/// Counters accumulated while a dataset is streamed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DatasetStats {
    /// Examples yielded so far.
    pub examples: u64,
    /// Examples whose `source` was absent (when provenance is not required).
    pub missing_source: u64,
    /// Examples whose `licence` was absent (when provenance is not required).
    pub missing_licence: u64,
    /// Examples whose `redacted` was absent (when provenance is not required).
    pub missing_redacted: u64,
    /// Bytes consumed from the file so far.
    pub bytes_read: u64,
}

impl DatasetStats {
    /// Total number of missing provenance fields counted.
    pub fn missing_total(&self) -> u64 {
        self.missing_source + self.missing_licence + self.missing_redacted
    }

    /// One-line report, or `None` when every example had full provenance.
    ///
    /// Callers log this once after the iterator is exhausted; the loader never
    /// logs per line.
    pub fn provenance_summary(&self) -> Option<String> {
        if self.missing_total() == 0 {
            return None;
        }
        Some(format!(
            "{} missing provenance field(s) across {} example(s): source={}, licence={}, redacted={}",
            self.missing_total(),
            self.examples,
            self.missing_source,
            self.missing_licence,
            self.missing_redacted
        ))
    }
}

/// Errors produced while loading a dataset.
#[derive(Debug)]
pub enum DatasetError {
    /// The file could not be opened or read.
    Io { path: PathBuf, line: Option<u64>, source: std::io::Error },
    /// zstd input is not supported.
    CompressedInput { path: PathBuf },
    /// A JSONL line is not valid JSON.
    InvalidJson { line: u64, source: serde_json::Error },
    /// A JSONL line is valid JSON but not a usable example.
    InvalidExample { line: u64, message: String },
    /// `--require-provenance` was set and a field is absent.
    MissingProvenance { line: u64, field: &'static str },
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DatasetError::Io { path, line, source } => match line {
                Some(line) => {
                    write!(f, "line {}: failed to read {}: {}", line, path.display(), source)
                }
                None => write!(f, "failed to read {}: {}", path.display(), source),
            },
            DatasetError::CompressedInput { path } => {
                let suggestion = path.with_extension("");
                write!(
                    f,
                    "refusing zstd-compressed input `{}`: zstd input is not supported. \
                     Decompress it first, for example: zstd -dc `{}` > `{}`",
                    path.display(),
                    path.display(),
                    suggestion.display()
                )
            }
            DatasetError::InvalidJson { line, source } => {
                write!(f, "line {}: invalid JSON: {}", line, source)
            }
            DatasetError::InvalidExample { line, message } => {
                write!(f, "line {}: invalid example: {}", line, message)
            }
            DatasetError::MissingProvenance { line, field } => write!(
                f,
                "line {}: example is missing required provenance field `{}` \
                 (--require-provenance)",
                line, field
            ),
        }
    }
}

impl std::error::Error for DatasetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DatasetError::Io { source, .. } => Some(source),
            DatasetError::InvalidJson { source, .. } => Some(source),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Stream a dataset with default options.
///
/// The returned [`DatasetIter`] is the `impl Iterator<Item = Result<TrainingExample>>`;
/// it is a named type so callers can read [`DatasetIter::stats`] afterwards.
pub fn load_dataset(path: &Path, format: DataFormat) -> Result<DatasetIter, DatasetError> {
    load_dataset_with(path, format, &DatasetOptions::default())
}

/// Stream a dataset with explicit options.
pub fn load_dataset_with(
    path: &Path,
    format: DataFormat,
    options: &DatasetOptions,
) -> Result<DatasetIter, DatasetError> {
    let format = format.resolve(path)?;

    let mut file = File::open(path).map_err(|e| DatasetError::Io {
        path: path.to_path_buf(),
        line: None,
        source: e,
    })?;

    // Refuse zstd content even when the extension lies.
    let mut magic = [0u8; 4];
    let mut filled = 0usize;
    while filled < magic.len() {
        match file.read(&mut magic[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                return Err(DatasetError::Io { path: path.to_path_buf(), line: None, source: e })
            }
        }
    }
    if filled == magic.len() && magic == ZSTD_MAGIC {
        return Err(DatasetError::CompressedInput { path: path.to_path_buf() });
    }

    file.seek(SeekFrom::Start(0)).map_err(|e| DatasetError::Io {
        path: path.to_path_buf(),
        line: None,
        source: e,
    })?;

    let inner = match format {
        DataFormat::Auto => unreachable!("resolve() never returns Auto"),
        DataFormat::Text => {
            let mut text = String::new();
            file.read_to_string(&mut text).map_err(|e| DatasetError::Io {
                path: path.to_path_buf(),
                line: None,
                source: e,
            })?;
            let bytes = text.len() as u64;
            Inner::Text {
                example: Some(TrainingExample::Text {
                    text,
                    source: None,
                    licence: None,
                    redacted: None,
                }),
                bytes,
            }
        }
        DataFormat::Jsonl => Inner::Jsonl(JsonlReader {
            path: path.to_path_buf(),
            reader: BufReader::new(file),
            line_no: 0,
            consumed: 0,
            eof: false,
        }),
    };

    let mut iter =
        DatasetIter { inner, options: *options, stats: DatasetStats::default(), finished: false };
    if let Inner::Text { bytes, .. } = &iter.inner {
        iter.stats.bytes_read = *bytes;
    }
    Ok(iter)
}

#[derive(Debug)]
enum Inner {
    Text { example: Option<TrainingExample>, bytes: u64 },
    Jsonl(JsonlReader),
}

#[derive(Debug)]
struct JsonlReader {
    path: PathBuf,
    reader: BufReader<File>,
    line_no: u64,
    consumed: u64,
    eof: bool,
}

impl JsonlReader {
    /// Read the next non-blank line. Returns `None` at EOF.
    fn next_line(&mut self) -> Option<Result<(u64, String), DatasetError>> {
        loop {
            if self.eof {
                return None;
            }
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    self.eof = true;
                    return None;
                }
                Ok(n) => {
                    self.consumed += n as u64;
                    self.line_no += 1;
                    if line.trim().is_empty() {
                        continue;
                    }
                    return Some(Ok((self.line_no, line)));
                }
                Err(e) => {
                    self.eof = true;
                    return Some(Err(DatasetError::Io {
                        path: self.path.clone(),
                        line: Some(self.line_no + 1),
                        source: e,
                    }));
                }
            }
        }
    }

    fn next_example(&mut self) -> Option<Result<(u64, TrainingExample), DatasetError>> {
        let (line_no, line) = match self.next_line()? {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        Some(parse_jsonl_line(line_no, &line).map(|example| (line_no, example)))
    }
}

/// Iterator over a dataset. See [`load_dataset`].
#[derive(Debug)]
pub struct DatasetIter {
    inner: Inner,
    options: DatasetOptions,
    stats: DatasetStats,
    finished: bool,
}

impl DatasetIter {
    /// Counters accumulated so far. Read after the loop for the final totals.
    pub fn stats(&self) -> &DatasetStats {
        &self.stats
    }

    /// Bytes consumed from the file so far. Useful to prove streaming.
    pub fn bytes_consumed(&self) -> u64 {
        match &self.inner {
            Inner::Text { bytes, .. } => *bytes,
            Inner::Jsonl(reader) => reader.consumed,
        }
    }

    fn check_provenance(
        &mut self,
        line: u64,
        example: &TrainingExample,
    ) -> Result<(), DatasetError> {
        let missing = example.missing_provenance();
        if missing.is_empty() {
            return Ok(());
        }
        if self.options.require_provenance {
            return Err(DatasetError::MissingProvenance { line, field: missing[0] });
        }
        for field in missing {
            match field {
                "source" => self.stats.missing_source += 1,
                "licence" => self.stats.missing_licence += 1,
                "redacted" => self.stats.missing_redacted += 1,
                _ => {}
            }
        }
        Ok(())
    }
}

impl Iterator for DatasetIter {
    type Item = Result<TrainingExample, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let next = match &mut self.inner {
            Inner::Text { example, .. } => example.take().map(|e| Ok((1, e))),
            Inner::Jsonl(reader) => reader.next_example(),
        };

        match next {
            None => {
                self.finished = true;
                None
            }
            Some(Err(e)) => {
                self.finished = true;
                Some(Err(e))
            }
            Some(Ok((line, example))) => {
                if let Err(e) = self.check_provenance(line, &example) {
                    self.finished = true;
                    return Some(Err(e));
                }
                self.stats.examples += 1;
                Some(Ok(example))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawExample {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    messages: Option<Vec<Message>>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default, alias = "license")]
    licence: Option<String>,
    #[serde(default)]
    redacted: Option<bool>,
}

impl RawExample {
    fn into_example(self, line: u64) -> Result<TrainingExample, DatasetError> {
        let provenance = (self.source, self.licence, self.redacted);

        match (self.text, self.messages) {
            (Some(text), None) => {
                if self.tools.is_some() || self.reasoning.is_some() {
                    return Err(DatasetError::InvalidExample {
                        line,
                        message: "`tools` and `reasoning` are only valid together with `messages`"
                            .to_string(),
                    });
                }
                Ok(TrainingExample::Text {
                    text,
                    source: provenance.0,
                    licence: provenance.1,
                    redacted: provenance.2,
                })
            }
            (None, Some(messages)) => {
                if messages.is_empty() {
                    return Err(DatasetError::InvalidExample {
                        line,
                        message: "`messages` must not be empty".to_string(),
                    });
                }
                for (i, message) in messages.iter().enumerate() {
                    if message.role.trim().is_empty() {
                        return Err(DatasetError::InvalidExample {
                            line,
                            message: format!("messages[{}].role must not be empty", i),
                        });
                    }
                }
                Ok(TrainingExample::Chat {
                    messages,
                    tools: self.tools,
                    reasoning: self.reasoning,
                    source: provenance.0,
                    licence: provenance.1,
                    redacted: provenance.2,
                })
            }
            (Some(_), Some(_)) => Err(DatasetError::InvalidExample {
                line,
                message: "example has both `text` and `messages`; use exactly one".to_string(),
            }),
            (None, None) => Err(DatasetError::InvalidExample {
                line,
                message: "example has neither `text` nor `messages`".to_string(),
            }),
        }
    }
}

fn parse_jsonl_line(line_no: u64, line: &str) -> Result<TrainingExample, DatasetError> {
    let raw: RawExample = serde_json::from_str(line)
        .map_err(|source| DatasetError::InvalidJson { line: line_no, source })?;
    raw.into_example(line_no)
}

/// Accept a JSON string, `null`, or a missing field; reject arrays/objects with
/// a message that names the unsupported multimodal case.
fn string_or_null<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;

    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    match value {
        None | Some(serde_json::Value::Null) => Ok(String::new()),
        Some(serde_json::Value::String(s)) => Ok(s),
        Some(serde_json::Value::Array(_)) | Some(serde_json::Value::Object(_)) => Err(
            D::Error::custom(
                "message `content` must be a string or null; multimodal content arrays are not supported",
            ),
        ),
        Some(other) => Ok(other.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn render_chat(
    template: ChatTemplate,
    messages: &[Message],
    tools: Option<&serde_json::Value>,
    reasoning: Option<&str>,
) -> String {
    match template {
        ChatTemplate::Gemma => render_gemma(messages, tools, reasoning),
        ChatTemplate::Qwen => render_chatml(None, messages, tools, reasoning, ToolsStyle::Qwen),
        ChatTemplate::Lfm => {
            render_chatml(Some("<|startoftext|>"), messages, tools, reasoning, ToolsStyle::Lfm)
        }
        ChatTemplate::Plain => render_plain(messages, tools, reasoning),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ToolsStyle {
    Qwen,
    Lfm,
}

/// ChatML renderer shared by Qwen and LFM. LFM differs only by its BOS token,
/// tool-list syntax and tool-call tokens.
fn render_chatml(
    bos: Option<&str>,
    messages: &[Message],
    tools: Option<&serde_json::Value>,
    reasoning: Option<&str>,
    style: ToolsStyle,
) -> String {
    let mut out = String::new();
    if let Some(bos) = bos {
        out.push_str(bos);
    }

    let last_assistant = messages.iter().rposition(|m| m.role == "assistant");
    let first_system = messages.iter().position(|m| m.role == "system");

    // Tool definitions: fold into the first system turn, or synthesise one.
    let tools_block = tools.map(|tools| match style {
        ToolsStyle::Qwen => format!(
            "\n\n# Tools\n\n\
             You may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n\
             <tools>\n{}\n</tools>\n\n\
             For each function call, return a JSON object with the function name and \
             arguments within <tool_call></tool_call> XML tags.",
            pretty_json(tools)
        ),
        ToolsStyle::Lfm => format!("\n\n{}", lfm_tool_list(tools)),
    });

    if first_system.is_none() {
        if let Some(block) = tools_block.as_ref() {
            out.push_str("<|im_start|>system\n");
            out.push_str("You are a helpful assistant.");
            out.push_str(block);
            out.push_str("<|im_end|>\n");
        }
    }

    for (i, message) in messages.iter().enumerate() {
        out.push_str("<|im_start|>");
        out.push_str(&message.role);
        out.push('\n');

        if message.role == "system" {
            out.push_str(&message.content);
            if first_system == Some(i) {
                if let Some(block) = tools_block.as_ref() {
                    out.push_str(block);
                }
            }
        } else if message.role == "assistant" {
            if last_assistant == Some(i) {
                if let Some(reasoning) = reasoning {
                    out.push_str(" thinking\n");
                    out.push_str(reasoning);
                    out.push_str("\n</think>\n\n");
                }
            }
            out.push_str(&message.content);
            if let Some(tool_calls) = message.tool_calls.as_ref() {
                render_tool_calls_chatml(&mut out, tool_calls, style);
            }
        } else {
            out.push_str(&message.content);
        }

        out.push_str("<|im_end|>\n");
    }

    out
}

fn render_tool_calls_chatml(out: &mut String, tool_calls: &serde_json::Value, style: ToolsStyle) {
    let calls = normalize_tool_calls(tool_calls);
    match style {
        ToolsStyle::Qwen => {
            for call in calls {
                out.push_str("\n<tool_call>\n");
                out.push_str(&compact_json(&call));
                out.push_str("\n</tool_call>");
            }
        }
        ToolsStyle::Lfm => {
            out.push_str("\n<|tool_call_start|>[");
            for (i, call) in calls.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&compact_json(call));
            }
            out.push_str("]<|tool_call_end|>");
        }
    }
}

/// Gemma 2/3 `<start_of_turn>` renderer.
fn render_gemma(
    messages: &[Message],
    tools: Option<&serde_json::Value>,
    _reasoning: Option<&str>,
) -> String {
    let mut out = String::new();

    // Gemma has no system role: fold a leading system message and the tool
    // definitions into the first user turn.
    let mut preamble: Option<String> = None;
    let rest = if messages.first().map(|m| m.role == "system").unwrap_or(false) {
        preamble = Some(messages[0].content.clone());
        &messages[1..]
    } else {
        messages
    };

    if let Some(tools) = tools {
        let block = format!("Available tools (JSON):\n<tools>\n{}\n</tools>", pretty_json(tools));
        preamble = Some(match preamble {
            Some(system) => format!("{}\n\n{}", system, block),
            None => block,
        });
    }

    let has_user = rest.iter().any(|m| m.role == "user");
    if !has_user {
        if let Some(preamble) = preamble.take() {
            out.push_str("<start_of_turn>user\n");
            out.push_str(&preamble);
            out.push_str("<end_of_turn>\n");
        }
    }

    for message in rest.iter() {
        let role = match message.role.as_str() {
            "assistant" => "model",
            "tool" | "system" => "user",
            other => other,
        };

        out.push_str("<start_of_turn>");
        out.push_str(role);
        out.push('\n');

        if message.role == "user" {
            if let Some(preamble) = preamble.take() {
                out.push_str(&preamble);
                out.push_str("\n\n");
            }
        }
        if message.role == "tool" {
            out.push_str("Tool response:\n");
        }
        out.push_str(&message.content);

        if message.role == "assistant" {
            // Gemma 2/3 has no reasoning channel, so `reasoning` is dropped.
            if let Some(tool_calls) = message.tool_calls.as_ref() {
                for call in normalize_tool_calls(tool_calls) {
                    out.push_str("\n<tool_call>\n");
                    out.push_str(&compact_json(&call));
                    out.push_str("\n</tool_call>");
                }
            }
        }

        out.push_str("<end_of_turn>\n");
    }

    out
}

fn render_plain(
    messages: &[Message],
    tools: Option<&serde_json::Value>,
    reasoning: Option<&str>,
) -> String {
    let mut out = String::new();
    if let Some(tools) = tools {
        out.push_str("tools: ");
        out.push_str(&compact_json(tools));
        out.push('\n');
    }

    let last_assistant = messages.iter().rposition(|m| m.role == "assistant");
    for (i, message) in messages.iter().enumerate() {
        if last_assistant == Some(i) {
            if let Some(reasoning) = reasoning {
                out.push_str("reasoning: ");
                out.push_str(reasoning);
                out.push('\n');
            }
        }
        out.push_str(&message.role);
        out.push_str(": ");
        out.push_str(&message.content);
        out.push('\n');
        if let Some(tool_calls) = message.tool_calls.as_ref() {
            out.push_str("tool_calls: ");
            out.push_str(&compact_json(tool_calls));
            out.push('\n');
        }
    }
    out
}

/// Flatten a tool-call value into `[{"name", "arguments"}]` objects.
///
/// Accepts a single object or an array, and unwraps the OpenAI
/// `{"type": "function", "function": {"name", "arguments"}}` envelope.
/// `arguments` strings that contain JSON are parsed so the rendered text is
/// stable regardless of which shape the dataset used.
fn normalize_tool_calls(tool_calls: &serde_json::Value) -> Vec<serde_json::Value> {
    let items: Vec<&serde_json::Value> = match tool_calls {
        serde_json::Value::Array(items) => items.iter().collect(),
        other => vec![other],
    };

    items
        .into_iter()
        .map(|item| {
            let function = item.get("function");
            let Some(function) = function else {
                return item.clone();
            };

            let mut call = serde_json::Map::new();
            if let Some(name) = function.get("name") {
                call.insert("name".to_string(), name.clone());
            }
            if let Some(arguments) = function.get("arguments") {
                let arguments = match arguments {
                    serde_json::Value::String(s) => serde_json::from_str::<serde_json::Value>(s)
                        .unwrap_or_else(|_| arguments.clone()),
                    other => other.clone(),
                };
                call.insert("arguments".to_string(), arguments);
            }
            if let Some(id) = item.get("id") {
                call.insert("id".to_string(), id.clone());
            }
            serde_json::Value::Object(call)
        })
        .collect()
}

fn lfm_tool_list(tools: &serde_json::Value) -> String {
    match tools {
        serde_json::Value::Array(items) => {
            let rendered: Vec<String> = items.iter().map(compact_json).collect();
            format!("List of tools: [{}]", rendered.join(", "))
        }
        other => format!("List of tools: [{}]", compact_json(other)),
    }
}

fn compact_json(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
}

fn pretty_json(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tokenizer::HfTokenizer;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    /// Unique scratch directory; tests clean up after themselves.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "ferrisres_dataset_{}_{}_{}",
                tag,
                std::process::id(),
                id
            ));
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }

        fn file(&self, name: &str, contents: &str) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, contents).unwrap();
            path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Minimal but valid HuggingFace tokenizer used for the equality test.
    fn tiny_tokenizer(dir: &TempDir) -> HfTokenizer {
        let json = serde_json::json!({
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "[UNK]": 0,
                    "<bos>": 1,
                    "<eos>": 2,
                    "hello": 3,
                    "world": 4,
                    "second": 5,
                    "line": 6
                },
                "unk_token": "[UNK]"
            },
            "pre_tokenizer": { "type": "Whitespace" },
            "added_tokens": [
                { "id": 0, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
                { "id": 1, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
                { "id": 2, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true }
            ]
        });
        let path = dir.file("tokenizer.json", &serde_json::to_string_pretty(&json).unwrap());
        HfTokenizer::from_tokenizer_json(&path).expect("tiny tokenizer loads")
    }

    fn collect(iter: &mut DatasetIter) -> Vec<TrainingExample> {
        iter.by_ref().map(|r| r.expect("example loads")).collect()
    }

    #[test]
    fn text_file_loads_as_one_example_and_tokenises_like_the_old_path() {
        let dir = TempDir::new("text_equality");
        let text = "hello world\nsecond line\n";
        let path = dir.file("corpus.txt", text);
        let tokenizer = tiny_tokenizer(&dir);

        // The path this change replaces.
        let old_ids = tokenizer.encode_raw(&std::fs::read_to_string(&path).unwrap());

        let mut iter = load_dataset(&path, DataFormat::Auto).unwrap();
        let example = iter.next().unwrap().unwrap();
        assert!(iter.next().is_none(), "a .txt file must yield exactly one example");
        assert!(matches!(example, TrainingExample::Text { .. }));
        assert_eq!(iter.stats().examples, 1);

        for template in
            [ChatTemplate::Gemma, ChatTemplate::Qwen, ChatTemplate::Lfm, ChatTemplate::Plain]
        {
            let rendered = example.render(template);
            assert_eq!(rendered, text, "text must be identity under {:?}", template);
            assert_eq!(
                tokenizer.encode_raw(&rendered),
                old_ids,
                "token ids must match the old whole-file path under {:?}",
                template
            );
        }
    }

    #[test]
    fn jsonl_text_lines_load_one_example_per_line() {
        let dir = TempDir::new("jsonl_text");
        let path = dir.file(
            "corpus.jsonl",
            "{\"text\":\"hello\"}\n{\"text\":\"world\"}\n\n{\"text\":\"second\"}\n",
        );

        let mut iter = load_dataset(&path, DataFormat::Auto).unwrap();
        let examples = collect(&mut iter);
        assert_eq!(examples.len(), 3);
        assert_eq!(examples[0], TrainingExample::text("hello"));
        assert_eq!(examples[1], TrainingExample::text("world"));
        assert_eq!(examples[2], TrainingExample::text("second"));
        assert_eq!(iter.stats().examples, 3);
    }

    fn chat_example() -> TrainingExample {
        let mut assistant = Message::new("assistant", "The answer is 4.");
        assistant.tool_calls = Some(serde_json::json!([
            { "id": "call_1", "type": "function",
              "function": { "name": "add", "arguments": "{\"a\":2,\"b\":2}" } }
        ]));

        TrainingExample::Chat {
            messages: vec![
                Message::new("system", "Be terse."),
                Message::new("user", "What is 2 + 2?"),
                assistant,
                Message::new("tool", "4"),
            ],
            tools: Some(serde_json::json!([
                { "type": "function",
                  "function": { "name": "add", "parameters": { "type": "object" } } }
            ])),
            reasoning: Some("I should add the two numbers.".to_string()),
            source: Some("unit-test".to_string()),
            licence: Some("cc0-1.0".to_string()),
            redacted: Some(false),
        }
    }

    #[test]
    fn chat_renders_with_each_template_and_its_special_tokens() {
        let example = chat_example();

        let gemma = example.render(ChatTemplate::Gemma);
        assert!(gemma.contains("<start_of_turn>user"), "gemma: {gemma}");
        assert!(gemma.contains("<start_of_turn>model"), "gemma: {gemma}");
        assert!(gemma.contains("<end_of_turn>"), "gemma: {gemma}");
        assert!(gemma.contains("Be terse."), "system must be folded into the user turn");
        assert!(gemma.contains("<tools>"), "tool definitions must be rendered");
        assert!(gemma.contains("<tool_call>"), "tool calls must be rendered");
        assert!(gemma.contains("Tool response:"), "tool results must be rendered");
        assert!(
            !gemma.contains("I should add the two numbers."),
            "gemma has no reasoning channel, so reasoning is dropped"
        );

        let qwen = example.render(ChatTemplate::Qwen);
        assert!(qwen.contains("<|im_start|>system"), "qwen: {qwen}");
        assert!(qwen.contains("<|im_start|>user"), "qwen: {qwen}");
        assert!(qwen.contains("<|im_start|>assistant"), "qwen: {qwen}");
        assert!(qwen.contains("<|im_end|>"), "qwen: {qwen}");
        assert!(qwen.contains("<tools>"), "qwen: {qwen}");
        assert!(qwen.contains("<tool_call>"), "qwen: {qwen}");
        assert!(qwen.contains(" thinking"), "qwen carries reasoning: {qwen}");
        assert!(qwen.contains("I should add the two numbers."), "qwen: {qwen}");

        let lfm = example.render(ChatTemplate::Lfm);
        assert!(lfm.starts_with("<|startoftext|>"), "lfm: {lfm}");
        assert!(lfm.contains("<|im_start|>user"), "lfm: {lfm}");
        assert!(lfm.contains("<|im_end|>"), "lfm: {lfm}");
        assert!(lfm.contains("List of tools: ["), "lfm: {lfm}");
        assert!(lfm.contains("<|tool_call_start|>["), "lfm: {lfm}");
        assert!(lfm.contains("<|tool_call_end|>"), "lfm: {lfm}");
        assert!(lfm.contains("I should add the two numbers."), "lfm: {lfm}");

        let plain = example.render(ChatTemplate::Plain);
        assert!(plain.contains("system: Be terse."), "plain: {plain}");
        assert!(plain.contains("user: What is 2 + 2?"), "plain: {plain}");
        assert!(plain.contains("assistant: The answer is 4."), "plain: {plain}");
        assert!(plain.contains("tool_calls: "), "plain: {plain}");
        assert!(plain.contains("reasoning: "), "plain: {plain}");
    }

    #[test]
    fn chat_jsonl_renders_with_each_template() {
        let dir = TempDir::new("jsonl_chat");
        let path = dir.file(
            "chat.jsonl",
            concat!(
                "{\"messages\":[{\"role\":\"system\",\"content\":\"Be terse.\"},",
                "{\"role\":\"user\",\"content\":\"Hi\"},",
                "{\"role\":\"assistant\",\"content\":\"Hello.\"}],",
                "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"now\"}}],",
                "\"reasoning\":\"Greet back.\",",
                "\"source\":\"synthetic\",\"licence\":\"cc-by-4.0\",\"redacted\":true}\n"
            ),
        );

        let mut iter = load_dataset(&path, DataFormat::Auto).unwrap();
        let examples = collect(&mut iter);
        assert_eq!(examples.len(), 1);
        assert!(matches!(examples[0], TrainingExample::Chat { .. }));

        assert!(examples[0].render(ChatTemplate::Gemma).contains("<start_of_turn>user"));
        assert!(examples[0].render(ChatTemplate::Qwen).contains("<|im_start|>assistant"));
        assert!(examples[0].render(ChatTemplate::Lfm).contains("<|startoftext|>"));
        assert!(examples[0].render(ChatTemplate::Plain).contains("user: Hi"));
    }

    #[test]
    fn malformed_line_is_refused_with_its_line_number() {
        let dir = TempDir::new("malformed");
        let path =
            dir.file("bad.jsonl", "{\"text\":\"ok\"}\n{\"text\":\n{\"text\":\"never reached\"}\n");

        let mut iter = load_dataset(&path, DataFormat::Auto).unwrap();
        assert!(iter.next().unwrap().is_ok());
        let err = iter.next().unwrap().unwrap_err();
        let message = err.to_string();
        assert!(message.contains("line 2"), "got: {message}");
        assert!(message.contains("invalid JSON"), "got: {message}");
        assert!(iter.next().is_none(), "the iterator stops at the first error");
    }

    #[test]
    fn example_with_both_text_and_messages_is_refused() {
        let dir = TempDir::new("both");
        let path = dir.file(
            "both.jsonl",
            "{\"text\":\"a\",\"messages\":[{\"role\":\"user\",\"content\":\"b\"}]}\n",
        );

        let err =
            load_dataset(&path, DataFormat::Auto).unwrap().next().unwrap().unwrap_err().to_string();
        assert!(err.contains("line 1"), "got: {err}");
        assert!(err.contains("both `text` and `messages`"), "got: {err}");
    }

    #[test]
    fn multimodal_content_is_refused_with_a_clear_message() {
        let dir = TempDir::new("multimodal");
        let path = dir.file(
            "mm.jsonl",
            "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}]}\n",
        );

        let err =
            load_dataset(&path, DataFormat::Auto).unwrap().next().unwrap().unwrap_err().to_string();
        assert!(err.contains("line 1"), "got: {err}");
        assert!(err.contains("multimodal content arrays are not supported"), "got: {err}");
    }

    #[test]
    fn zst_path_is_refused_naming_zstd_dc() {
        let dir = TempDir::new("zst");
        let path = dir.file("corpus.jsonl.zst", "not really zstd");
        let err = DataFormat::Auto.resolve(&path).unwrap_err().to_string();
        assert!(err.contains("zstd -dc"), "got: {err}");
        assert!(err.contains("corpus.jsonl.zst"), "got: {err}");

        // The loader refuses it too, even when the format is explicit.
        let err = load_dataset(&path, DataFormat::Jsonl).unwrap_err().to_string();
        assert!(err.contains("zstd -dc"), "got: {err}");
    }

    #[test]
    fn zstd_magic_bytes_are_refused_even_with_a_txt_extension() {
        let dir = TempDir::new("zstd_magic");
        let path = dir.0.join("disguised.txt");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&ZSTD_MAGIC).unwrap();
        file.write_all(b"rest of a zstd frame").unwrap();
        drop(file);

        let err = load_dataset(&path, DataFormat::Text).unwrap_err().to_string();
        assert!(err.contains("zstd -dc"), "got: {err}");
    }

    #[test]
    fn require_provenance_refuses_missing_licence_and_names_field_and_line() {
        let dir = TempDir::new("provenance_strict");
        let path = dir.file(
            "prov.jsonl",
            concat!(
                "{\"text\":\"ok\",\"source\":\"s\",\"licence\":\"mit\",\"redacted\":false}\n",
                "{\"text\":\"bad\",\"source\":\"s\",\"redacted\":true}\n"
            ),
        );

        let options = DatasetOptions { require_provenance: true };
        let mut iter = load_dataset_with(&path, DataFormat::Auto, &options).unwrap();
        assert!(iter.next().unwrap().is_ok());

        let err = iter.next().unwrap().unwrap_err();
        match &err {
            DatasetError::MissingProvenance { line, field } => {
                assert_eq!(*line, 2);
                assert_eq!(*field, "licence");
            }
            other => panic!("expected MissingProvenance, got {:?}", other),
        }
        let message = err.to_string();
        assert!(message.contains("line 2"), "got: {message}");
        assert!(message.contains("licence"), "got: {message}");
        assert!(iter.next().is_none(), "the loader refuses at the first bad example");
    }

    #[test]
    fn require_provenance_refuses_a_txt_file_at_line_one() {
        let dir = TempDir::new("provenance_text");
        let path = dir.file("corpus.txt", "hello\n");
        let options = DatasetOptions { require_provenance: true };

        let err = load_dataset_with(&path, DataFormat::Auto, &options)
            .unwrap()
            .next()
            .unwrap()
            .unwrap_err()
            .to_string();
        assert!(err.contains("line 1"), "got: {err}");
        assert!(err.contains("source"), "got: {err}");
    }

    #[test]
    fn missing_provenance_is_counted_without_the_flag() {
        let dir = TempDir::new("provenance_count");
        let path = dir.file(
            "prov.jsonl",
            concat!(
                "{\"text\":\"a\",\"source\":\"s\",\"licence\":\"mit\",\"redacted\":false}\n",
                "{\"text\":\"b\",\"source\":\"s\",\"redacted\":true}\n",
                "{\"text\":\"c\",\"licence\":\"mit\",\"redacted\":false}\n"
            ),
        );

        let mut iter = load_dataset(&path, DataFormat::Auto).unwrap();
        assert_eq!(collect(&mut iter).len(), 3);
        let stats = iter.stats();
        assert_eq!(stats.examples, 3);
        assert_eq!(stats.missing_licence, 1);
        assert_eq!(stats.missing_source, 1);
        assert_eq!(stats.missing_redacted, 0);
        assert_eq!(stats.missing_total(), 2);

        let summary = stats.provenance_summary().unwrap();
        assert!(summary.contains("source=1"), "got: {summary}");
        assert!(summary.contains("licence=1"), "got: {summary}");
    }

    #[test]
    fn ten_thousand_line_file_is_streamed() {
        let dir = TempDir::new("streaming");
        let path = dir.0.join("big.jsonl");
        {
            let file = std::fs::File::create(&path).unwrap();
            let mut writer = std::io::BufWriter::new(file);
            for i in 0..10_000 {
                writeln!(writer, "{{\"text\":\"line {:05} {}\"}}", i, "x".repeat(80)).unwrap();
            }
            writer.flush().unwrap();
        }
        let file_len = std::fs::metadata(&path).unwrap().len();
        assert!(file_len > 500_000, "fixture should be large: {} bytes", file_len);

        let mut iter = load_dataset(&path, DataFormat::Auto).unwrap();

        // Proof of streaming: the first example is produced after reading only
        // the first line (plus BufReader's internal 8 KiB buffer), not the file.
        let first = iter.next().unwrap().unwrap();
        assert!(matches!(first, TrainingExample::Text { .. }));
        assert_eq!(iter.stats().examples, 1);
        assert!(
            iter.bytes_consumed() <= 64 * 1024,
            "first example consumed {} of {} bytes; the loader must not read the whole file",
            iter.bytes_consumed(),
            file_len
        );
        assert!(
            iter.bytes_consumed() < file_len / 4,
            "first example consumed {} of {} bytes",
            iter.bytes_consumed(),
            file_len
        );

        // The rest still streams to completion.
        let remaining = iter.map(|r| r.unwrap()).count();
        assert_eq!(remaining + 1, 10_000);
    }

    #[test]
    fn auto_format_uses_extensions() {
        assert_eq!(DataFormat::Auto.resolve(Path::new("a.txt")).unwrap(), DataFormat::Text);
        assert_eq!(DataFormat::Auto.resolve(Path::new("a.jsonl")).unwrap(), DataFormat::Jsonl);
        assert_eq!(DataFormat::Auto.resolve(Path::new("a.json")).unwrap(), DataFormat::Jsonl);
        assert_eq!(DataFormat::Auto.resolve(Path::new("no_extension")).unwrap(), DataFormat::Text);
    }

    #[test]
    fn from_str_rejects_unknown_values() {
        assert_eq!("auto".parse::<DataFormat>().unwrap(), DataFormat::Auto);
        assert_eq!("jsonl".parse::<DataFormat>().unwrap(), DataFormat::Jsonl);
        assert!("parquet".parse::<DataFormat>().is_err());

        assert_eq!("gemma".parse::<ChatTemplate>().unwrap(), ChatTemplate::Gemma);
        assert_eq!("lfm".parse::<ChatTemplate>().unwrap(), ChatTemplate::Lfm);
        assert!("llama".parse::<ChatTemplate>().is_err());
    }

    #[test]
    fn tool_calls_are_normalised_and_stable() {
        let raw = serde_json::json!([
            { "function": { "name": "add", "arguments": "{\"b\":2,\"a\":1}" } },
            { "name": "now", "arguments": {} }
        ]);
        let normalized = normalize_tool_calls(&raw);
        assert_eq!(normalized.len(), 2);
        // serde_json sorts object keys, so the rendered JSON is stable.
        assert_eq!(
            compact_json(&normalized[0]),
            "{\"arguments\":{\"a\":1,\"b\":2},\"name\":\"add\"}"
        );
        assert_eq!(compact_json(&normalized[1]), "{\"arguments\":{},\"name\":\"now\"}");
    }

    #[test]
    fn plain_renders_text_unchanged() {
        let example = TrainingExample::text("hello world");
        assert_eq!(example.render(ChatTemplate::Plain), "hello world");
        assert_eq!(example.render(ChatTemplate::Gemma), "hello world");
    }
}
