//! Jev-style decision head: a small calibrated readout on a frozen hidden state.
//!
//! The head answers one typed question per forward pass and produces no text.
//! It maps a hidden vector through a linear map `W: [hidden, labels]`, applies
//! an optional soft cap (like an LM head), divides by a fitted temperature and
//! normalises over the labels that belong to the question:
//!
//! - [`Question::Choice`] — softmax over the option tokens.
//! - [`Question::Score`] — softmax over ordered level tokens, plus the
//!   probability-weighted level.
//! - [`Question::Noul`] — sigmoid over the yes/no token pair.
//!
//! Each option, level and the yes/no pair are single tokens, so the tokenizer
//! that produced the token ids is part of the artefact ([`DecisionHead::tokenizer_id`]).
//!
//! This module is deliberately self-contained: no LM, no tokenizer, no I/O
//! beyond a small versioned byte format. The synthetic tests train and
//! calibrate a head on Gaussian class prototypes, which proves that the maths
//! and the artefact format work; it does not claim that a head answers real
//! decisions for a real model.

use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Magic bytes of the head artefact.
pub const MAGIC: [u8; 8] = *b"FRDECHD\0";
/// Current artefact version.
pub const FORMAT_VERSION: u16 = 1;
/// A temperature below this would amplify instead of calibrate.
pub const MIN_TEMPERATURE: f32 = 0.01;
/// Maximum number of options a `Choice` question may carry.
pub const MAX_CHOICE_OPTIONS: usize = 255;
/// Maximum number of ordered levels a `Score` question may carry.
pub const MAX_SCORE_LEVELS: usize = 10;

// ---------------------------------------------------------------------------
// Questions
// ---------------------------------------------------------------------------

/// One option of a [`Question::Choice`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionSpec {
    /// The single-token key the option is scored on, for example `"R"`.
    pub key: String,
    /// Optional human-readable description. Not sent through the head.
    pub description: Option<String>,
}

impl OptionSpec {
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            description: None,
        }
    }

    pub fn with_description(key: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            description: Some(description.into()),
        }
    }
}

/// A bounded question answered against one hidden state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Question {
    /// Pick one of 1..=255 options.
    Choice { options: Vec<OptionSpec> },
    /// Pick one of 2..=10 ordered levels.
    Score { levels: Vec<String> },
    /// Answer a yes/no statement.
    Noul { statement: String },
}

impl Question {
    /// Validate the question and return the label keys in answer order.
    ///
    /// For `Noul` the order is `["no", "yes"]`, so `probabilities[1]` is the
    /// probability of yes.
    pub fn keys(&self) -> Result<Vec<&str>, DecisionHeadError> {
        match self {
            Question::Choice { options } => {
                if options.is_empty() {
                    return Err(DecisionHeadError::InvalidQuestion(
                        "choice needs at least one option".to_string(),
                    ));
                }
                if options.len() > MAX_CHOICE_OPTIONS {
                    return Err(DecisionHeadError::InvalidQuestion(format!(
                        "choice carries {} options, the maximum is {}",
                        options.len(),
                        MAX_CHOICE_OPTIONS
                    )));
                }
                let mut seen = HashSet::with_capacity(options.len());
                for option in options {
                    if option.key.is_empty() {
                        return Err(DecisionHeadError::InvalidQuestion(
                            "option key is empty".to_string(),
                        ));
                    }
                    if !seen.insert(option.key.as_str()) {
                        return Err(DecisionHeadError::InvalidQuestion(format!(
                            "duplicate option key `{}`",
                            option.key
                        )));
                    }
                }
                Ok(options.iter().map(|o| o.key.as_str()).collect())
            }
            Question::Score { levels } => {
                if !(2..=MAX_SCORE_LEVELS).contains(&levels.len()) {
                    return Err(DecisionHeadError::InvalidQuestion(format!(
                        "score carries {} levels, the range is 2..={}",
                        levels.len(),
                        MAX_SCORE_LEVELS
                    )));
                }
                let mut seen = HashSet::with_capacity(levels.len());
                for level in levels {
                    if level.is_empty() {
                        return Err(DecisionHeadError::InvalidQuestion(
                            "score level is empty".to_string(),
                        ));
                    }
                    if !seen.insert(level.as_str()) {
                        return Err(DecisionHeadError::InvalidQuestion(format!(
                            "duplicate score level `{}`",
                            level
                        )));
                    }
                }
                Ok(levels.iter().map(|l| l.as_str()).collect())
            }
            Question::Noul { statement } => {
                if statement.trim().is_empty() {
                    return Err(DecisionHeadError::InvalidQuestion(
                        "noul statement is empty".to_string(),
                    ));
                }
                Ok(vec!["no", "yes"])
            }
        }
    }

    /// Whether answers to this question should be normalised as a binary pair.
    pub fn is_binary(&self) -> bool {
        matches!(self, Question::Noul { .. })
    }
}

// ---------------------------------------------------------------------------
// Artefact types
// ---------------------------------------------------------------------------

/// A single-token label: the token id that is scored and the key that binds it
/// to an option, level or the yes/no pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelSpec {
    pub token_id: u32,
    /// `"yes"`, `"no"`, an option key or a level string.
    pub key: String,
}

impl LabelSpec {
    pub fn new(token_id: u32, key: impl Into<String>) -> Self {
        Self {
            token_id,
            key: key.into(),
        }
    }
}

/// A positive answer to one question.
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    /// Probability per option/level, aligned with the question. For `Noul`
    /// this is `[p(no), p(yes)]`.
    pub probabilities: Vec<f32>,
    /// Index of the most likely option/level. For `Noul`, `1` means yes.
    pub selected: usize,
    /// Maximum probability. Never below `1 / probabilities.len()`.
    pub confidence: f32,
    /// `Some(Σ p_i · i)` for `Score`, `None` otherwise. Levels are zero-based,
    /// matching the order in the question.
    pub expected_level: Option<f32>,
}

/// The trained head.
///
/// `weights` is row-major `[hidden, labels]`: `weights[hidden_index * labels.len() + label_index]`.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionHead {
    pub weights: Vec<f32>,
    /// Fitted temperature, always `>= MIN_TEMPERATURE`.
    pub temperature: f32,
    /// Optional soft cap: `(x / cap).tanh() * cap`.
    pub softcap: Option<f32>,
    /// The tokenizer whose ids the labels refer to.
    pub tokenizer_id: u16,
    pub labels: Vec<LabelSpec>,
}

/// Named failures of the head, its questions and its artefact format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionHeadError {
    /// The weight matrix, hidden dimension and label count disagree.
    Shape(String),
    /// The tokenizer id / label table does not define a key the question names.
    UnknownLabel(String),
    /// The question violates its own cardinality or uniqueness rules.
    InvalidQuestion(String),
    /// Temperature is not finite or is below [`MIN_TEMPERATURE`].
    InvalidTemperature(u32),
    /// Soft cap is not finite or is not strictly positive.
    InvalidSoftcap(u32),
    /// The artefact does not start with [`MAGIC`].
    BadMagic([u8; 8]),
    /// The artefact version is not [`FORMAT_VERSION`].
    UnsupportedVersion(u16),
    /// The soft-cap flag byte is neither 0 nor 1.
    BadSoftcapFlag(u8),
    /// A label key is not valid UTF-8.
    InvalidUtf8,
    /// The artefact ends before the header promised.
    Truncated { expected: usize, found: usize },
    /// Bytes remain after the declared labels.
    TrailingBytes(usize),
    /// Reading or writing the artefact failed.
    Io(String),
}

impl fmt::Display for DecisionHeadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecisionHeadError::Shape(why) => write!(f, "decision head shape error: {}", why),
            DecisionHeadError::UnknownLabel(key) => {
                write!(f, "no label token is declared for key `{}`", key)
            }
            DecisionHeadError::InvalidQuestion(why) => {
                write!(f, "invalid decision question: {}", why)
            }
            DecisionHeadError::InvalidTemperature(bits) => write!(
                f,
                "invalid decision head temperature {} (minimum {})",
                f32::from_bits(*bits),
                MIN_TEMPERATURE
            ),
            DecisionHeadError::InvalidSoftcap(bits) => write!(
                f,
                "invalid decision head soft cap {} (must be finite and > 0)",
                f32::from_bits(*bits)
            ),
            DecisionHeadError::BadMagic(found) => {
                write!(f, "bad decision head magic {:?}", found)
            }
            DecisionHeadError::UnsupportedVersion(version) => {
                write!(f, "unsupported decision head version {}", version)
            }
            DecisionHeadError::BadSoftcapFlag(flag) => {
                write!(f, "bad decision head soft-cap flag {}", flag)
            }
            DecisionHeadError::InvalidUtf8 => {
                write!(f, "decision head label key is not valid UTF-8")
            }
            DecisionHeadError::Truncated { expected, found } => write!(
                f,
                "decision head artefact truncated: needed {} bytes, found {}",
                expected, found
            ),
            DecisionHeadError::TrailingBytes(count) => {
                write!(f, "decision head artefact has {} trailing bytes", count)
            }
            DecisionHeadError::Io(why) => write!(f, "decision head io error: {}", why),
        }
    }
}

impl std::error::Error for DecisionHeadError {}

impl From<std::io::Error> for DecisionHeadError {
    fn from(e: std::io::Error) -> Self {
        DecisionHeadError::Io(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Head
// ---------------------------------------------------------------------------

impl DecisionHead {
    /// Build and validate a head.
    pub fn new(
        weights: Vec<f32>,
        temperature: f32,
        softcap: Option<f32>,
        tokenizer_id: u16,
        labels: Vec<LabelSpec>,
    ) -> Result<Self, DecisionHeadError> {
        let head = Self {
            weights,
            temperature,
            softcap,
            tokenizer_id,
            labels,
        };
        head.validate()?;
        Ok(head)
    }

    /// Check the internal invariants and return a named error on mismatch.
    pub fn validate(&self) -> Result<(), DecisionHeadError> {
        if self.labels.is_empty() {
            return Err(DecisionHeadError::Shape(
                "a head needs at least one label".to_string(),
            ));
        }
        if !self.temperature.is_finite() || self.temperature < MIN_TEMPERATURE {
            return Err(DecisionHeadError::InvalidTemperature(
                self.temperature.to_bits(),
            ));
        }
        if let Some(cap) = self.softcap {
            if !cap.is_finite() || cap <= 0.0 {
                return Err(DecisionHeadError::InvalidSoftcap(cap.to_bits()));
            }
        }
        let expected = self
            .labels
            .len()
            .checked_mul(self.expected_hidden()?)
            .ok_or_else(|| DecisionHeadError::Shape("weight matrix size overflows".to_string()))?;
        if self.weights.len() != expected {
            return Err(DecisionHeadError::Shape(format!(
                "weights hold {} values, {} hidden x {} labels needs {}",
                self.weights.len(),
                self.expected_hidden()?,
                self.labels.len(),
                expected
            )));
        }
        let mut seen = HashSet::with_capacity(self.labels.len());
        for label in &self.labels {
            if !seen.insert(label.key.as_str()) {
                return Err(DecisionHeadError::Shape(format!(
                    "duplicate label key `{}`",
                    label.key
                )));
            }
        }
        Ok(())
    }

    /// Hidden dimension inferred from the weight matrix.
    pub fn expected_hidden(&self) -> Result<usize, DecisionHeadError> {
        if self.labels.is_empty() || self.weights.is_empty() {
            return Err(DecisionHeadError::Shape(
                "weights and labels must both be non-empty".to_string(),
            ));
        }
        if self.weights.len() % self.labels.len() != 0 {
            return Err(DecisionHeadError::Shape(format!(
                "{} weights do not divide into {} labels",
                self.weights.len(),
                self.labels.len()
            )));
        }
        Ok(self.weights.len() / self.labels.len())
    }

    /// Number of labels (columns of `W`).
    pub fn label_count(&self) -> usize {
        self.labels.len()
    }

    fn label_index(&self, key: &str) -> Option<usize> {
        self.labels.iter().position(|label| label.key == key)
    }

    /// Soft-capped logits for the question's labels, before temperature.
    ///
    /// The order matches [`Question::keys`]; `Noul` is `[no, yes]`.
    pub fn logits(
        &self,
        hidden: &[f32],
        question: &Question,
    ) -> Result<Vec<f32>, DecisionHeadError> {
        let expected_hidden = self.expected_hidden()?;
        if hidden.len() != expected_hidden {
            return Err(DecisionHeadError::Shape(format!(
                "hidden state has {} values, the head expects {}",
                hidden.len(),
                expected_hidden
            )));
        }
        let n_labels = self.labels.len();
        let mut out = Vec::new();
        for key in question.keys()? {
            let label = self.label_index(key).ok_or_else(|| {
                DecisionHeadError::UnknownLabel(key.to_string())
            })?;
            let mut sum = 0.0f32;
            for (h_idx, &value) in hidden.iter().enumerate() {
                sum += value * self.weights[h_idx * n_labels + label];
            }
            out.push(apply_softcap(sum, self.softcap));
        }
        Ok(out)
    }

    /// Answer one question from one hidden state.
    pub fn forward(
        &self,
        hidden: &[f32],
        question: &Question,
    ) -> Result<Answer, DecisionHeadError> {
        if !self.temperature.is_finite() || self.temperature < MIN_TEMPERATURE {
            return Err(DecisionHeadError::InvalidTemperature(
                self.temperature.to_bits(),
            ));
        }
        let raw = self.logits(hidden, question)?;
        let scaled: Vec<f32> = raw
            .iter()
            .map(|logit| logit / self.temperature)
            .collect();

        if question.is_binary() {
            // Binary softmax over [no, yes] is the sigmoid of the difference.
            let diff = scaled[1] - scaled[0];
            let yes = sigmoid(diff);
            let probabilities = vec![1.0 - yes, yes];
            let selected = if yes >= 0.5 { 1 } else { 0 };
            let confidence = probabilities[selected];
            return Ok(Answer {
                probabilities,
                selected,
                confidence,
                expected_level: None,
            });
        }

        let probabilities = softmax(&scaled);
        let (selected, confidence) = argmax(&probabilities);
        let expected_level = match question {
            Question::Score { .. } => Some(
                probabilities
                    .iter()
                    .enumerate()
                    .map(|(index, p)| index as f32 * p)
                    .sum(),
            ),
            _ => None,
        };
        Ok(Answer {
            probabilities,
            selected,
            confidence,
            expected_level,
        })
    }

    /// Stable 64-bit hash of the head's contents, useful for shadow logs and
    /// for asserting that saving and loading did not change anything.
    pub fn fingerprint(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |byte: u8| {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        };
        for value in &self.weights {
            for byte in value.to_bits().to_le_bytes() {
                mix(byte);
            }
        }
        for byte in self.temperature.to_bits().to_le_bytes() {
            mix(byte);
        }
        match self.softcap {
            Some(cap) => {
                mix(1);
                for byte in cap.to_bits().to_le_bytes() {
                    mix(byte);
                }
            }
            None => mix(0),
        }
        for byte in self.tokenizer_id.to_le_bytes() {
            mix(byte);
        }
        for label in &self.labels {
            for byte in label.token_id.to_le_bytes() {
                mix(byte);
            }
            for byte in label.key.as_bytes() {
                mix(*byte);
            }
        }
        hash
    }

    /// Write the versioned artefact.
    ///
    /// Layout: magic (8), version (u16), tokenizer id (u16), hidden (u32),
    /// label count (u32), temperature (f32), soft-cap flag (u8) and optional
    /// soft cap (f32), weights (`hidden * labels` f32), then per label a token
    /// id (u32), key length (u16) and key bytes. All integers little-endian.
    pub fn save<W: Write>(&self, mut writer: W) -> Result<(), DecisionHeadError> {
        self.validate()?;
        let hidden = self.expected_hidden()?;
        writer.write_all(&MAGIC)?;
        writer.write_all(&FORMAT_VERSION.to_le_bytes())?;
        writer.write_all(&self.tokenizer_id.to_le_bytes())?;
        writer.write_all(&(hidden as u32).to_le_bytes())?;
        writer.write_all(&(self.labels.len() as u32).to_le_bytes())?;
        writer.write_all(&self.temperature.to_le_bytes())?;
        match self.softcap {
            Some(cap) => {
                writer.write_all(&[1u8])?;
                writer.write_all(&cap.to_le_bytes())?;
            }
            None => writer.write_all(&[0u8])?,
        }
        for weight in &self.weights {
            writer.write_all(&weight.to_le_bytes())?;
        }
        for label in &self.labels {
            if label.key.len() > u16::MAX as usize {
                return Err(DecisionHeadError::Shape(format!(
                    "label key is {} bytes, the maximum is {}",
                    label.key.len(),
                    u16::MAX
                )));
            }
            writer.write_all(&label.token_id.to_le_bytes())?;
            writer.write_all(&(label.key.len() as u16).to_le_bytes())?;
            writer.write_all(label.key.as_bytes())?;
        }
        Ok(())
    }

    /// Save to a path.
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), DecisionHeadError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.save(&mut writer)?;
        writer.flush()?;
        Ok(())
    }

    /// Read a versioned artefact and reject any internal mismatch.
    pub fn load<R: Read>(mut reader: R) -> Result<Self, DecisionHeadError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Load from a path.
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, DecisionHeadError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::load(&mut reader)
    }

    /// Parse an in-memory artefact.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DecisionHeadError> {
        let mut cursor = 0usize;
        let magic: [u8; 8] = take(bytes, &mut cursor, 8)?.try_into().unwrap();
        if magic != MAGIC {
            return Err(DecisionHeadError::BadMagic(magic));
        }
        let version = read_u16(bytes, &mut cursor)?;
        if version != FORMAT_VERSION {
            return Err(DecisionHeadError::UnsupportedVersion(version));
        }
        let tokenizer_id = read_u16(bytes, &mut cursor)?;
        let hidden = read_u32(bytes, &mut cursor)? as usize;
        let label_count = read_u32(bytes, &mut cursor)? as usize;
        if hidden == 0 || label_count == 0 {
            return Err(DecisionHeadError::Shape(
                "artefact declares zero hidden dimension or zero labels".to_string(),
            ));
        }
        let temperature = read_f32(bytes, &mut cursor)?;
        let softcap_flag = take(bytes, &mut cursor, 1)?[0];
        let softcap = match softcap_flag {
            0 => None,
            1 => Some(read_f32(bytes, &mut cursor)?),
            other => return Err(DecisionHeadError::BadSoftcapFlag(other)),
        };

        let weight_count = hidden.checked_mul(label_count).ok_or_else(|| {
            DecisionHeadError::Shape("artefact weight count overflows".to_string())
        })?;
        let weight_bytes = weight_count.checked_mul(4).ok_or_else(|| {
            DecisionHeadError::Shape("artefact weight byte count overflows".to_string())
        })?;
        let remaining = bytes.len().saturating_sub(cursor);
        if remaining < weight_bytes {
            return Err(DecisionHeadError::Truncated {
                expected: cursor + weight_bytes,
                found: bytes.len(),
            });
        }
        let mut weights = Vec::with_capacity(weight_count);
        for _ in 0..weight_count {
            weights.push(read_f32(bytes, &mut cursor)?);
        }

        let mut labels = Vec::with_capacity(label_count);
        for _ in 0..label_count {
            let token_id = read_u32(bytes, &mut cursor)?;
            let key_len = read_u16(bytes, &mut cursor)? as usize;
            let key_bytes = take(bytes, &mut cursor, key_len)?;
            let key = std::str::from_utf8(key_bytes)
                .map_err(|_| DecisionHeadError::InvalidUtf8)?
                .to_string();
            labels.push(LabelSpec { token_id, key });
        }
        if cursor != bytes.len() {
            return Err(DecisionHeadError::TrailingBytes(bytes.len() - cursor));
        }

        let head = Self::new(weights, temperature, softcap, tokenizer_id, labels)?;
        // The declared shape must match what was read; `new` validates the
        // weight count against the actual labels, so a header that lied about
        // `hidden` will not survive the roundtrip.
        if head.expected_hidden()? != hidden {
            return Err(DecisionHeadError::Shape(format!(
                "artefact declares hidden {}, weights imply {}",
                hidden,
                head.expected_hidden()?
            )));
        }
        Ok(head)
    }
}

// ---------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------

fn apply_softcap(value: f32, cap: Option<f32>) -> f32 {
    match cap {
        Some(cap) => (value / cap).tanh() * cap,
        None => value,
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically stable softmax.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let total: f32 = out.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        // Degenerate logits (NaN/overflow): fall back to a uniform answer.
        let uniform = 1.0 / out.len() as f32;
        return vec![uniform; out.len()];
    }
    for p in &mut out {
        *p /= total;
    }
    out
}

fn argmax(probabilities: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_p = f32::NEG_INFINITY;
    for (index, &p) in probabilities.iter().enumerate() {
        if p > best_p {
            best_p = p;
            best = index;
        }
    }
    (best, best_p)
}

/// Cross-entropy of one probability vector against a target index.
pub fn cross_entropy(probabilities: &[f32], target: usize) -> f32 {
    let p = probabilities.get(target).copied().unwrap_or(0.0);
    -(p.max(f32::MIN_POSITIVE)).ln()
}

/// Expected calibration error over equal-width confidence bins.
///
/// `probabilities` is one distribution per example and `targets` the correct
/// index. Confidence is the maximum probability. An empty set has ECE 0.
pub fn expected_calibration_error(
    probabilities: &[Vec<f32>],
    targets: &[usize],
    bins: usize,
) -> f32 {
    if probabilities.is_empty() || bins == 0 {
        return 0.0;
    }
    let mut counts = vec![0usize; bins];
    let mut correct = vec![0usize; bins];
    let mut confidence_sum = vec![0.0f32; bins];
    for (distribution, &target) in probabilities.iter().zip(targets) {
        let (selected, confidence) = argmax(distribution);
        let mut bucket = (confidence * bins as f32) as usize;
        if bucket >= bins {
            bucket = bins - 1;
        }
        counts[bucket] += 1;
        confidence_sum[bucket] += confidence;
        if selected == target {
            correct[bucket] += 1;
        }
    }
    let total = probabilities.len() as f32;
    let mut ece = 0.0f32;
    for bucket in 0..bins {
        if counts[bucket] == 0 {
            continue;
        }
        let accuracy = correct[bucket] as f32 / counts[bucket] as f32;
        let confidence = confidence_sum[bucket] / counts[bucket] as f32;
        ece += (counts[bucket] as f32 / total) * (accuracy - confidence).abs();
    }
    ece
}

/// Fit one temperature by minimising negative log-likelihood on the supplied
/// logits. The caller must pass the calibration split only; there is no
/// internal train/test split here.
pub fn fit_temperature(raw_logits: &[Vec<f32>], targets: &[usize]) -> f32 {
    if raw_logits.is_empty() {
        return 1.0;
    }
    const CANDIDATES: usize = 240;
    let low = MIN_TEMPERATURE.ln();
    let high = 20.0f32.ln();
    let mut best_temp = 1.0f32;
    let mut best_nll = f32::INFINITY;
    for step in 0..CANDIDATES {
        let fraction = step as f32 / (CANDIDATES - 1) as f32;
        let temperature = (low + (high - low) * fraction).exp();
        let mut nll = 0.0f32;
        for (logits, &target) in raw_logits.iter().zip(targets) {
            let scaled: Vec<f32> = logits.iter().map(|x| x / temperature).collect();
            nll += cross_entropy(&softmax(&scaled), target);
        }
        if nll < best_nll {
            best_nll = nll;
            best_temp = temperature;
        }
    }
    best_temp.max(MIN_TEMPERATURE)
}

// ---------------------------------------------------------------------------
// Synthetic training
// ---------------------------------------------------------------------------

/// One labelled hidden state for linear-head training.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingExample {
    pub hidden: Vec<f32>,
    pub label: usize,
}

/// Train a linear softmax head with cross-entropy and Adam.
///
/// The weights are all the head has; the frozen model that produced `hidden`
/// is not involved. Returns row-major `[hidden, labels]` weights.
pub fn train_linear_head(
    examples: &[TrainingExample],
    hidden_dim: usize,
    num_labels: usize,
    epochs: usize,
    batch_size: usize,
    learning_rate: f32,
    seed: u64,
) -> Result<Vec<f32>, DecisionHeadError> {
    if examples.is_empty() || hidden_dim == 0 || num_labels == 0 {
        return Err(DecisionHeadError::Shape(
            "training needs examples, a hidden dimension and labels".to_string(),
        ));
    }
    for example in examples {
        if example.hidden.len() != hidden_dim {
            return Err(DecisionHeadError::Shape(format!(
                "training example has {} hidden values, expected {}",
                example.hidden.len(),
                hidden_dim
            )));
        }
        if example.label >= num_labels {
            return Err(DecisionHeadError::Shape(format!(
                "training label {} is outside 0..{}",
                example.label, num_labels
            )));
        }
    }
    if batch_size == 0 || epochs == 0 {
        return Err(DecisionHeadError::Shape(
            "training needs a non-zero epoch and batch count".to_string(),
        ));
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(DecisionHeadError::Shape(
            "learning rate must be finite and positive".to_string(),
        ));
    }

    let mut weights = vec![0.0f32; hidden_dim * num_labels];
    let mut first_moment = vec![0.0f32; weights.len()];
    let mut second_moment = vec![0.0f32; weights.len()];
    let mut rng = Rng::new(seed);
    let mut order: Vec<usize> = (0..examples.len()).collect();
    let mut gradient = vec![0.0f32; weights.len()];
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.999;
    const EPSILON: f32 = 1e-8;
    let mut step = 0u64;

    for _ in 0..epochs {
        // Deterministic Fisher-Yates shuffle for this epoch.
        for index in (1..order.len()).rev() {
            let swap = (rng.next_u64() % (index as u64 + 1)) as usize;
            order.swap(index, swap);
        }
        for batch in order.chunks(batch_size) {
            gradient.fill(0.0);
            let inv_batch = 1.0 / batch.len() as f32;
            for &sample in batch {
                let example = &examples[sample];
                let mut logits = vec![0.0f32; num_labels];
                for (label, logit) in logits.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for (h_idx, &value) in example.hidden.iter().enumerate() {
                        sum += value * weights[h_idx * num_labels + label];
                    }
                    *logit = sum;
                }
                let mut probs = softmax(&logits);
                probs[example.label] -= 1.0;
                for (h_idx, &value) in example.hidden.iter().enumerate() {
                    for (label, &delta) in probs.iter().enumerate() {
                        gradient[h_idx * num_labels + label] += value * delta * inv_batch;
                    }
                }
            }

            step += 1;
            let bias1 = 1.0 - BETA1.powi(step as i32);
            let bias2 = 1.0 - BETA2.powi(step as i32);
            for index in 0..weights.len() {
                let g = gradient[index];
                first_moment[index] = BETA1 * first_moment[index] + (1.0 - BETA1) * g;
                second_moment[index] = BETA2 * second_moment[index] + (1.0 - BETA2) * g * g;
                let m_hat = first_moment[index] / bias1;
                let v_hat = second_moment[index] / bias2;
                weights[index] -= learning_rate * m_hat / (v_hat.sqrt() + EPSILON);
            }
        }
    }
    Ok(weights)
}

/// Deterministic SplitMix64 plus Box-Muller. No external RNG so a fixed seed
/// produces exactly the same synthetic data and training run on every platform.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1]`.
    #[cfg(test)]
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        ((bits as f32) + 1.0) / ((1u32 << 24) as f32)
    }

    #[cfg(test)]
    fn normal(&mut self) -> f32 {
        let u1 = self.next_f32();
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Byte helpers
// ---------------------------------------------------------------------------

fn take<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    count: usize,
) -> Result<&'a [u8], DecisionHeadError> {
    let end = cursor
        .checked_add(count)
        .ok_or_else(|| DecisionHeadError::Shape("artifact cursor overflows".to_string()))?;
    if end > bytes.len() {
        return Err(DecisionHeadError::Truncated {
            expected: end,
            found: bytes.len(),
        });
    }
    let slice = &bytes[*cursor..end];
    *cursor = end;
    Ok(slice)
}

fn read_u16(bytes: &[u8], cursor: &mut usize) -> Result<u16, DecisionHeadError> {
    let raw = take(bytes, cursor, 2)?;
    Ok(u16::from_le_bytes([raw[0], raw[1]]))
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, DecisionHeadError> {
    let raw = take(bytes, cursor, 4)?;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

fn read_f32(bytes: &[u8], cursor: &mut usize) -> Result<f32, DecisionHeadError> {
    let raw = take(bytes, cursor, 4)?;
    Ok(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

// ---------------------------------------------------------------------------
// Tests: synthetic training, calibration and artefact format
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const HIDDEN: usize = 64;
    const CLASSES: usize = 8;
    const TRAIN: usize = 2_000;
    const CALIBRATION: usize = 1_000;
    const PROTOTYPE_SCALE: f32 = 0.6;
    const NOISE_SCALE: f32 = 1.4;

    fn class_question() -> Question {
        Question::Choice {
            options: (0..CLASSES)
                .map(|c| OptionSpec::new(format!("c{}", c)))
                .collect(),
        }
    }

    fn class_labels() -> Vec<LabelSpec> {
        (0..CLASSES)
            .map(|c| LabelSpec::new(1_000 + c as u32, format!("c{}", c)))
            .collect()
    }

    fn head_with_weights(weights: Vec<f32>, temperature: f32, softcap: Option<f32>) -> DecisionHead {
        DecisionHead::new(weights, temperature, softcap, 7, class_labels()).expect("valid head")
    }

    fn zero_head() -> DecisionHead {
        head_with_weights(vec![0.0; HIDDEN * CLASSES], 1.0, None)
    }

    fn prototypes(seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Rng::new(seed);
        (0..CLASSES)
            .map(|_| {
                (0..HIDDEN)
                    .map(|_| rng.normal() * PROTOTYPE_SCALE)
                    .collect()
            })
            .collect()
    }

    fn synthetic(count: usize, seed: u64, prototypes: &[Vec<f32>]) -> Vec<TrainingExample> {
        let mut rng = Rng::new(seed);
        (0..count)
            .map(|_| {
                let label = (rng.next_u64() % CLASSES as u64) as usize;
                let hidden = prototypes[label]
                    .iter()
                    .map(|&prototype| prototype + rng.normal() * NOISE_SCALE)
                    .collect();
                TrainingExample { hidden, label }
            })
            .collect()
    }

    fn accuracy(
        head: &DecisionHead,
        examples: &[TrainingExample],
        question: &Question,
    ) -> f32 {
        let mut correct = 0usize;
        for example in examples {
            let answer = head.forward(&example.hidden, question).expect("forward");
            if answer.selected == example.label {
                correct += 1;
            }
        }
        correct as f32 / examples.len() as f32
    }

    fn raw_logits(
        head: &DecisionHead,
        examples: &[TrainingExample],
        question: &Question,
    ) -> Vec<Vec<f32>> {
        examples
            .iter()
            .map(|example| head.logits(&example.hidden, question).expect("logits"))
            .collect()
    }

    fn targets(examples: &[TrainingExample]) -> Vec<usize> {
        examples.iter().map(|example| example.label).collect()
    }

    fn calibrated_probabilities(
        logits: &[Vec<f32>],
        temperature: f32,
    ) -> Vec<Vec<f32>> {
        logits
            .iter()
            .map(|row| {
                let scaled: Vec<f32> = row.iter().map(|x| x / temperature).collect();
                softmax(&scaled)
            })
            .collect()
    }

    /// The proof the module exists for: train on prototypes plus noise, hit
    /// the accuracy gate, then fit one temperature on a separate split and
    /// bring ECE below 0.05.
    #[test]
    fn synthetic_head_learns_and_calibrates() {
        let question = class_question();
        let prototypes = prototypes(0x5eed_1234);
        let train = synthetic(TRAIN, 11, &prototypes);
        let calibration = synthetic(CALIBRATION, 22, &prototypes);

        let weights = train_linear_head(
            &train, HIDDEN, CLASSES, 80, 32, 0.02, 0xabc_def,
        )
        .expect("train");
        let mut head = head_with_weights(weights, 1.0, None);

        let train_accuracy = accuracy(&head, &train, &question);
        let held_out_accuracy = accuracy(&head, &calibration, &question);
        assert!(
            train_accuracy > 0.90,
            "training accuracy {:.3} should exceed 0.90",
            train_accuracy
        );
        assert!(
            held_out_accuracy > 0.90,
            "held-out accuracy {:.3} should exceed 0.90",
            held_out_accuracy
        );

        let logits = raw_logits(&head, &calibration, &question);
        let targets = targets(&calibration);
        let before = expected_calibration_error(
            &calibrated_probabilities(&logits, head.temperature),
            &targets,
            10,
        );
        let fitted = fit_temperature(&logits, &targets);
        assert!(fitted >= MIN_TEMPERATURE);
        let after = expected_calibration_error(
            &calibrated_probabilities(&logits, fitted),
            &targets,
            10,
        );
        assert!(
            after < before,
            "temperature scaling must improve ECE (before {:.4}, after {:.4}, T={:.3})",
            before,
            after,
            fitted
        );
        assert!(
            after <= 0.05,
            "calibrated ECE {:.4} must be at most 0.05 (before {:.4}, T={:.3})",
            after,
            before,
            fitted
        );

        head.temperature = fitted;
        let answer = head
            .forward(&calibration[0].hidden, &question)
            .expect("forward");
        assert_eq!(answer.probabilities.len(), CLASSES);
        let sum: f32 = answer.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probabilities sum to {}", sum);
        println!(
            "decision-head synthetic: train_acc={:.4} heldout_acc={:.4} ece_before={:.4} ece_after={:.4} temperature={:.4}",
            train_accuracy, held_out_accuracy, before, after, fitted
        );
    }

    #[test]
    fn choice_answer_has_one_probability_per_option() {
        let head = zero_head();
        let options: Vec<OptionSpec> = (0..CLASSES)
            .map(|c| OptionSpec::with_description(format!("c{}", c), format!("class {}", c)))
            .collect();
        let question = Question::Choice { options };
        let answer = head
            .forward(&vec![0.25; HIDDEN], &question)
            .expect("forward");
        assert_eq!(answer.probabilities.len(), CLASSES);
        assert!(answer.selected < CLASSES);
        assert!(answer.expected_level.is_none());
        assert!((answer.confidence - 1.0 / CLASSES as f32).abs() < 1e-6);
        let sum: f32 = answer.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn score_answer_returns_a_weighted_level() {
        let labels = vec![
            LabelSpec::new(20, "low"),
            LabelSpec::new(21, "medium"),
            LabelSpec::new(22, "high"),
            LabelSpec::new(23, "critical"),
        ];
        let head = DecisionHead::new(
            vec![0.1; HIDDEN * labels.len()],
            1.0,
            None,
            3,
            labels,
        )
        .expect("head");
        let question = Question::Score {
            levels: vec![
                "low".to_string(),
                "medium".to_string(),
                "high".to_string(),
                "critical".to_string(),
            ],
        };
        let answer = head
            .forward(&vec![0.5; HIDDEN], &question)
            .expect("forward");
        assert_eq!(answer.probabilities.len(), 4);
        let expected = answer.expected_level.expect("score has an expected level");
        assert!(
            (0.0..=3.0).contains(&expected),
            "expected level {} is outside the ordered range",
            expected
        );
        let sum: f32 = answer.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn noul_answer_is_a_binary_pair() {
        let labels = vec![LabelSpec::new(30, "no"), LabelSpec::new(31, "yes")];
        let head = DecisionHead::new(
            vec![0.05; HIDDEN * labels.len()],
            1.0,
            None,
            4,
            labels,
        )
        .expect("head");
        let question = Question::Noul {
            statement: "this action is read-only".to_string(),
        };
        let answer = head
            .forward(&vec![1.0; HIDDEN], &question)
            .expect("forward");
        assert_eq!(answer.probabilities.len(), 2);
        assert!((answer.probabilities[0] + answer.probabilities[1] - 1.0).abs() < 1e-6);
        assert!(answer.selected <= 1);
        assert!((answer.confidence - answer.probabilities[answer.selected]).abs() < 1e-6);
        assert!(answer.expected_level.is_none());
    }

    #[test]
    fn noul_prefers_yes_when_the_yes_logit_is_larger() {
        let labels = vec![LabelSpec::new(30, "no"), LabelSpec::new(31, "yes")];
        // Column 0 is "no", column 1 is "yes"; a positive hidden favours yes.
        let mut weights = vec![0.0f32; 2 * 2];
        weights[0 * 2 + 1] = 1.0;
        weights[1 * 2 + 1] = 1.0;
        let head =
            DecisionHead::new(weights, 1.0, None, 4, labels).expect("valid head");
        let question = Question::Noul {
            statement: "is it safe".to_string(),
        };
        let answer = head.forward(&[1.0, 1.0], &question).expect("forward");
        assert_eq!(answer.selected, 1);
        assert!(answer.probabilities[1] > 0.5);
        let low = head.forward(&[-1.0, -1.0], &question).expect("forward");
        assert_eq!(low.selected, 0);
    }

    #[test]
    fn softcap_bounds_every_logit() {
        let cap = 3.0f32;
        let mut weights = vec![0.0f32; HIDDEN * CLASSES];
        for w in &mut weights {
            *w = 10.0;
        }
        let head = head_with_weights(weights, 1.0, Some(cap));
        let logits = head
            .logits(&vec![1.0; HIDDEN], &class_question())
            .expect("logits");
        for logit in logits {
            assert!(
                logit.abs() <= cap + 1e-6,
                "soft cap {} must bound logit {}",
                cap,
                logit
            );
            assert!(
                logit.is_finite(),
                "soft cap must keep the logit finite"
            );
        }
    }

    #[test]
    fn temperature_scaling_preserves_the_argmax() {
        let head = zero_head();
        let hidden: Vec<f32> = (0..HIDDEN)
            .map(|index| (index as f32 * 0.13).sin())
            .collect();
        let question = class_question();
        let reference = head.forward(&hidden, &question).expect("forward");
        for temperature in [0.05f32, 0.5, 1.0, 2.0, 10.0, 50.0] {
            let mut scaled = head.clone();
            scaled.temperature = temperature;
            let answer = scaled.forward(&hidden, &question).expect("forward");
            assert_eq!(
                answer.selected, reference.selected,
                "temperature {} changed the argmax",
                temperature
            );
        }
    }

    #[test]
    fn training_is_deterministic_for_a_fixed_seed() {
        let prototypes = prototypes(99);
        let train = synthetic(300, 7, &prototypes);
        let first =
            train_linear_head(&train, HIDDEN, CLASSES, 5, 32, 0.02, 42).expect("train");
        let second =
            train_linear_head(&train, HIDDEN, CLASSES, 5, 32, 0.02, 42).expect("train");
        assert_eq!(first, second);
        let third =
            train_linear_head(&train, HIDDEN, CLASSES, 5, 32, 0.02, 43).expect("train");
        assert_ne!(first, third, "a different seed must change the run");

        let first_head = head_with_weights(first, 1.0, None);
        let second_head = head_with_weights(second, 1.0, None);
        assert_eq!(first_head.fingerprint(), second_head.fingerprint());
        let example = &train[0];
        assert_eq!(
            first_head
                .forward(&example.hidden, &class_question())
                .expect("forward"),
            second_head
                .forward(&example.hidden, &class_question())
                .expect("forward")
        );
    }

    #[test]
    fn save_load_roundtrip_preserves_the_head() {
        let prototypes = prototypes(5);
        let train = synthetic(200, 6, &prototypes);
        let weights =
            train_linear_head(&train, HIDDEN, CLASSES, 3, 32, 0.02, 123).expect("train");
        let head = head_with_weights(weights, 1.75, Some(6.0));

        let mut bytes = Vec::new();
        head.save(&mut bytes).expect("save to memory");
        let loaded = DecisionHead::from_bytes(&bytes).expect("load from bytes");
        assert_eq!(loaded, head);
        assert_eq!(loaded.fingerprint(), head.fingerprint());

        let path = std::env::temp_dir().join(format!(
            "ferrisres-decision-head-{}.bin",
            std::process::id()
        ));
        head.save_to_file(&path).expect("save to file");
        let from_file = DecisionHead::load_from_file(&path).expect("load from file");
        let _ = std::fs::remove_file(&path);
        assert_eq!(from_file, head);

        // Temperature and soft cap survive exactly, so argmax and probabilities
        // do too.
        let hidden: Vec<f32> = (0..HIDDEN).map(|i| i as f32 / HIDDEN as f32).collect();
        assert_eq!(
            head.forward(&hidden, &class_question()).expect("forward"),
            loaded.forward(&hidden, &class_question()).expect("forward")
        );
    }

    #[test]
    fn loader_rejects_bad_magic_version_and_shape() {
        let head = zero_head();
        let mut bytes = Vec::new();
        head.save(&mut bytes).expect("save");

        let mut bad_magic = bytes.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            DecisionHead::from_bytes(&bad_magic),
            Err(DecisionHeadError::BadMagic(_))
        ));

        let mut bad_version = bytes.clone();
        bad_version[8..10].copy_from_slice(&99u16.to_le_bytes());
        assert!(matches!(
            DecisionHead::from_bytes(&bad_version),
            Err(DecisionHeadError::UnsupportedVersion(99))
        ));

        // Claim more labels than the weight block can satisfy.
        let mut bad_shape = bytes.clone();
        bad_shape[14..18].copy_from_slice(&(CLASSES as u32 + 1).to_le_bytes());
        let result = DecisionHead::from_bytes(&bad_shape);
        assert!(
            matches!(
                result,
                Err(DecisionHeadError::Truncated { .. })
                    | Err(DecisionHeadError::Shape(_))
            ),
            "a lying label count must be rejected, got {:?}",
            result
        );

        // A truncated weight block is not a smaller head.
        let mut truncated = bytes.clone();
        truncated.truncate(bytes.len() - 9);
        assert!(matches!(
            DecisionHead::from_bytes(&truncated),
            Err(DecisionHeadError::Truncated { .. })
        ));

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(matches!(
            DecisionHead::from_bytes(&trailing),
            Err(DecisionHeadError::TrailingBytes(1))
        ));
    }

    #[test]
    fn questions_of_the_wrong_shape_are_rejected() {
        let head = zero_head();

        let empty = Question::Choice { options: vec![] };
        assert!(matches!(
            head.forward(&vec![0.0; HIDDEN], &empty),
            Err(DecisionHeadError::InvalidQuestion(_))
        ));

        let too_many = Question::Choice {
            options: (0..=MAX_CHOICE_OPTIONS)
                .map(|index| OptionSpec::new(format!("o{}", index)))
                .collect(),
        };
        assert!(matches!(
            head.forward(&vec![0.0; HIDDEN], &too_many),
            Err(DecisionHeadError::InvalidQuestion(_))
        ));

        let one_level = Question::Score {
            levels: vec!["only".to_string()],
        };
        assert!(matches!(
            head.forward(&vec![0.0; HIDDEN], &one_level),
            Err(DecisionHeadError::InvalidQuestion(_))
        ));

        let missing = Question::Choice {
            options: vec![OptionSpec::new("not-a-label")],
        };
        assert!(matches!(
            head.forward(&vec![0.0; HIDDEN], &missing),
            Err(DecisionHeadError::UnknownLabel(_))
        ));

        assert!(matches!(
            head.forward(&vec![0.0; HIDDEN - 1], &class_question()),
            Err(DecisionHeadError::Shape(_))
        ));
    }

    #[test]
    fn softmax_and_ece_are_sane() {
        let uniform = softmax(&[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(uniform, vec![0.25; 4]);
        let peaked = softmax(&[100.0, 0.0, 0.0, 0.0]);
        assert!(peaked[0] > 0.999);
        assert!((peaked.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        let probs = vec![vec![0.9, 0.1], vec![0.6, 0.4], vec![0.5, 0.5]];
        let targets = vec![0, 1, 0];
        let ece = expected_calibration_error(&probs, &targets, 10);
        assert!(ece > 0.0 && ece < 1.0);
        assert_eq!(expected_calibration_error(&[], &[], 10), 0.0);
    }
}
