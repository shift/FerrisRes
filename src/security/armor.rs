//! Armor orchestrator: ArmorLayer tying L0-L3 + self-learning feedback.
//!
//! The ArmorLayer provides a unified security interface:
//! - `verify_input()`: L0 regex+bloom + L1 neural scanner
//! - `verify_hidden()`: L2 RepE safety probe on hidden states
//! - `sanitize_output()`: L3 PII redaction
//! - Self-learning feedback loop via ViolationRecord storage

use crate::security::armor_l0::L0Scanner;
use crate::security::armor_l1::L1Scanner;
use crate::security::armor_l2::L2Prober;
use crate::security::armor_l3::L3Sanitizer;

// ---------------------------------------------------------------------------
// Security verdict
// ---------------------------------------------------------------------------

/// Overall security verdict.
#[derive(Debug, Clone, PartialEq)]
pub enum SecurityVerdict {
    /// Input/output is safe.
    Allow,
    /// Input/output is blocked with a reason.
    Block(String),
    /// Output contains redacted content.
    Redact(String),
}

// ---------------------------------------------------------------------------
// Armor configuration
// ---------------------------------------------------------------------------

/// Armor configuration.
#[derive(Debug, Clone)]
pub struct ArmorConfig {
    /// Enable L0 regex + bloom checks.
    pub l0_enabled: bool,
    /// Enable L1 neural scanner.
    pub l1_enabled: bool,
    /// Enable L2 representation probe.
    pub l2_enabled: bool,
    /// Enable L3 output sanitizer.
    pub l3_enabled: bool,
    /// L1 injection score threshold (0.0-1.0).
    pub l1_threshold: f32,
    /// L2 unsafe score threshold (0.0-1.0).
    pub l2_threshold: f32,
    /// Maximum violation history size.
    pub max_violation_history: usize,
}

impl Default for ArmorConfig {
    fn default() -> Self {
        Self {
            l0_enabled: true,
            l1_enabled: true,
            l2_enabled: true,
            l3_enabled: true,
            l1_threshold: 0.7,
            l2_threshold: 0.7,
            max_violation_history: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// Violation record — self-learning feedback
// ---------------------------------------------------------------------------

/// Source layer that detected the violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationSource {
    L0Regex,
    L0Bloom,
    L1Neural,
    L2Probe,
    L3Redaction,
    WasmSandbox,
    External,
}

/// A recorded security violation for self-learning feedback.
#[derive(Debug, Clone)]
pub struct ViolationRecord {
    /// The input text that triggered the violation.
    pub input: String,
    /// Which layer detected it.
    pub source: ViolationSource,
    /// Score from the detecting layer (0.0-1.0).
    pub score: f32,
    /// Reason string.
    pub reason: String,
    /// Unix timestamp (seconds).
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// ArmorLayer — the orchestrator
// ---------------------------------------------------------------------------

/// The Armor security layer orchestrator.
///
/// Ties L0-L3 together for comprehensive input/output security,
/// with self-learning feedback for continuous improvement.
pub struct ArmorLayer {
    /// L0: regex + bloom static scanner.
    pub l0: L0Scanner,
    /// L1: neural injection scanner.
    pub l1: L1Scanner,
    /// L2: representation safety prober.
    pub l2: L2Prober,
    /// L3: output sanitizer.
    pub l3: L3Sanitizer,
    /// Configuration.
    pub config: ArmorConfig,
    /// Violation history for self-learning feedback.
    violation_history: Vec<ViolationRecord>,
    /// Total violations recorded.
    total_violations: u64,
    /// Total inputs checked.
    total_checks: u64,
    /// Total inputs allowed.
    total_allowed: u64,
}

impl ArmorLayer {
    /// Create with default config and scanners.
    pub fn new() -> Self {
        Self {
            l0: L0Scanner::new(),
            l1: L1Scanner::new(),
            l2: L2Prober::new(),
            l3: L3Sanitizer::default(),
            config: ArmorConfig::default(),
            violation_history: Vec::new(),
            total_violations: 0,
            total_checks: 0,
            total_allowed: 0,
        }
    }

    /// Create with custom config.
    pub fn with_config(config: ArmorConfig) -> Self {
        Self {
            l0: L0Scanner::new(),
            l1: L1Scanner::new(),
            l2: L2Prober::new(),
            l3: L3Sanitizer::default(),
            config,
            violation_history: Vec::new(),
            total_violations: 0,
            total_checks: 0,
            total_allowed: 0,
        }
    }

    /// Check an input prompt before generation.
    ///
    /// Runs L0 (regex + bloom) and L1 (neural scanner) in sequence.
    /// Returns Allow, Block(reason), or the first block encountered.
    pub fn verify_input(&mut self, prompt: &str) -> SecurityVerdict {
        self.total_checks += 1;

        // L0: regex + bloom
        if self.config.l0_enabled {
            let l0_result = self.l0.scan(prompt);
            if !l0_result.safe {
                self.record_violation(ViolationRecord {
                    input: prompt.to_string(),
                    source: if l0_result.bloom_hit { ViolationSource::L0Bloom } else { ViolationSource::L0Regex },
                    score: 1.0,
                    reason: format!(
                        "L0: {} PII hits, bloom={}",
                        l0_result.pii_hits.len(), l0_result.bloom_hit
                    ),
                    timestamp: 0,
                });
                self.total_violations += 1;
                return SecurityVerdict::Block(format!(
                    "L0: PII or blocklist detected ({} hits, bloom={})",
                    l0_result.pii_hits.len(), l0_result.bloom_hit
                ));
            }
        }

        // L1: neural scanner
        if self.config.l1_enabled {
            let l1_result = self.l1.scan(prompt);
            if !l1_result.safe {
                self.record_violation(ViolationRecord {
                    input: prompt.to_string(),
                    source: ViolationSource::L1Neural,
                    score: l1_result.injection_score,
                    reason: format!("L1: Injection score {:.2}", l1_result.injection_score),
                    timestamp: 0,
                });
                self.total_violations += 1;
                return SecurityVerdict::Block(format!(
                    "L1: Injection detected (score={:.2})",
                    l1_result.injection_score
                ));
            }
        }

        self.total_allowed += 1;
        SecurityVerdict::Allow
    }

    /// Check hidden states during generation (L2 probe).
    pub fn verify_hidden(&mut self, hidden: &[f32]) -> SecurityVerdict {
        if !self.config.l2_enabled {
            return SecurityVerdict::Allow;
        }

        let l2_result = self.l2.probe(hidden);
        if !l2_result.safe {
            self.record_violation(ViolationRecord {
                input: format!("[hidden state, {} dims]", hidden.len()),
                source: ViolationSource::L2Probe,
                score: l2_result.unsafe_score,
                reason: format!(
                    "L2: Unsafe score {:.2}, category={:?}",
                    l2_result.unsafe_score, l2_result.top_category
                ),
                timestamp: 0,
            });
            self.total_violations += 1;
            return SecurityVerdict::Block(format!(
                "L2: Unsafe hidden state (score={:.2}, category={:?})",
                l2_result.unsafe_score, l2_result.top_category
            ));
        }

        SecurityVerdict::Allow
    }

    /// Sanitize output after generation (L3 redaction).
    pub fn sanitize_output(&self, output: &str) -> SecurityVerdict {
        if !self.config.l3_enabled {
            return SecurityVerdict::Allow;
        }

        let l3_result = self.l3.redact(output);
        if l3_result.redaction_count > 0 {
            return SecurityVerdict::Redact(l3_result.redacted);
        }

        SecurityVerdict::Allow
    }

    /// Record a violation for self-learning feedback.
    pub fn record_violation(&mut self, record: ViolationRecord) {
        self.violation_history.push(record);
        // Trim to max size
        if self.violation_history.len() > self.config.max_violation_history {
            self.violation_history.remove(0);
        }
    }

    /// Record an external violation (e.g., from WASM sandbox).
    pub fn feedback_violation(&mut self, input: &str, source: ViolationSource, reason: &str) {
        self.record_violation(ViolationRecord {
            input: input.to_string(),
            source,
            score: 1.0,
            reason: reason.to_string(),
            timestamp: 0,
        });
        self.total_violations += 1;

        // Also add to bloom filter if it's a string-based violation
        if matches!(source, ViolationSource::WasmSandbox | ViolationSource::External) {
            for word in input.split_whitespace() {
                self.l0.block(word);
            }
        }
    }

    /// Get recent violation history.
    pub fn violation_history(&self) -> &[ViolationRecord] {
        &self.violation_history
    }

    /// Get violation statistics.
    pub fn stats(&self) -> ArmorStats {
        let total = self.total_checks;
        ArmorStats {
            total_checks: total,
            total_allowed: self.total_allowed,
            total_blocked: self.total_violations,
            block_rate: if total > 0 { self.total_violations as f32 / total as f32 } else { 0.0 },
            violation_history_len: self.violation_history.len(),
        }
    }
}

impl Default for ArmorLayer {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Armor statistics
// ---------------------------------------------------------------------------

/// Running statistics for the ArmorLayer.
#[derive(Debug, Clone)]
pub struct ArmorStats {
    pub total_checks: u64,
    pub total_allowed: u64,
    pub total_blocked: u64,
    pub block_rate: f32,
    pub violation_history_len: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_armor_default_config() {
        let config = ArmorConfig::default();
        assert!(config.l0_enabled);
        assert!(config.l1_enabled);
        assert!(config.l2_enabled);
        assert!(config.l3_enabled);
        assert!((config.l1_threshold - 0.7).abs() < 0.01);
        assert!((config.l2_threshold - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_armor_safe_input() {
        let mut armor = ArmorLayer::new();
        let verdict = armor.verify_input("What is the weather?");
        assert_eq!(verdict, SecurityVerdict::Allow);
    }

    #[test]
    fn test_armor_pii_blocked() {
        let mut armor = ArmorLayer::new();
        let verdict = armor.verify_input("My SSN is 123-45-6789");
        assert!(matches!(verdict, SecurityVerdict::Block(_)));
    }

    #[test]
    fn test_armor_sanitize_clean() {
        let armor = ArmorLayer::new();
        let verdict = armor.sanitize_output("The weather is sunny.");
        assert_eq!(verdict, SecurityVerdict::Allow);
    }

    #[test]
    fn test_armor_sanitize_redacts() {
        let armor = ArmorLayer::new();
        let verdict = armor.sanitize_output("Email me at user@example.com");
        assert!(matches!(verdict, SecurityVerdict::Redact(_)));
        if let SecurityVerdict::Redact(text) = verdict {
            assert!(text.contains("[REDACTED]"));
            assert!(!text.contains("user@example.com"));
        }
    }

    #[test]
    fn test_armor_l0_disabled() {
        let config = ArmorConfig { l0_enabled: false, ..Default::default() };
        let mut armor = ArmorLayer::with_config(config);
        let verdict = armor.verify_input("My SSN is 123-45-6789");
        assert_eq!(verdict, SecurityVerdict::Allow);
    }

    #[test]
    fn test_armor_bloom_blocked() {
        let mut armor = ArmorLayer::new();
        armor.l0.block("evil_command");
        let verdict = armor.verify_input("Run evil_command now");
        assert!(matches!(verdict, SecurityVerdict::Block(_)));
    }

    #[test]
    fn test_armor_verify_hidden_safe() {
        let mut armor = ArmorLayer::new();
        let hidden = vec![0.0f32; 256];
        let verdict = armor.verify_hidden(&hidden);
        assert_eq!(verdict, SecurityVerdict::Allow);
    }

    #[test]
    fn test_armor_verify_hidden_disabled() {
        let config = ArmorConfig { l2_enabled: false, ..Default::default() };
        let mut armor = ArmorLayer::with_config(config);
        let hidden = vec![0.5f32; 256];
        let verdict = armor.verify_hidden(&hidden);
        assert_eq!(verdict, SecurityVerdict::Allow);
    }

    #[test]
    fn test_armor_feedback_violation() {
        let mut armor = ArmorLayer::new();
        armor.feedback_violation(
            "malicious code execution",
            ViolationSource::WasmSandbox,
            "Attempted filesystem escape",
        );
        assert_eq!(armor.violation_history().len(), 1);
        // Words should be added to bloom
        assert!(armor.l0.bloom.check("malicious"));
    }

    #[test]
    fn test_armor_stats() {
        let mut armor = ArmorLayer::new();
        armor.verify_input("safe input");
        armor.verify_input("another safe");
        let stats = armor.stats();
        assert_eq!(stats.total_checks, 2);
        assert_eq!(stats.total_allowed, 2);
        assert_eq!(stats.total_blocked, 0);
        assert!((stats.block_rate - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_armor_stats_with_blocks() {
        let mut armor = ArmorLayer::new();
        armor.verify_input("safe");
        armor.verify_input("SSN: 123-45-6789");
        armor.verify_input("safe again");
        let stats = armor.stats();
        assert_eq!(stats.total_checks, 3);
        assert_eq!(stats.total_allowed, 2);
        assert!(stats.total_blocked >= 1);
    }

    #[test]
    fn test_armor_violation_history_trim() {
        let config = ArmorConfig { max_violation_history: 3, ..Default::default() };
        let mut armor = ArmorLayer::with_config(config);
        for i in 0..5 {
            armor.feedback_violation(&format!("violation_{}", i), ViolationSource::External, "test");
        }
        assert_eq!(armor.violation_history().len(), 3);
        // Should keep most recent
        assert!(armor.violation_history().last().unwrap().input.contains("violation_4"));
    }

    #[test]
    fn test_armor_verdict_equality() {
        assert_eq!(SecurityVerdict::Allow, SecurityVerdict::Allow);
        assert_eq!(SecurityVerdict::Block("test".into()), SecurityVerdict::Block("test".into()));
        assert_ne!(SecurityVerdict::Allow, SecurityVerdict::Block("test".into()));
    }

    #[test]
    fn test_violation_source_variants() {
        assert_ne!(ViolationSource::L0Regex, ViolationSource::L0Bloom);
        assert_ne!(ViolationSource::L1Neural, ViolationSource::L2Probe);
    }
}
