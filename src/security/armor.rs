//! Armor orchestrator: ArmorLayer tying L0-L3 + self-learning feedback.
//!
//! Placeholder module — to be implemented.
//! Target: parallel L0+L1 pre-generation check, L2 mid-generation probe,
//! L3 post-generation redaction. Self-learning via ConceptMap + LoRA.

use crate::security::armor_l0::L0Scanner;
use crate::security::armor_l1::L1Scanner;
use crate::security::armor_l2::L2Prober;
use crate::security::armor_l3::L3Sanitizer;

/// Overall security verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecurityVerdict {
    /// Input/output is safe.
    Allow,
    /// Input/output is blocked.
    Block(String),
    /// Output contains redacted content.
    Redact(String),
}

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
        }
    }
}

/// The Armor security layer orchestrator.
///
/// Ties L0-L3 together for comprehensive input/output security.
pub struct ArmorLayer {
    pub l0: L0Scanner,
    pub l1: L1Scanner,
    pub l2: L2Prober,
    pub l3: L3Sanitizer,
    pub config: ArmorConfig,
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
        }
    }

    /// Check an input prompt before generation.
    pub fn verify_input(&self, prompt: &str) -> SecurityVerdict {
        // L0: regex + bloom
        if self.config.l0_enabled {
            let l0_result = self.l0.scan(prompt);
            if !l0_result.safe {
                return SecurityVerdict::Block(format!(
                    "L0: PII or blocklist detected ({} hits, bloom={})",
                    l0_result.pii_hits.len(), l0_result.bloom_hit
                ));
            }
        }

        // L1: neural scanner (stub always returns safe)
        if self.config.l1_enabled {
            let l1_result = self.l1.scan(prompt);
            if !l1_result.safe {
                return SecurityVerdict::Block(format!(
                    "L1: Injection detected (score={:.2})", l1_result.injection_score
                ));
            }
        }

        SecurityVerdict::Allow
    }

    /// Sanitize output after generation.
    pub fn sanitize_output(&self, output: &str) -> SecurityVerdict {
        // L3: redaction (stub passes through)
        if self.config.l3_enabled {
            let l3_result = self.l3.redact(output);
            if l3_result.redaction_count > 0 {
                return SecurityVerdict::Redact(l3_result.redacted);
            }
        }
        SecurityVerdict::Allow
    }
}

impl Default for ArmorLayer {
    fn default() -> Self { Self::new() }
}

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
    }

    #[test]
    fn test_armor_safe_input() {
        let armor = ArmorLayer::new();
        let verdict = armor.verify_input("What is the weather?");
        assert_eq!(verdict, SecurityVerdict::Allow);
    }

    #[test]
    fn test_armor_pii_blocked() {
        let armor = ArmorLayer::new();
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
    fn test_armor_l0_disabled() {
        let config = ArmorConfig { l0_enabled: false, ..Default::default() };
        let armor = ArmorLayer::with_config(config);
        // With L0 disabled, PII input passes through
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
}
