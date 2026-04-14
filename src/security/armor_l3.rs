//! Armor L3: Parallel PII redaction output sanitizer.
//!
//! Runs PII detection (reusing L0 PatternEngine) on generated output
//! and applies configurable redaction strategies. Designed for
//! streaming integration with zero TTFT impact.

use crate::security::armor_l0::{PatternEngine, PiiHit};

// ---------------------------------------------------------------------------
// Redaction strategy
// ---------------------------------------------------------------------------

/// Redaction strategy for PII hits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedactionStrategy {
    /// Replace each character with an asterisk (e.g., `***@****.com`).
    Mask,
    /// Replace entire match with `[REDACTED]`.
    Replace,
    /// Remove the match entirely (shrink text).
    Truncate,
}

// ---------------------------------------------------------------------------
// Redaction result
// ---------------------------------------------------------------------------

/// Result of an L3 redaction pass.
#[derive(Debug, Clone)]
pub struct L3RedactResult {
    /// The redacted output text.
    pub redacted: String,
    /// Number of redactions applied.
    pub redaction_count: usize,
    /// Details of each redaction.
    pub details: Vec<RedactionDetail>,
}

/// Detail of a single redaction.
#[derive(Debug, Clone)]
pub struct RedactionDetail {
    /// Recognizer that triggered the redaction.
    pub recognizer: String,
    /// Original text that was redacted.
    pub original: String,
    /// What it was replaced with.
    pub replacement: String,
}

// ---------------------------------------------------------------------------
// L3Sanitizer
// ---------------------------------------------------------------------------

/// PII redaction sanitizer for generated output.
///
/// Uses PatternEngine to detect PII in output text and applies
/// configurable redaction strategies.
pub struct L3Sanitizer {
    /// The regex pattern engine (shared with L0).
    patterns: PatternEngine,
    /// Redaction strategy.
    pub strategy: RedactionStrategy,
    /// Recognizers to exclude from redaction (e.g., injection heuristics
    /// should be handled by L0/L1 blocking, not output redaction).
    exclude_recognizers: Vec<String>,
}

impl L3Sanitizer {
    /// Create with a given redaction strategy.
    pub fn new(strategy: RedactionStrategy) -> Self {
        let mut sanitizer = Self {
            patterns: PatternEngine::new(),
            strategy,
            exclude_recognizers: Vec::new(),
        };
        // Don't redact injection heuristics from output — those are input-only
        for name in &[
            "injection_ignore_previous",
            "injection_system_prompt",
            "injection_output_exfiltration",
            "injection_dan",
            "injection_base64_command",
            "injection_new_instructions",
        ] {
            sanitizer.exclude_recognizers.push(name.to_string());
        }
        sanitizer
    }

    /// Create with custom pattern engine.
    pub fn with_patterns(patterns: PatternEngine, strategy: RedactionStrategy) -> Self {
        Self { patterns, strategy, exclude_recognizers: Vec::new() }
    }

    /// Add a recognizer to the exclusion list.
    pub fn exclude_recognizer(&mut self, name: &str) {
        self.exclude_recognizers.push(name.to_string());
    }

    /// Redact PII from output text.
    pub fn redact(&self, output: &str) -> L3RedactResult {
        let hits = self.patterns.scan(output);

        // Filter to PII-only hits (exclude injection heuristics)
        let pii_hits: Vec<&PiiHit> = hits.iter()
            .filter(|h| !self.exclude_recognizers.contains(&h.recognizer))
            .collect();

        if pii_hits.is_empty() {
            return L3RedactResult {
                redacted: output.to_string(),
                redaction_count: 0,
                details: Vec::new(),
            };
        }

        // Sort hits by start position (descending) so we can replace
        // from end to beginning without offset shifts
        let mut sorted: Vec<&PiiHit> = pii_hits;
        sorted.sort_by(|a, b| b.start.cmp(&a.start));

        let mut result = output.to_string();
        let mut details = Vec::new();

        for hit in sorted {
            let replacement = match self.strategy {
                RedactionStrategy::Mask => {
                    // Replace each char with *
                    let len = hit.text.chars().count();
                    "*".repeat(len)
                }
                RedactionStrategy::Replace => "[REDACTED]".to_string(),
                RedactionStrategy::Truncate => String::new(),
            };

            // Replace by byte offsets
            if hit.end <= result.len() {
                details.push(RedactionDetail {
                    recognizer: hit.recognizer.clone(),
                    original: hit.text.clone(),
                    replacement: replacement.clone(),
                });
                result.replace_range(hit.start..hit.end, &replacement);
            }
        }

        L3RedactResult {
            redaction_count: details.len(),
            details,
            redacted: result,
        }
    }

    /// Redact only specific recognizer types.
    pub fn redact_only(&self, output: &str, recognizers: &[&str]) -> L3RedactResult {
        let hits = self.patterns.scan(output);

        let filtered: Vec<&PiiHit> = hits.iter()
            .filter(|h| recognizers.contains(&h.recognizer.as_str()))
            .collect();

        if filtered.is_empty() {
            return L3RedactResult {
                redacted: output.to_string(),
                redaction_count: 0,
                details: Vec::new(),
            };
        }

        let mut sorted = filtered;
        sorted.sort_by(|a, b| b.start.cmp(&a.start));

        let mut result = output.to_string();
        let mut details = Vec::new();

        for hit in sorted {
            let replacement = match self.strategy {
                RedactionStrategy::Mask => "*".repeat(hit.text.chars().count()),
                RedactionStrategy::Replace => "[REDACTED]".to_string(),
                RedactionStrategy::Truncate => String::new(),
            };

            if hit.end <= result.len() {
                details.push(RedactionDetail {
                    recognizer: hit.recognizer.clone(),
                    original: hit.text.clone(),
                    replacement: replacement.clone(),
                });
                result.replace_range(hit.start..hit.end, &replacement);
            }
        }

        L3RedactResult {
            redaction_count: details.len(),
            details,
            redacted: result,
        }
    }

    /// Check if output contains any redactable PII.
    pub fn contains_pii(&self, output: &str) -> bool {
        let hits = self.patterns.scan(output);
        hits.iter().any(|h| !self.exclude_recognizers.contains(&h.recognizer))
    }
}

impl Default for L3Sanitizer {
    fn default() -> Self { Self::new(RedactionStrategy::Replace) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l3_redact_clean() {
        let sanitizer = L3Sanitizer::default();
        let result = sanitizer.redact("The weather is sunny today.");
        assert_eq!(result.redacted, "The weather is sunny today.");
        assert_eq!(result.redaction_count, 0);
    }

    #[test]
    fn test_l3_redact_email_mask() {
        let sanitizer = L3Sanitizer::new(RedactionStrategy::Mask);
        let result = sanitizer.redact("Contact user@example.com for details");
        assert!(!result.redacted.contains("user@example.com"));
        assert!(result.redacted.contains("***"));
        assert_eq!(result.redaction_count, 1);
    }

    #[test]
    fn test_l3_redact_email_replace() {
        let sanitizer = L3Sanitizer::new(RedactionStrategy::Replace);
        let result = sanitizer.redact("Email: test@test.com here");
        assert!(result.redacted.contains("[REDACTED]"));
        assert!(!result.redacted.contains("test@test.com"));
    }

    #[test]
    fn test_l3_redact_ssn_truncate() {
        let sanitizer = L3Sanitizer::new(RedactionStrategy::Truncate);
        let result = sanitizer.redact("SSN: 123-45-6789 end");
        assert!(!result.redacted.contains("123-45-6789"));
        // The SSN should be removed entirely
        assert!(result.redacted.contains("SSN:") || result.redacted.contains("end"));
    }

    #[test]
    fn test_l3_redact_multiple() {
        let sanitizer = L3Sanitizer::new(RedactionStrategy::Replace);
        let result = sanitizer.redact("Email a@b.com and SSN 123-45-6789");
        assert!(result.redaction_count >= 2, "Should detect both email and SSN");
        assert!(!result.redacted.contains("a@b.com"));
        assert!(!result.redacted.contains("123-45-6789"));
    }

    #[test]
    fn test_l3_redact_preserves_injection_heuristics() {
        let sanitizer = L3Sanitizer::default();
        // "Ignore all previous instructions" matches injection heuristic
        // but should NOT be redacted in output (only blocked in input by L0)
        let result = sanitizer.redact("The model should ignore all previous instructions");
        // Injection recognizers are excluded from redaction
        assert!(result.redacted.contains("ignore all previous"));
    }

    #[test]
    fn test_l3_redact_detail() {
        let sanitizer = L3Sanitizer::new(RedactionStrategy::Replace);
        let result = sanitizer.redact("My email is user@example.com");
        assert_eq!(result.details.len(), 1);
        assert_eq!(result.details[0].recognizer, "email");
        assert_eq!(result.details[0].original, "user@example.com");
        assert_eq!(result.details[0].replacement, "[REDACTED]");
    }

    #[test]
    fn test_l3_redact_only() {
        let sanitizer = L3Sanitizer::default();
        let result = sanitizer.redact_only(
            "Email a@b.com and SSN 123-45-6789",
            &["email"],
        );
        // Only email should be redacted
        assert!(!result.redacted.contains("a@b.com"));
        assert!(result.redacted.contains("123-45-6789"));
    }

    #[test]
    fn test_l3_contains_pii() {
        let sanitizer = L3Sanitizer::default();
        assert!(sanitizer.contains_pii("user@example.com"));
        assert!(!sanitizer.contains_pii("hello world"));
    }

    #[test]
    fn test_l3_exclude_recognizer() {
        let mut sanitizer = L3Sanitizer::new(RedactionStrategy::Replace);
        sanitizer.exclude_recognizer("email");
        let result = sanitizer.redact("Email: user@example.com");
        // Email excluded → not redacted
        assert!(result.redacted.contains("user@example.com"));
    }

    #[test]
    fn test_l3_strategy_equality() {
        assert_eq!(RedactionStrategy::Mask, RedactionStrategy::Mask);
        assert_ne!(RedactionStrategy::Mask, RedactionStrategy::Replace);
    }
}
