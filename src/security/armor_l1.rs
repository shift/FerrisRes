//! Armor L1: Distilled neural injection scanner (~5M params).
//!
//! Placeholder module — to be implemented.
//! Target: 4-layer BERT variant (hidden=256) distilled from
//! protectai/deberta-v3-base-prompt-injection. Binary SAFE/INJECTION.

/// Result of an L1 neural scan.
#[derive(Debug, Clone)]
pub struct L1ScanResult {
    /// Classification score [0.0, 1.0] where 1.0 = INJECTION.
    pub injection_score: f32,
    /// Whether the input is classified as safe.
    pub safe: bool,
}

/// Placeholder L1 scanner (stub).
pub struct L1Scanner;

impl L1Scanner {
    pub fn new() -> Self { Self }
    pub fn scan(&self, _input: &str) -> L1ScanResult {
        L1ScanResult { injection_score: 0.0, safe: true }
    }
}

impl Default for L1Scanner {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_stub_scan() {
        let scanner = L1Scanner::new();
        let result = scanner.scan("hello");
        assert!(result.safe);
        assert_eq!(result.injection_score, 0.0);
    }
}
