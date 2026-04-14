//! Armor L2: RepE safety probe on BlockSummary hidden states.
//!
//! Placeholder module — to be implemented.
//! Target: linear logistic regression on 256-dim BlockSummaryLayer outputs.
//! <1ms per probe.

/// Result of an L2 safety probe.
#[derive(Debug, Clone)]
pub struct L2ProbeResult {
    /// Safety score [0.0, 1.0] where 1.0 = UNSAFE.
    pub unsafe_score: f32,
    /// Whether the hidden state is classified as safe.
    pub safe: bool,
}

/// Placeholder L2 prober (stub).
pub struct L2Prober;

impl L2Prober {
    pub fn new() -> Self { Self }
    pub fn probe(&self, _hidden: &[f32]) -> L2ProbeResult {
        L2ProbeResult { unsafe_score: 0.0, safe: true }
    }
}

impl Default for L2Prober {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_stub_probe() {
        let prober = L2Prober::new();
        let result = prober.probe(&[0.0; 256]);
        assert!(result.safe);
    }
}
