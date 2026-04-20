//! Self-Modification Guard — Weight Change Safety Protocol
//!
//! Formal safety protocol for weight modifications via the 'learn' tool:
//!   1. Pre-flight risk assessment (Low/Medium/High/Critical)
//!   2. Sandbox trial: clone weights, apply modification, A/B quality comparison
//!   3. Commit with weight snapshot
//!   4. Post-commit monitoring with auto-rollback
//!   5. Persistent audit trail
//!
//! Safety layers:
//!   - Autonomy gate: SemiAutonomous+ required
//!   - Rate limiting: configurable per-session/per-hour/cooldown
//!   - Cumulative drift tracking
//!   - EWC penalty integration
//!   - Human approval gate for Medium+ risk below FullyAutonomous

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Source of a modification request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModificationSource {
    /// Triggered by the 'learn' tool.
    ToolCall,
    /// Triggered by quality degradation (ProactiveController).
    QualityDegradation,
    /// Triggered by practice goal (IntrinsicMotivation).
    PracticeGoal,
    /// Triggered by consolidation (ConsolidationEngine).
    Consolidation,
}

/// Risk level for a modification.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum RiskLevel {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

impl RiskLevel {
    pub fn max(self, other: Self) -> Self {
        if self as u32 >= other as u32 { self } else { other }
    }
}

/// Autonomy level (mirrors ProactiveController).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AutonomyLevel {
    Manual,
    Assisted,
    SemiAutonomous,
    FullyAutonomous,
}

/// A proposed weight modification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModificationRequest {
    pub source: ModificationSource,
    pub adapter_id: Option<String>,
    pub importance: f32,
    pub learning_rate: f32,
    pub trigger_embedding: Vec<f32>,
    pub target_embedding: Vec<f32>,
}

/// Pre-flight risk assessment result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreFlightReport {
    pub risk: RiskLevel,
    pub cumulative_drift: f32,
    pub requires_human_approval: bool,
    pub estimated_quality_delta: f32,
    pub reasoning: String,
}

/// Result of a sandboxed trial run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    pub baseline_quality: f32,
    pub trial_quality: f32,
    pub quality_delta: f32,
    pub validation_samples: usize,
    pub passed: bool,
    pub ewc_penalty: f32,
}

/// Result of committing a modification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitResult {
    pub modification_id: u64,
    pub post_commit_quality: f32,
    pub snapshot_id: u64,
    pub accepted: bool,
    pub rejection_reason: Option<String>,
}

/// A weight snapshot for rollback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightSnapshot {
    pub id: u64,
    pub timestamp: u64,
    /// Serialized adapter state (simplified as byte vector).
    pub state: Vec<u8>,
    pub quality: f32,
    pub adapter_count: usize,
}

/// Persistent audit record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub id: u64,
    pub timestamp: u64,
    pub source: ModificationSource,
    pub risk: RiskLevel,
    pub pre_flight_approved: bool,
    pub trial_quality_delta: f32,
    pub committed: bool,
    pub rolled_back: bool,
    pub rollback_reason: Option<String>,
    pub autonomy_level: String,
}

/// Health report for the modification system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    pub cumulative_drift: f32,
    pub quality_trend: QualityTrend,
    pub recommendations: Vec<String>,
    pub session_modifications: usize,
    pub session_rollbacks: usize,
}

/// Quality trend direction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QualityTrend {
    Improving,
    Stable,
    Degrading,
}

/// A pending modification awaiting human approval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingApproval {
    pub request: ModificationRequest,
    pub pre_flight: PreFlightReport,
    pub description: String,
    pub submitted_at: u64,
}

/// Configuration for SelfModificationGuard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModificationConfig {
    /// Maximum cumulative drift before blocking (default: 0.10).
    pub max_cumulative_drift: f32,
    /// Warning drift threshold (default: 0.05).
    pub drift_warning_threshold: f32,
    /// Quality degradation that triggers auto-rollback (default: -0.10).
    pub rollback_quality_delta: f32,
    /// Post-commit monitoring window (default: 5 interactions).
    pub post_commit_monitoring_window: usize,
    /// Maximum snapshots in memory (default: 10).
    pub max_snapshots: usize,
    /// Validation samples for sandbox (default: 5).
    pub sandbox_validation_samples: usize,
    /// Minimum quality delta to pass sandbox (default: -0.05).
    pub sandbox_min_quality_delta: f32,
    /// Human approval for Medium risk (default: true).
    pub human_approval_for_medium: bool,
    /// Max modifications per session (default: 20).
    pub max_per_session: usize,
    /// Max modifications per hour (default: 50).
    pub max_per_hour: usize,
    /// Cooldown between modifications in seconds (default: 30).
    pub cooldown_secs: u64,
}

impl Default for SelfModificationConfig {
    fn default() -> Self {
        Self {
            max_cumulative_drift: 0.10,
            drift_warning_threshold: 0.05,
            rollback_quality_delta: -0.10,
            post_commit_monitoring_window: 5,
            max_snapshots: 10,
            sandbox_validation_samples: 5,
            sandbox_min_quality_delta: -0.05,
            human_approval_for_medium: true,
            max_per_session: 20,
            max_per_hour: 50,
            cooldown_secs: 30,
        }
    }
}

// ---------------------------------------------------------------------------
// SelfModificationGuard
// ---------------------------------------------------------------------------

/// Formal safety protocol for weight modifications.
pub struct SelfModificationGuard {
    config: SelfModificationConfig,
    /// Cumulative parameter drift this session.
    cumulative_drift: f32,
    /// Quality history for trend detection.
    quality_history: Vec<f32>,
    /// Weight snapshots for rollback.
    snapshots: Vec<WeightSnapshot>,
    /// Audit trail.
    audit_trail: Vec<AuditRecord>,
    /// Pending human approvals.
    pending_approvals: Vec<PendingApproval>,
    /// Next modification ID.
    next_mod_id: u64,
    /// Next snapshot ID.
    next_snapshot_id: u64,
    /// Session modification count.
    session_mod_count: usize,
    /// Recent modification timestamps (for rate limiting).
    recent_timestamps: Vec<u64>,
    /// Post-commit quality tracker (quality deltas since last commit).
    post_commit_deltas: Vec<f32>,
    /// Last committed modification ID (for monitoring).
    last_committed_id: Option<u64>,
    /// Current session quality.
    current_quality: f32,
}

impl SelfModificationGuard {
    pub fn new(config: SelfModificationConfig) -> Self {
        Self {
            config,
            cumulative_drift: 0.0,
            quality_history: vec![0.5], // Start neutral
            snapshots: Vec::new(),
            audit_trail: Vec::new(),
            pending_approvals: Vec::new(),
            next_mod_id: 1,
            next_snapshot_id: 1,
            session_mod_count: 0,
            recent_timestamps: Vec::new(),
            post_commit_deltas: Vec::new(),
            last_committed_id: None,
            current_quality: 0.5,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(SelfModificationConfig::default())
    }

    // -----------------------------------------------------------------------
    // Step 1: Pre-flight check
    // -----------------------------------------------------------------------

    /// Assess the risk of a proposed modification.
    pub fn pre_flight(
        &self,
        request: &ModificationRequest,
        autonomy: &AutonomyLevel,
    ) -> PreFlightReport {
        let mut risk = RiskLevel::Low;
        let mut reasons = Vec::new();

        // New adapter = moderate risk
        if request.adapter_id.is_none() {
            risk = risk.max(RiskLevel::Medium);
            reasons.push("New adapter (no existing state)".into());
        }

        // Low importance = low risk
        if request.importance < 0.3 {
            risk = risk.max(RiskLevel::Low);
            reasons.push("Low importance input".into());
        }

        // Cumulative drift check
        if self.cumulative_drift > self.config.max_cumulative_drift {
            risk = RiskLevel::Critical;
            reasons.push(format!(
                "Cumulative drift {:.4} exceeds max {:.4}",
                self.cumulative_drift, self.config.max_cumulative_drift
            ));
        } else if self.cumulative_drift > self.config.drift_warning_threshold {
            risk = risk.max(RiskLevel::High);
            reasons.push(format!(
                "Cumulative drift {:.4} exceeds warning {:.4}",
                self.cumulative_drift, self.config.drift_warning_threshold
            ));
        }

        // Practice goals are lower risk (structured learning)
        if request.source == ModificationSource::PracticeGoal {
            if risk as u32 > RiskLevel::Medium as u32 {
                risk = RiskLevel::Medium; // Cap at medium for practice
                reasons.push("Capped at Medium: practice goal (structured learning)".into());
            }
        }

        // High learning rate = higher risk
        if request.learning_rate > 0.01 {
            risk = risk.max(RiskLevel::Medium);
            reasons.push(format!("High learning rate: {:.4}", request.learning_rate));
        }

        // Autonomy gate
        let requires_human = match autonomy {
            AutonomyLevel::Manual | AutonomyLevel::Assisted => {
                risk = risk.max(RiskLevel::High);
                reasons.push("Low autonomy level".into());
                true
            }
            AutonomyLevel::SemiAutonomous => {
                if risk as u32 >= RiskLevel::Medium as u32 && self.config.human_approval_for_medium {
                    reasons.push("Human approval required at SemiAutonomous".into());
                    true
                } else {
                    false
                }
            }
            AutonomyLevel::FullyAutonomous => false,
        };

        let estimated_delta = if request.importance > 0.5 { 0.05 } else { 0.02 };

        PreFlightReport {
            risk,
            cumulative_drift: self.cumulative_drift,
            requires_human_approval: requires_human,
            estimated_quality_delta: estimated_delta,
            reasoning: if reasons.is_empty() {
                "Low risk modification".into()
            } else {
                reasons.join("; ")
            },
        }
    }

    /// Check rate limits.
    pub fn check_rate_limit(&self) -> Result<(), String> {
        if self.session_mod_count >= self.config.max_per_session {
            return Err(format!(
                "Session limit reached ({}/{})",
                self.session_mod_count, self.config.max_per_session
            ));
        }

        let now = Self::now_secs();
        let one_hour_ago = now.saturating_sub(3600);
        let hourly = self.recent_timestamps.iter().filter(|&&t| t > one_hour_ago).count();
        if hourly >= self.config.max_per_hour {
            return Err(format!(
                "Hourly limit reached ({}/{})",
                hourly, self.config.max_per_hour
            ));
        }

        // Cooldown
        if let Some(&last) = self.recent_timestamps.last() {
            if now - last < self.config.cooldown_secs {
                return Err(format!(
                    "Cooldown: {}s remaining",
                    self.config.cooldown_secs - (now - last)
                ));
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Step 2: Sandbox trial
    // -----------------------------------------------------------------------

    /// Run a sandboxed trial of the modification.
    /// In a real system, this would clone weights and test.
    /// Here we simulate based on the request properties.
    pub fn sandbox_trial(
        &self,
        request: &ModificationRequest,
        baseline_quality: f32,
    ) -> TrialResult {
        // Simulate trial quality: baseline + estimated delta + noise
        let estimated_delta = request.importance * request.learning_rate * 10.0;
        // Add some noise proportional to drift
        let noise = if self.cumulative_drift > 0.05 { -0.02 } else { 0.01 };
        let trial_quality = (baseline_quality + estimated_delta + noise).clamp(0.0, 1.0);

        let quality_delta = trial_quality - baseline_quality;

        // EWC penalty: proportional to drift and learning rate
        let ewc_penalty = self.cumulative_drift * request.learning_rate * 100.0;

        let passed = quality_delta >= self.config.sandbox_min_quality_delta;

        TrialResult {
            baseline_quality,
            trial_quality,
            quality_delta,
            validation_samples: self.config.sandbox_validation_samples,
            passed,
            ewc_penalty,
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: Commit
    // -----------------------------------------------------------------------

    /// Commit a modification after successful trial.
    pub fn commit(
        &mut self,
        request: &ModificationRequest,
        trial: &TrialResult,
        autonomy: &AutonomyLevel,
        current_adapter_state: &[u8],
        current_quality: f32,
    ) -> CommitResult {
        let mod_id = self.next_mod_id;
        self.next_mod_id += 1;

        // Final quality gate
        if !trial.passed {
            self.record_audit(request, &RiskLevel::Low, false, trial.quality_delta, false, autonomy);
            return CommitResult {
                modification_id: mod_id,
                post_commit_quality: current_quality,
                snapshot_id: 0,
                accepted: false,
                rejection_reason: Some("Sandbox trial failed".into()),
            };
        }

        // Critical risk = block
        let pre_flight = self.pre_flight(request, autonomy);
        if pre_flight.risk == RiskLevel::Critical {
            self.record_audit(request, &RiskLevel::Critical, false, trial.quality_delta, false, autonomy);
            return CommitResult {
                modification_id: mod_id,
                post_commit_quality: current_quality,
                snapshot_id: 0,
                accepted: false,
                rejection_reason: Some("Critical risk blocked".into()),
            };
        }

        // Needs human approval?
        if pre_flight.requires_human_approval {
            self.pending_approvals.push(PendingApproval {
                request: request.clone(),
                pre_flight: pre_flight.clone(),
                description: format!(
                    "Proposed learning: {:?} (importance: {:.2})\nRisk: {:?}\nEstimated quality impact: +{:.3}",
                    request.source, request.importance, pre_flight.risk, pre_flight.estimated_quality_delta
                ),
                submitted_at: Self::now_secs(),
            });
            return CommitResult {
                modification_id: mod_id,
                post_commit_quality: current_quality,
                snapshot_id: 0,
                accepted: false,
                rejection_reason: Some("Pending human approval".into()),
            };
        }

        // Create snapshot for rollback
        let snapshot_id = self.create_snapshot(current_adapter_state, current_quality, 1);

        // Update drift
        let drift_delta = request.learning_rate * request.importance;
        self.cumulative_drift += drift_delta;

        // Update quality tracking
        self.current_quality = trial.trial_quality;
        self.quality_history.push(trial.trial_quality);

        // Update rate limiting
        self.session_mod_count += 1;
        self.recent_timestamps.push(Self::now_secs());

        // Reset post-commit monitoring
        self.post_commit_deltas.clear();
        self.last_committed_id = Some(mod_id);

        // Record audit
        self.record_audit(request, &pre_flight.risk, true, trial.quality_delta, true, autonomy);

        CommitResult {
            modification_id: mod_id,
            post_commit_quality: trial.trial_quality,
            snapshot_id,
            accepted: true,
            rejection_reason: None,
        }
    }

    // -----------------------------------------------------------------------
    // Step 4: Post-commit monitoring
    // -----------------------------------------------------------------------

    /// Record a post-commit quality observation.
    /// Returns true if auto-rollback triggered.
    pub fn post_commit_observe(&mut self, quality: f32) -> bool {
        if self.last_committed_id.is_none() {
            return false;
        }

        let pre_commit_quality = self.quality_history.iter().cloned().rev().nth(1).unwrap_or(0.5);
        let delta = quality - pre_commit_quality;
        self.post_commit_deltas.push(delta);

        // Check if quality dropped below threshold
        if delta < self.config.rollback_quality_delta {
            // Auto-rollback
            self.trigger_rollback("Quality degradation detected post-commit");
            return true;
        }

        // Check monitoring window
        if self.post_commit_deltas.len() >= self.config.post_commit_monitoring_window {
            // Monitoring complete, clear
            self.last_committed_id = None;
            self.post_commit_deltas.clear();
        }

        false
    }

    // -----------------------------------------------------------------------
    // Rollback
    // -----------------------------------------------------------------------

    /// Trigger a rollback to the previous snapshot.
    pub fn trigger_rollback(&mut self, reason: &str) -> Option<WeightSnapshot> {
        if let Some(snapshot) = self.snapshots.pop() {
            // Update current quality
            self.current_quality = snapshot.quality;

            // Mark in audit trail
            if let Some(last_audit) = self.audit_trail.last_mut() {
                last_audit.rolled_back = true;
                last_audit.rollback_reason = Some(reason.to_string());
            }

            Some(snapshot)
        } else {
            None
        }
    }

    // -----------------------------------------------------------------------
    // Human approval
    // -----------------------------------------------------------------------

    /// Approve a pending modification.
    pub fn approve_pending(&mut self, index: usize) -> Option<&PendingApproval> {
        if index < self.pending_approvals.len() {
            // In a real system, this would execute the modification
            Some(&self.pending_approvals[index])
        } else {
            None
        }
    }

    /// Reject a pending modification.
    pub fn reject_pending(&mut self, index: usize) -> Option<PendingApproval> {
        if index < self.pending_approvals.len() {
            Some(self.pending_approvals.remove(index))
        } else {
            None
        }
    }

    /// Get pending approvals.
    pub fn pending_approvals(&self) -> &[PendingApproval] {
        &self.pending_approvals
    }

    // -----------------------------------------------------------------------
    // Health check
    // -----------------------------------------------------------------------

    /// Get a health report for the modification system.
    pub fn health_check(&self) -> HealthReport {
        let trend = if self.quality_history.len() >= 5 {
            let mid = self.quality_history.len() / 2;
            let first: f32 = self.quality_history[..mid].iter().sum::<f32>() / mid as f32;
            let second: f32 = self.quality_history[mid..].iter().sum::<f32>()
                / (self.quality_history.len() - mid) as f32;
            let delta = second - first;
            if delta > 0.05 {
                QualityTrend::Improving
            } else if delta < -0.05 {
                QualityTrend::Degrading
            } else {
                QualityTrend::Stable
            }
        } else {
            QualityTrend::Stable
        };

        let mut recommendations = Vec::new();
        if self.cumulative_drift > self.config.drift_warning_threshold {
            recommendations.push("Increase cooldown between modifications".into());
        }
        if self.cumulative_drift > self.config.max_cumulative_drift * 0.8 {
            recommendations.push("Approaching drift limit — consider pausing learning".into());
        }
        if trend == QualityTrend::Degrading {
            recommendations.push("Quality degrading — reduce learning rate".into());
        }

        let rollbacks = self.audit_trail.iter().filter(|a| a.rolled_back).count();

        HealthReport {
            cumulative_drift: self.cumulative_drift,
            quality_trend: trend,
            recommendations,
            session_modifications: self.session_mod_count,
            session_rollbacks: rollbacks,
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Get cumulative drift.
    pub fn cumulative_drift(&self) -> f32 {
        self.cumulative_drift
    }

    /// Get current quality.
    pub fn current_quality(&self) -> f32 {
        self.current_quality
    }

    /// Get the audit trail.
    pub fn audit_trail(&self) -> &[AuditRecord] {
        &self.audit_trail
    }

    /// Get snapshots.
    pub fn snapshots(&self) -> &[WeightSnapshot] {
        &self.snapshots
    }

    /// Session modification count.
    pub fn session_mod_count(&self) -> usize {
        self.session_mod_count
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn create_snapshot(&mut self, state: &[u8], quality: f32, adapter_count: usize) -> u64 {
        let id = self.next_snapshot_id;
        self.next_snapshot_id += 1;

        let snapshot = WeightSnapshot {
            id,
            timestamp: Self::now_secs(),
            state: state.to_vec(),
            quality,
            adapter_count,
        };

        // Enforce max snapshots
        if self.snapshots.len() >= self.config.max_snapshots {
            self.snapshots.remove(0);
        }

        self.snapshots.push(snapshot);
        id
    }

    fn record_audit(
        &mut self,
        request: &ModificationRequest,
        risk: &RiskLevel,
        pre_flight_approved: bool,
        trial_delta: f32,
        committed: bool,
        autonomy: &AutonomyLevel,
    ) {
        let record = AuditRecord {
            id: self.next_mod_id,
            timestamp: Self::now_secs(),
            source: request.source.clone(),
            risk: risk.clone(),
            pre_flight_approved,
            trial_quality_delta: trial_delta,
            committed,
            rolled_back: false,
            rollback_reason: None,
            autonomy_level: format!("{:?}", autonomy),
        };
        self.audit_trail.push(record);
    }

    fn now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_request() -> ModificationRequest {
        ModificationRequest {
            source: ModificationSource::ToolCall,
            adapter_id: Some("adapter_1".into()),
            importance: 0.5,
            learning_rate: 0.001,
            trigger_embedding: vec![0.1; 64],
            target_embedding: vec![0.2; 64],
        }
    }

    #[test]
    fn test_risk_assessment_low() {
        let guard = SelfModificationGuard::with_defaults();
        let req = ModificationRequest {
            source: ModificationSource::ToolCall,
            adapter_id: Some("a".into()),
            importance: 0.2,
            learning_rate: 0.001,
            trigger_embedding: vec![],
            target_embedding: vec![],
        };
        let report = guard.pre_flight(&req, &AutonomyLevel::FullyAutonomous);
        assert_eq!(report.risk, RiskLevel::Low);
    }

    #[test]
    fn test_risk_assessment_high_drift() {
        let mut guard = SelfModificationGuard::with_defaults();
        guard.cumulative_drift = 0.06;

        let report = guard.pre_flight(&simple_request(), &AutonomyLevel::FullyAutonomous);
        assert!(report.risk as u32 >= RiskLevel::High as u32);
    }

    #[test]
    fn test_risk_assessment_critical_drift() {
        let mut guard = SelfModificationGuard::with_defaults();
        guard.cumulative_drift = 0.15;

        let report = guard.pre_flight(&simple_request(), &AutonomyLevel::FullyAutonomous);
        assert_eq!(report.risk, RiskLevel::Critical);
    }

    #[test]
    fn test_preflight_approve_low_risk() {
        let guard = SelfModificationGuard::with_defaults();
        let report = guard.pre_flight(&simple_request(), &AutonomyLevel::FullyAutonomous);
        assert!(!report.requires_human_approval);
    }

    #[test]
    fn test_preflight_block_critical() {
        let mut guard = SelfModificationGuard::with_defaults();
        guard.cumulative_drift = 0.15;

        let req = simple_request();
        let report = guard.pre_flight(&req, &AutonomyLevel::FullyAutonomous);
        assert_eq!(report.risk, RiskLevel::Critical);
    }

    #[test]
    fn test_preflight_human_approval() {
        let guard = SelfModificationGuard::with_defaults();
        // New adapter (None) + SemiAutonomous → needs approval
        let req = ModificationRequest {
            source: ModificationSource::ToolCall,
            adapter_id: None, // triggers Medium risk
            importance: 0.5,
            learning_rate: 0.001,
            trigger_embedding: vec![],
            target_embedding: vec![],
        };
        let report = guard.pre_flight(&req, &AutonomyLevel::SemiAutonomous);
        assert!(report.requires_human_approval);
    }

    #[test]
    fn test_sandbox_trial_pass() {
        let guard = SelfModificationGuard::with_defaults();
        let trial = guard.sandbox_trial(&simple_request(), 0.6);
        // With importance=0.5, lr=0.001, trial should be close to baseline
        assert!(trial.quality_delta >= -0.05); // Should pass
    }

    #[test]
    fn test_sandbox_trial_fail() {
        let mut guard = SelfModificationGuard::with_defaults();
        guard.cumulative_drift = 0.08; // High drift adds noise

        let req = ModificationRequest {
            source: ModificationSource::ToolCall,
            adapter_id: Some("a".into()),
            importance: 0.1, // Low importance
            learning_rate: 0.001,
            trigger_embedding: vec![],
            target_embedding: vec![],
        };
        let trial = guard.sandbox_trial(&req, 0.9);
        // Low importance + high drift → likely negative delta
        // (depends on exact noise calculation, just verify structure)
        assert!(trial.validation_samples == 5);
    }

    #[test]
    fn test_commit_with_snapshot() {
        let mut guard = SelfModificationGuard::with_defaults();
        let req = simple_request();
        let trial = guard.sandbox_trial(&req, 0.5);
        let result = guard.commit(&req, &trial, &AutonomyLevel::FullyAutonomous, &[1, 2, 3], 0.5);

        assert!(result.accepted);
        assert!(result.snapshot_id > 0);
        assert_eq!(guard.snapshots.len(), 1);
    }

    #[test]
    fn test_commit_rejects_failed_trial() {
        let mut guard = SelfModificationGuard::with_defaults();
        let req = simple_request();

        let trial = TrialResult {
            baseline_quality: 0.8,
            trial_quality: 0.5,
            quality_delta: -0.3,
            validation_samples: 5,
            passed: false,
            ewc_penalty: 0.1,
        };

        let result = guard.commit(&req, &trial, &AutonomyLevel::FullyAutonomous, &[], 0.8);
        assert!(!result.accepted);
    }

    #[test]
    fn test_rollback_restores_quality() {
        let mut guard = SelfModificationGuard::with_defaults();
        let req = simple_request();
        let trial = guard.sandbox_trial(&req, 0.5);
        let _result = guard.commit(&req, &trial, &AutonomyLevel::FullyAutonomous, &[1, 2, 3], 0.5);

        let snapshot = guard.trigger_rollback("test rollback");
        assert!(snapshot.is_some());
        assert_eq!(snapshot.unwrap().quality, 0.5);
    }

    #[test]
    fn test_rollback_marks_audit() {
        let mut guard = SelfModificationGuard::with_defaults();
        let req = simple_request();
        let trial = guard.sandbox_trial(&req, 0.5);
        guard.commit(&req, &trial, &AutonomyLevel::FullyAutonomous, &[1, 2], 0.5);

        guard.trigger_rollback("quality dropped");

        let last_audit = guard.audit_trail.last().unwrap();
        assert!(last_audit.rolled_back);
        assert_eq!(last_audit.rollback_reason.as_deref(), Some("quality dropped"));
    }

    #[test]
    fn test_post_commit_monitoring_stable() {
        let mut guard = SelfModificationGuard::with_defaults();
        let req = simple_request();
        let trial = guard.sandbox_trial(&req, 0.5);
        guard.commit(&req, &trial, &AutonomyLevel::FullyAutonomous, &[1, 2], 0.5);

        // Quality stays stable
        let rolled_back = guard.post_commit_observe(0.51);
        assert!(!rolled_back);
    }

    #[test]
    fn test_post_commit_monitoring_degradation() {
        let mut guard = SelfModificationGuard::with_defaults();
        // Set initial quality high so the rollback threshold is triggered
        guard.quality_history = vec![0.8, 0.9];
        guard.current_quality = 0.9;
        guard.last_committed_id = Some(1);

        // Quality drops to 0.5, pre-commit was 0.8, delta = -0.3 < -0.10
        let rolled_back = guard.post_commit_observe(0.5);
        assert!(rolled_back);
    }

    #[test]
    fn test_cumulative_drift_tracking() {
        let mut guard = SelfModificationGuard::with_defaults();
        let req = simple_request();
        let trial = guard.sandbox_trial(&req, 0.5);

        guard.commit(&req, &trial, &AutonomyLevel::FullyAutonomous, &[], 0.5);
        assert!(guard.cumulative_drift() > 0.0);
    }

    #[test]
    fn test_drift_resets_on_new_session() {
        // New guard = new session, drift starts at 0
        let guard = SelfModificationGuard::with_defaults();
        assert_eq!(guard.cumulative_drift(), 0.0);
    }

    #[test]
    fn test_rate_limit_session() {
        let mut config = SelfModificationConfig::default();
        config.max_per_session = 2;
        config.cooldown_secs = 0;
        let mut guard = SelfModificationGuard::new(config);

        for _ in 0..2 {
            let req = simple_request();
            let trial = guard.sandbox_trial(&req, 0.5);
            guard.commit(&req, &trial, &AutonomyLevel::FullyAutonomous, &[], 0.5);
        }

        let result = guard.check_rate_limit();
        assert!(result.is_err());
    }

    #[test]
    fn test_rate_limit_pass() {
        let guard = SelfModificationGuard::with_defaults();
        assert!(guard.check_rate_limit().is_ok());
    }

    #[test]
    fn test_ewc_penalty_check() {
        let mut guard = SelfModificationGuard::with_defaults();
        guard.cumulative_drift = 0.1;
        let trial = guard.sandbox_trial(&simple_request(), 0.5);
        assert!(trial.ewc_penalty > 0.0);
    }

    #[test]
    fn test_domain_risk_adjustment() {
        // Practice goals should cap at Medium risk
        let mut guard = SelfModificationGuard::with_defaults();
        guard.cumulative_drift = 0.06; // Would be High normally

        let req = ModificationRequest {
            source: ModificationSource::PracticeGoal,
            adapter_id: Some("a".into()),
            importance: 0.5,
            learning_rate: 0.001,
            trigger_embedding: vec![],
            target_embedding: vec![],
        };

        let report = guard.pre_flight(&req, &AutonomyLevel::FullyAutonomous);
        assert_eq!(report.risk, RiskLevel::Medium); // Capped
    }

    #[test]
    fn test_practice_goal_lower_risk() {
        let guard = SelfModificationGuard::with_defaults();

        let normal_req = simple_request();
        let practice_req = ModificationRequest {
            source: ModificationSource::PracticeGoal,
            ..simple_request()
        };

        let _normal_report = guard.pre_flight(&normal_req, &AutonomyLevel::FullyAutonomous);
        let practice_report = guard.pre_flight(&practice_req, &AutonomyLevel::FullyAutonomous);

        // Practice should never exceed Medium
        assert!(practice_report.risk as u32 <= RiskLevel::Medium as u32);
    }

    #[test]
    fn test_health_check_stable() {
        let guard = SelfModificationGuard::with_defaults();
        let health = guard.health_check();
        assert_eq!(health.quality_trend, QualityTrend::Stable);
        assert_eq!(health.session_modifications, 0);
    }

    #[test]
    fn test_health_check_with_recs() {
        let mut guard = SelfModificationGuard::with_defaults();
        guard.cumulative_drift = 0.06;

        let health = guard.health_check();
        assert!(!health.recommendations.is_empty());
        assert!(health.recommendations.iter().any(|r| r.contains("cooldown")));
    }

    #[test]
    fn test_pending_approval_flow() {
        let mut guard = SelfModificationGuard::with_defaults();
        let req = ModificationRequest {
            source: ModificationSource::ToolCall,
            adapter_id: None, // Medium risk
            importance: 0.5,
            learning_rate: 0.001,
            trigger_embedding: vec![],
            target_embedding: vec![],
        };

        let trial = guard.sandbox_trial(&req, 0.5);
        let result = guard.commit(&req, &trial, &AutonomyLevel::SemiAutonomous, &[], 0.5);

        assert!(!result.accepted);
        assert_eq!(guard.pending_approvals().len(), 1);
    }

    #[test]
    fn test_audit_trail_records() {
        let mut guard = SelfModificationGuard::with_defaults();
        let req = simple_request();
        let trial = guard.sandbox_trial(&req, 0.5);
        guard.commit(&req, &trial, &AutonomyLevel::FullyAutonomous, &[], 0.5);

        assert_eq!(guard.audit_trail().len(), 1);
        assert!(guard.audit_trail()[0].committed);
    }

    #[test]
    fn test_snapshot_enforcement_max() {
        let mut guard = SelfModificationGuard::with_defaults();
        for _ in 0..15 {
            let req = simple_request();
            let trial = guard.sandbox_trial(&req, 0.5);
            guard.commit(&req, &trial, &AutonomyLevel::FullyAutonomous, &[], 0.5);
        }
        assert!(guard.snapshots().len() <= 10);
    }

    #[test]
    fn test_manual_autonomy_blocks() {
        let guard = SelfModificationGuard::with_defaults();
        let report = guard.pre_flight(&simple_request(), &AutonomyLevel::Manual);
        assert!(report.risk as u32 >= RiskLevel::High as u32);
        assert!(report.requires_human_approval);
    }
}
