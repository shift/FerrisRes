# Research: Safe Self-Modification Protocol — Weight Change Safety

## Summary

Research into a formal safety protocol for weight modifications via FerrisRes's 'learn' tool. Phase 8 implemented a 5-layer safety stack (autonomy gate, rate limit, pre-flight check, EWC quality gate, audit log). This research covers strengthening that into a formal protocol with sandboxed trial runs, A/B quality comparison, formal rollback, human approval gates, and integration with the Phase 9 modules (SubgoalGenerator, PlanCache, ToolBootstrapper).

## Current State (Phase 8)

The existing `execute_learn()` in `cognitive_pipeline.rs` implements 5 safety layers:

```
Layer 1: Autonomy gate — SemiAutonomous or FullyAutonomous required
Layer 2: Rate limiting — 5/session, 20/hour, 60s cooldown
Layer 3: (placeholder for pre-flight quality check)
Layer 4: EWC quality gate — ToolTriggeredLora checks new_quality >= 0.6
Layer 5: Audit log — tracing::info! with adapter_id, quality_gate_passed
```

ToolTriggeredLora provides:
- **FisherDiagonal**: EWC penalty λ/2 × Σ fisher_i × (θ_i - θ*_i)² prevents catastrophic forgetting
- **LearningEvent**: records loss_before, loss_after, quality_gate_passed, adapter_id
- **quality_gate_threshold**: 0.6 default, rejects weight changes that degrade quality
- **Max adapters**: caps total learned adapters
- **Per-adapter stats**: usage_count, success_rate, quality_pass_rate

## Gaps in Phase 8 Safety

1. **No sandboxing**: Weight changes are applied directly. No "try before commit".
2. **No A/B comparison**: We check `new_quality >= threshold` but don't compare against the *baseline* quality for this specific input.
3. **No formal rollback**: If a weight change causes downstream degradation, there's no mechanism to revert it.
4. **No pre-flight simulation**: We don't simulate the modification's impact before committing.
5. **Audit trail is ephemeral**: `tracing::info!` lines aren't persisted across sessions.
6. **No human approval gate**: The autonomy check only gates by level, not by risk assessment.
7. **No cumulative impact tracking**: Each learning event is evaluated in isolation. Accumulated drift across many small changes isn't monitored.

## Proposed: SelfModificationGuard Module

### Architecture

```
SelfModificationGuard
├── pre_flight(request) → PreFlightReport
│   ├── Risk assessment: is this a safe modification?
│   ├── Cumulative drift check: how much has the model changed this session?
│   └── Human approval gate (if autonomy < FullyAutonomous and risk > medium)
├── sandbox_trial(request) → TrialResult
│   ├── Clone current weights → apply modification → test against validation set
│   ├── Compare trial quality vs baseline quality
│   └── Return A/B comparison with confidence interval
├── commit(request, trial_result) → CommitResult
│   ├── Final quality gate (EWC + A/B delta)
│   ├── Snapshot weights for rollback
│   ├── Apply modification
│   └── Persist audit record
├── rollback(reason) → RollbackResult
│   ├── Restore last snapshot
│   ├── Mark failed modification in audit trail
│   └── Alert ProactiveController
└── health_check() → HealthReport
    ├── Cumulative drift since last checkpoint
    ├── Quality trend (improving/stable/degrading)
    └── Recommendations (pause learning, increase cooldown, etc.)
```

### Data Structures

```rust
/// A proposed weight modification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModificationRequest {
    /// What triggered this modification.
    pub source: ModificationSource,
    /// The LoRA adapter to modify (or create).
    pub adapter_id: Option<String>,
    /// Input embedding that triggered learning.
    pub trigger_embedding: Vec<f32>,
    /// Target embedding (desired behavior).
    pub target_embedding: Vec<f32>,
    /// Estimated importance (from IntrinsicMotivation).
    pub importance: f32,
    /// Requested learning rate.
    pub learning_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Pre-flight risk assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreFlightReport {
    /// Risk level of this modification.
    pub risk: RiskLevel,
    /// Cumulative parameter drift this session (0.0 = no change, 1.0 = complete overwrite).
    pub cumulative_drift: f32,
    /// Whether human approval is required.
    pub requires_human_approval: bool,
    /// Estimated quality impact.
    pub estimated_quality_delta: f32,
    /// Reason for the risk assessment.
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RiskLevel {
    /// Low risk: small parameter change, high-confidence domain.
    Low,
    /// Medium risk: moderate change, or unfamiliar domain.
    Medium,
    /// High risk: large change, or core capabilities affected.
    High,
    /// Critical: would affect safety-critical weights (Armor, etc.).
    Critical,
}

/// Result of a sandboxed trial run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    /// Baseline quality (before modification).
    pub baseline_quality: f32,
    /// Trial quality (after modification on same inputs).
    pub trial_quality: f32,
    /// Quality delta (positive = improvement).
    pub quality_delta: f32,
    /// Number of validation samples tested.
    pub validation_samples: usize,
    /// Whether the trial passed the quality gate.
    pub passed: bool,
    /// EWC penalty incurred by this modification.
    pub ewc_penalty: f32,
}

/// Result of committing a modification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitResult {
    /// The committed modification's ID.
    pub modification_id: u64,
    /// Quality after commit.
    pub post_commit_quality: f32,
    /// Snapshot ID for rollback.
    pub snapshot_id: u64,
    /// Whether the commit was accepted.
    pub accepted: bool,
    /// Reason if rejected.
    pub rejection_reason: Option<String>,
}

/// Persistent audit record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    /// Unique ID.
    pub id: u64,
    /// Timestamp.
    pub timestamp: u64,
    /// The modification source.
    pub source: ModificationSource,
    /// Risk level assigned.
    pub risk: RiskLevel,
    /// Pre-flight decision.
    pub pre_flight_approved: bool,
    /// Trial result.
    pub trial_quality_delta: f32,
    /// Whether committed.
    pub committed: bool,
    /// Whether later rolled back.
    pub rolled_back: bool,
    /// Rollback reason (if any).
    pub rollback_reason: Option<String>,
    /// Autonomy level at time of modification.
    pub autonomy_level: String,
}
```

### Safety Protocol Steps

**Step 1: Pre-flight Check**

```
risk = assess_risk(request):
  if request.adapter_id is None (new adapter):
    risk = Medium  // new adapter = moderate risk
  if request.importance < 0.3:
    risk = Low     // unimportant input = low risk
  if cumulative_drift > 0.1:
    risk = High    // already changed a lot this session
  if cumulative_drift > 0.3:
    risk = Critical // too much drift, block
  
  human_required = (risk >= Medium && autonomy < FullyAutonomous)
                  || (risk >= High)
```

**Step 2: Sandbox Trial**

```
trial = sandbox_trial(request):
  1. Clone the current ToolTriggeredLora state (shallow copy of weights)
  2. Apply the learning event to the clone
  3. Run forward pass on last N validation samples (N=5 by default)
  4. Compare quality: baseline_quality vs trial_quality
  5. If quality_delta < -0.05: FAIL (any degradation > 5%)
  6. If quality_delta < 0.0: WARN (slight regression, allow if EWC penalty low)
  7. If quality_delta >= 0.0: PASS
```

The sandbox is cheap — it reuses the existing CPU forward pass. No GPU needed.

**Step 3: Commit**

```
commit(request, trial):
  1. Snapshot current weights (for rollback)
  2. Apply modification to real ToolTriggeredLora
  3. Run EWC quality gate (existing logic, threshold 0.6)
  4. Record audit entry
  5. If EWC gate fails: rollback immediately
```

**Step 4: Post-commit Monitoring**

After committing, track quality over the next K interactions:
- If quality drops > 0.1 below pre-commit baseline over 5 interactions → auto-rollback
- If quality stable or improving → confirm commit

### Rollback Mechanism

```rust
struct WeightSnapshot {
    id: u64,
    timestamp: u64,
    /// Serialized ToolTriggeredLora state.
    lora_state: Vec<u8>,
    /// Quality at time of snapshot.
    quality: f32,
    /// Number of adapters at time of snapshot.
    adapter_count: usize,
}
```

Snapshots are stored in memory (Vec<WeightSnapshot>, max 10). On rollback:
1. Pop the latest snapshot
2. Deserialize into ToolTriggeredLora
3. Log rollback event with quality evidence
4. IntrinsicMotivation gets a practice goal: "Why did this learning fail?"

### Cumulative Drift Tracking

Drift = how much the model has changed from its initial state this session.

```
drift = Σ (|Δθ_i| / |θ_i|) for all modified parameters

Where:
  Δθ_i = parameter change from last snapshot
  θ_i = parameter value at session start

Drift thresholds:
  < 0.01: Low (normal learning)
  0.01–0.05: Medium (monitor closely)
  0.05–0.10: High (increase cooldown, reduce learning rate)
  > 0.10: Critical (block further learning this session)
```

### Human Approval Gate

When autonomy < FullyAutonomous AND risk >= Medium:
1. The modification is queued, not executed
2. A human-readable explanation is generated:
   ```
   "Proposed learning: adapt 'math_eval' tool for calculus problems.
    Risk: Medium (new adapter in unfamiliar domain).
    Estimated quality impact: +0.08.
    Approval required: autonomy is SemiAutonomous."
   ```
3. The human can approve, reject, or defer
4. Deferred modifications expire after 1 hour

### Audit Trail Persistence

Audit records are persisted to JSON alongside the cognitive pipeline state:
```
~/.ferrisres/audit/
  session_<timestamp>.json   — per-session audit log
  cumulative.json            — cross-session drift + quality trends
```

### Integration with Phase 9 Modules

- **SubgoalGenerator**: Learning requests from subgoal decomposition are tagged `ModificationSource::PracticeGoal` and get lower risk assessment (structured learning is safer than ad-hoc).
- **PlanCache**: If a plan's quality drops after a weight change, PlanCache flags the affected plan for re-evaluation.
- **ToolBootstrapper**: Composed tool creation goes through SelfModificationGuard if it involves weight changes. Bootstrapping safety (level limit, monotonicity) is enforced here.
- **DomainDetector**: Cross-domain learning is higher risk (risk = Medium+ instead of Low). Domain confidence < 0.5 bumps risk by one level.

### Configuration

```rust
pub struct SelfModificationConfig {
    /// Maximum cumulative drift before blocking (default: 0.10).
    pub max_cumulative_drift: f32,
    /// Warning drift threshold (default: 0.05).
    pub drift_warning_threshold: f32,
    /// Quality degradation that triggers auto-rollback (default: -0.10).
    pub rollback_quality_delta: f32,
    /// Number of post-commit interactions to monitor (default: 5).
    pub post_commit_monitoring_window: usize,
    /// Maximum snapshots kept in memory (default: 10).
    pub max_snapshots: usize,
    /// Number of validation samples for sandbox trial (default: 5).
    pub sandbox_validation_samples: usize,
    /// Minimum quality delta to pass sandbox (default: -0.05).
    pub sandbox_min_quality_delta: f32,
    /// Whether human approval is required for Medium risk (default: true).
    pub human_approval_for_medium: bool,
    /// Whether to persist audit records (default: true).
    pub persist_audit: bool,
    /// Path for audit persistence.
    pub audit_path: Option<PathBuf>,
}
```

### Test Plan

1. `test_risk_assessment_low` — familiar domain, small change → Low
2. `test_risk_assessment_high_drift` — cumulative drift > 0.05 → High
3. `test_risk_assessment_critical_drift` — cumulative drift > 0.10 → Critical
4. `test_preflight_approve_low_risk` — Low risk passes without approval
5. `test_preflight_block_critical` — Critical risk blocked
6. `test_preflight_human_approval` — Medium risk + SemiAutonomous requires approval
7. `test_sandbox_trial_pass` — quality improvement → pass
8. `test_sandbox_trial_fail` — quality degradation > 5% → fail
9. `test_commit_with_snapshot` — snapshot created before commit
10. `test_rollback_restores_weights` — rollback restores previous state
11. `test_rollback_marks_audit` — audit record shows rolled_back = true
12. `test_post_commit_monitoring_stable` — quality stable → confirmed
13. `test_post_commit_monitoring_degradation` — quality drops → auto-rollback
14. `test_cumulative_drift_tracking` — drift accumulates across modifications
15. `test_drift_resets_on_new_session` — new session starts fresh
16. `test_audit_persistence_roundtrip` — save/load audit records
17. `test_rate_limit_integration` — existing rate limits still enforced
18. `test_ewc_penalty_check` — high EWC penalty bumps risk
19. `test_domain_risk_adjustment` — unfamiliar domain → +1 risk level
20. `test_practice_goal_lower_risk` — structured learning gets lower risk

## References

- Kirkpatrick, J. et al. (2017). "Overcoming Catastrophic Forgetting in Neural Networks" — EWC foundations
- Nguyen, C. et al. (2019). "Variational Continual Learning" — principled weight protection
- Masse, N. et al. (2018). "Alleviating Catastrophic Forgetting Through Bayesian Neurophysiological Regulation" — synaptic consolidation
- Riemer, M. et al. (2019). "Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference" — gradient-based safety
- Xu, J. & Zhu, Z. (2018). "Reinforced Continual Learning" — safe exploration in weight space
- Amodei, D. et al. (2016). "Concrete Problems in AI Safety" — modification safety taxonomy
- Hadfield-Menell, D. et al. (2017). "The Off-Switch Game" — human oversight in self-modifying systems
- Everitt, T. et al. (2018). "Agent Foundations for Safe Reinforcement Learning" — self-modification safety protocols
