//! Tool Bootstrapper — Recursive Tool Composition
//!
//! Enables the system to compose lower-level tools into higher-level tools:
//!   Level 0: Host tools (built-in)
//!   Level 1: Model-created tools (single-purpose)
//!   Level 2: Composed tools (Level 0/1 combined into pipelines)
//!   Level 3: Meta-tools (tools that create/configure other tools)
//!
//! Safety:
//!   - Quality gate: composed tool must pass MirrorTest with quality > 0.7
//!   - Monotonicity: composed >= best constituent - 0.05
//!   - Composition budget: max 5 per session
//!   - Level limit: max 3
//!   - Divergence detection: rolling quality per level

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Tool composition level.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompositionLevel {
    /// Built-in host tool.
    Host = 0,
    /// Model-created single-purpose tool.
    ModelCreated = 1,
    /// Composed from existing tools.
    Composed = 2,
    /// Meta-tool (creates other tools).
    Meta = 3,
}

/// A composed tool — a plan template registered as a regular tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposedTool {
    /// Tool name (appears in registry).
    pub name: String,
    /// Description (for retrieval and model prompting).
    pub description: String,
    /// The plan template that implements this tool.
    pub plan_template: Vec<PlanStepDef>,
    /// Input parameter mapping.
    pub input_params: Vec<ParamDef>,
    /// Output extraction: which step's output is the return value.
    pub output_step: usize,
    /// Composition level.
    pub level: CompositionLevel,
    /// Tool version.
    pub version: u32,
}

/// A plan step definition for a composed tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStepDef {
    pub tool_name: String,
    pub arg_template: String,
}

/// A parameter definition for a composed tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamDef {
    pub name: String,
    pub description: String,
    pub required: bool,
}

/// Tool implementation variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolImpl {
    /// Built-in host tool.
    Host(String),
    /// Model-generated WASM code.
    Generated { code: String, spec: String },
    /// Composed from other tools.
    Composed(ComposedTool),
}

/// A tool version with quality tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolVersion {
    pub tool_name: String,
    pub version: u32,
    pub created_at: u64,
    pub quality: f32,
    pub success_count: u32,
    pub failure_count: u32,
    pub change_description: String,
    pub implementation: ToolImpl,
    /// Whether this version is quarantined (failed quality check).
    pub quarantined: bool,
}

/// EMA tracker for quality per composition level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmaTracker {
    pub value: f32,
    pub alpha: f32,
    pub samples: u32,
}

impl EmaTracker {
    pub fn new(alpha: f32) -> Self {
        Self { value: 0.5, alpha, samples: 0 }
    }

    pub fn update(&mut self, value: f32) {
        self.samples += 1;
        self.value = self.alpha * value + (1.0 - self.alpha) * self.value;
    }
}

/// A candidate for composition (detected pattern).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionCandidate {
    /// Tool names in the repeated sequence.
    pub tool_sequence: Vec<String>,
    /// Co-occurrence count.
    pub co_occurrence: u32,
    /// Average success rate.
    pub avg_success_rate: f32,
    /// Suggested name for the composed tool.
    pub suggested_name: String,
}

/// Divergence alert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceAlert {
    pub level: CompositionLevel,
    pub level_quality: f32,
    pub lower_level_quality: f32,
    pub message: String,
}

/// Configuration for ToolBootstrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapConfig {
    /// Min quality for composed tool registration (default: 0.7).
    pub min_quality: f32,
    /// Max composed tools per session (default: 5).
    pub max_compositions_per_session: u32,
    /// Max composition level (default: 3).
    pub max_level: u32,
    /// Min quality delta: composed >= best_constituent - delta (default: 0.05).
    pub min_quality_delta: f32,
    /// Past executions for sandbox testing (default: 10).
    pub sandbox_test_count: usize,
    /// Max versions per tool (default: 5).
    pub max_versions_per_tool: usize,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            min_quality: 0.7,
            max_compositions_per_session: 5,
            max_level: 3,
            min_quality_delta: 0.05,
            sandbox_test_count: 10,
            max_versions_per_tool: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// ToolBootstrapper
// ---------------------------------------------------------------------------

/// Orchestrates recursive tool composition and bootstrapping.
pub struct ToolBootstrapper {
    config: BootstrapConfig,
    /// Composed tools created this session.
    session_compositions: u32,
    /// Quality tracking per composition level.
    level_quality: HashMap<u32, EmaTracker>,
    /// Tool registry: name → versions.
    registry: HashMap<String, Vec<ToolVersion>>,
    /// Active version per tool.
    active_version: HashMap<String, u32>,
}

impl ToolBootstrapper {
    pub fn new(config: BootstrapConfig) -> Self {
        Self {
            config,
            session_compositions: 0,
            level_quality: HashMap::new(),
            registry: HashMap::new(),
            active_version: HashMap::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(BootstrapConfig::default())
    }

    // -----------------------------------------------------------------------
    // Tool registration
    // -----------------------------------------------------------------------

    /// Register a host tool (Level 0).
    pub fn register_host_tool(&mut self, name: &str, description: &str) {
        let version = ToolVersion {
            tool_name: name.to_string(),
            version: 1,
            created_at: Self::now_secs(),
            quality: 1.0, // host tools assumed perfect
            success_count: 0,
            failure_count: 0,
            change_description: "Initial host tool".into(),
            implementation: ToolImpl::Host(description.to_string()),
            quarantined: false,
        };

        self.registry.insert(name.to_string(), vec![version]);
        self.active_version.insert(name.to_string(), 1);
    }

    // -----------------------------------------------------------------------
    // Composition
    // -----------------------------------------------------------------------

    /// Create a composed tool from a plan template.
    pub fn compose(
        &mut self,
        name: &str,
        description: &str,
        plan_template: Vec<PlanStepDef>,
        input_params: Vec<ParamDef>,
        output_step: usize,
        level: CompositionLevel,
    ) -> Result<ComposedTool, String> {
        // Check level limit
        if (level as u32) > self.config.max_level {
            return Err(format!(
                "Composition level {} exceeds max {}",
                level as u32, self.config.max_level
            ));
        }

        // Check session budget
        if self.session_compositions >= self.config.max_compositions_per_session {
            return Err(format!(
                "Session composition budget exhausted ({}/{})",
                self.session_compositions, self.config.max_compositions_per_session
            ));
        }

        // Check that constituent tools exist
        for step in &plan_template {
            if !self.registry.contains_key(&step.tool_name) {
                return Err(format!("Constituent tool '{}' not found in registry", step.tool_name));
            }
        }

        // Determine version
        let existing_versions = self.registry.get(name).map(|v| v.len()).unwrap_or(0);
        let version = (existing_versions as u32) + 1;

        let composed = ComposedTool {
            name: name.to_string(),
            description: description.to_string(),
            plan_template,
            input_params,
            output_step,
            level,
            version,
        };

        self.session_compositions += 1;
        Ok(composed)
    }

    /// Register a composed tool (with quality validation).
    pub fn register_composed(
        &mut self,
        tool: ComposedTool,
        quality: f32,
        constituent_qualities: &[f32],
    ) -> Result<(), String> {
        // Quality gate
        if quality < self.config.min_quality {
            return Err(format!(
                "Quality {:.3} below minimum {:.3}",
                quality, self.config.min_quality
            ));
        }

        // Monotonicity check: composed must be >= best constituent - delta
        if !constituent_qualities.is_empty() {
            let best_constituent = constituent_qualities.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if quality < best_constituent - self.config.min_quality_delta {
                return Err(format!(
                    "Monotonicity violation: quality {:.3} < best constituent {:.3} - {:.3}",
                    quality, best_constituent, self.config.min_quality_delta
                ));
            }
        }

        // Determine version
        let existing_versions = self.registry.get(&tool.name).map(|v| v.len()).unwrap_or(0);
        let version = (existing_versions as u32) + 1;

        let tool_version = ToolVersion {
            tool_name: tool.name.clone(),
            version,
            created_at: Self::now_secs(),
            quality,
            success_count: 0,
            failure_count: 0,
            change_description: format!("Composed tool (level {})", tool.level as u32),
            implementation: ToolImpl::Composed(tool.clone()),
            quarantined: false,
        };

        // Enforce max versions
        let versions = self.registry.entry(tool.name.clone()).or_default();
        if versions.len() >= self.config.max_versions_per_tool {
            versions.remove(0);
        }
        versions.push(tool_version);
        self.active_version.insert(tool.name.clone(), version);

        // Update level quality tracker
        let tracker = self.level_quality.entry(tool.level as u32).or_insert_with(|| EmaTracker::new(0.2));
        tracker.update(quality);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Pattern detection
    // -----------------------------------------------------------------------

    /// Detect repeated tool usage patterns suitable for composition.
    pub fn detect_pattern(
        &self,
        tool_sequences: &[(Vec<String>, bool, f32)],
    ) -> Vec<CompositionCandidate> {
        // Count co-occurring tool sequences
        let mut pattern_counts: HashMap<Vec<String>, (u32, f32)> = HashMap::new();

        for (seq, success, quality) in tool_sequences {
            if *success && seq.len() >= 2 {
                let entry = pattern_counts.entry(seq.clone()).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 = (entry.1 + quality) / 2.0;
            }
        }

        let mut candidates: Vec<CompositionCandidate> = pattern_counts
            .into_iter()
            .filter(|(_, (count, _))| *count >= 3) // Must appear 3+ times
            .map(|(seq, (count, avg_q))| CompositionCandidate {
                suggested_name: seq.join("_"),
                tool_sequence: seq,
                co_occurrence: count,
                avg_success_rate: avg_q,
            })
            .collect();

        candidates.sort_by(|a, b| b.co_occurrence.cmp(&a.co_occurrence));
        candidates
    }

    // -----------------------------------------------------------------------
    // Divergence detection
    // -----------------------------------------------------------------------

    /// Check for divergence at any composition level.
    pub fn check_divergence(&self) -> Option<DivergenceAlert> {
        for level in 2..=3u32 {
            let current_quality = self.level_quality.get(&level).map(|t| t.value);
            let lower_quality = self.level_quality.get(&(level - 1)).map(|t| t.value);

            if let (Some(cq), Some(lq)) = (current_quality, lower_quality) {
                if cq < lq - 0.1 {
                    return Some(DivergenceAlert {
                        level: match level {
                            2 => CompositionLevel::Composed,
                            3 => CompositionLevel::Meta,
                            _ => CompositionLevel::Host,
                        },
                        level_quality: cq,
                        lower_level_quality: lq,
                        message: format!(
                            "Level {} quality ({:.3}) significantly below level {} ({:.3})",
                            level, cq, level - 1, lq
                        ),
                    });
                }
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Tool quality tracking
    // -----------------------------------------------------------------------

    /// Record a tool usage outcome.
    pub fn record_outcome(&mut self, tool_name: &str, success: bool) {
        if let Some(&version_num) = self.active_version.get(tool_name) {
            if let Some(versions) = self.registry.get_mut(tool_name) {
                if let Some(version) = versions.iter_mut().find(|v| v.version == version_num) {
                    if success {
                        version.success_count += 1;
                    } else {
                        version.failure_count += 1;
                    }
                    // Update quality with EMA
                    let alpha = 0.2;
                    version.quality = alpha * (success as u32 as f32) + (1.0 - alpha) * version.quality;

                    // Auto-rollback if quality drops too much
                    if version.quality < 0.3 && version.success_count + version.failure_count >= 5 {
                        self.auto_rollback(tool_name);
                    }
                }
            }
        }
    }

    /// Get the quality of a tool's active version.
    pub fn tool_quality(&self, tool_name: &str) -> Option<f32> {
        let &v = self.active_version.get(tool_name)?;
        let versions = self.registry.get(tool_name)?;
        versions.iter().find(|ver| ver.version == v).map(|ver| ver.quality)
    }

    // -----------------------------------------------------------------------
    // Version management
    // -----------------------------------------------------------------------

    /// Auto-rollback a tool to its previous version.
    pub fn auto_rollback(&mut self, tool_name: &str) -> bool {
        if let Some(&current_v) = self.active_version.get(tool_name) {
            if current_v > 1 {
                // Quarantine current version
                if let Some(versions) = self.registry.get_mut(tool_name) {
                    if let Some(v) = versions.iter_mut().find(|v| v.version == current_v) {
                        v.quarantined = true;
                    }
                }
                // Roll back to previous
                self.active_version.insert(tool_name.to_string(), current_v - 1);
                return true;
            }
        }
        false
    }

    /// Get the active version number for a tool.
    pub fn active_version(&self, tool_name: &str) -> Option<u32> {
        self.active_version.get(tool_name).copied()
    }

    /// Get all versions for a tool.
    pub fn tool_versions(&self, tool_name: &str) -> Vec<&ToolVersion> {
        self.registry.get(tool_name).map(|v| v.iter().collect()).unwrap_or_default()
    }

    /// Check if a tool is quarantined.
    pub fn is_quarantined(&self, tool_name: &str) -> bool {
        if let Some(&v) = self.active_version.get(tool_name) {
            if let Some(versions) = self.registry.get(tool_name) {
                return versions.iter().any(|ver| ver.version == v && ver.quarantined);
            }
        }
        false
    }

    /// Number of tools in registry.
    pub fn tool_count(&self) -> usize {
        self.registry.len()
    }

    /// Session composition count.
    pub fn session_composition_count(&self) -> u32 {
        self.session_compositions
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

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

    fn make_composed(name: &str, level: CompositionLevel) -> ComposedTool {
        ComposedTool {
            name: name.to_string(),
            description: format!("Composed tool {}", name),
            plan_template: vec![PlanStepDef {
                tool_name: "host_tool".into(),
                arg_template: "${param_0}".into(),
            }],
            input_params: vec![ParamDef {
                name: "param_0".into(),
                description: "input".into(),
                required: true,
            }],
            output_step: 0,
            level,
            version: 1,
        }
    }

    #[test]
    fn test_composed_tool_creation() {
        let mut bs = ToolBootstrapper::with_defaults();
        bs.register_host_tool("host_tool", "A host tool");

        let tool = bs.compose(
            "composed_1",
            "Composed tool",
            vec![PlanStepDef { tool_name: "host_tool".into(), arg_template: "x".into() }],
            vec![],
            0,
            CompositionLevel::Composed,
        ).unwrap();

        assert_eq!(tool.name, "composed_1");
        assert_eq!(tool.level, CompositionLevel::Composed);
        assert_eq!(tool.version, 1);
    }

    #[test]
    fn test_level_assignment() {
        assert_eq!(CompositionLevel::Host as u32, 0);
        assert_eq!(CompositionLevel::ModelCreated as u32, 1);
        assert_eq!(CompositionLevel::Composed as u32, 2);
        assert_eq!(CompositionLevel::Meta as u32, 3);
    }

    #[test]
    fn test_compose_missing_constituent() {
        let mut bs = ToolBootstrapper::with_defaults();
        let result = bs.compose(
            "bad_composed",
            "Uses nonexistent tool",
            vec![PlanStepDef { tool_name: "nonexistent".into(), arg_template: "x".into() }],
            vec![],
            0,
            CompositionLevel::Composed,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_quality_gate_pass() {
        let mut bs = ToolBootstrapper::with_defaults();
        bs.register_host_tool("host_tool", "desc");

        let tool = make_composed("good_tool", CompositionLevel::Composed);
        let result = bs.register_composed(tool, 0.8, &[0.7]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quality_gate_fail() {
        let mut bs = ToolBootstrapper::with_defaults();
        let tool = make_composed("bad_tool", CompositionLevel::Composed);
        let result = bs.register_composed(tool, 0.5, &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("below minimum"));
    }

    #[test]
    fn test_monotonicity_check() {
        let mut bs = ToolBootstrapper::with_defaults();
        let tool = make_composed("mono_fail", CompositionLevel::Composed);
        // Quality 0.75 but best constituent is 0.95 → 0.75 < 0.95 - 0.05 = 0.90
        let result = bs.register_composed(tool, 0.75, &[0.95]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Monotonicity"));
    }

    #[test]
    fn test_composition_budget() {
        let mut config = BootstrapConfig::default();
        config.max_compositions_per_session = 2;
        let mut bs = ToolBootstrapper::new(config);
        bs.register_host_tool("h", "desc");

        for i in 0..2 {
            bs.compose(
                &format!("tool_{}", i),
                "desc",
                vec![PlanStepDef { tool_name: "h".into(), arg_template: "x".into() }],
                vec![],
                0,
                CompositionLevel::Composed,
            ).unwrap();
        }

        // Third should fail
        let result = bs.compose(
            "tool_2",
            "desc",
            vec![PlanStepDef { tool_name: "h".into(), arg_template: "x".into() }],
            vec![],
            0,
            CompositionLevel::Composed,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("budget"));
    }

    #[test]
    fn test_level_limit() {
        let mut config = BootstrapConfig::default();
        config.max_level = 2;
        let mut bs = ToolBootstrapper::new(config);
        bs.register_host_tool("h", "desc");

        let result = bs.compose(
            "meta_tool",
            "desc",
            vec![PlanStepDef { tool_name: "h".into(), arg_template: "x".into() }],
            vec![],
            0,
            CompositionLevel::Meta, // level 3 > max 2
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("level"));
    }

    #[test]
    fn test_divergence_detection() {
        let mut config = BootstrapConfig::default();
        config.min_quality = 0.5; // Lower for test
        let mut bs = ToolBootstrapper::new(config);
        bs.register_host_tool("h", "desc");

        // Register a good level-1 tool
        let tool1 = make_composed("t1", CompositionLevel::ModelCreated);
        bs.register_composed(tool1, 0.9, &[]).unwrap();

        // Register a bad level-2 tool (no constituent check since [] passed)
        let tool2 = make_composed("t2", CompositionLevel::Composed);
        bs.register_composed(tool2, 0.6, &[]).unwrap();

        // Record many failures for level-2
        for _ in 0..15 {
            bs.level_quality.get_mut(&2).unwrap().update(0.3);
        }

        let alert = bs.check_divergence();
        assert!(alert.is_some());
        assert!(alert.unwrap().message.contains("significantly below"));
    }

    #[test]
    fn test_version_tracking() {
        let mut bs = ToolBootstrapper::with_defaults();
        let tool_v1 = make_composed("my_tool", CompositionLevel::Composed);
        bs.register_composed(tool_v1, 0.8, &[]).unwrap();

        let tool_v2 = make_composed("my_tool", CompositionLevel::Composed);
        bs.register_composed(tool_v2, 0.85, &[]).unwrap();

        let versions = bs.tool_versions("my_tool");
        assert_eq!(versions.len(), 2);
        assert_eq!(bs.active_version("my_tool"), Some(2));
    }

    #[test]
    fn test_auto_rollback() {
        let mut bs = ToolBootstrapper::with_defaults();
        let tool_v1 = make_composed("rollback_test", CompositionLevel::Composed);
        bs.register_composed(tool_v1, 0.8, &[]).unwrap();

        let tool_v2 = make_composed("rollback_test", CompositionLevel::Composed);
        bs.register_composed(tool_v2, 0.9, &[]).unwrap();
        assert_eq!(bs.active_version("rollback_test"), Some(2));

        // Simulate many failures
        for _ in 0..5 {
            bs.record_outcome("rollback_test", false);
        }

        // Should have rolled back to v1
        assert_eq!(bs.active_version("rollback_test"), Some(1));
        // v2 should be quarantined
        let versions = bs.tool_versions("rollback_test");
        let v2 = versions.iter().find(|v| v.version == 2);
        assert!(v2.is_some_and(|v| v.quarantined));
    }

    #[test]
    fn test_replace_vs_extend_same_tool() {
        let mut bs = ToolBootstrapper::with_defaults();
        // Register v1
        let t1 = make_composed("tool", CompositionLevel::ModelCreated);
        bs.register_composed(t1, 0.7, &[]).unwrap();
        // Register v2 with same name → replaces (new version)
        let t2 = make_composed("tool", CompositionLevel::ModelCreated);
        bs.register_composed(t2, 0.8, &[]).unwrap();

        assert_eq!(bs.tool_versions("tool").len(), 2);
        assert_eq!(bs.active_version("tool"), Some(2));
    }

    #[test]
    fn test_max_versions() {
        let mut config = BootstrapConfig::default();
        config.max_versions_per_tool = 3;
        config.min_quality = 0.4; // Lower for test
        let mut bs = ToolBootstrapper::new(config);

        for i in 0..5 {
            let t = make_composed("capped_tool", CompositionLevel::ModelCreated);
            bs.register_composed(t, 0.5 + (i as f32 * 0.1), &[]).unwrap();
        }

        // Should have at most 3 versions (old ones pruned)
        assert!(bs.tool_versions("capped_tool").len() <= 3);
    }

    #[test]
    fn test_pattern_detection() {
        let bs = ToolBootstrapper::with_defaults();
        let sequences = vec![
            (vec!["a".into(), "b".into(), "c".into()], true, 0.9),
            (vec!["a".into(), "b".into(), "c".into()], true, 0.8),
            (vec!["a".into(), "b".into(), "c".into()], true, 0.85),
            (vec!["x".into(), "y".into()], true, 0.7),
        ];

        let candidates = bs.detect_pattern(&sequences);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].tool_sequence, vec!["a", "b", "c"]);
        assert_eq!(candidates[0].co_occurrence, 3);
    }

    #[test]
    fn test_pattern_detection_insufficient_occurrences() {
        let bs = ToolBootstrapper::with_defaults();
        let sequences = vec![
            (vec!["a".into(), "b".into()], true, 0.9),
            (vec!["a".into(), "b".into()], true, 0.8),
        ];

        let candidates = bs.detect_pattern(&sequences);
        assert!(candidates.is_empty()); // Only 2 occurrences, need 3
    }

    #[test]
    fn test_record_outcome_quality_update() {
        let mut bs = ToolBootstrapper::with_defaults();
        let t = make_composed("tracked", CompositionLevel::ModelCreated);
        bs.register_composed(t, 0.8, &[]).unwrap();

        bs.record_outcome("tracked", true);
        bs.record_outcome("tracked", true);
        bs.record_outcome("tracked", false);

        let quality = bs.tool_quality("tracked").unwrap();
        assert!(quality > 0.5 && quality < 0.9);
    }

    #[test]
    fn test_quarantine_flag() {
        let mut bs = ToolBootstrapper::with_defaults();
        let t1 = make_composed("q_test", CompositionLevel::Composed);
        bs.register_composed(t1, 0.8, &[]).unwrap();
        let t2 = make_composed("q_test", CompositionLevel::Composed);
        bs.register_composed(t2, 0.9, &[]).unwrap();

        // Trigger rollback
        for _ in 0..5 {
            bs.record_outcome("q_test", false);
        }

        // Version 2 should be quarantined
        let versions = bs.tool_versions("q_test");
        let v2 = versions.iter().find(|v| v.version == 2);
        assert!(v2.is_some_and(|v| v.quarantined));
    }

    #[test]
    fn test_ema_tracker() {
        let mut tracker = EmaTracker::new(0.3);
        tracker.update(1.0);
        tracker.update(0.0);
        tracker.update(1.0);
        assert!(tracker.value > 0.5);
        assert_eq!(tracker.samples, 3);
    }

    #[test]
    fn test_session_composition_count() {
        let mut bs = ToolBootstrapper::with_defaults();
        bs.register_host_tool("h", "desc");

        bs.compose("c1", "d", vec![PlanStepDef { tool_name: "h".into(), arg_template: "x".into() }], vec![], 0, CompositionLevel::Composed).unwrap();
        bs.compose("c2", "d", vec![PlanStepDef { tool_name: "h".into(), arg_template: "x".into() }], vec![], 0, CompositionLevel::Composed).unwrap();

        assert_eq!(bs.session_composition_count(), 2);
    }

    #[test]
    fn test_host_tool_quality_is_one() {
        let mut bs = ToolBootstrapper::with_defaults();
        bs.register_host_tool("built_in", "A built-in tool");
        assert_eq!(bs.tool_quality("built_in"), Some(1.0));
    }
}
