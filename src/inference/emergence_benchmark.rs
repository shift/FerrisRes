//! Emergence Benchmark — Quantitative Emergence Measurement
//!
//! Based on BIG-Bench, grokking/phase transitions, and capability overhang.
//! Measures emergent behaviors across 6 categories:
//!   1. Skill Acquisition — improvement over repeated attempts
//!   2. Self-Correction — MirrorTest error recurrence rate
//!   3. Self-Extension — self-created tools, reuse, chain depth
//!   4. Cognitive Scaffolding — concept count vs task diversity correlation
//!   5. Planning — plan success rate and depth over time
//!   6. Abstraction — compression ratio and generalization quality
//!
//! Compares baseline (no cognitive pipeline) vs augmented (with pipeline)
//! to quantify the delta = emergent capability from the cognitive architecture.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Emergence categories
// ---------------------------------------------------------------------------

/// Category of emergent behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EmergenceCategory {
    /// Task performance improvement over repeated attempts.
    SkillAcquisition,
    /// MirrorTest pass rate, error recurrence.
    SelfCorrection,
    /// Self-created tools, reuse frequency, chain depth.
    SelfExtension,
    /// Concept count vs task diversity correlation.
    CognitiveScaffolding,
    /// Plan success rate and depth over time.
    Planning,
    /// Compression ratio and generalization quality.
    Abstraction,
}

impl EmergenceCategory {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            EmergenceCategory::SkillAcquisition => "Skill Acquisition",
            EmergenceCategory::SelfCorrection => "Self-Correction",
            EmergenceCategory::SelfExtension => "Self-Extension",
            EmergenceCategory::CognitiveScaffolding => "Cognitive Scaffolding",
            EmergenceCategory::Planning => "Planning",
            EmergenceCategory::Abstraction => "Abstraction",
        }
    }

    /// All categories.
    pub fn all() -> &'static [EmergenceCategory] {
        &[
            EmergenceCategory::SkillAcquisition,
            EmergenceCategory::SelfCorrection,
            EmergenceCategory::SelfExtension,
            EmergenceCategory::CognitiveScaffolding,
            EmergenceCategory::Planning,
            EmergenceCategory::Abstraction,
        ]
    }
}

// ---------------------------------------------------------------------------
// Measurement sample
// ---------------------------------------------------------------------------

/// A single measurement sample for an emergence metric.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Measurement {
    /// Category being measured.
    pub category: EmergenceCategory,
    /// Metric name (e.g., "improvement_rate", "correction_rate").
    pub metric: String,
    /// Baseline value (no cognitive pipeline).
    pub baseline: f32,
    /// Augmented value (with cognitive pipeline).
    pub augmented: f32,
    /// Delta (augmented - baseline).
    pub delta: f32,
    /// Timestamp.
    pub timestamp: u64,
    /// Additional context.
    pub context: HashMap<String, f32>,
}

impl Measurement {
    /// Create a new measurement.
    pub fn new(category: EmergenceCategory, metric: impl Into<String>, baseline: f32, augmented: f32) -> Self {
        let delta = augmented - baseline;
        Self {
            category,
            metric: metric.into(),
            baseline,
            augmented,
            delta,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            context: HashMap::new(),
        }
    }

    /// Add context value.
    pub fn with_context(mut self, key: impl Into<String>, value: f32) -> Self {
        self.context.insert(key.into(), value);
        self
    }

    /// Emergence ratio (augmented / baseline). Returns 0 if baseline is 0.
    pub fn emergence_ratio(&self) -> f32 {
        if self.baseline.abs() < 1e-6 {
            if self.augmented.abs() < 1e-6 {
                1.0 // Both zero = no emergence
            } else {
                f32::INFINITY // Baseline zero, augmented nonzero = full emergence
            }
        } else {
            self.augmented / self.baseline
        }
    }
}

// ---------------------------------------------------------------------------
// Emergence score
// ---------------------------------------------------------------------------

/// Aggregated emergence score for a category.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmergenceScore {
    /// Category.
    pub category: EmergenceCategory,
    /// Number of measurements.
    pub samples: usize,
    /// Average delta.
    pub avg_delta: f32,
    /// Average emergence ratio.
    pub avg_ratio: f32,
    /// Whether emergence was detected (delta > threshold).
    pub emerged: bool,
    /// Confidence (0.0–1.0 based on sample count).
    pub confidence: f32,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the emergence benchmark.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmergenceConfig {
    /// Minimum delta to count as emergence per category.
    pub emergence_thresholds: HashMap<EmergenceCategory, f32>,
    /// Minimum samples for confident detection.
    pub min_samples: usize,
    /// Maximum history per category.
    pub max_history: usize,
}

impl Default for EmergenceConfig {
    fn default() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert(EmergenceCategory::SkillAcquisition, 0.1);
        thresholds.insert(EmergenceCategory::SelfCorrection, 0.15);
        thresholds.insert(EmergenceCategory::SelfExtension, 0.05);
        thresholds.insert(EmergenceCategory::CognitiveScaffolding, 0.3);
        thresholds.insert(EmergenceCategory::Planning, 0.1);
        thresholds.insert(EmergenceCategory::Abstraction, 0.1);

        Self {
            emergence_thresholds: thresholds,
            min_samples: 3,
            max_history: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// Snapshot data
// ---------------------------------------------------------------------------

/// A snapshot of the system state for measuring emergence.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SystemSnapshot {
    /// Number of concepts in ConceptMap.
    pub concept_count: usize,
    /// Number of episodes in EpisodicMemory.
    pub episode_count: usize,
    /// Number of self-created tools.
    pub created_tool_count: usize,
    /// Average tool reuse (times a created tool was invoked).
    pub avg_tool_reuse: f32,
    /// Maximum tool chain depth observed.
    pub max_chain_depth: usize,
    /// MirrorTest pass rate (0.0–1.0).
    pub mirror_pass_rate: f32,
    /// Error recurrence rate (0.0–1.0, lower is better).
    pub error_recurrence_rate: f32,
    /// Plan success rate (0.0–1.0).
    pub plan_success_rate: f32,
    /// Average plan depth (steps).
    pub avg_plan_depth: f32,
    /// Abstraction compression ratio.
    pub compression_ratio: f32,
    /// Number of abstraction levels reached.
    pub abstraction_levels: usize,
    /// Number of mastered concepts.
    pub mastered_count: usize,
    /// Task diversity (unique prompt categories).
    pub task_diversity: usize,
    /// Improvement rate (Δ quality / attempts).
    pub improvement_rate: f32,
    /// Baseline quality (without cognitive pipeline).
    pub baseline_quality: f32,
    /// Augmented quality (with cognitive pipeline).
    pub augmented_quality: f32,
}

// ---------------------------------------------------------------------------
// EmergenceBenchmark
// ---------------------------------------------------------------------------

/// Quantitative emergence measurement across cognitive architecture layers.
pub struct EmergenceBenchmark {
    config: EmergenceConfig,
    /// Measurement history, by category.
    history: HashMap<EmergenceCategory, Vec<Measurement>>,
    /// Snapshots over time.
    snapshots: Vec<(u64, SystemSnapshot)>,
    /// Computed scores cache.
    scores: HashMap<EmergenceCategory, EmergenceScore>,
}

impl EmergenceBenchmark {
    /// Create a new benchmark with the given configuration.
    pub fn new(config: EmergenceConfig) -> Self {
        Self {
            config,
            history: HashMap::new(),
            snapshots: Vec::new(),
            scores: HashMap::new(),
        }
    }

    /// Create with default configuration.
    pub fn default_benchmark() -> Self {
        Self::new(EmergenceConfig::default())
    }

    /// Get configuration.
    pub fn config(&self) -> &EmergenceConfig {
        &self.config
    }

    /// Total measurements recorded.
    pub fn total_measurements(&self) -> usize {
        self.history.values().map(|v| v.len()).sum()
    }

    /// Total snapshots recorded.
    pub fn total_snapshots(&self) -> usize {
        self.snapshots.len()
    }

    // ---- Main API ----

    /// Record a snapshot of the system state.
    pub fn record_snapshot(&mut self, snapshot: SystemSnapshot) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        self.snapshots.push((timestamp, snapshot));

        // Trim if needed
        if self.snapshots.len() > self.config.max_history {
            let excess = self.snapshots.len() - self.config.max_history;
            self.snapshots.drain(..excess);
        }
    }

    /// Record a measurement for a category.
    pub fn record_measurement(&mut self, measurement: Measurement) {
        let category = measurement.category;
        self.history
            .entry(category)
            .or_default()
            .push(measurement);

        // Trim per-category history
        if let Some(history) = self.history.get_mut(&category) {
            if history.len() > self.config.max_history {
                let excess = history.len() - self.config.max_history;
                history.drain(..excess);
            }
        }

        // Recompute score for this category
        self.recompute_score(category);
    }

    /// Measure skill acquisition from snapshot data.
    pub fn measure_skill_acquisition(&mut self, baseline_improvement: f32, augmented_improvement: f32) -> &EmergenceScore {
        let m = Measurement::new(
            EmergenceCategory::SkillAcquisition,
            "improvement_rate",
            baseline_improvement,
            augmented_improvement,
        );
        self.record_measurement(m);
        self.scores.get(&EmergenceCategory::SkillAcquisition).unwrap()
    }

    /// Measure self-correction capability.
    pub fn measure_self_correction(&mut self, error_recurrence_rate: f32, mirror_pass_rate: f32) -> &EmergenceScore {
        // Baseline: no correction (recurrence = 1.0, pass rate depends)
        // Augmented: with correction loop
        let baseline_correction = 1.0 - 1.0; // No correction → rate = 0
        let augmented_correction = 1.0 - error_recurrence_rate;

        let m = Measurement::new(
            EmergenceCategory::SelfCorrection,
            "correction_rate",
            baseline_correction,
            augmented_correction,
        )
            .with_context("mirror_pass_rate", mirror_pass_rate)
            .with_context("error_recurrence", error_recurrence_rate);
        self.record_measurement(m);
        self.scores.get(&EmergenceCategory::SelfCorrection).unwrap()
    }

    /// Measure self-extension (tool creation and reuse).
    pub fn measure_self_extension(&mut self, tool_count: usize, avg_reuse: f32, max_depth: usize) -> &EmergenceScore {
        // Extension score = tool_count * avg_reuse * max_depth
        let augmented = (tool_count as f32).max(0.0) * avg_reuse.max(0.0) * (max_depth as f32).max(1.0);
        let baseline = 0.0; // No self-created tools without cognitive pipeline

        let m = Measurement::new(
            EmergenceCategory::SelfExtension,
            "extension_score",
            baseline,
            augmented,
        )
            .with_context("tool_count", tool_count as f32)
            .with_context("avg_reuse", avg_reuse)
            .with_context("max_depth", max_depth as f32);
        self.record_measurement(m);
        self.scores.get(&EmergenceCategory::SelfExtension).unwrap()
    }

    /// Measure cognitive scaffolding (concept count vs task diversity).
    pub fn measure_scaffolding(&mut self, concept_count: usize, task_diversity: usize) -> &EmergenceScore {
        let augmented = if concept_count > 0 && task_diversity > 0 {
            (concept_count as f32).ln() * (task_diversity as f32).ln()
        } else {
            0.0
        };
        let baseline = 0.0;

        let m = Measurement::new(
            EmergenceCategory::CognitiveScaffolding,
            "scaffolding_score",
            baseline,
            augmented,
        )
            .with_context("concept_count", concept_count as f32)
            .with_context("task_diversity", task_diversity as f32);
        self.record_measurement(m);
        self.scores.get(&EmergenceCategory::CognitiveScaffolding).unwrap()
    }

    /// Measure planning capability.
    pub fn measure_planning(&mut self, baseline_success: f32, augmented_success: f32, avg_depth: f32) -> &EmergenceScore {
        let m = Measurement::new(
            EmergenceCategory::Planning,
            "plan_success_rate",
            baseline_success,
            augmented_success,
        )
            .with_context("avg_depth", avg_depth);
        self.record_measurement(m);
        self.scores.get(&EmergenceCategory::Planning).unwrap()
    }

    /// Measure abstraction capability.
    pub fn measure_abstraction(&mut self, compression_ratio: f32, levels_reached: usize) -> &EmergenceScore {
        let baseline = 1.0; // No compression without abstraction
        let augmented = compression_ratio * (levels_reached as f32).max(1.0);

        let m = Measurement::new(
            EmergenceCategory::Abstraction,
            "abstraction_score",
            baseline,
            augmented,
        )
            .with_context("compression_ratio", compression_ratio)
            .with_context("levels", levels_reached as f32);
        self.record_measurement(m);
        self.scores.get(&EmergenceCategory::Abstraction).unwrap()
    }

    /// Run a full measurement pass from a snapshot.
    pub fn measure_from_snapshot(&mut self, baseline_quality: f32, snapshot: &SystemSnapshot) {
        // Skill Acquisition
        self.measure_skill_acquisition(
            baseline_quality,
            snapshot.augmented_quality,
        );

        // Self-Correction
        self.measure_self_correction(
            snapshot.error_recurrence_rate,
            snapshot.mirror_pass_rate,
        );

        // Self-Extension
        self.measure_self_extension(
            snapshot.created_tool_count,
            snapshot.avg_tool_reuse,
            snapshot.max_chain_depth,
        );

        // Cognitive Scaffolding
        self.measure_scaffolding(
            snapshot.concept_count,
            snapshot.task_diversity,
        );

        // Planning
        self.measure_planning(
            baseline_quality * 0.5, // Estimate: baseline plans succeed less
            snapshot.plan_success_rate,
            snapshot.avg_plan_depth,
        );

        // Abstraction
        self.measure_abstraction(
            snapshot.compression_ratio,
            snapshot.abstraction_levels,
        );

        self.record_snapshot(snapshot.clone());
    }

    // ---- Scoring ----

    fn recompute_score(&mut self, category: EmergenceCategory) {
        let measurements = self.history.get(&category).map(|v| v.as_slice()).unwrap_or(&[]);
        let samples = measurements.len();

        if samples == 0 {
            self.scores.remove(&category);
            return;
        }

        let avg_delta = measurements.iter().map(|m| m.delta).sum::<f32>() / samples as f32;
        let valid_ratios: Vec<f32> = measurements.iter()
            .filter_map(|m| {
                let r = m.emergence_ratio();
                if r.is_finite() { Some(r) } else { None }
            })
            .collect();
        let avg_ratio = if valid_ratios.is_empty() {
            0.0
        } else {
            valid_ratios.iter().sum::<f32>() / valid_ratios.len() as f32
        };

        let threshold = self.config.emergence_thresholds.get(&category).copied().unwrap_or(0.1);
        let emerged = avg_delta > threshold;

        // Confidence increases with sample count (logarithmic)
        let confidence = (1.0 + samples as f32).ln() / (1.0 + self.config.min_samples as f32).ln();
        let confidence = confidence.min(1.0);

        self.scores.insert(category, EmergenceScore {
            category,
            samples,
            avg_delta,
            avg_ratio,
            emerged,
            confidence,
        });
    }

    /// Get the emergence score for a category.
    pub fn score(&self, category: EmergenceCategory) -> Option<&EmergenceScore> {
        self.scores.get(&category)
    }

    /// Get all emergence scores.
    pub fn all_scores(&self) -> Vec<&EmergenceScore> {
        EmergenceCategory::all().iter()
            .filter_map(|&cat| self.scores.get(&cat))
            .collect()
    }

    /// Check if emergence was detected in any category.
    pub fn any_emerged(&self) -> bool {
        self.scores.values().any(|s| s.emerged)
    }

    /// Count categories where emergence was detected.
    pub fn emerged_count(&self) -> usize {
        self.scores.values().filter(|s| s.emerged).count()
    }

    /// Compute a composite emergence index (0.0–1.0).
    pub fn emergence_index(&self) -> f32 {
        let scores = self.all_scores();
        if scores.is_empty() {
            return 0.0;
        }

        let weighted_sum: f32 = scores.iter()
            .filter(|s| s.confidence >= 0.5) // Only confident measurements
            .map(|s| {
                let weight = s.confidence;
                let emerged_value = if s.emerged { s.avg_delta } else { 0.0 };
                weight * emerged_value
            })
            .sum();

        let total_weight: f32 = scores.iter()
            .filter(|s| s.confidence >= 0.5)
            .map(|s| s.confidence)
            .sum();

        if total_weight > 0.0 {
            (weighted_sum / total_weight).min(1.0)
        } else {
            0.0
        }
    }

    /// Get trend analysis: is emergence increasing, stable, or decreasing?
    pub fn trend(&self, category: EmergenceCategory) -> EmergenceTrend {
        let measurements = self.history.get(&category).map(|v| v.as_slice()).unwrap_or(&[]);
        if measurements.len() < 3 {
            return EmergenceTrend::InsufficientData;
        }

        // Compare last third to first third
        let n = measurements.len();
        let first_third = &measurements[..n / 3];
        let last_third = &measurements[n - n / 3..];

        let first_avg = first_third.iter().map(|m| m.delta).sum::<f32>() / first_third.len() as f32;
        let last_avg = last_third.iter().map(|m| m.delta).sum::<f32>() / last_third.len() as f32;

        let diff = last_avg - first_avg;
        if diff > 0.05 {
            EmergenceTrend::Increasing(diff)
        } else if diff < -0.05 {
            EmergenceTrend::Decreasing(diff)
        } else {
            EmergenceTrend::Stable
        }
    }

    // ---- Report ----

    /// Generate a human-readable emergence report.
    pub fn report(&self) -> EmergenceReport {
        let scores = EmergenceCategory::all().iter()
            .filter_map(|&cat| self.score(cat).cloned())
            .collect();

        EmergenceReport {
            total_measurements: self.total_measurements(),
            total_snapshots: self.total_snapshots(),
            emerged_categories: self.emerged_count(),
            total_categories: EmergenceCategory::all().len(),
            emergence_index: self.emergence_index(),
            scores,
        }
    }

    // ---- Persistence ----

    /// Save benchmark state to JSON.
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let data = BenchmarkState {
            config: self.config.clone(),
            history: self.history.clone(),
            snapshots: self.snapshots.clone(),
        };
        let json = serde_json::to_string_pretty(&data).map_err(|e| format!("Serialize error: {}", e))?;

        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let tmp_path = format!("{}.tmp", path.display());
        std::fs::write(&tmp_path, &json).map_err(|e| format!("Write error: {}", e))?;
        std::fs::rename(&tmp_path, path).map_err(|e| format!("Rename error: {}", e))?;

        Ok(())
    }

    /// Load benchmark state from JSON.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| format!("Read error: {}", e))?;
        let data: BenchmarkState = serde_json::from_str(&json).map_err(|e| format!("Deserialize error: {}", e))?;

        let mut benchmark = Self::new(data.config);
        // Replay measurements to rebuild scores
        for measurements in data.history.values() {
            for m in measurements {
                let cat = m.category;
                benchmark.history
                    .entry(cat)
                    .or_default()
                    .push(m.clone());
            }
        }
        benchmark.snapshots = data.snapshots;

        // Recompute all scores
        for &cat in EmergenceCategory::all() {
            benchmark.recompute_score(cat);
        }

        Ok(benchmark)
    }

    // ---- Stats ----

    /// Get benchmark statistics.
    pub fn stats(&self) -> BenchmarkStats {
        BenchmarkStats {
            total_measurements: self.total_measurements(),
            total_snapshots: self.total_snapshots(),
            emerged_categories: self.emerged_count(),
            emergence_index: self.emergence_index(),
        }
    }
}

// ---------------------------------------------------------------------------
// Trend
// ---------------------------------------------------------------------------

/// Trend direction for an emergence metric.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum EmergenceTrend {
    /// Not enough data.
    InsufficientData,
    /// Emergence delta is increasing.
    Increasing(f32),
    /// Emergence delta is stable.
    Stable,
    /// Emergence delta is decreasing.
    Decreasing(f32),
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Full emergence report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmergenceReport {
    pub total_measurements: usize,
    pub total_snapshots: usize,
    pub emerged_categories: usize,
    pub total_categories: usize,
    pub emergence_index: f32,
    pub scores: Vec<EmergenceScore>,
}

// ---------------------------------------------------------------------------
// Serializable state
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
struct BenchmarkState {
    config: EmergenceConfig,
    history: HashMap<EmergenceCategory, Vec<Measurement>>,
    snapshots: Vec<(u64, SystemSnapshot)>,
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Benchmark statistics summary.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchmarkStats {
    pub total_measurements: usize,
    pub total_snapshots: usize,
    pub emerged_categories: usize,
    pub emergence_index: f32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measurement_creation() {
        let m = Measurement::new(EmergenceCategory::SkillAcquisition, "test_metric", 0.3, 0.7);
        assert_eq!(m.baseline, 0.3);
        assert_eq!(m.augmented, 0.7);
        assert!((m.delta - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_measurement_emergence_ratio() {
        let m = Measurement::new(EmergenceCategory::SkillAcquisition, "test", 0.5, 1.0);
        assert!((m.emergence_ratio() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_measurement_zero_baseline() {
        let m = Measurement::new(EmergenceCategory::SelfExtension, "test", 0.0, 0.5);
        assert!(m.emergence_ratio().is_infinite());
    }

    #[test]
    fn test_measurement_both_zero() {
        let m = Measurement::new(EmergenceCategory::SkillAcquisition, "test", 0.0, 0.0);
        assert!((m.emergence_ratio() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_measurement_with_context() {
        let m = Measurement::new(EmergenceCategory::Planning, "test", 0.3, 0.8)
            .with_context("depth", 5.0);
        assert_eq!(m.context.get("depth"), Some(&5.0));
    }

    #[test]
    fn test_category_names() {
        assert_eq!(EmergenceCategory::SkillAcquisition.name(), "Skill Acquisition");
        assert_eq!(EmergenceCategory::SelfCorrection.name(), "Self-Correction");
        assert_eq!(EmergenceCategory::SelfExtension.name(), "Self-Extension");
        assert_eq!(EmergenceCategory::CognitiveScaffolding.name(), "Cognitive Scaffolding");
        assert_eq!(EmergenceCategory::Planning.name(), "Planning");
        assert_eq!(EmergenceCategory::Abstraction.name(), "Abstraction");
    }

    #[test]
    fn test_all_categories() {
        assert_eq!(EmergenceCategory::all().len(), 6);
    }

    #[test]
    fn test_measure_skill_acquisition() {
        let mut bench = EmergenceBenchmark::default_benchmark();
        let score = bench.measure_skill_acquisition(0.2, 0.6);

        assert_eq!(score.samples, 1);
        assert!((score.avg_delta - 0.4).abs() < 0.01);
        assert!(score.emerged); // 0.4 > threshold 0.1
    }

    #[test]
    fn test_measure_self_correction() {
        let mut bench = EmergenceBenchmark::default_benchmark();
        let score = bench.measure_self_correction(0.2, 0.8);

        assert_eq!(score.samples, 1);
        assert!(score.avg_delta > 0.0);
        assert!(score.emerged); // delta > threshold
    }

    #[test]
    fn test_measure_self_extension() {
        let mut bench = EmergenceBenchmark::default_benchmark();
        let score = bench.measure_self_extension(3, 2.5, 4);

        assert_eq!(score.samples, 1);
        assert!(score.avg_delta > 0.0); // self-extension always has positive delta
        assert!(score.emerged); // Any self-created tools = emergence
    }

    #[test]
    fn test_measure_scaffolding() {
        let mut bench = EmergenceBenchmark::default_benchmark();
        let score = bench.measure_scaffolding(50, 10);

        assert_eq!(score.samples, 1);
        assert!(score.avg_delta > 0.0); // scaffolding has positive delta
    }

    #[test]
    fn test_measure_planning() {
        let mut bench = EmergenceBenchmark::default_benchmark();
        let score = bench.measure_planning(0.3, 0.7, 3.0);

        assert_eq!(score.samples, 1);
        assert!((score.avg_delta - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_measure_abstraction() {
        let mut bench = EmergenceBenchmark::default_benchmark();
        let score = bench.measure_abstraction(3.0, 2);

        assert_eq!(score.samples, 1);
        assert!(score.avg_delta > 0.0); // augmented > baseline → positive delta
    }

    #[test]
    fn test_measure_from_snapshot() {
        let mut bench = EmergenceBenchmark::default_benchmark();

        let snapshot = SystemSnapshot {
            concept_count: 50,
            episode_count: 100,
            created_tool_count: 5,
            avg_tool_reuse: 2.0,
            max_chain_depth: 3,
            mirror_pass_rate: 0.85,
            error_recurrence_rate: 0.15,
            plan_success_rate: 0.8,
            avg_plan_depth: 3.5,
            compression_ratio: 4.0,
            abstraction_levels: 2,
            mastered_count: 10,
            task_diversity: 8,
            improvement_rate: 0.3,
            baseline_quality: 0.4,
            augmented_quality: 0.75,
        };

        bench.measure_from_snapshot(0.4, &snapshot);

        assert_eq!(bench.total_measurements(), 6); // One per category
        assert_eq!(bench.total_snapshots(), 1);
    }

    #[test]
    fn test_emerged_count() {
        let mut bench = EmergenceBenchmark::default_benchmark();
        assert_eq!(bench.emerged_count(), 0);

        bench.measure_skill_acquisition(0.2, 0.8); // Big delta → emerged
        assert!(bench.emerged_count() >= 1);
    }

    #[test]
    fn test_any_emerged() {
        let mut bench = EmergenceBenchmark::default_benchmark();
        assert!(!bench.any_emerged());

        bench.measure_skill_acquisition(0.2, 0.8);
        assert!(bench.any_emerged());
    }

    #[test]
    fn test_emergence_index() {
        let mut bench = EmergenceBenchmark::default_benchmark();
        assert_eq!(bench.emergence_index(), 0.0);

        // Add several measurements with high confidence
        for _ in 0..5 {
            bench.measure_skill_acquisition(0.2, 0.7);
        }

        assert!(bench.emergence_index() > 0.0);
    }

    #[test]
    fn test_trend_increasing() {
        let mut bench = EmergenceBenchmark::default_benchmark();

        // Increasing deltas
        for i in 0..10 {
            bench.measure_skill_acquisition(0.1, 0.1 + i as f32 * 0.05);
        }

        match bench.trend(EmergenceCategory::SkillAcquisition) {
            EmergenceTrend::Increasing(_) => {} // Expected
            other => panic!("Expected Increasing, got {:?}", other),
        }
    }

    #[test]
    fn test_trend_insufficient_data() {
        let bench = EmergenceBenchmark::default_benchmark();
        match bench.trend(EmergenceCategory::Planning) {
            EmergenceTrend::InsufficientData => {}
            other => panic!("Expected InsufficientData, got {:?}", other),
        }
    }

    #[test]
    fn test_all_scores() {
        let mut bench = EmergenceBenchmark::default_benchmark();
        bench.measure_skill_acquisition(0.2, 0.6);
        bench.measure_self_correction(0.3, 0.7);

        let scores = bench.all_scores();
        assert!(scores.len() >= 2);
    }

    #[test]
    fn test_report() {
        let mut bench = EmergenceBenchmark::default_benchmark();
        bench.measure_skill_acquisition(0.2, 0.7);

        let report = bench.report();
        assert_eq!(report.total_measurements, 1);
        assert!(report.emergence_index > 0.0);
    }

    #[test]
    fn test_stats() {
        let mut bench = EmergenceBenchmark::default_benchmark();
        bench.measure_skill_acquisition(0.2, 0.7);
        bench.measure_self_correction(0.3, 0.8);

        let stats = bench.stats();
        assert_eq!(stats.total_measurements, 2);
    }

    #[test]
    fn test_record_snapshot() {
        let mut bench = EmergenceBenchmark::default_benchmark();
        let snapshot = SystemSnapshot {
            concept_count: 10,
            ..Default::default()
        };
        bench.record_snapshot(snapshot);
        assert_eq!(bench.total_snapshots(), 1);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = std::env::temp_dir().join("ferrisres_emergence_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("emergence.json");

        let mut bench = EmergenceBenchmark::default_benchmark();
        bench.measure_skill_acquisition(0.2, 0.7);
        bench.measure_self_correction(0.3, 0.8);
        bench.measure_from_snapshot(0.3, &SystemSnapshot {
            concept_count: 20,
            created_tool_count: 3,
            ..Default::default()
        });

        bench.save(&path).unwrap();

        let loaded = EmergenceBenchmark::load(&path).unwrap();
        assert_eq!(loaded.total_measurements(), bench.total_measurements());
        assert_eq!(loaded.total_snapshots(), bench.total_snapshots());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_confidence_increases_with_samples() {
        let mut bench = EmergenceBenchmark::default_benchmark();

        bench.measure_skill_acquisition(0.2, 0.6);
        let c1 = bench.score(EmergenceCategory::SkillAcquisition).unwrap().confidence;

        for _ in 0..10 {
            bench.measure_skill_acquisition(0.2, 0.6);
        }
        let c2 = bench.score(EmergenceCategory::SkillAcquisition).unwrap().confidence;

        assert!(c2 > c1);
    }

    #[test]
    fn test_no_emergence_when_no_delta() {
        let mut bench = EmergenceBenchmark::default_benchmark();
        let score = bench.measure_skill_acquisition(0.5, 0.5); // No improvement

        assert!(!score.emerged); // delta = 0, threshold = 0.1
    }

    #[test]
    fn test_snapshot_trim() {
        let config = EmergenceConfig {
            max_history: 5,
            ..Default::default()
        };
        let mut bench = EmergenceBenchmark::new(config);

        for i in 0..10 {
            bench.record_snapshot(SystemSnapshot {
                concept_count: i,
                ..Default::default()
            });
        }

        assert!(bench.total_snapshots() <= 5);
    }
}
