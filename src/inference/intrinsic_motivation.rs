//! Intrinsic Motivation — Self-Directed Learning & Uncertainty Tracking
//!
//! Based on curiosity-driven learning (ICM), competence-based motivation,
//! and AlphaZero self-play. The system estimates its own uncertainty,
//! selects practice goals in the Zone of Proximal Development, and
//! tracks learning progress over time.
//!
//! Pipeline:
//!   1. ESTIMATE UNCERTAINTY per concept (entropy + mirror quality + retrieval distance)
//!   2. SELECT PRACTICE GOAL from ZPD (not too easy, not too hard)
//!   3. EXECUTE practice through cognitive pipeline
//!   4. UPDATE learning progress (Δ quality / Δ time)
//!   5. MARK mastered concepts (quality > 0.95 sustained) → stop practicing

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Uncertainty estimate
// ---------------------------------------------------------------------------

/// Per-concept uncertainty estimate.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UncertaintyEstimate {
    /// Average logit entropy.
    pub entropy_avg: f32,
    /// Average MirrorTest quality.
    pub mirror_test_avg: f32,
    /// Average concept retrieval distance.
    pub retrieval_distance: f32,
    /// Number of samples.
    pub sample_count: usize,
    /// Last updated timestamp.
    pub last_updated: u64,
}

impl UncertaintyEstimate {
    /// Create a new estimate.
    pub fn new() -> Self {
        Self {
            entropy_avg: 0.0,
            mirror_test_avg: 0.0,
            retrieval_distance: 0.0,
            sample_count: 0,
            last_updated: 0,
        }
    }

    /// Update with a new observation.
    pub fn update(&mut self, entropy: f32, mirror_quality: f32, retrieval_distance: f32, timestamp: u64) {
        let alpha = 0.1; // EMA decay
        self.entropy_avg = (1.0 - alpha) * self.entropy_avg + alpha * entropy;
        self.mirror_test_avg = (1.0 - alpha) * self.mirror_test_avg + alpha * mirror_quality;
        self.retrieval_distance = (1.0 - alpha) * self.retrieval_distance + alpha * retrieval_distance;
        self.sample_count += 1;
        self.last_updated = timestamp;
    }

    /// Combined uncertainty score (0 = certain, 1 = very uncertain).
    pub fn uncertainty(&self) -> f32 {
        // High entropy → uncertain
        // Low mirror quality → uncertain
        // High retrieval distance → uncertain
        let entropy_component = (self.entropy_avg / 5.0).min(1.0); // Normalize entropy
        let quality_component = 1.0 - self.mirror_test_avg;
        let distance_component = self.retrieval_distance.min(1.0);

        (entropy_component + quality_component + distance_component) / 3.0
    }
}

// ---------------------------------------------------------------------------
// Goal
// ---------------------------------------------------------------------------

/// A self-generated practice goal.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Goal {
    /// Unique ID.
    pub id: String,
    /// Goal description.
    pub description: String,
    /// Source of the goal.
    pub source: GoalSource,
    /// Priority (0.0–1.0).
    pub priority: f32,
    /// Difficulty level (0.0–1.0).
    pub difficulty: f32,
    /// Concept IDs targeted.
    pub target_concepts: Vec<String>,
    /// Creation timestamp.
    pub created_at: u64,
    /// Deadline (if any).
    pub deadline: Option<u64>,
    /// Whether this goal was completed.
    pub completed: bool,
}

/// Source of a goal.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum GoalSource {
    /// High uncertainty on specific concepts.
    UncertaintyReduction(Vec<String>),
    /// Concept in the Zone of Proximal Development.
    ZpdPractice(String),
    /// Review a stale concept (spaced repetition).
    SpacedRepetition(String),
    /// Self-generated challenge.
    SelfChallenge(String),
    /// Maintenance task.
    Maintenance(String),
}

// ---------------------------------------------------------------------------
// Goal outcome
// ---------------------------------------------------------------------------

/// Result of attempting a goal.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum GoalOutcome {
    /// Successfully completed.
    Success,
    /// Partially completed with quality score.
    PartialSuccess(f32),
    /// Failed to complete.
    Failure,
    /// Abandoned (too hard or interrupted).
    Abandoned,
}

impl GoalOutcome {
    /// Numeric quality (0.0–1.0).
    pub fn quality(&self) -> f32 {
        match self {
            GoalOutcome::Success => 1.0,
            GoalOutcome::PartialSuccess(q) => *q,
            GoalOutcome::Failure => 0.0,
            GoalOutcome::Abandoned => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Goal attempt record
// ---------------------------------------------------------------------------

/// Record of a goal attempt.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GoalAttempt {
    /// The goal that was attempted.
    pub goal_id: String,
    /// Goal description.
    pub goal_description: String,
    /// Difficulty.
    pub difficulty: f32,
    /// Outcome.
    pub outcome: GoalOutcome,
    /// Duration in ms.
    pub duration_ms: u64,
    /// Timestamp.
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// Learning progress
// ---------------------------------------------------------------------------

/// Per-concept learning progress tracker.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LearningProgress {
    /// Concept ID.
    pub concept_id: String,
    /// Quality history (recent samples).
    pub quality_history: Vec<(u64, f32)>,
    /// Learning progress rate (Δ quality / Δ time).
    pub progress_rate: f32,
    /// Whether this concept is mastered.
    pub mastered: bool,
    /// Consecutive successes above mastery threshold.
    pub consecutive_successes: usize,
}

impl LearningProgress {
    fn new(concept_id: String) -> Self {
        Self {
            concept_id,
            quality_history: Vec::new(),
            progress_rate: 0.0,
            mastered: false,
            consecutive_successes: 0,
        }
    }

    /// Record a quality sample.
    fn record(&mut self, timestamp: u64, quality: f32, mastery_threshold: f32) {
        self.quality_history.push((timestamp, quality));

        // Keep only last 20 samples
        if self.quality_history.len() > 20 {
            self.quality_history.remove(0);
        }

        // Update progress rate
        if self.quality_history.len() >= 2 {
            let first = self.quality_history.first().unwrap();
            let last = self.quality_history.last().unwrap();
            let dt = (last.0 - first.0) as f32;
            if dt > 0.0 {
                self.progress_rate = (last.1 - first.1) / dt;
            }
        }

        // Update mastery tracking
        if quality >= mastery_threshold {
            self.consecutive_successes += 1;
        } else {
            self.consecutive_successes = 0;
        }
        self.mastered = self.consecutive_successes >= 5;
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for intrinsic motivation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntrinsicMotivationConfig {
    /// Zone of Proximal Development: lower bound (below = too easy/boring).
    pub zpd_low: f32,
    /// Zone of Proximal Development: upper bound (above = too hard/frustrating).
    pub zpd_high: f32,
    /// Quality threshold for mastery (boredom signal).
    pub mastery_threshold: f32,
    /// Consecutive successes needed for mastery.
    pub mastery_streak: usize,
    /// Maximum active goals.
    pub max_active_goals: usize,
    /// Spaced repetition interval (seconds).
    pub spaced_repetition_interval: u64,
    /// Minimum uncertainty to generate a practice goal.
    pub min_uncertainty_for_goal: f32,
}

impl Default for IntrinsicMotivationConfig {
    fn default() -> Self {
        Self {
            zpd_low: 0.3,
            zpd_high: 0.8,
            mastery_threshold: 0.95,
            mastery_streak: 5,
            max_active_goals: 10,
            spaced_repetition_interval: 3600, // 1 hour
            min_uncertainty_for_goal: 0.4,
        }
    }
}

// ---------------------------------------------------------------------------
// IntrinsicMotivation
// ---------------------------------------------------------------------------

/// Self-directed learning through uncertainty tracking and practice goal generation.
pub struct IntrinsicMotivation {
    config: IntrinsicMotivationConfig,
    /// Per-concept uncertainty.
    uncertainties: HashMap<String, UncertaintyEstimate>,
    /// Per-concept learning progress.
    learning_progress: HashMap<String, LearningProgress>,
    /// Mastered concept IDs.
    mastered: HashSet<String>,
    /// Active goals.
    active_goals: Vec<Goal>,
    /// Goal attempt history.
    goal_history: Vec<GoalAttempt>,
    /// Goal counter for unique IDs.
    next_goal_id: u64,
}

impl IntrinsicMotivation {
    /// Create a new intrinsic motivation module.
    pub fn new(config: IntrinsicMotivationConfig) -> Self {
        Self {
            config,
            uncertainties: HashMap::new(),
            learning_progress: HashMap::new(),
            mastered: HashSet::new(),
            active_goals: Vec::new(),
            goal_history: Vec::new(),
            next_goal_id: 0,
        }
    }

    /// Create with default configuration.
    pub fn default_motivation() -> Self {
        Self::new(IntrinsicMotivationConfig::default())
    }

    /// Get configuration.
    pub fn config(&self) -> &IntrinsicMotivationConfig {
        &self.config
    }

    /// Number of tracked concepts.
    pub fn tracked_concepts(&self) -> usize {
        self.uncertainties.len()
    }

    /// Number of mastered concepts.
    pub fn mastered_count(&self) -> usize {
        self.mastered.len()
    }

    /// Number of active goals.
    pub fn active_goal_count(&self) -> usize {
        self.active_goals.len()
    }

    /// Number of goal attempts.
    pub fn goal_attempt_count(&self) -> usize {
        self.goal_history.len()
    }

    // ---- Main API ----

    /// Record an observation for a concept.
    pub fn record_observation(
        &mut self,
        concept_id: &str,
        entropy: f32,
        mirror_quality: f32,
        retrieval_distance: f32,
    ) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        self.uncertainties
            .entry(concept_id.to_string())
            .or_insert_with(UncertaintyEstimate::new)
            .update(entropy, mirror_quality, retrieval_distance, timestamp);

        // Update learning progress
        self.learning_progress
            .entry(concept_id.to_string())
            .or_insert_with(|| LearningProgress::new(concept_id.to_string()))
            .record(timestamp, mirror_quality, self.config.mastery_threshold);

        // Check mastery
        if let Some(progress) = self.learning_progress.get(concept_id) {
            if progress.mastered {
                self.mastered.insert(concept_id.to_string());
            } else {
                self.mastered.remove(concept_id);
            }
        }
    }

    /// Get uncertainty for a concept.
    pub fn get_uncertainty(&self, concept_id: &str) -> Option<&UncertaintyEstimate> {
        self.uncertainties.get(concept_id)
    }

    /// Get learning progress for a concept.
    pub fn get_progress(&self, concept_id: &str) -> Option<&LearningProgress> {
        self.learning_progress.get(concept_id)
    }

    /// Check if a concept is mastered.
    pub fn is_mastered(&self, concept_id: &str) -> bool {
        self.mastered.contains(concept_id)
    }

    /// Find concepts in the Zone of Proximal Development.
    pub fn zpd_concepts(&self) -> Vec<(String, f32)> {
        let mut in_zpd: Vec<(String, f32)> = self.uncertainties.iter()
            .filter(|(id, _)| !self.mastered.contains(*id))
            .filter_map(|(id, est)| {
                let u = est.uncertainty();
                if u >= self.config.zpd_low && u <= self.config.zpd_high {
                    Some((id.clone(), u))
                } else {
                    None
                }
            })
            .collect();

        // Sort by learning progress (highest first — most room for improvement)
        in_zpd.sort_by(|a, b| {
            let pa = self.learning_progress.get(&a.0).map(|p| p.progress_rate).unwrap_or(0.0);
            let pb = self.learning_progress.get(&b.0).map(|p| p.progress_rate).unwrap_or(0.0);
            pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
        });

        in_zpd
    }

    /// Generate a practice goal.
    pub fn generate_goal(&mut self) -> Option<Goal> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        if self.active_goals.len() >= self.config.max_active_goals {
            return None;
        }

        // Priority 1: ZPD concepts with highest learning progress
        let zpd = self.zpd_concepts();
        if let Some((concept_id, uncertainty)) = zpd.first() {
            let goal = Goal {
                id: format!("goal_{}", self.next_goal_id),
                description: format!("Practice concept: {}", concept_id),
                source: GoalSource::ZpdPractice(concept_id.to_string()),
                priority: 0.8,
                difficulty: *uncertainty,
                target_concepts: vec![concept_id.to_string()],
                created_at: timestamp,
                deadline: None,
                completed: false,
            };
            self.next_goal_id += 1;
            self.active_goals.push(goal.clone());
            return Some(goal);
        }

        // Priority 2: High uncertainty concepts
        let high_uncertainty: Vec<_> = self.uncertainties.iter()
            .filter(|(id, _)| !self.mastered.contains(*id))
            .filter(|(_, est)| est.uncertainty() >= self.config.min_uncertainty_for_goal)
            .collect();

        if let Some((concept_id, est)) = high_uncertainty.first() {
            let goal = Goal {
                id: format!("goal_{}", self.next_goal_id),
                description: format!("Reduce uncertainty on: {}", concept_id),
                source: GoalSource::UncertaintyReduction(vec![concept_id.to_string()]),
                priority: 0.6,
                difficulty: est.uncertainty(),
                target_concepts: vec![concept_id.to_string()],
                created_at: timestamp,
                deadline: None,
                completed: false,
            };
            self.next_goal_id += 1;
            self.active_goals.push(goal.clone());
            return Some(goal);
        }

        None
    }

    /// Record the outcome of a goal attempt.
    pub fn record_goal_outcome(
        &mut self,
        goal_id: &str,
        outcome: GoalOutcome,
        duration_ms: u64,
    ) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Find the goal
        if let Some(goal) = self.active_goals.iter_mut().find(|g| g.id == goal_id) {
            goal.completed = matches!(outcome, GoalOutcome::Success);

            let attempt = GoalAttempt {
                goal_id: goal_id.to_string(),
                goal_description: goal.description.clone(),
                difficulty: goal.difficulty,
                outcome: outcome.clone(),
                duration_ms,
                timestamp,
            };
            self.goal_history.push(attempt);

            // Update uncertainty based on outcome
            for concept_id in &goal.target_concepts {
                let quality = outcome.quality();
                // Successful practice reduces uncertainty
                if let Some(est) = self.uncertainties.get_mut(concept_id) {
                    let new_entropy = est.entropy_avg * (1.0 - quality * 0.1);
                    est.update(new_entropy, quality, est.retrieval_distance, timestamp);
                }
                // Update learning progress
                self.learning_progress
                    .entry(concept_id.clone())
                    .or_insert_with(|| LearningProgress::new(concept_id.clone()))
                    .record(timestamp, quality, self.config.mastery_threshold);

                if let Some(progress) = self.learning_progress.get(concept_id) {
                    if progress.mastered {
                        self.mastered.insert(concept_id.clone());
                    }
                }
            }
        }
    }

    /// Get practice candidates: concepts that would benefit from practice.
    pub fn practice_candidates(&self) -> Vec<(String, f32)> {
        self.zpd_concepts()
    }

    /// Get concepts with highest uncertainty.
    pub fn most_uncertain(&self, top_k: usize) -> Vec<(String, f32)> {
        let mut results: Vec<_> = self.uncertainties.iter()
            .filter(|(id, _)| !self.mastered.contains(*id))
            .map(|(id, est)| (id.clone(), est.uncertainty()))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    // ---- Stats ----

    /// Get motivation statistics.
    pub fn stats(&self) -> MotivationStats {
        let avg_uncertainty = if self.uncertainties.is_empty() {
            0.0
        } else {
            self.uncertainties.values()
                .map(|e| e.uncertainty())
                .sum::<f32>() / self.uncertainties.len() as f32
        };

        let success_rate = if self.goal_history.is_empty() {
            0.0
        } else {
            self.goal_history.iter()
                .filter(|a| matches!(a.outcome, GoalOutcome::Success))
                .count() as f32 / self.goal_history.len() as f32
        };

        MotivationStats {
            tracked_concepts: self.uncertainties.len(),
            mastered_concepts: self.mastered.len(),
            active_goals: self.active_goals.len(),
            total_goal_attempts: self.goal_history.len(),
            avg_uncertainty,
            goal_success_rate: success_rate,
        }
    }
}

/// Motivation statistics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MotivationStats {
    pub tracked_concepts: usize,
    pub mastered_concepts: usize,
    pub active_goals: usize,
    pub total_goal_attempts: usize,
    pub avg_uncertainty: f32,
    pub goal_success_rate: f32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertainty_estimate() {
        let mut est = UncertaintyEstimate::new();
        assert_eq!(est.sample_count, 0);

        est.update(2.0, 0.8, 0.3, 100);
        assert_eq!(est.sample_count, 1);

        est.update(1.5, 0.9, 0.2, 200);
        assert_eq!(est.sample_count, 2);

        // Uncertainty should be moderate
        let u = est.uncertainty();
        assert!(u > 0.0 && u < 1.0);
    }

    #[test]
    fn test_uncertainty_high_entropy() {
        let mut est = UncertaintyEstimate::new();
        // Multiple updates to accumulate high uncertainty
        for _ in 0..20 {
            est.update(10.0, 0.1, 0.9, 100);
        }
        assert!(est.uncertainty() > 0.5);
    }

    #[test]
    fn test_uncertainty_low_entropy() {
        let mut est = UncertaintyEstimate::new();
        // Multiple updates to accumulate low uncertainty
        for _ in 0..20 {
            est.update(0.1, 0.99, 0.05, 100);
        }
        assert!(est.uncertainty() < 0.3);
    }

    #[test]
    fn test_record_observation() {
        let mut im = IntrinsicMotivation::default_motivation();
        im.record_observation("concept_a", 2.0, 0.7, 0.3);
        assert_eq!(im.tracked_concepts(), 1);
        assert!(im.get_uncertainty("concept_a").is_some());
    }

    #[test]
    fn test_zpd_concepts() {
        let mut im = IntrinsicMotivation::default_motivation();

        // In ZPD (uncertainty ~0.5)
        im.record_observation("in_zpd", 3.0, 0.5, 0.5);

        // Too easy (uncertainty < 0.3)
        for _ in 0..5 {
            im.record_observation("too_easy", 0.1, 0.99, 0.05);
        }

        // Too hard (uncertainty > 0.8)
        im.record_observation("too_hard", 10.0, 0.1, 0.9);

        let zpd = im.zpd_concepts();
        assert!(!zpd.is_empty());
        // "in_zpd" should be in the ZPD
        assert!(zpd.iter().any(|(id, _)| id == "in_zpd"));
    }

    #[test]
    fn test_mastery() {
        let mut im = IntrinsicMotivation::default_motivation();

        // Record 5 high-quality observations
        for _ in 0..6 {
            im.record_observation("mastered_concept", 0.1, 0.97, 0.05);
        }

        assert!(im.is_mastered("mastered_concept"));
        assert_eq!(im.mastered_count(), 1);
    }

    #[test]
    fn test_mastery_broken_streak() {
        let mut im = IntrinsicMotivation::default_motivation();

        for _ in 0..4 {
            im.record_observation("streak", 0.1, 0.97, 0.05);
        }
        // Break the streak
        im.record_observation("streak", 0.1, 0.5, 0.05);
        assert!(!im.is_mastered("streak"));
    }

    #[test]
    fn test_generate_goal() {
        let mut im = IntrinsicMotivation::default_motivation();
        im.record_observation("uncertain_concept", 3.0, 0.5, 0.5);

        let goal = im.generate_goal();
        assert!(goal.is_some());
        assert_eq!(im.active_goal_count(), 1);
    }

    #[test]
    fn test_generate_goal_no_candidates() {
        let mut im = IntrinsicMotivation::default_motivation();
        let goal = im.generate_goal();
        assert!(goal.is_none());
    }

    #[test]
    fn test_generate_goal_max_active() {
        let config = IntrinsicMotivationConfig {
            max_active_goals: 2,
            ..Default::default()
        };
        let mut im = IntrinsicMotivation::new(config);

        for i in 0..5 {
            im.record_observation(&format!("c{}", i), 3.0, 0.5, 0.5);
        }

        let g1 = im.generate_goal();
        let g2 = im.generate_goal();
        let g3 = im.generate_goal(); // Should be None

        assert!(g1.is_some());
        assert!(g2.is_some());
        assert!(g3.is_none());
    }

    #[test]
    fn test_record_goal_outcome() {
        let mut im = IntrinsicMotivation::default_motivation();
        im.record_observation("target", 3.0, 0.5, 0.5);

        let goal = im.generate_goal().unwrap();
        im.record_goal_outcome(&goal.id, GoalOutcome::Success, 100);

        assert_eq!(im.goal_attempt_count(), 1);
    }

    #[test]
    fn test_goal_outcome_quality() {
        assert_eq!(GoalOutcome::Success.quality(), 1.0);
        assert_eq!(GoalOutcome::PartialSuccess(0.6).quality(), 0.6);
        assert_eq!(GoalOutcome::Failure.quality(), 0.0);
        assert_eq!(GoalOutcome::Abandoned.quality(), 0.0);
    }

    #[test]
    fn test_most_uncertain() {
        let mut im = IntrinsicMotivation::default_motivation();
        im.record_observation("certain", 0.1, 0.99, 0.05);
        im.record_observation("uncertain", 10.0, 0.1, 0.9);
        im.record_observation("moderate", 3.0, 0.5, 0.5);

        let top = im.most_uncertain(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "uncertain");
    }

    #[test]
    fn test_practice_candidates() {
        let mut im = IntrinsicMotivation::default_motivation();
        im.record_observation("zpd_concept", 3.0, 0.5, 0.5);

        let candidates = im.practice_candidates();
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_stats() {
        let mut im = IntrinsicMotivation::default_motivation();
        im.record_observation("concept1", 2.0, 0.7, 0.3);
        im.record_observation("concept2", 1.0, 0.9, 0.1);

        let stats = im.stats();
        assert_eq!(stats.tracked_concepts, 2);
        assert_eq!(stats.mastered_concepts, 0);
        assert!(stats.avg_uncertainty > 0.0);
    }

    #[test]
    fn test_learning_progress_rate() {
        let mut im = IntrinsicMotivation::default_motivation();

        // Quality improving over time
        for i in 0..5 {
            im.record_observation("improving", 2.0, 0.5 + i as f32 * 0.1, 0.3);
        }

        let progress = im.get_progress("improving").unwrap();
        assert!(progress.progress_rate >= 0.0); // Improving
    }

    #[test]
    fn test_mastered_concepts_not_in_zpd() {
        let mut im = IntrinsicMotivation::default_motivation();

        // Master a concept
        for _ in 0..6 {
            im.record_observation("mastered", 0.1, 0.97, 0.05);
        }
        assert!(im.is_mastered("mastered"));

        // Should not appear in ZPD
        let zpd = im.zpd_concepts();
        assert!(!zpd.iter().any(|(id, _)| id == "mastered"));
    }
}
