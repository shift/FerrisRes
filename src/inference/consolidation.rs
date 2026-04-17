//! Consolidation Engine — Sleep-like Memory Replay for Learning Reinforcement
//!
//! Inspired by sleep consolidation in biological systems (McClelland et al. 1995):
//! important memories are replayed during "idle time" to strengthen learning
//! and transfer episodic memories into longer-term conceptual knowledge.
//!
//! Pipeline:
//!   1. SELECT important episodes (high importance, not yet consolidated)
//!   2. REPLAY them — score by quality, categorize by outcome
//!   3. STRENGTHEN concepts that were reinforced by replay
//!   4. FORM new concepts from episode clusters (pattern extraction)
//!   5. MARK episodes as consolidated
//!   6. PRUNE low-importance consolidated episodes to free capacity
//!
//! Triggers:
//!   - Capacity pressure (episodic memory > 80% full)
//!   - Time-based (configurable interval)
//!   - Quality-based (after a particularly bad generation)

use std::collections::HashMap;

use crate::inference::episodic_memory::EpisodeOutcome;

// ---------------------------------------------------------------------------
// Consolidation result
// ---------------------------------------------------------------------------

/// Result of a consolidation pass.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConsolidationResult {
    /// Number of episodes replayed.
    pub episodes_replayed: usize,
    /// Number of concepts strengthened.
    pub concepts_strengthened: usize,
    /// Number of new concepts formed from replay.
    pub concepts_formed: usize,
    /// Number of episodes marked consolidated.
    pub episodes_consolidated: usize,
    /// Number of low-importance episodes pruned.
    pub episodes_pruned: usize,
    /// Average quality of replayed episodes.
    pub avg_replay_quality: f32,
    /// Duration in ms.
    pub duration_ms: u64,
    /// Timestamp.
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// Consolidation policy
// ---------------------------------------------------------------------------

/// Policy for selecting which episodes to consolidate.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConsolidationPolicy {
    /// Maximum episodes to replay per consolidation pass.
    pub max_replay: usize,
    /// Minimum importance score to select for replay.
    pub min_importance: f32,
    /// Quality threshold for considering a replay "strengthening".
    pub strengthen_threshold: f32,
    /// Whether to prune consolidated episodes with low importance.
    pub prune_consolidated: bool,
    /// Importance below which consolidated episodes are pruned.
    pub prune_importance_threshold: f32,
    /// Whether to attempt concept formation from replay patterns.
    pub form_concepts: bool,
    /// Minimum similar episodes to form a new concept.
    pub min_similar_for_concept: usize,
    /// Similarity threshold for grouping episodes into a concept.
    pub concept_similarity_threshold: f32,
    /// Never prune episodes with quality above this threshold (preserve exemplars).
    pub preserve_quality_threshold: f32,
}

impl Default for ConsolidationPolicy {
    fn default() -> Self {
        Self {
            max_replay: 20,
            min_importance: 0.3,
            strengthen_threshold: 0.6,
            prune_consolidated: true,
            prune_importance_threshold: 0.1,
            form_concepts: true,
            min_similar_for_concept: 3,
            concept_similarity_threshold: 0.8,
            preserve_quality_threshold: 0.9,
        }
    }
}

// ---------------------------------------------------------------------------
// Episode summary (input from EpisodicMemory)
// ---------------------------------------------------------------------------

/// Lightweight episode view for consolidation selection.
#[derive(Debug, Clone)]
pub struct EpisodeSummary {
    /// Episode ID.
    pub id: u64,
    /// Importance score (0.0–1.0).
    pub importance: f32,
    /// MirrorTest quality score (0.0–1.0).
    pub quality_score: f32,
    /// Outcome classification.
    pub outcome: EpisodeOutcome,
    /// Tags for categorical grouping.
    pub tags: Vec<String>,
    /// Embedding vector.
    pub embedding: Vec<f32>,
    /// Prompt text.
    pub prompt: String,
    /// Output text.
    pub output: String,
    /// Whether already consolidated.
    pub consolidated: bool,
    /// Timestamp.
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// Conceptual learning (output)
// ---------------------------------------------------------------------------

/// A concept formed during consolidation from recurring episode patterns.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FormedConcept {
    /// Description derived from episode prompts.
    pub description: String,
    /// Tags common to all source episodes.
    pub tags: Vec<String>,
    /// Centroid embedding of source episodes.
    pub embedding: Vec<f32>,
    /// Number of episodes that formed this concept.
    pub source_count: usize,
    /// Average quality of source episodes.
    pub avg_quality: f32,
}

// ---------------------------------------------------------------------------
// ConsolidationEngine
// ---------------------------------------------------------------------------

/// Performs sleep-like consolidation: replay important episodes,
/// strengthen concepts, form new abstractions, prune stale memories.
pub struct ConsolidationEngine {
    policy: ConsolidationPolicy,
    /// History of consolidation passes.
    history: Vec<ConsolidationResult>,
    /// Accumulated formed concepts awaiting registration.
    pending_concepts: Vec<FormedConcept>,
}

impl ConsolidationEngine {
    /// Create with a specific policy.
    pub fn new(policy: ConsolidationPolicy) -> Self {
        Self {
            policy,
            history: Vec::new(),
            pending_concepts: Vec::new(),
        }
    }

    /// Create with default policy.
    pub fn default_engine() -> Self {
        Self::new(ConsolidationPolicy::default())
    }

    /// Get policy reference.
    pub fn policy(&self) -> &ConsolidationPolicy {
        &self.policy
    }

    /// Number of consolidation passes performed.
    pub fn pass_count(&self) -> usize {
        self.history.len()
    }

    /// Pending concepts not yet registered.
    pub fn pending_concept_count(&self) -> usize {
        self.pending_concepts.len()
    }

    /// Drain pending concepts (caller registers them in ConceptMap).
    pub fn drain_pending_concepts(&mut self) -> Vec<FormedConcept> {
        std::mem::take(&mut self.pending_concepts)
    }

    /// Consolidation history.
    pub fn history(&self) -> &[ConsolidationResult] {
        &self.history
    }

    // ---- Main API ----

    /// Run a consolidation pass over the given episodes.
    ///
    /// Returns the consolidation result with stats about what was done.
    /// The caller is responsible for:
    ///   - Marking selected episodes as consolidated in EpisodicMemory
    ///   - Registering pending concepts in ConceptMap
    ///   - Removing pruned episodes from EpisodicMemory
    ///   - Persisting state
    pub fn consolidate(&mut self, episodes: &[EpisodeSummary]) -> ConsolidationResult {
        let start = std::time::Instant::now();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Step 1: Select episodes for replay
        let candidates: Vec<&EpisodeSummary> = episodes.iter()
            .filter(|e| !e.consolidated)
            .filter(|e| e.importance >= self.policy.min_importance)
            .collect();

        // Sort by importance (descending)
        let mut sorted_candidates: Vec<&EpisodeSummary> = candidates;
        sorted_candidates.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));

        let mut episodes_replayed = 0;
        let mut concepts_strengthened = 0;
        let mut total_quality = 0.0f32;

        // Step 2: Simulate replay — score and categorize
        for episode in sorted_candidates.iter().take(self.policy.max_replay) {
            episodes_replayed += 1;
            total_quality += episode.quality_score;

            // High-quality replays strengthen concepts
            if episode.quality_score >= self.policy.strengthen_threshold {
                concepts_strengthened += 1;
            }
        }

        let avg_replay_quality = if episodes_replayed > 0 {
            total_quality / episodes_replayed as f32
        } else {
            0.0
        };

        // Episodes consolidated = all replayed
        let episodes_consolidated = episodes_replayed;

        // Step 3: Prune low-importance consolidated episodes
        let episodes_pruned = if self.policy.prune_consolidated {
            episodes.iter()
                .filter(|e| e.consolidated)
                .filter(|e| e.importance < self.policy.prune_importance_threshold)
                .filter(|e| e.quality_score < self.policy.preserve_quality_threshold)
                .count()
        } else {
            0
        };

        // Step 4: Attempt concept formation from patterns
        let concepts_formed = if self.policy.form_concepts {
            self.form_concepts_from_episodes(sorted_candidates.iter().copied())
        } else {
            0
        };

        let result = ConsolidationResult {
            episodes_replayed,
            concepts_strengthened,
            concepts_formed,
            episodes_consolidated,
            episodes_pruned,
            avg_replay_quality,
            duration_ms: start.elapsed().as_millis() as u64,
            timestamp,
        };

        self.history.push(result.clone());
        result
    }

    /// Select which episode IDs should be marked consolidated.
    pub fn select_for_consolidation(&self, episodes: &[EpisodeSummary]) -> Vec<u64> {
        let mut candidates: Vec<&EpisodeSummary> = episodes.iter()
            .filter(|e| !e.consolidated)
            .filter(|e| e.importance >= self.policy.min_importance)
            .collect();
        candidates.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));
        candidates.iter()
            .take(self.policy.max_replay)
            .map(|e| e.id)
            .collect()
    }

    /// Select which episode IDs should be pruned.
    pub fn select_for_pruning(&self, episodes: &[EpisodeSummary]) -> Vec<u64> {
        if !self.policy.prune_consolidated {
            return vec![];
        }

        episodes.iter()
            .filter(|e| e.consolidated)
            .filter(|e| e.importance < self.policy.prune_importance_threshold)
            .filter(|e| e.quality_score < self.policy.preserve_quality_threshold)
            .map(|e| e.id)
            .collect()
    }

    /// Attempt to form new concepts from episode clusters.
    fn form_concepts_from_episodes<'a>(
        &mut self,
        episodes: impl Iterator<Item = &'a EpisodeSummary>,
    ) -> usize {
        let episodes: Vec<&EpisodeSummary> = episodes.collect();
        let mut clusters: Vec<Vec<&EpisodeSummary>> = Vec::new();
        let mut assigned: Vec<bool> = vec![false; episodes.len()];

        // Simple greedy clustering by cosine similarity
        for i in 0..episodes.len() {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![episodes[i]];
            assigned[i] = true;

            for j in (i + 1)..episodes.len() {
                if assigned[j] {
                    continue;
                }

                let sim = cosine_similarity(&episodes[i].embedding, &episodes[j].embedding);
                if sim >= self.policy.concept_similarity_threshold {
                    cluster.push(episodes[j]);
                    assigned[j] = true;
                }
            }

            if cluster.len() >= self.policy.min_similar_for_concept {
                clusters.push(cluster);
            }
        }

        // Form concepts from clusters
        let mut formed = 0;
        for cluster in &clusters {
            // Find common tags (present in >= half of episodes)
            let mut tag_counts: HashMap<String, usize> = HashMap::new();
            for ep in cluster {
                for tag in &ep.tags {
                    *tag_counts.entry(tag.clone()).or_insert(0) += 1;
                }
            }
            let common_tags: Vec<String> = tag_counts.iter()
                .filter(|(_, &count)| count >= cluster.len() / 2)
                .map(|(tag, _)| tag.clone())
                .collect();

            // Compute centroid embedding
            let dim = cluster[0].embedding.len();
            let mut centroid = vec![0.0f32; dim];
            for ep in cluster {
                for (i, &v) in ep.embedding.iter().enumerate() {
                    centroid[i] += v;
                }
            }
            for v in centroid.iter_mut() {
                *v /= cluster.len() as f32;
            }

            let avg_quality: f32 = cluster.iter().map(|e| e.quality_score).sum::<f32>() / cluster.len() as f32;
            let prompt_summary: String = cluster.iter()
                .take(3)
                .map(|e| e.prompt.chars().take(60).collect::<String>())
                .collect::<Vec<_>>()
                .join("; ");

            self.pending_concepts.push(FormedConcept {
                description: format!(
                    "Pattern from {} episodes: {}",
                    cluster.len(),
                    prompt_summary
                ),
                tags: common_tags,
                embedding: centroid,
                source_count: cluster.len(),
                avg_quality,
            });
            formed += 1;
        }

        formed
    }

    /// Get consolidation statistics.
    pub fn stats(&self) -> ConsolidationStats {
        let total_replayed: usize = self.history.iter().map(|r| r.episodes_replayed).sum();
        let total_strengthened: usize = self.history.iter().map(|r| r.concepts_strengthened).sum();
        let total_formed: usize = self.history.iter().map(|r| r.concepts_formed).sum();
        let total_pruned: usize = self.history.iter().map(|r| r.episodes_pruned).sum();

        ConsolidationStats {
            total_passes: self.history.len(),
            total_episodes_replayed: total_replayed,
            total_concepts_strengthened: total_strengthened,
            total_concepts_formed: total_formed,
            total_episodes_pruned: total_pruned,
            pending_concepts: self.pending_concepts.len(),
        }
    }
}

/// Consolidation statistics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConsolidationStats {
    pub total_passes: usize,
    pub total_episodes_replayed: usize,
    pub total_concepts_strengthened: usize,
    pub total_concepts_formed: usize,
    pub total_episodes_pruned: usize,
    pub pending_concepts: usize,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_episode(id: u64, importance: f32, quality: f32, consolidated: bool) -> EpisodeSummary {
        EpisodeSummary {
            id,
            importance,
            quality_score: quality,
            outcome: if quality >= 0.7 { EpisodeOutcome::Success } else { EpisodeOutcome::Failure },
            tags: vec!["test".into()],
            embedding: vec![1.0, 0.0, 0.0],
            prompt: format!("prompt {}", id),
            output: format!("output {}", id),
            consolidated,
            timestamp: 100,
        }
    }

    fn make_episode_with_embedding(id: u64, embedding: Vec<f32>, tags: Vec<&str>) -> EpisodeSummary {
        EpisodeSummary {
            id,
            importance: 0.8,
            quality_score: 0.9,
            outcome: EpisodeOutcome::Success,
            tags: tags.iter().map(|t| t.to_string()).collect(),
            embedding,
            prompt: format!("prompt {}", id),
            output: format!("output {}", id),
            consolidated: false,
            timestamp: 100,
        }
    }

    #[test]
    fn test_consolidate_selects_important_unconsolidated() {
        let mut engine = ConsolidationEngine::default_engine();
        let episodes = vec![
            make_episode(1, 0.8, 0.9, false),
            make_episode(2, 0.1, 0.2, false),  // Too low importance
            make_episode(3, 0.7, 0.8, true),   // Already consolidated
            make_episode(4, 0.6, 0.7, false),
        ];

        let result = engine.consolidate(&episodes);
        // Episodes 1 and 4 are eligible (not consolidated, importance >= 0.3)
        assert!(result.episodes_replayed >= 2);
        assert!(result.episodes_consolidated >= 2);
        assert_eq!(engine.pass_count(), 1);
    }

    #[test]
    fn test_consolidate_empty() {
        let mut engine = ConsolidationEngine::default_engine();
        let result = engine.consolidate(&[]);
        assert_eq!(result.episodes_replayed, 0);
        assert_eq!(result.avg_replay_quality, 0.0);
    }

    #[test]
    fn test_consolidate_respects_max_replay() {
        let policy = ConsolidationPolicy { max_replay: 2, ..Default::default() };
        let mut engine = ConsolidationEngine::new(policy);
        let episodes: Vec<EpisodeSummary> = (0..10)
            .map(|i| make_episode(i, 0.8, 0.9, false))
            .collect();

        let result = engine.consolidate(&episodes);
        assert!(result.episodes_replayed <= 2);
    }

    #[test]
    fn test_select_for_consolidation() {
        let engine = ConsolidationEngine::default_engine();
        let episodes = vec![
            make_episode(1, 0.8, 0.9, false),
            make_episode(2, 0.1, 0.2, false),
            make_episode(3, 0.5, 0.6, false),
        ];

        let selected = engine.select_for_consolidation(&episodes);
        assert!(selected.contains(&1));
        assert!(selected.contains(&3));
        assert!(!selected.contains(&2));
    }

    #[test]
    fn test_select_for_pruning() {
        let engine = ConsolidationEngine::default_engine();
        let episodes = vec![
            make_episode(1, 0.05, 0.1, true),
            make_episode(2, 0.8, 0.9, true),
            make_episode(3, 0.02, 0.1, true),
            make_episode(4, 0.5, 0.6, false),
        ];

        let pruned = engine.select_for_pruning(&episodes);
        assert!(pruned.contains(&1));
        assert!(pruned.contains(&3));
        assert!(!pruned.contains(&2));
        assert!(!pruned.contains(&4));
    }

    #[test]
    fn test_select_for_pruning_disabled() {
        let policy = ConsolidationPolicy { prune_consolidated: false, ..Default::default() };
        let engine = ConsolidationEngine::new(policy);
        let episodes = vec![make_episode(1, 0.01, 0.1, true)];

        let pruned = engine.select_for_pruning(&episodes);
        assert!(pruned.is_empty());
    }

    #[test]
    fn test_concept_formation() {
        let policy = ConsolidationPolicy {
            form_concepts: true,
            min_similar_for_concept: 2,
            concept_similarity_threshold: 0.9,
            ..Default::default()
        };
        let mut engine = ConsolidationEngine::new(policy);

        let episodes = vec![
            make_episode_with_embedding(1, vec![1.0, 0.0], vec!["sort", "array"]),
            make_episode_with_embedding(2, vec![0.95, 0.05], vec!["sort", "array"]),
            make_episode_with_embedding(3, vec![0.0, 1.0], vec!["search"]), // Different cluster
        ];

        let result = engine.consolidate(&episodes);
        assert!(result.concepts_formed >= 1, "Should form at least 1 concept, got {}", result.concepts_formed);
        assert!(engine.pending_concept_count() >= 1);

        let concepts = engine.drain_pending_concepts();
        let sort_concept = concepts.iter().find(|c| c.tags.contains(&"sort".to_string())).unwrap();
        assert!(sort_concept.source_count >= 2);
        assert!(sort_concept.description.contains("episodes"));
    }

    #[test]
    fn test_concept_formation_disabled() {
        let policy = ConsolidationPolicy { form_concepts: false, ..Default::default() };
        let mut engine = ConsolidationEngine::new(policy);
        let episodes = vec![
            make_episode_with_embedding(1, vec![1.0, 0.0], vec!["test"]),
            make_episode_with_embedding(2, vec![0.95, 0.05], vec!["test"]),
        ];

        let result = engine.consolidate(&episodes);
        assert_eq!(result.concepts_formed, 0);
    }

    #[test]
    fn test_strengthen_on_high_quality() {
        let mut engine = ConsolidationEngine::default_engine();
        let episodes = vec![
            make_episode(1, 0.8, 0.9, false), // High quality → strengthen
            make_episode(2, 0.7, 0.3, false), // Low quality → no strengthen
        ];

        let result = engine.consolidate(&episodes);
        assert!(result.concepts_strengthened >= 1);
    }

    #[test]
    fn test_drain_pending_concepts() {
        let policy = ConsolidationPolicy {
            form_concepts: true,
            min_similar_for_concept: 2,
            concept_similarity_threshold: 0.9,
            ..Default::default()
        };
        let mut engine = ConsolidationEngine::new(policy);

        let episodes = vec![
            make_episode_with_embedding(1, vec![1.0], vec!["a"]),
            make_episode_with_embedding(2, vec![0.99], vec!["a"]),
        ];
        engine.consolidate(&episodes);

        let concepts = engine.drain_pending_concepts();
        assert!(!concepts.is_empty());
        assert_eq!(engine.pending_concept_count(), 0);
    }

    #[test]
    fn test_stats() {
        let mut engine = ConsolidationEngine::default_engine();
        engine.consolidate(&[make_episode(1, 0.8, 0.9, false)]);
        engine.consolidate(&[make_episode(2, 0.7, 0.8, false)]);

        let stats = engine.stats();
        assert_eq!(stats.total_passes, 2);
        assert!(stats.total_episodes_replayed >= 2);
    }

    #[test]
    fn test_history_accumulates() {
        let mut engine = ConsolidationEngine::default_engine();
        engine.consolidate(&[make_episode(1, 0.8, 0.9, false)]);
        engine.consolidate(&[make_episode(2, 0.7, 0.8, false)]);

        assert_eq!(engine.history().len(), 2);
    }

    #[test]
    fn test_duration_recorded() {
        let mut engine = ConsolidationEngine::default_engine();
        let result = engine.consolidate(&[make_episode(1, 0.8, 0.9, false)]);
        assert!(result.duration_ms < 1000);
    }

    #[test]
    fn test_prune_low_importance_consolidated() {
        let mut engine = ConsolidationEngine::default_engine();
        let episodes = vec![
            make_episode(1, 0.05, 0.1, true),
            make_episode(2, 0.02, 0.1, true),
        ];

        let result = engine.consolidate(&episodes);
        assert_eq!(result.episodes_pruned, 2);
    }

    #[test]
    fn test_preserve_exemplars() {
        // High-quality episodes should NOT be pruned even if low importance
        let engine = ConsolidationEngine::default_engine();
        let episodes = vec![
            EpisodeSummary {
                id: 1,
                importance: 0.05,
                quality_score: 0.95, // Above preserve threshold
                outcome: EpisodeOutcome::Success,
                tags: vec![],
                embedding: vec![1.0],
                prompt: "test".into(),
                output: "test".into(),
                consolidated: true,
                timestamp: 100,
            },
        ];

        let pruned = engine.select_for_pruning(&episodes);
        assert!(pruned.is_empty(), "High-quality episode should be preserved");
    }

    #[test]
    fn test_sorted_by_importance() {
        let engine = ConsolidationEngine::default_engine();
        let episodes = vec![
            make_episode(1, 0.5, 0.7, false),
            make_episode(2, 0.9, 0.8, false),
            make_episode(3, 0.3, 0.6, false),
        ];

        let selected = engine.select_for_consolidation(&episodes);
        // Should be sorted by importance: 2 (0.9), 1 (0.5), 3 (0.3)
        assert_eq!(selected[0], 2);
    }
}
