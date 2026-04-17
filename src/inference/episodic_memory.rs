//! Episodic Memory — event-based experience storage for self-extending models.
//!
//! Unlike ConceptMap (semantic memory — stores facts/skills), EpisodicMemory stores
//! *experiences*: what happened, what was attempted, what the outcome was, and what
//! can be learned from it.
//!
//! Key principles (from episodic_memory.md research):
//!   1. Event-based, not token-based — stores salient events, not every token
//!   2. Sparse — importance filter rejects mundane experiences
//!   3. Content-based retrieval — find similar past situations via embedding
//!   4. Editable/compressible — similar episodes merge into generalized ones
//!   5. Tool-connected — episodes record tool usage traces
//!   6. Supports meta-learning — tracking improvement over time
//!   7. Multi-modal — can store text, code, structured data
//!   8. Persistent — save/load across sessions
//!   9. Safe/bounded — capacity limits, quality gates
//!  10. Curriculum-generating — high-uncertainty episodes become practice targets

use std::collections::VecDeque;
use std::path::Path;

/// How a tool was used within an episode.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolTrace {
    /// Name of the tool invoked.
    pub tool_name: String,
    /// Arguments passed to the tool.
    pub args: String,
    /// Output from the tool.
    pub output: String,
    /// Whether the tool call succeeded.
    pub success: bool,
    /// Step number within the plan (0-indexed).
    pub step: usize,
}

/// Outcome of an episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EpisodeOutcome {
    /// Goal achieved, quality above threshold.
    Success,
    /// Partially achieved, quality above minimum but below ideal.
    PartialSuccess,
    /// Failed to achieve goal.
    Failure,
}

impl EpisodeOutcome {
    /// Numerical score for importance computation.
    pub fn weight(&self) -> f32 {
        match self {
            EpisodeOutcome::Success => 0.5,
            EpisodeOutcome::PartialSuccess => 0.8, // More informative than pure success
            EpisodeOutcome::Failure => 1.0,        // Failures are most informative
        }
    }
}

/// A single episodic memory — one salient event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Episode {
    /// Unique identifier (monotonically increasing).
    pub id: u64,
    /// Embedding of the context/prompt (for content-based retrieval).
    pub context_embedding: Vec<f32>,
    /// The original prompt or situation that triggered this episode.
    pub prompt: String,
    /// Tools used during this episode, in order.
    pub tools_used: Vec<ToolTrace>,
    /// Description of the plan that was executed (if any).
    pub plan_description: Option<String>,
    /// The final output/response.
    pub output: String,
    /// Outcome classification.
    pub outcome: EpisodeOutcome,
    /// MirrorTest quality score (0.0–1.0).
    pub quality_score: f32,
    /// Average logit entropy during generation (higher = more uncertain).
    pub logit_entropy: f32,
    /// Model's confidence in its output (0.0–1.0).
    pub confidence: f32,
    /// Computed importance score (0.0–1.0).
    pub importance: f32,
    /// Timestamp (seconds since epoch).
    pub timestamp: u64,
    /// Tags for categorical retrieval.
    pub tags: Vec<String>,
    /// If this episode was produced by compressing others, these are the source IDs.
    pub compressed_from: Vec<u64>,
    /// Whether this episode has been "consolidated" (replayed for learning).
    pub consolidated: bool,
}

impl Episode {
    /// Compute the importance score from intrinsic properties.
    ///
    /// importance = surprise × uncertainty × outcome_magnitude
    ///
    /// - surprise: 1 - confidence (low confidence = high surprise)
    /// - uncertainty: entropy normalized to [0, 1]
    /// - outcome_magnitude: failure=1.0, partial=0.8, success=0.5
    pub fn compute_importance(&mut self) {
        let surprise = 1.0 - self.confidence;
        let uncertainty = (self.logit_entropy / 5.0).min(1.0); // Normalize entropy
        let magnitude = self.outcome.weight();

        self.importance = (surprise * uncertainty * magnitude).clamp(0.0, 1.0);
    }
}

/// Result of retrieving episodes from memory.
#[derive(Debug, Clone)]
pub struct RetrievedEpisode {
    /// The retrieved episode.
    pub episode: Episode,
    /// Similarity score to the query.
    pub similarity: f32,
}

/// Configuration for episodic memory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EpisodicMemoryConfig {
    /// Embedding dimension for context vectors.
    pub embedding_dim: usize,
    /// Maximum number of recent episodes to keep in the fast buffer.
    pub recent_capacity: usize,
    /// Maximum total episodes (recent + persistent).
    pub max_episodes: usize,
    /// Importance threshold below which episodes are not stored.
    pub importance_threshold: f32,
    /// Cosine similarity threshold for compression (merge if > this).
    pub compression_similarity: f32,
    /// Recency bias: weight multiplier for more recent episodes during retrieval.
    pub recency_bias: f32,
    /// Maximum number of episodes returned by retrieval.
    pub max_retrieved: usize,
}

impl Default for EpisodicMemoryConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 64,
            recent_capacity: 100,
            max_episodes: 10000,
            importance_threshold: 0.1,
            compression_similarity: 0.85,
            recency_bias: 0.1,
            max_retrieved: 10,
        }
    }
}

/// Episodic Memory — event-based experience storage.
///
/// Two-tier architecture:
/// - Recent buffer: VecDeque of the last N episodes (fast, in-memory)
/// - Persistent store: all episodes, including compressed/generalized ones
pub struct EpisodicMemory {
    config: EpisodicMemoryConfig,
    /// Fast-access buffer of recent episodes.
    recent: VecDeque<Episode>,
    /// Full persistent store (includes compressed episodes).
    persistent: Vec<Episode>,
    /// Monotonically increasing ID counter.
    next_id: u64,
    /// Total number of episodes stored (for stats).
    total_stored: u64,
    /// Number of episodes that were compressed away.
    total_compressed: u64,
}

impl EpisodicMemory {
    /// Create a new episodic memory with the given configuration.
    pub fn new(config: EpisodicMemoryConfig) -> Self {
        Self {
            config,
            recent: VecDeque::with_capacity(100),
            persistent: Vec::with_capacity(1000),
            next_id: 1,
            total_stored: 0,
            total_compressed: 0,
        }
    }

    /// Store a new episode.
    ///
    /// Returns the episode ID, or None if the episode was below the importance
    /// threshold.
    pub fn store(&mut self, mut episode: Episode) -> Option<u64> {
        // Compute importance if not already set
        if episode.importance == 0.0 {
            episode.compute_importance();
        }

        // Filter by importance
        if episode.importance < self.config.importance_threshold {
            tracing::debug!(
                event = "episode_rejected",
                importance = episode.importance,
                "Episode below importance threshold"
            );
            return None;
        }

        // Assign ID and timestamp
        episode.id = self.next_id;
        self.next_id += 1;
        self.total_stored += 1;

        if episode.timestamp == 0 {
            episode.timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
        }

        let id = episode.id;

        // Add to recent buffer
        if self.recent.len() >= self.config.recent_capacity {
            // Evict oldest from recent (stays in persistent)
            if let Some(evicted) = self.recent.pop_front() {
                self.persistent.push(evicted);
            }
        }
        self.recent.push_back(episode);

        tracing::debug!(
            event = "episode_stored",
            id,
            importance = self.recent.back().map(|e| e.importance).unwrap_or(0.0),
            outcome = ?self.recent.back().map(|e| e.outcome).unwrap_or(EpisodeOutcome::Success),
            "Stored episode"
        );

        Some(id)
    }

    /// Retrieve episodes similar to a query embedding.
    ///
    /// Uses cosine similarity with a recency bias to favor recent experiences.
    pub fn retrieve(&self, query_embedding: &[f32], top_k: usize) -> Vec<RetrievedEpisode> {
        let top_k = top_k.min(self.config.max_retrieved).max(1);
        let query_norm = norm(query_embedding);
        if query_norm < 1e-8 {
            return Vec::new();
        }

        let mut scored: Vec<RetrievedEpisode> = Vec::new();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let score_episode = |ep: &Episode| -> f32 {
            let sim = cosine_similarity(query_embedding, &ep.context_embedding);
            // Recency bias: exponential decay
            let age_secs = now.saturating_sub(ep.timestamp) as f32;
            let age_hours = age_secs / 3600.0;
            let recency_factor = (-self.config.recency_bias * age_hours).exp();
            // Combine: similarity weighted by recency and importance
            sim * (0.7 + 0.3 * recency_factor) * (0.5 + 0.5 * ep.importance)
        };

        // Score recent episodes
        for ep in &self.recent {
            scored.push(RetrievedEpisode {
                similarity: score_episode(ep),
                episode: ep.clone(),
            });
        }

        // Score persistent episodes
        for ep in &self.persistent {
            scored.push(RetrievedEpisode {
                similarity: score_episode(ep),
                episode: ep.clone(),
            });
        }

        // Sort by similarity descending
        scored.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
    }

    /// Retrieve all episodes where a specific tool was used.
    pub fn retrieve_by_tool(&self, tool_name: &str, top_k: usize) -> Vec<RetrievedEpisode> {
        let top_k = top_k.min(self.config.max_retrieved).max(1);
        let mut results = Vec::new();

        let check = |ep: &Episode| -> bool {
            ep.tools_used.iter().any(|t| t.tool_name == tool_name)
        };

        for ep in self.recent.iter().chain(self.persistent.iter()) {
            if check(ep) {
                results.push(RetrievedEpisode {
                    similarity: 1.0, // Exact match
                    episode: ep.clone(),
                });
            }
        }

        // Sort by recency (newest first)
        results.sort_by(|a, b| b.episode.timestamp.cmp(&a.episode.timestamp));
        results.truncate(top_k);
        results
    }

    /// Retrieve failure episodes for contexts similar to the query.
    ///
    /// Useful for avoiding past mistakes.
    pub fn retrieve_failures(&self, query_embedding: &[f32], top_k: usize) -> Vec<RetrievedEpisode> {
        let top_k = top_k.min(self.config.max_retrieved).max(1);
        let mut scored = Vec::new();

        let check = |ep: &Episode| {
            matches!(ep.outcome, EpisodeOutcome::Failure | EpisodeOutcome::PartialSuccess)
        };

        for ep in self.recent.iter().chain(self.persistent.iter()) {
            if check(ep) {
                let sim = cosine_similarity(query_embedding, &ep.context_embedding);
                scored.push(RetrievedEpisode {
                    similarity: sim,
                    episode: ep.clone(),
                });
            }
        }

        scored.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Retrieve episodes that are good candidates for practice (high uncertainty, recent).
    pub fn retrieve_practice_candidates(&self, top_k: usize) -> Vec<RetrievedEpisode> {
        let top_k = top_k.min(self.config.max_retrieved).max(1);
        let mut candidates = Vec::new();

        for ep in self.recent.iter().chain(self.persistent.iter()) {
            // High uncertainty and not yet consolidated
            if !ep.consolidated && ep.logit_entropy > 2.0 {
                candidates.push(RetrievedEpisode {
                    similarity: ep.importance, // Use importance as score
                    episode: ep.clone(),
                });
            }
        }

        candidates.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(top_k);
        candidates
    }

    /// Compress similar episodes into generalized ones.
    ///
    /// Finds clusters of episodes with cosine similarity > threshold and merges
    /// them into a single generalized episode.
    ///
    /// Returns the number of episodes compressed.
    pub fn compress(&mut self) -> usize {
        let threshold = self.config.compression_similarity;
        let mut to_remove = std::collections::HashSet::new();
        let mut new_episodes = Vec::new();

        // Collect all episodes for clustering
        let all_episodes: Vec<&Episode> = self.recent.iter()
            .chain(self.persistent.iter())
            .collect();

        // Simple greedy clustering: for each episode, find all similar ones
        for i in 0..all_episodes.len() {
            if to_remove.contains(&all_episodes[i].id) {
                continue;
            }

            let mut cluster = vec![all_episodes[i].id];
            for j in (i + 1)..all_episodes.len() {
                if to_remove.contains(&all_episodes[j].id) {
                    continue;
                }

                let sim = cosine_similarity(
                    &all_episodes[i].context_embedding,
                    &all_episodes[j].context_embedding,
                );

                if sim > threshold {
                    cluster.push(all_episodes[j].id);
                }
            }

            // Merge cluster if it has 3+ members
            if cluster.len() >= 3 {
                let members: Vec<&Episode> = all_episodes.iter()
                    .filter(|e| cluster.contains(&e.id))
                    .copied()
                    .collect();

                // Create generalized episode
                let centroid = compute_centroid(
                    members.iter().map(|e| e.context_embedding.as_slice()).collect()
                );

                let avg_quality: f32 = members.iter().map(|e| e.quality_score).sum::<f32>() / members.len() as f32;
                let avg_entropy: f32 = members.iter().map(|e| e.logit_entropy).sum::<f32>() / members.len() as f32;
                let avg_confidence: f32 = members.iter().map(|e| e.confidence).sum::<f32>() / members.len() as f32;
                let max_importance: f32 = members.iter().map(|e| e.importance).fold(0.0f32, f32::max);

                let prompts: Vec<&str> = members.iter().map(|e| e.prompt.as_str()).collect();
                let generalized_prompt = format!(
                    "[Generalized from {} episodes] {}",
                    members.len(),
                    prompts.first().unwrap_or(&""),
                );

                let mut generalized = Episode {
                    id: 0, // Will be assigned by store
                    context_embedding: centroid,
                    prompt: generalized_prompt,
                    tools_used: members.iter().flat_map(|e| e.tools_used.clone()).collect(),
                    plan_description: Some(format!(
                        "Generalized from {} similar experiences",
                        members.len(),
                    )),
                    output: String::new(),
                    outcome: if avg_quality > 0.7 {
                        EpisodeOutcome::Success
                    } else if avg_quality > 0.4 {
                        EpisodeOutcome::PartialSuccess
                    } else {
                        EpisodeOutcome::Failure
                    },
                    quality_score: avg_quality,
                    logit_entropy: avg_entropy,
                    confidence: avg_confidence,
                    importance: max_importance,
                    timestamp: members.iter().map(|e| e.timestamp).max().unwrap_or(0),
                    tags: members.iter().flat_map(|e| e.tags.clone()).collect(),
                    compressed_from: cluster.clone(),
                    consolidated: false,
                };
                generalized.compute_importance();

                new_episodes.push(generalized);
                for &id in &cluster {
                    to_remove.insert(id);
                }
            }
        }

        let compressed_count = to_remove.len();

        // Remove compressed episodes from both stores
        self.recent.retain(|e| !to_remove.contains(&e.id));
        self.persistent.retain(|e| !to_remove.contains(&e.id));

        // Add generalized episodes
        for ep in new_episodes {
            self.store(ep);
        }

        self.total_compressed += compressed_count as u64;

        if compressed_count > 0 {
            tracing::info!(
                event = "episodes_compressed",
                compressed = compressed_count,
                total_stored = self.total_stored,
                total_compressed = self.total_compressed,
                "Compressed {} episodes into generalizations",
                compressed_count,
            );
        }

        compressed_count
    }

    /// Get statistics about the episodic memory.
    pub fn stats(&self) -> EpisodicMemoryStats {
        let success_count = self.recent.iter().chain(self.persistent.iter())
            .filter(|e| e.outcome == EpisodeOutcome::Success)
            .count();
        let failure_count = self.recent.iter().chain(self.persistent.iter())
            .filter(|e| e.outcome == EpisodeOutcome::Failure)
            .count();
        let avg_quality = self.recent.iter().chain(self.persistent.iter())
            .map(|e| e.quality_score)
            .sum::<f32>() / (self.len() as f32).max(1.0);

        EpisodicMemoryStats {
            total_episodes: self.len(),
            recent_count: self.recent.len(),
            persistent_count: self.persistent.len(),
            total_stored: self.total_stored,
            total_compressed: self.total_compressed,
            success_count,
            failure_count,
            avg_quality,
        }
    }

    /// Number of episodes currently in memory.
    pub fn len(&self) -> usize {
        self.recent.len() + self.persistent.len()
    }

    /// Whether memory is empty.
    pub fn is_empty(&self) -> bool {
        self.recent.is_empty() && self.persistent.is_empty()
    }

    /// Check if we're near capacity and should compress.
    pub fn should_compress(&self) -> bool {
        self.len() > (self.config.max_episodes as f64 * 0.8) as usize
    }

    /// Mark an episode as consolidated (replayed for learning).
    pub fn mark_consolidated(&mut self, episode_id: u64) {
        for ep in self.recent.iter_mut().chain(self.persistent.iter_mut()) {
            if ep.id == episode_id {
                ep.consolidated = true;
                break;
            }
        }
    }

    /// Save episodic memory to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let data = EpisodicMemoryData {
            config: self.config.clone(),
            recent: self.recent.iter().cloned().collect(),
            persistent: self.persistent.clone(),
            next_id: self.next_id,
            total_stored: self.total_stored,
            total_compressed: self.total_compressed,
        };

        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let tmp_path = format!("{}.tmp", path.display());
        let json = serde_json::to_string(&data).map_err(|e| format!("Serialize: {}", e))?;
        std::fs::write(&tmp_path, &json).map_err(|e| format!("Write: {}", e))?;
        std::fs::rename(&tmp_path, path).map_err(|e| format!("Rename: {}", e))?;

        Ok(())
    }

    /// Load episodic memory from a JSON file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| format!("Read: {}", e))?;
        let data: EpisodicMemoryData = serde_json::from_str(&json).map_err(|e| format!("Deserialize: {}", e))?;

        Ok(Self {
            config: data.config,
            recent: data.recent.into_iter().collect(),
            persistent: data.persistent,
            next_id: data.next_id,
            total_stored: data.total_stored,
            total_compressed: data.total_compressed,
        })
    }

    /// Get an episode by ID.
    pub fn get(&self, id: u64) -> Option<&Episode> {
        self.recent.iter().chain(self.persistent.iter()).find(|e| e.id == id)
    }

    /// Get a mutable reference to an episode by ID.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut Episode> {
        self.recent.iter_mut().chain(self.persistent.iter_mut()).find(|e| e.id == id)
    }

    /// Iterate over all episodes.
    pub fn iter(&self) -> impl Iterator<Item = &Episode> {
        self.recent.iter().chain(self.persistent.iter())
    }

    /// Get the configuration.
    pub fn config(&self) -> &EpisodicMemoryConfig {
        &self.config
    }
}

/// Serialization helper.
#[derive(serde::Serialize, serde::Deserialize)]
struct EpisodicMemoryData {
    config: EpisodicMemoryConfig,
    recent: Vec<Episode>,
    persistent: Vec<Episode>,
    next_id: u64,
    total_stored: u64,
    total_compressed: u64,
}

/// Statistics about episodic memory.
#[derive(Debug, Clone)]
pub struct EpisodicMemoryStats {
    pub total_episodes: usize,
    pub recent_count: usize,
    pub persistent_count: usize,
    pub total_stored: u64,
    pub total_compressed: u64,
    pub success_count: usize,
    pub failure_count: usize,
    pub avg_quality: f32,
}

impl std::fmt::Display for EpisodicMemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EpisodicMemory: {} episodes ({} recent, {} persistent), {} stored total, {} compressed, success={}/{} (avg quality {:.2})",
            self.total_episodes,
            self.recent_count,
            self.persistent_count,
            self.total_stored,
            self.total_compressed,
            self.success_count,
            self.success_count + self.failure_count,
            self.avg_quality,
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let a_norm = norm(a);
    let b_norm = norm(b);
    if a_norm < 1e-8 || b_norm < 1e-8 {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (a_norm * b_norm)
}

fn compute_centroid(embeddings: Vec<&[f32]>) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    let dim = embeddings[0].len();
    let mut centroid = vec![0.0f32; dim];
    for emb in &embeddings {
        for (i, &v) in emb.iter().enumerate() {
            centroid[i] += v;
        }
    }
    let n = embeddings.len() as f32;
    for v in centroid.iter_mut() {
        *v /= n;
    }
    // Normalize
    let n = norm(&centroid);
    if n > 1e-8 {
        for v in centroid.iter_mut() {
            *v /= n;
        }
    }
    centroid
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(dim: usize, seed: f32) -> Vec<f32> {
        let mut emb = vec![0.0f32; dim];
        for (i, v) in emb.iter_mut().enumerate() {
            *v = ((seed + i as f32) * 0.1).sin();
        }
        let n = norm(&emb);
        if n > 1e-8 {
            for v in emb.iter_mut() {
                *v /= n;
            }
        }
        emb
    }

    fn make_episode(id: u64, prompt: &str, embedding_seed: f32, outcome: EpisodeOutcome) -> Episode {
        Episode {
            id,
            context_embedding: make_embedding(64, embedding_seed),
            prompt: prompt.to_string(),
            tools_used: vec![],
            plan_description: None,
            output: format!("output for {}", prompt),
            outcome,
            quality_score: match outcome {
                EpisodeOutcome::Success => 0.9,
                EpisodeOutcome::PartialSuccess => 0.6,
                EpisodeOutcome::Failure => 0.2,
            },
            logit_entropy: 2.5,
            confidence: match outcome {
                EpisodeOutcome::Success => 0.9,
                EpisodeOutcome::PartialSuccess => 0.6,
                EpisodeOutcome::Failure => 0.3,
            },
            importance: 0.5,
            timestamp: 1000 + id,
            tags: vec!["test".into()],
            compressed_from: vec![],
            consolidated: false,
        }
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut mem = EpisodicMemory::new(EpisodicMemoryConfig {
            embedding_dim: 64,
            importance_threshold: 0.0,
            ..Default::default()
        });

        let ep = make_episode(0, "sort algorithm", 1.0, EpisodeOutcome::Success);
        let query = ep.context_embedding.clone();
        mem.store(ep).unwrap();

        let results = mem.retrieve(&query, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].episode.prompt, "sort algorithm");
    }

    #[test]
    fn test_importance_filter() {
        let mut mem = EpisodicMemory::new(EpisodicMemoryConfig {
            importance_threshold: 0.5,
            ..Default::default()
        });

        // Low importance → rejected
        let mut low = make_episode(0, "low", 1.0, EpisodeOutcome::Success);
        low.importance = 0.1;
        low.confidence = 0.99;
        low.logit_entropy = 0.1;
        assert!(mem.store(low).is_none());

        // High importance → accepted
        let mut high = make_episode(0, "high", 2.0, EpisodeOutcome::Failure);
        high.importance = 0.8;
        assert!(mem.store(high).is_some());
    }

    #[test]
    fn test_importance_computation() {
        let mut ep = make_episode(0, "test", 1.0, EpisodeOutcome::Failure);
        ep.confidence = 0.2; // Low confidence = high surprise
        ep.logit_entropy = 4.0; // High entropy
        ep.compute_importance();

        // surprise = 0.8, uncertainty = 0.8, outcome_weight = 1.0
        let expected = 0.8 * 0.8 * 1.0;
        assert!((ep.importance - expected).abs() < 0.01, "Expected {}, got {}", expected, ep.importance);
    }

    #[test]
    fn test_retrieve_by_tool() {
        let mut mem = EpisodicMemory::new(EpisodicMemoryConfig {
            importance_threshold: 0.0,
            ..Default::default()
        });

        let mut ep1 = make_episode(0, "calc", 1.0, EpisodeOutcome::Success);
        ep1.tools_used.push(ToolTrace {
            tool_name: "calm_execute".into(),
            args: "add 3 5".into(),
            output: "8".into(),
            success: true,
            step: 0,
        });
        mem.store(ep1).unwrap();

        let ep2 = make_episode(0, "other", 2.0, EpisodeOutcome::Success);
        mem.store(ep2).unwrap();

        let results = mem.retrieve_by_tool("calm_execute", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].episode.prompt, "calc");
    }

    #[test]
    fn test_retrieve_failures() {
        let mut mem = EpisodicMemory::new(EpisodicMemoryConfig {
            importance_threshold: 0.0,
            ..Default::default()
        });

        let success = make_episode(0, "good", 1.0, EpisodeOutcome::Success);
        let failure = make_episode(0, "bad", 1.0, EpisodeOutcome::Failure);
        let query = failure.context_embedding.clone();

        mem.store(success).unwrap();
        mem.store(failure).unwrap();

        let results = mem.retrieve_failures(&query, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].episode.prompt, "bad");
    }

    #[test]
    fn test_compression() {
        let mut mem = EpisodicMemory::new(EpisodicMemoryConfig {
            importance_threshold: 0.0,
            compression_similarity: 0.8,
            recent_capacity: 100,
            ..Default::default()
        });

        // Store 5 very similar episodes
        for i in 0..5 {
            let mut ep = make_episode(0, &format!("similar {}", i), 1.0, EpisodeOutcome::Success);
            // Tiny perturbation so they cluster but are different
            ep.context_embedding[0] += (i as f32) * 0.001;
            mem.store(ep).unwrap();
        }

        // Store 1 different episode
        let different = make_episode(0, "different", 10.0, EpisodeOutcome::Success);
        mem.store(different).unwrap();

        assert_eq!(mem.len(), 6);

        let compressed = mem.compress();
        assert!(compressed >= 3, "Should compress at least 3 similar episodes, got {}", compressed);
        // The generalized episode replaces the cluster
        assert!(mem.len() < 6, "Should have fewer episodes after compression");
    }

    #[test]
    fn test_recent_buffer_overflow() {
        let mut mem = EpisodicMemory::new(EpisodicMemoryConfig {
            recent_capacity: 3,
            importance_threshold: 0.0,
            ..Default::default()
        });

        for i in 0..5 {
            let ep = make_episode(0, &format!("ep {}", i), i as f32, EpisodeOutcome::Success);
            mem.store(ep).unwrap();
        }

        // Recent should be capped at 3, overflow goes to persistent
        assert_eq!(mem.recent.len(), 3);
        assert_eq!(mem.persistent.len(), 2);
        assert_eq!(mem.len(), 5);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = std::env::temp_dir().join("ferrisres_episodic_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("episodes.json");

        {
            let mut mem = EpisodicMemory::new(EpisodicMemoryConfig {
                importance_threshold: 0.0,
                ..Default::default()
            });

            for i in 0..3 {
                let ep = make_episode(0, &format!("ep {}", i), i as f32, EpisodeOutcome::Success);
                mem.store(ep).unwrap();
            }
            mem.save(&path).unwrap();
        }

        {
            let mem = EpisodicMemory::load(&path).unwrap();
            assert_eq!(mem.len(), 3);
            assert_eq!(mem.total_stored, 3);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_stats() {
        let mut mem = EpisodicMemory::new(EpisodicMemoryConfig {
            importance_threshold: 0.0,
            ..Default::default()
        });

        mem.store(make_episode(0, "s1", 1.0, EpisodeOutcome::Success)).unwrap();
        mem.store(make_episode(0, "s2", 2.0, EpisodeOutcome::Success)).unwrap();
        mem.store(make_episode(0, "f1", 3.0, EpisodeOutcome::Failure)).unwrap();

        let stats = mem.stats();
        assert_eq!(stats.total_episodes, 3);
        assert_eq!(stats.success_count, 2);
        assert_eq!(stats.failure_count, 1);
    }

    #[test]
    fn test_episode_outcome_weights() {
        assert_eq!(EpisodeOutcome::Success.weight(), 0.5);
        assert_eq!(EpisodeOutcome::PartialSuccess.weight(), 0.8);
        assert_eq!(EpisodeOutcome::Failure.weight(), 1.0);
    }

    #[test]
    fn test_get_by_id() {
        let mut mem = EpisodicMemory::new(EpisodicMemoryConfig {
            importance_threshold: 0.0,
            ..Default::default()
        });

        let ep = make_episode(0, "find me", 1.0, EpisodeOutcome::Success);
        mem.store(ep.clone()).unwrap();
        let id = mem.recent.back().unwrap().id;

        assert!(mem.get(id).is_some());
        assert!(mem.get(999).is_none());
    }

    #[test]
    fn test_mark_consolidated() {
        let mut mem = EpisodicMemory::new(EpisodicMemoryConfig {
            importance_threshold: 0.0,
            ..Default::default()
        });

        mem.store(make_episode(0, "ep", 1.0, EpisodeOutcome::Success)).unwrap();
        let id = mem.recent.back().unwrap().id;

        assert!(!mem.get(id).unwrap().consolidated);
        mem.mark_consolidated(id);
        assert!(mem.get(id).unwrap().consolidated);
    }

    #[test]
    fn test_practice_candidates() {
        let mut mem = EpisodicMemory::new(EpisodicMemoryConfig {
            importance_threshold: 0.0,
            ..Default::default()
        });

        // High entropy, not consolidated → candidate
        let mut high_unc = make_episode(0, "uncertain", 1.0, EpisodeOutcome::PartialSuccess);
        high_unc.logit_entropy = 3.5;
        mem.store(high_unc).unwrap();

        // Low entropy → not a candidate
        let mut low_unc = make_episode(0, "certain", 2.0, EpisodeOutcome::Success);
        low_unc.logit_entropy = 0.5;
        mem.store(low_unc).unwrap();

        let candidates = mem.retrieve_practice_candidates(5);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].episode.prompt, "uncertain");
    }

    #[test]
    fn test_should_compress() {
        let mut mem = EpisodicMemory::new(EpisodicMemoryConfig {
            max_episodes: 10,
            importance_threshold: 0.0,
            ..Default::default()
        });

        for i in 0..9 {
            mem.store(make_episode(0, &format!("ep {}", i), i as f32, EpisodeOutcome::Success)).unwrap();
        }

        assert!(mem.should_compress()); // 9 > 80% of 10
    }
}
