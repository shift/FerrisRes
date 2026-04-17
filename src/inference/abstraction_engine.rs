//! Abstraction Engine — Concept Compression & Generalization
//!
//! Based on DreamCoder, SOAR chunking, and prototype theory.
//! Periodically scans ConceptMap for clusters of similar concepts,
//! computes generalized meta-concepts (prototypes), and compresses
//! N specific concepts into 1 abstract concept.
//!
//! Pipeline:
//!   1. SCAN: Find clusters of similar concepts (cosine > 0.8)
//!   2. EXTRACT: Compute generalized meta-concept (centroid + parameterized code)
//!   3. VALIDATE: Check abstraction quality
//!   4. COMPRESS: Replace N specific → 1 abstract, free ConceptMap capacity

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Abstraction levels
// ---------------------------------------------------------------------------

/// Hierarchical concept level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ConceptLevel {
    /// Raw observation / instance.
    Instance,
    /// Abstracted from instances.
    Pattern,
    /// Abstracted from patterns.
    Principle,
    /// Cross-domain abstraction.
    MetaPrinciple,
}

impl ConceptLevel {
    /// Numeric level for comparison.
    pub fn depth(&self) -> usize {
        match self {
            ConceptLevel::Instance => 0,
            ConceptLevel::Pattern => 1,
            ConceptLevel::Principle => 2,
            ConceptLevel::MetaPrinciple => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Concept cluster
// ---------------------------------------------------------------------------

/// A cluster of similar concepts found during scanning.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConceptCluster {
    /// IDs of concepts in the cluster.
    pub member_ids: Vec<String>,
    /// Tags shared by all members.
    pub common_tags: Vec<String>,
    /// Centroid embedding of the cluster.
    pub centroid: Vec<f32>,
    /// Maximum pairwise distance within cluster.
    pub max_distance: f32,
    /// Average pairwise distance within cluster.
    pub avg_distance: f32,
    /// Category of the cluster.
    pub category: String,
}

// ---------------------------------------------------------------------------
// Abstraction result
// ---------------------------------------------------------------------------

/// Result of abstracting a cluster into a meta-concept.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AbstractionResult {
    /// The cluster that was abstracted.
    pub source_cluster: ConceptCluster,
    /// Description of the meta-concept.
    pub description: String,
    /// Tags for the meta-concept.
    pub tags: Vec<String>,
    /// Centroid embedding.
    pub embedding: Vec<f32>,
    /// Level of the abstraction.
    pub level: ConceptLevel,
    /// Number of concepts compressed.
    pub concepts_compressed: usize,
    /// Compression ratio (original / abstracted).
    pub compression_ratio: f32,
    /// Quality score of the abstraction.
    pub quality: f32,
    /// Timestamp.
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the abstraction engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AbstractionConfig {
    /// Minimum cosine similarity to group concepts into a cluster.
    pub cluster_threshold: f32,
    /// Minimum cluster size to abstract.
    pub min_cluster_size: usize,
    /// Maximum cluster size (split larger clusters).
    pub max_cluster_size: usize,
    /// Minimum compression ratio to accept abstraction.
    pub min_compression_ratio: f32,
    /// Quality threshold for accepting an abstraction.
    pub quality_threshold: f32,
    /// Maximum abstraction level to produce.
    pub max_level: ConceptLevel,
    /// Minimum concepts in ConceptMap before abstraction triggers.
    pub min_concepts_for_abstraction: usize,
}

impl Default for AbstractionConfig {
    fn default() -> Self {
        Self {
            cluster_threshold: 0.8,
            min_cluster_size: 3,
            max_cluster_size: 20,
            min_compression_ratio: 2.0,
            quality_threshold: 0.6,
            max_level: ConceptLevel::Principle,
            min_concepts_for_abstraction: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Concept summary (lightweight view for clustering)
// ---------------------------------------------------------------------------

/// Lightweight concept representation for clustering.
#[derive(Debug, Clone)]
pub struct ConceptSummary {
    pub id: String,
    pub embedding: Vec<f32>,
    pub tags: Vec<String>,
    pub quality: f32,
    pub category: String,
    pub description: String,
}

// ---------------------------------------------------------------------------
// AbstractionEngine
// ---------------------------------------------------------------------------

/// Scans concepts, finds clusters, and creates generalized abstractions.
pub struct AbstractionEngine {
    config: AbstractionConfig,
    /// History of abstractions performed.
    history: Vec<AbstractionResult>,
}

impl AbstractionEngine {
    /// Create a new abstraction engine.
    pub fn new(config: AbstractionConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Create with default configuration.
    pub fn default_engine() -> Self {
        Self::new(AbstractionConfig::default())
    }

    /// Get configuration.
    pub fn config(&self) -> &AbstractionConfig {
        &self.config
    }

    /// Number of abstractions performed.
    pub fn abstraction_count(&self) -> usize {
        self.history.len()
    }

    /// Get abstraction history.
    pub fn history(&self) -> &[AbstractionResult] {
        &self.history
    }

    // ---- Main API ----

    /// Scan concepts and find clusters for abstraction.
    pub fn scan_for_clusters(&self, concepts: &[ConceptSummary]) -> Vec<ConceptCluster> {
        if concepts.len() < self.config.min_cluster_size {
            return vec![];
        }

        // Build similarity graph and find connected components
        let n = concepts.len();
        let mut visited = vec![false; n];
        let mut clusters = Vec::new();

        for i in 0..n {
            if visited[i] {
                continue;
            }

            // BFS to find connected component
            let mut component = vec![i];
            visited[i] = true;
            let mut queue = vec![i];

            while let Some(current) = queue.pop() {
                for j in 0..n {
                    if visited[j] {
                        continue;
                    }
                    let sim = cosine_similarity(&concepts[current].embedding, &concepts[j].embedding);
                    if sim >= self.config.cluster_threshold {
                        visited[j] = true;
                        component.push(j);
                        queue.push(j);
                    }
                }
            }

            // Only keep clusters of sufficient size
            if component.len() >= self.config.min_cluster_size {
                let cluster = self.build_cluster(&component, concepts);
                clusters.push(cluster);
            }
        }

        // Split oversized clusters
        let mut result = Vec::new();
        for cluster in clusters {
            if cluster.member_ids.len() > self.config.max_cluster_size {
                // Simple split: take first max_cluster_size, then remainder
                let chunks = cluster.member_ids.chunks(self.config.max_cluster_size);
                for chunk in chunks {
                    if chunk.len() >= self.config.min_cluster_size {
                        let indices: Vec<usize> = chunk.iter().filter_map(|id| {
                            concepts.iter().position(|c| c.id == *id)
                        }).collect();
                        result.push(self.build_cluster(&indices, concepts));
                    }
                }
            } else {
                result.push(cluster);
            }
        }

        // Sort by avg_distance ascending (tightest clusters first)
        result.sort_by(|a, b| a.avg_distance.partial_cmp(&b.avg_distance).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Abstract a cluster into a meta-concept.
    pub fn abstract_cluster(&mut self, cluster: &ConceptCluster, concepts: &[ConceptSummary]) -> AbstractionResult {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Determine level
        let max_member_level = concepts.iter()
            .filter(|c| cluster.member_ids.contains(&c.id))
            .map(|_| ConceptLevel::Instance.depth()) // simplified
            .max()
            .unwrap_or(0);
        let level = match max_member_level {
            0 => ConceptLevel::Pattern,
            1 => ConceptLevel::Principle,
            _ => ConceptLevel::MetaPrinciple,
        };

        if level.depth() > self.config.max_level.depth() {
            // Can't abstract further
            return AbstractionResult {
                source_cluster: cluster.clone(),
                description: String::new(),
                tags: vec![],
                embedding: cluster.centroid.clone(),
                level,
                concepts_compressed: 0,
                compression_ratio: 1.0,
                quality: 0.0,
                timestamp,
            };
        }

        // Compute centroid description
        let descriptions: Vec<&str> = concepts.iter()
            .filter(|c| cluster.member_ids.contains(&c.id))
            .map(|c| c.description.as_str())
            .collect();

        let description = Self::generalize_descriptions(&descriptions);
        let tags = cluster.common_tags.clone();
        let compression_ratio = cluster.member_ids.len() as f32; // N concepts → 1

        // Quality: tighter clusters produce better abstractions
        let tightness = 1.0 - cluster.avg_distance;
        let quality = tightness.min(1.0).max(0.0);

        let result = AbstractionResult {
            source_cluster: cluster.clone(),
            description,
            tags,
            embedding: cluster.centroid.clone(),
            level,
            concepts_compressed: cluster.member_ids.len(),
            compression_ratio,
            quality,
            timestamp,
        };

        self.history.push(result.clone());
        result
    }

    /// Full abstraction pass: scan, abstract, return results.
    pub fn run_abstraction(&mut self, concepts: &[ConceptSummary]) -> Vec<AbstractionResult> {
        let clusters = self.scan_for_clusters(concepts);
        let mut results = Vec::new();

        for cluster in &clusters {
            let result = self.abstract_cluster(cluster, concepts);
            if result.quality >= self.config.quality_threshold
                && result.compression_ratio >= self.config.min_compression_ratio
            {
                results.push(result);
            }
        }

        results
    }

    // ---- Internal ----

    fn build_cluster(&self, indices: &[usize], concepts: &[ConceptSummary]) -> ConceptCluster {
        let members: Vec<&ConceptSummary> = indices.iter().map(|&i| &concepts[i]).collect();

        // Compute centroid
        let dim = members[0].embedding.len();
        let mut centroid = vec![0.0f32; dim];
        for m in &members {
            for (i, &v) in m.embedding.iter().enumerate() {
                centroid[i] += v;
            }
        }
        for v in centroid.iter_mut() {
            *v /= members.len() as f32;
        }

        // Compute distances
        let mut max_dist = 0.0f32;
        let mut total_dist = 0.0f32;
        let mut pairs = 0;
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                let dist = 1.0 - cosine_similarity(&members[i].embedding, &members[j].embedding);
                max_dist = max_dist.max(dist);
                total_dist += dist;
                pairs += 1;
            }
        }
        let avg_dist = if pairs > 0 { total_dist / pairs as f32 } else { 0.0 };

        // Common tags: intersection
        let mut common_tags = members[0].tags.clone();
        for m in &members[1..] {
            common_tags.retain(|t| m.tags.contains(t));
        }

        // Category: most common
        let mut cat_counts: HashMap<String, usize> = HashMap::new();
        for m in &members {
            *cat_counts.entry(m.category.clone()).or_insert(0) += 1;
        }
        let category = cat_counts.into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(c, _)| c)
            .unwrap_or_else(|| "general".to_string());

        ConceptCluster {
            member_ids: members.iter().map(|m| m.id.clone()).collect(),
            common_tags,
            centroid,
            max_distance: max_dist,
            avg_distance: avg_dist,
            category,
        }
    }

    /// Generalize a set of descriptions into a single description.
    fn generalize_descriptions(descriptions: &[&str]) -> String {
        if descriptions.is_empty() {
            return String::new();
        }
        if descriptions.len() == 1 {
            return descriptions[0].to_string();
        }

        // Find common words
        let word_sets: Vec<Vec<String>> = descriptions.iter()
            .map(|d| {
                d.to_lowercase()
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|s| s.len() > 2)
                    .map(String::from)
                    .collect()
            })
            .collect();

        let mut common_words: Vec<String> = word_sets[0].iter()
            .filter(|w| word_sets[1..].iter().all(|ws| ws.contains(w)))
            .cloned()
            .collect();
        common_words.sort();
        common_words.dedup();

        if common_words.is_empty() {
            format!("Generalized from {} instances", descriptions.len())
        } else {
            format!(
                "Generalized: {} (from {} instances)",
                common_words.join(" "),
                descriptions.len()
            )
        }
    }

    // ---- Stats ----

    /// Get abstraction statistics.
    pub fn stats(&self) -> AbstractionStats {
        let total = self.history.len();
        let total_compressed: usize = self.history.iter().map(|r| r.concepts_compressed).sum();
        let avg_quality = if total > 0 {
            self.history.iter().map(|r| r.quality).sum::<f32>() / total as f32
        } else {
            0.0
        };

        AbstractionStats {
            total_abstractions: total,
            total_concepts_compressed: total_compressed,
            avg_quality,
            avg_compression_ratio: if total > 0 {
                self.history.iter().map(|r| r.compression_ratio).sum::<f32>() / total as f32
            } else {
                0.0
            },
        }
    }
}

/// Statistics about abstractions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AbstractionStats {
    pub total_abstractions: usize,
    pub total_concepts_compressed: usize,
    pub avg_quality: f32,
    pub avg_compression_ratio: f32,
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

    fn make_concept(id: &str, embedding: Vec<f32>, tags: Vec<&str>) -> ConceptSummary {
        ConceptSummary {
            id: id.to_string(),
            embedding,
            tags: tags.iter().map(|t| t.to_string()).collect(),
            quality: 0.8,
            category: "test".to_string(),
            description: format!("Concept {}", id),
        }
    }

    #[test]
    fn test_scan_finds_clusters() {
        let engine = AbstractionEngine::default_engine();

        // Three similar concepts (high cosine similarity)
        let concepts = vec![
            make_concept("a", vec![1.0, 0.0, 0.0], vec!["math"]),
            make_concept("b", vec![0.95, 0.05, 0.0], vec!["math"]),
            make_concept("c", vec![0.9, 0.1, 0.0], vec!["math"]),
            // One different concept
            make_concept("d", vec![0.0, 0.0, 1.0], vec!["text"]),
        ];

        let clusters = engine.scan_for_clusters(&concepts);
        assert_eq!(clusters.len(), 1);
        assert!(clusters[0].member_ids.len() >= 3);
    }

    #[test]
    fn test_scan_no_clusters_if_too_few() {
        let engine = AbstractionEngine::default_engine();
        let concepts = vec![
            make_concept("a", vec![1.0, 0.0], vec!["math"]),
            make_concept("b", vec![0.95, 0.05], vec!["math"]),
        ];

        let clusters = engine.scan_for_clusters(&concepts);
        assert!(clusters.is_empty()); // min_cluster_size=3
    }

    #[test]
    fn test_abstract_cluster() {
        let mut engine = AbstractionEngine::default_engine();

        let concepts = vec![
            make_concept("a", vec![1.0, 0.0], vec!["math", "add"]),
            make_concept("b", vec![0.95, 0.05], vec!["math", "add"]),
            make_concept("c", vec![0.9, 0.1], vec!["math", "add"]),
        ];

        let clusters = engine.scan_for_clusters(&concepts);
        assert!(!clusters.is_empty());

        let result = engine.abstract_cluster(&clusters[0], &concepts);
        assert!(result.concepts_compressed >= 3);
        assert!(result.compression_ratio >= 2.0);
        assert!(result.source_cluster.common_tags.contains(&"math".to_string()));
        assert_eq!(result.level, ConceptLevel::Pattern);
    }

    #[test]
    fn test_run_abstraction() {
        let mut engine = AbstractionEngine::default_engine();

        let mut concepts = Vec::new();
        for i in 0..5 {
            concepts.push(make_concept(
                &format!("c{}", i),
                vec![1.0 + i as f32 * 0.01, 0.0],
                vec!["math"],
            ));
        }
        concepts.push(make_concept("other", vec![0.0, 1.0], vec!["text"]));

        let results = engine.run_abstraction(&concepts);
        assert!(!results.is_empty());
        assert!(engine.abstraction_count() > 0);
    }

    #[test]
    fn test_generalize_descriptions() {
        let desc = vec!["sort array ascending", "sort array descending", "sort array random"];
        let gen = AbstractionEngine::generalize_descriptions(&desc);
        assert!(gen.contains("sort"));
        assert!(gen.contains("array"));
        assert!(gen.contains("3 instances"));
    }

    #[test]
    fn test_generalize_descriptions_no_common() {
        let desc = vec!["alpha beta", "gamma delta"];
        let gen = AbstractionEngine::generalize_descriptions(&desc);
        assert!(gen.contains("2 instances"));
    }

    #[test]
    fn test_concept_level_depth() {
        assert_eq!(ConceptLevel::Instance.depth(), 0);
        assert_eq!(ConceptLevel::Pattern.depth(), 1);
        assert_eq!(ConceptLevel::Principle.depth(), 2);
        assert_eq!(ConceptLevel::MetaPrinciple.depth(), 3);
    }

    #[test]
    fn test_cluster_common_tags() {
        let engine = AbstractionEngine::default_engine();
        let concepts = vec![
            make_concept("a", vec![1.0, 0.0], vec!["math", "fast"]),
            make_concept("b", vec![0.95, 0.05], vec!["math", "fast"]),
            make_concept("c", vec![0.9, 0.1], vec!["math", "slow"]),
        ];

        let clusters = engine.scan_for_clusters(&concepts);
        assert!(!clusters.is_empty());
        assert!(clusters[0].common_tags.contains(&"math".to_string()));
    }

    #[test]
    fn test_stats() {
        let mut engine = AbstractionEngine::default_engine();

        let concepts = vec![
            make_concept("a", vec![1.0, 0.0], vec!["math"]),
            make_concept("b", vec![0.95, 0.05], vec!["math"]),
            make_concept("c", vec![0.9, 0.1], vec!["math"]),
        ];

        let clusters = engine.scan_for_clusters(&concepts);
        if !clusters.is_empty() {
            engine.abstract_cluster(&clusters[0], &concepts);
        }

        let stats = engine.stats();
        assert!(stats.total_abstractions > 0 || clusters.is_empty());
    }

    #[test]
    fn test_empty_concepts() {
        let engine = AbstractionEngine::default_engine();
        let clusters = engine.scan_for_clusters(&[]);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_quality_threshold() {
        let config = AbstractionConfig {
            quality_threshold: 1.0, // Impossible threshold
            ..Default::default()
        };
        let mut engine = AbstractionEngine::new(config);

        let concepts = vec![
            make_concept("a", vec![1.0, 0.0], vec!["math"]),
            make_concept("b", vec![0.95, 0.05], vec!["math"]),
            make_concept("c", vec![0.9, 0.1], vec!["math"]),
        ];

        let results = engine.run_abstraction(&concepts);
        assert!(results.is_empty()); // Quality too low for threshold
    }
}
