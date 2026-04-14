//! Persistent Concept Memory via Hull-KV Cache
//!
//! Stores learned logic patterns (bytecode snippets, algorithms, solution templates)
//! indexed by their high-level reasoning embedding. The model can retrieve exact
//! programs without re-reasoning, saving compute cycles across sessions.
//!
//! Architecture:
//!   ConceptMap: reasoning_embedding → concept_id → ConceptEntry
//!   ConceptEntry: metadata + bytecode + usage stats + quality score
//!   Retrieval: cosine similarity between query embedding and stored embeddings
//!   Persistence: serialize to JSON/binary for disk storage

use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Concept types
// ---------------------------------------------------------------------------

/// Unique identifier for a concept.
pub type ConceptId = u64;

/// A learned concept — a reusable pattern, algorithm, or solution template.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConceptEntry {
    /// Unique ID.
    pub id: ConceptId,
    /// Human-readable name.
    pub name: String,
    /// The concept's reasoning embedding (used for retrieval).
    pub embedding: Vec<f32>,
    /// The actual content — bytecode, code snippet, algorithm description.
    pub content: ConceptContent,
    /// Tags for categorization.
    pub tags: Vec<String>,
    /// Quality score (higher = more reliable). Updated with usage.
    pub quality: f32,
    /// Number of times this concept has been used.
    pub usage_count: u32,
    /// Number of successful uses (test passed, correct output).
    pub success_count: u32,
    /// Creation timestamp.
    pub created_at: u64,
    /// Last access timestamp.
    pub last_accessed: u64,
    /// Source of the concept (model-generated, user-provided, etc.).
    pub source: ConceptSource,
}

/// The content of a concept.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ConceptContent {
    /// A code snippet in some language.
    Code { language: String, code: String },
    /// A bytecode program for the CALM VM.
    Bytecode { instructions: Vec<u8>, description: String },
    /// A natural language algorithm description.
    Algorithm { steps: Vec<String>, complexity: String },
    /// A mathematical formula or pattern.
    Formula { latex: String, variables: Vec<String> },
    /// A retrieved web result (for RAG concepts).
    WebResult { url: String, summary: String, content_hash: String },
    /// A test pattern (for mirror test reuse).
    TestPattern { language: String, template: String, applies_to: Vec<String> },
}

/// Source of a concept.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ConceptSource {
    /// Model generated this concept during training.
    ModelGenerated { step: u64, loss_at_creation: f32 },
    /// User explicitly provided this concept.
    UserProvided { user_id: String },
    /// Discovered via self-verification (mirror test).
    SelfVerified { test_result: String },
    /// Retrieved from external source (RAG).
    External { url: String },
}

// ---------------------------------------------------------------------------
// Concept Map
// ---------------------------------------------------------------------------

/// The concept memory store. Maps reasoning embeddings to reusable concepts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConceptMap {
    /// All concepts indexed by ID.
    concepts: HashMap<ConceptId, ConceptEntry>,
    /// Embedding dimensionality.
    embedding_dim: usize,
    /// Next available ID.
    next_id: ConceptId,
    /// Maximum number of concepts (LRU eviction when exceeded).
    max_concepts: usize,
    /// Quality threshold for retrieval (only return concepts above this).
    quality_threshold: f32,
}

impl ConceptMap {
    /// Create a new empty concept map.
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            concepts: HashMap::new(),
            embedding_dim,
            next_id: 1,
            max_concepts: 10000,
            quality_threshold: 0.3,
        }
    }

    /// Create with custom capacity.
    pub fn with_capacity(embedding_dim: usize, max_concepts: usize) -> Self {
        Self {
            concepts: HashMap::new(),
            embedding_dim,
            next_id: 1,
            max_concepts,
            quality_threshold: 0.3,
        }
    }

    /// Store a new concept.
    pub fn store(
        &mut self,
        name: String,
        embedding: Vec<f32>,
        content: ConceptContent,
        tags: Vec<String>,
        source: ConceptSource,
    ) -> ConceptId {
        let id = self.next_id;
        self.next_id += 1;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let entry = ConceptEntry {
            id,
            name,
            embedding,
            content,
            tags,
            quality: 0.5, // Initial quality — will be updated with usage
            usage_count: 0,
            success_count: 0,
            created_at: now,
            last_accessed: now,
            source,
        };

        // Evict lowest-quality concept if at capacity
        if self.concepts.len() >= self.max_concepts {
            self.evict_lowest_quality();
        }

        self.concepts.insert(id, entry);
        id
    }

    /// Retrieve concepts similar to the query embedding.
    ///
    /// Returns concepts sorted by similarity (most similar first).
    /// Only returns concepts above the quality threshold.
    pub fn retrieve(&mut self, query: &[f32], top_k: usize) -> Vec<RetrievedConcept> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut scored: Vec<(ConceptId, f32, f32)> = self.concepts.iter()
            .filter(|(_, c)| c.quality >= self.quality_threshold)
            .map(|(id, c)| {
                let sim = cosine_similarity(query, &c.embedding);
                // Combine similarity with quality score
                let score = sim * 0.7 + c.quality * 0.3;
                (*id, sim, score)
            })
            .collect();

        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored.into_iter()
            .filter_map(|(id, similarity, score)| {
                let entry = self.concepts.get_mut(&id)?;
                entry.usage_count += 1;
                entry.last_accessed = now;
                Some(RetrievedConcept {
                    concept: entry.clone(),
                    similarity,
                    score,
                })
            })
            .collect()
    }

    /// Update a concept's quality based on whether its use was successful.
    pub fn update_quality(&mut self, id: ConceptId, success: bool) {
        if let Some(entry) = self.concepts.get_mut(&id) {
            entry.usage_count += 1;
            entry.success_count += if success { 1 } else { 0 };
            // Exponential moving average of quality
            let alpha = 0.1;
            let new_quality = if entry.usage_count > 0 {
                entry.success_count as f32 / entry.usage_count as f32
            } else {
                0.5
            };
            entry.quality = entry.quality * (1.0 - alpha) + new_quality * alpha;
        }
    }

    /// Get a concept by ID.
    pub fn get(&self, id: ConceptId) -> Option<&ConceptEntry> {
        self.concepts.get(&id)
    }

    /// Get a mutable concept by ID.
    pub fn get_mut(&mut self, id: ConceptId) -> Option<&mut ConceptEntry> {
        self.concepts.get_mut(&id)
    }

    /// Remove a concept.
    pub fn remove(&mut self, id: ConceptId) -> Option<ConceptEntry> {
        self.concepts.remove(&id)
    }

    /// Number of stored concepts.
    pub fn len(&self) -> usize {
        self.concepts.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.concepts.is_empty()
    }

    /// Get all concept IDs.
    pub fn ids(&self) -> Vec<ConceptId> {
        self.concepts.keys().copied().collect()
    }

    /// Search concepts by tag.
    pub fn search_by_tag(&self, tag: &str) -> Vec<&ConceptEntry> {
        self.concepts.values()
            .filter(|c| c.tags.iter().any(|t| t.eq_ignore_ascii_case(tag)))
            .collect()
    }

    /// Get concepts sorted by quality (best first).
    pub fn top_by_quality(&self, n: usize) -> Vec<&ConceptEntry> {
        let mut entries: Vec<_> = self.concepts.values().collect();
        entries.sort_by(|a, b| b.quality.partial_cmp(&a.quality).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(n);
        entries
    }

    /// Evict the lowest-quality concept.
    fn evict_lowest_quality(&mut self) {
        if let Some((&id, _)) = self.concepts.iter()
            .min_by(|a, b| a.1.quality.partial_cmp(&b.1.quality).unwrap_or(std::cmp::Ordering::Equal))
        {
            self.concepts.remove(&id);
        }
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    /// Save the concept map to a file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string(self)
            .map_err(|e| format!("Serialize error: {}", e))?;
        std::fs::write(path, json)
            .map_err(|e| format!("Write error: {}", e))
    }

    /// Load a concept map from a file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("Read error: {}", e))?;
        serde_json::from_str(&json)
            .map_err(|e| format!("Deserialize error: {}", e))
    }
}

// ---------------------------------------------------------------------------
// Retrieved concept
// ---------------------------------------------------------------------------

/// A concept retrieved from the concept map.
#[derive(Debug, Clone)]
pub struct RetrievedConcept {
    pub concept: ConceptEntry,
    pub similarity: f32,
    pub score: f32,
}

// ---------------------------------------------------------------------------
// Hull-KV Cache integration
// ---------------------------------------------------------------------------

/// Bridge between ConceptMap and the Hull-KV cache.
///
/// Concepts can be stored in Hull-KV entries so they're available during
/// inference without loading the full concept map.
pub struct ConceptHullBridge {
    concept_map: ConceptMap,
}

impl ConceptHullBridge {
    pub fn new(concept_map: ConceptMap) -> Self {
        Self { concept_map }
    }

    /// Retrieve relevant concepts for a given reasoning context.
    pub fn retrieve_for_context(&mut self, context_embedding: &[f32], max_results: usize) -> Vec<RetrievedConcept> {
        self.concept_map.retrieve(context_embedding, max_results)
    }

    /// Store a new concept from the inference context.
    pub fn store_concept(
        &mut self,
        name: String,
        embedding: Vec<f32>,
        content: ConceptContent,
        tags: Vec<String>,
        source: ConceptSource,
    ) -> ConceptId {
        self.concept_map.store(name, embedding, content, tags, source)
    }

    /// Get the underlying concept map.
    pub fn concept_map(&self) -> &ConceptMap {
        &self.concept_map
    }

    /// Get mutable reference to the concept map.
    pub fn concept_map_mut(&mut self) -> &mut ConceptMap {
        &mut self.concept_map
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 { return 0.0; }
    dot / (na * nb)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use std::path::PathBuf;
    #[allow(unused_imports)]

    fn make_embedding(dim: usize, value: f32) -> Vec<f32> {
        vec![value; dim]
    }

    #[test]
    fn test_concept_store_and_retrieve() {
        let mut map = ConceptMap::new(64);
        let emb = make_embedding(64, 1.0);
        let id = map.store(
            "binary_search".into(),
            emb.clone(),
            ConceptContent::Code {
                language: "rust".into(),
                code: "fn binary_search(...)".into(),
            },
            vec!["algorithm".into(), "search".into()],
            ConceptSource::ModelGenerated { step: 0, loss_at_creation: 0.5 },
        );
        assert_eq!(id, 1);
        assert_eq!(map.len(), 1);

        // Retrieve by similar embedding
        let query = make_embedding(64, 0.9);
        let results = map.retrieve(&query, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].concept.id, 1);
        assert!(results[0].similarity > 0.99);
    }

    #[test]
    fn test_concept_no_retrieve_below_quality() {
        let mut map = ConceptMap::new(64);
        map.quality_threshold = 0.9;

        let _ = map.store(
            "bad_concept".into(),
            make_embedding(64, 1.0),
            ConceptContent::Algorithm { steps: vec![], complexity: "O(1)".into() },
            vec![],
            ConceptSource::ModelGenerated { step: 0, loss_at_creation: 1.0 },
        );

        // Initial quality is 0.5, below threshold
        let results = map.retrieve(&make_embedding(64, 1.0), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_concept_quality_update() {
        let mut map = ConceptMap::new(64);
        let id = map.store(
            "test_concept".into(),
            make_embedding(64, 1.0),
            ConceptContent::Formula { latex: "E=mc^2".into(), variables: vec!["E".into(), "m".into(), "c".into()] },
            vec!["physics".into()],
            ConceptSource::UserProvided { user_id: "test".into() },
        );

        // Simulate usage: 8 successes, 2 failures
        for _ in 0..8 { map.update_quality(id, true); }
        for _ in 0..2 { map.update_quality(id, false); }

        let concept = map.get(id).unwrap();
        assert!(concept.quality > 0.5, "Quality should increase with mostly successful uses");
        assert_eq!(concept.usage_count, 10);
        assert_eq!(concept.success_count, 8);
    }

    #[test]
    fn test_concept_tag_search() {
        let mut map = ConceptMap::new(64);
        map.store("c1".into(), make_embedding(64, 1.0),
            ConceptContent::Code { language: "rust".into(), code: "".into() },
            vec!["algorithm".into(), "sort".into()],
            ConceptSource::ModelGenerated { step: 0, loss_at_creation: 0.0 });
        map.store("c2".into(), make_embedding(64, 0.5),
            ConceptContent::Code { language: "python".into(), code: "".into() },
            vec!["algorithm".into(), "search".into()],
            ConceptSource::ModelGenerated { step: 1, loss_at_creation: 0.0 });
        map.store("c3".into(), make_embedding(64, 0.3),
            ConceptContent::Code { language: "rust".into(), code: "".into() },
            vec!["math".into()],
            ConceptSource::ModelGenerated { step: 2, loss_at_creation: 0.0 });

        let algo = map.search_by_tag("algorithm");
        assert_eq!(algo.len(), 2);

        let sort = map.search_by_tag("sort");
        assert_eq!(sort.len(), 1);
        assert_eq!(sort[0].name, "c1");
    }

    #[test]
    fn test_concept_eviction() {
        let mut map = ConceptMap::with_capacity(64, 3);

        // Store 3 concepts
        for i in 0..3 {
            map.store(format!("c{}", i), make_embedding(64, i as f32 / 3.0),
                ConceptContent::Algorithm { steps: vec![], complexity: "O(1)".into() },
                vec![],
                ConceptSource::ModelGenerated { step: i, loss_at_creation: 0.0 });
        }
        assert_eq!(map.len(), 3);

        // Store one more — should evict lowest quality
        map.store("c3".into(), make_embedding(64, 1.0),
            ConceptContent::Algorithm { steps: vec![], complexity: "O(n)".into() },
            vec![],
            ConceptSource::ModelGenerated { step: 3, loss_at_creation: 0.0 });
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn test_concept_persistence() {
        let dir = std::env::temp_dir().join("ferrisres_concept_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_concepts.json");

        let mut map = ConceptMap::new(64);
        map.store("test".into(), make_embedding(64, 1.0),
            ConceptContent::Code { language: "rust".into(), code: "fn main() {}".into() },
            vec!["test".into()],
            ConceptSource::ModelGenerated { step: 0, loss_at_creation: 0.0 });

        map.save(&path).expect("save should work");
        let loaded = ConceptMap::load(&path).expect("load should work");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.get(1).unwrap().name, "test");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_concept_hull_bridge() {
        let map = ConceptMap::new(64);
        let mut bridge = ConceptHullBridge::new(map);

        let bridge_id = bridge.store_concept(
            "sort_merge".into(),
            make_embedding(64, 1.0),
            ConceptContent::Code { language: "rust".into(), code: "fn merge_sort()".into() },
            vec!["sort".into()],
            ConceptSource::ModelGenerated { step: 100, loss_at_creation: 0.2 },
        );

        let results = bridge.retrieve_for_context(&make_embedding(64, 0.95), 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].concept.id, bridge_id);
    }

    #[test]
    fn test_concept_top_by_quality() {
        let mut map = ConceptMap::new(64);
        let ids: Vec<_> = (0..5).map(|i| {
            map.store(format!("c{}", i), make_embedding(64, i as f32),
                ConceptContent::Algorithm { steps: vec![], complexity: "O(1)".into() },
                vec![],
                ConceptSource::ModelGenerated { step: i, loss_at_creation: 0.0 })
        }).collect();

        // Boost quality of id 2
        for _ in 0..10 { map.update_quality(ids[2], true); }
        // Lower quality of id 3
        for _ in 0..5 { map.update_quality(ids[3], false); }

        let top = map.top_by_quality(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].id, ids[2]);
    }

    #[test]
    fn test_concept_content_types() {
        let mut map = ConceptMap::new(32);

        map.store("code".into(), make_embedding(32, 0.1),
            ConceptContent::Code { language: "rust".into(), code: "fn foo()".into() },
            vec![], ConceptSource::ModelGenerated { step: 0, loss_at_creation: 0.0 });

        map.store("bytecode".into(), make_embedding(32, 0.2),
            ConceptContent::Bytecode { instructions: vec![1, 2, 3], description: "add".into() },
            vec![], ConceptSource::ModelGenerated { step: 0, loss_at_creation: 0.0 });

        map.store("algo".into(), make_embedding(32, 0.3),
            ConceptContent::Algorithm { steps: vec!["step1".into()], complexity: "O(n log n)".into() },
            vec![], ConceptSource::ModelGenerated { step: 0, loss_at_creation: 0.0 });

        map.store("formula".into(), make_embedding(32, 0.4),
            ConceptContent::Formula { latex: "x^2".into(), variables: vec!["x".into()] },
            vec![], ConceptSource::ModelGenerated { step: 0, loss_at_creation: 0.0 });

        map.store("web".into(), make_embedding(32, 0.5),
            ConceptContent::WebResult { url: "https://example.com".into(), summary: "example".into(), content_hash: "abc".into() },
            vec![], ConceptSource::External { url: "https://example.com".into() });

        map.store("test".into(), make_embedding(32, 0.6),
            ConceptContent::TestPattern { language: "rust".into(), template: "#[test]".into(), applies_to: vec!["functions".into()] },
            vec![], ConceptSource::SelfVerified { test_result: "passed".into() });

        assert_eq!(map.len(), 6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);

        let c = vec![1.0, 1.0];
        assert!((cosine_similarity(&a, &c) - 0.7071).abs() < 0.01);
    }
}
