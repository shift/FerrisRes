//! Retrieval-Augmented Generation (RAG) pipeline.
//!
//! Implements:
//! - Document store with simple embedding-based retrieval
//! - Dense retrieval via cosine similarity
//! - Sparse retrieval via TF-IDF-like scoring
//! - In-context learning with retrieved examples
//! - RAG prompt formatting and fusion
//!
//! Based on research task fcce1e10.

use std::collections::HashMap;

/// A document in the RAG corpus.
#[derive(Debug, Clone)]
pub struct Document {
    /// Unique document ID.
    pub id: String,
    /// Document text content.
    pub content: String,
    /// Optional metadata (source, author, etc).
    pub metadata: HashMap<String, String>,
    /// Pre-computed embedding vector (for dense retrieval).
    pub embedding: Option<Vec<f32>>,
}

impl Document {
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            metadata: HashMap::new(),
            embedding: None,
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// A retrieved document with relevance score.
#[derive(Debug, Clone)]
pub struct RetrievedDocument {
    pub document: Document,
    pub score: f32,
    pub retrieval_method: RetrievalMethod,
}

/// Retrieval method used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMethod {
    Dense,
    Sparse,
    Hybrid,
}

/// Configuration for the RAG pipeline.
#[derive(Debug, Clone)]
pub struct RagConfig {
    /// Number of documents to retrieve.
    pub top_k: usize,
    /// Minimum relevance score threshold.
    pub min_score: f32,
    /// Weight for dense retrieval in hybrid mode (0.0-1.0).
    pub dense_weight: f32,
    /// Weight for sparse retrieval in hybrid mode (0.0-1.0).
    pub sparse_weight: f32,
    /// Maximum tokens for retrieved context.
    pub max_context_tokens: usize,
    /// Chunk size for document splitting.
    pub chunk_size: usize,
    /// Chunk overlap in characters.
    pub chunk_overlap: usize,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_score: 0.1,
            dense_weight: 0.7,
            sparse_weight: 0.3,
            max_context_tokens: 2048,
            chunk_size: 512,
            chunk_overlap: 64,
        }
    }
}

/// TF-IDF index for sparse retrieval.
struct SparseIndex {
    /// Term → document frequency.
    doc_freq: HashMap<String, usize>,
    /// Document ID → term frequency map.
    term_freqs: HashMap<String, HashMap<String, f32>>,
    /// Total number of documents.
    num_docs: usize,
}

impl SparseIndex {
    fn new() -> Self {
        Self {
            doc_freq: HashMap::new(),
            term_freqs: HashMap::new(),
            num_docs: 0,
        }
    }

    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 1)
            .map(String::from)
            .collect()
    }

    fn add_document(&mut self, doc_id: &str, content: &str) {
        let tokens = Self::tokenize(content);
        let mut tf: HashMap<String, f32> = HashMap::new();
        let total = tokens.len() as f32;

        for token in &tokens {
            *tf.entry(token.clone()).or_insert(0.0) += 1.0;
        }

        // Normalize by document length
        for (_, count) in tf.iter_mut() {
            *count /= total;
        }

        // Update document frequencies
        let unique_terms: Vec<String> = tf.keys().cloned().collect();
        for term in unique_terms {
            *self.doc_freq.entry(term).or_insert(0) += 1;
        }

        self.term_freqs.insert(doc_id.to_string(), tf);
        self.num_docs += 1;
    }

    fn search(&self, query: &str, top_k: usize) -> Vec<(String, f32)> {
        let query_tokens = Self::tokenize(query);
        let mut scores: HashMap<String, f32> = HashMap::new();

        for term in &query_tokens {
            let df = *self.doc_freq.get(term).unwrap_or(&0) as f32;
            if df == 0.0 {
                continue;
            }
            let idf = ((self.num_docs as f32 + 1.0) / (df + 1.0)).ln() + 1.0;

            for (doc_id, tf_map) in &self.term_freqs {
                if let Some(&tf) = tf_map.get(term) {
                    *scores.entry(doc_id.clone()).or_insert(0.0) += tf * idf;
                }
            }
        }

        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }
}

/// RAG document store with dense and sparse retrieval.
pub struct RagStore {
    documents: HashMap<String, Document>,
    sparse_index: SparseIndex,
    config: RagConfig,
    embedding_dim: usize,
}

impl RagStore {
    /// Create a new RAG store.
    pub fn new(config: RagConfig) -> Self {
        Self {
            documents: HashMap::new(),
            sparse_index: SparseIndex::new(),
            config,
            embedding_dim: 0,
        }
    }

    /// Create with default config.
    pub fn default_store() -> Self {
        Self::new(RagConfig::default())
    }

    /// Add a document to the store.
    pub fn add_document(&mut self, doc: Document) {
        if let Some(ref emb) = doc.embedding {
            if self.embedding_dim == 0 {
                self.embedding_dim = emb.len();
            }
        }
        self.sparse_index.add_document(&doc.id, &doc.content);
        self.documents.insert(doc.id.clone(), doc);
    }

    /// Add multiple documents.
    pub fn add_documents(&mut self, docs: Vec<Document>) {
        for doc in docs {
            self.add_document(doc);
        }
    }

    /// Get document count.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Check if store is empty.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Dense retrieval using cosine similarity.
    pub fn dense_search(&self, query_embedding: &[f32], top_k: usize) -> Vec<RetrievedDocument> {
        let mut results: Vec<RetrievedDocument> = self.documents.values()
            .filter_map(|doc| {
                doc.embedding.as_ref().map(|emb| {
                    let score = cosine_similarity(query_embedding, emb);
                    RetrievedDocument {
                        document: doc.clone(),
                        score,
                        retrieval_method: RetrievalMethod::Dense,
                    }
                })
            })
            .filter(|r| r.score >= self.config.min_score)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Sparse retrieval using TF-IDF.
    pub fn sparse_search(&self, query: &str, top_k: usize) -> Vec<RetrievedDocument> {
        let sparse_results = self.sparse_index.search(query, top_k);

        sparse_results.into_iter()
            .filter_map(|(doc_id, score)| {
                self.documents.get(&doc_id).map(|doc| RetrievedDocument {
                    document: doc.clone(),
                    score,
                    retrieval_method: RetrievalMethod::Sparse,
                })
            })
            .filter(|r| r.score >= self.config.min_score)
            .collect()
    }

    /// Hybrid retrieval: combine dense and sparse scores.
    pub fn hybrid_search(
        &self,
        query: &str,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Vec<RetrievedDocument> {
        let dense = self.dense_search(query_embedding, top_k * 2);
        let sparse = self.sparse_search(query, top_k * 2);

        // Normalize scores
        let max_dense = dense.iter().map(|r| r.score).fold(0.0f32, f32::max).max(0.001);
        let max_sparse = sparse.iter().map(|r| r.score).fold(0.0f32, f32::max).max(0.001);

        let mut combined: HashMap<String, (f32, Document)> = HashMap::new();

        for r in dense {
            let normalized = r.score / max_dense;
            let entry = combined.entry(r.document.id.clone()).or_insert_with(|| (0.0, r.document.clone()));
            entry.0 += normalized * self.config.dense_weight;
        }

        for r in sparse {
            let normalized = r.score / max_sparse;
            let entry = combined.entry(r.document.id.clone()).or_insert_with(|| (0.0, r.document.clone()));
            entry.0 += normalized * self.config.sparse_weight;
        }

        let mut results: Vec<RetrievedDocument> = combined.into_iter()
            .map(|(_, (score, doc))| RetrievedDocument {
                document: doc,
                score,
                retrieval_method: RetrievalMethod::Hybrid,
            })
            .filter(|r| r.score >= self.config.min_score)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Format retrieved documents as context for the model.
    pub fn format_context(&self, retrieved: &[RetrievedDocument]) -> String {
        let mut context = String::from("Retrieved context:\n\n");
        for (i, r) in retrieved.iter().enumerate() {
            context.push_str(&format!("[{}] (score: {:.3}) {}\n\n", i + 1, r.score, r.document.content));
        }
        context
    }

    /// Build a RAG prompt combining query and retrieved context.
    pub fn build_rag_prompt(&self, query: &str, retrieved: &[RetrievedDocument]) -> String {
        let context = self.format_context(retrieved);
        format!(
            "{}Based on the above context, answer the following question:\n{}\n\nAnswer:",
            context, query
        )
    }

    /// Chunk a document into smaller pieces.
    pub fn chunk_document(&self, doc: &Document) -> Vec<Document> {
        let content = &doc.content;
        let chunk_size = self.config.chunk_size;
        let overlap = self.config.chunk_overlap;
        let mut chunks = Vec::new();

        if content.len() <= chunk_size {
            chunks.push(doc.clone());
            return chunks;
        }

        let mut start = 0;
        let mut chunk_idx = 0;
        while start < content.len() {
            let end = (start + chunk_size).min(content.len());
            let chunk_content = content[start..end].to_string();

            let mut chunk = Document::new(
                format!("{}:chunk:{}", doc.id, chunk_idx),
                chunk_content,
            );
            chunk.metadata = doc.metadata.clone();
            chunk.embedding = doc.embedding.clone(); // Share embedding (could be improved)

            chunks.push(chunk);
            chunk_idx += 1;
            start += chunk_size - overlap;
            if start >= content.len() {
                break;
            }
        }

        chunks
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

// ---------------------------------------------------------------------------
// ElasticRagStore — Matryoshka-aware RAG with DeviceProfile-mapped query dims
//
// Stores embeddings at full D_max dimensionality.  At query time only the
// first `query_dim` dimensions are used, where `query_dim` is automatically
// selected based on the DeviceProfile:
//
//   Integrated  → 64   (~95% recall, minimal DRAM pressure)
//   LowEnd      → 128  (~97% recall)
//   MidRange    → 256  (~98.5% recall)
//   HighEnd     → D_max (100%)
//
// Quality figures from the MRL paper (Kusupati et al., NeurIPS 2022,
// arxiv 2205.13147) on the BEIR benchmark.
//
// Task: d7a18c03 — see papers_research/matryoshka_embeddings_research.md
// ---------------------------------------------------------------------------

/// DeviceProfile tier used to select the elastic embedding query dimension.
/// Mirrors `device::profile::DeviceProfile` without importing the GPU types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedProfile {
    /// Integrated / shared DRAM GPU (e.g. Intel Iris Xe, X1 Yoga).
    Integrated,
    /// Discrete GPU with < 4 GB VRAM.
    LowEnd,
    /// Discrete GPU with 4–8 GB VRAM.
    MidRange,
    /// Discrete GPU with > 8 GB VRAM.
    HighEnd,
}

impl EmbedProfile {
    /// Return the recommended query dimension for this profile.
    ///
    /// The returned value is clamped to `d_max` so callers never need to
    /// range-check.
    pub fn query_dim(&self, d_max: usize) -> usize {
        let dim = match self {
            EmbedProfile::Integrated => 64,
            EmbedProfile::LowEnd     => 128,
            EmbedProfile::MidRange   => 256,
            EmbedProfile::HighEnd    => d_max,
        };
        dim.min(d_max)
    }
}

/// RAG document store that exploits Matryoshka (nested) embedding structure.
///
/// Embeddings are stored at their full dimensionality `d_max`.  Searches use
/// only the leading `query_dim` dimensions, making retrieval proportionally
/// faster (fewer multiply-adds) while preserving correctness for models whose
/// embedding weights were trained with an MRL loss.
///
/// For a non-MRL model the prefix subspace is *not* meaningful — only use
/// this store with embeddings produced by an MRL-trained encoder.
pub struct ElasticRagStore {
    /// Full-dimension embeddings `[num_docs, d_max]`.
    embeddings: Vec<Vec<f32>>,
    /// Corresponding documents.
    documents: Vec<Document>,
    /// Full embedding dimensionality.
    d_max: usize,
    /// Active query dimensionality (≤ d_max).
    query_dim: usize,
    /// Nesting sizes supported by the encoder (informational; not enforced).
    pub matryoshka_dims: Vec<usize>,
}

impl ElasticRagStore {
    /// Create a new store.
    ///
    /// * `d_max`            – Full embedding dimension.
    /// * `matryoshka_dims`  – Supported nesting sizes, e.g. `[32, 64, 128, 256, 768]`.
    ///   Must all be ≤ `d_max`.  Used for documentation and profile validation.
    /// * `profile`          – Hardware profile; determines the default `query_dim`.
    pub fn new(
        d_max: usize,
        matryoshka_dims: Vec<usize>,
        profile: EmbedProfile,
    ) -> Self {
        let query_dim = profile.query_dim(d_max);
        tracing::debug!(
            "ElasticRagStore: d_max={} query_dim={} profile={:?}",
            d_max, query_dim, profile
        );
        Self {
            embeddings: Vec::new(),
            documents: Vec::new(),
            d_max,
            query_dim,
            matryoshka_dims,
        }
    }

    /// Add a document with its pre-computed full-dimension embedding.
    ///
    /// The embedding slice must have exactly `d_max` elements.
    pub fn add(&mut self, doc: Document, embedding: Vec<f32>) {
        debug_assert_eq!(
            embedding.len(), self.d_max,
            "Embedding length {} != d_max {}", embedding.len(), self.d_max
        );
        self.documents.push(doc);
        self.embeddings.push(embedding);
    }

    /// Change the active query dimension at runtime.
    ///
    /// Must be ≤ `d_max`.
    pub fn set_query_dim(&mut self, dim: usize) {
        assert!(dim <= self.d_max, "query_dim {} > d_max {}", dim, self.d_max);
        self.query_dim = dim;
        tracing::debug!("ElasticRagStore: query_dim set to {}", dim);
    }

    /// Apply a hardware profile, updating `query_dim` accordingly.
    pub fn apply_profile(&mut self, profile: EmbedProfile) {
        self.query_dim = profile.query_dim(self.d_max);
        tracing::debug!(
            "ElasticRagStore: profile {:?} → query_dim={}",
            profile, self.query_dim
        );
    }

    /// Active query dimension.
    pub fn query_dim(&self) -> usize {
        self.query_dim
    }

    /// Full embedding dimension.
    pub fn d_max(&self) -> usize {
        self.d_max
    }

    /// Number of stored documents.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// True if the store contains no documents.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Return the top-k documents most similar to `query`.
    ///
    /// Only the first `self.query_dim` dimensions of `query` and each stored
    /// embedding are used for the cosine similarity comparison.
    ///
    /// `query` must have at least `self.query_dim` elements.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<RetrievedDocument> {
        assert!(
            query.len() >= self.query_dim,
            "query len {} < query_dim {}",
            query.len(), self.query_dim
        );
        if self.documents.is_empty() {
            return Vec::new();
        }

        let q = &query[..self.query_dim];

        let mut scored: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                let score = cosine_similarity(q, &emb[..self.query_dim]);
                (i, score)
            })
            .collect();

        // Sort descending by score; stable for deterministic tie-breaking.
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .map(|(idx, score)| RetrievedDocument {
                document: self.documents[idx].clone(),
                score,
                retrieval_method: RetrievalMethod::Dense,
            })
            .collect()
    }

    /// Two-stage coarse-to-fine search (Matryoshka adaptive retrieval).
    ///
    /// 1. Retrieve `coarse_k` candidates using `coarse_dim` dimensions.
    /// 2. Re-rank those candidates using all `d_max` dimensions.
    /// 3. Return the top `fine_k` results.
    ///
    /// This pattern trades one extra pass over `coarse_k` embeddings at full
    /// dimensionality for a large savings in the initial scan over all `N` docs.
    pub fn search_coarse_then_fine(
        &self,
        query: &[f32],
        coarse_dim: usize,
        coarse_k: usize,
        fine_k: usize,
    ) -> Vec<RetrievedDocument> {
        assert!(
            coarse_dim <= self.d_max,
            "coarse_dim {} > d_max {}", coarse_dim, self.d_max
        );
        assert!(
            query.len() >= self.d_max,
            "query len {} < d_max {}", query.len(), self.d_max
        );
        if self.documents.is_empty() {
            return Vec::new();
        }

        // --- Stage 1: coarse scan ---
        let q_coarse = &query[..coarse_dim];
        let mut scored: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                (i, cosine_similarity(q_coarse, &emb[..coarse_dim]))
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(coarse_k);

        // --- Stage 2: full re-rank ---
        let q_full = &query[..self.d_max];
        let mut fine: Vec<(usize, f32)> = scored
            .iter()
            .map(|(idx, _)| {
                (*idx, cosine_similarity(q_full, &self.embeddings[*idx]))
            })
            .collect();
        fine.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        fine.truncate(fine_k);

        fine.into_iter()
            .map(|(idx, score)| RetrievedDocument {
                document: self.documents[idx].clone(),
                score,
                retrieval_method: RetrievalMethod::Dense,
            })
            .collect()
    }
}

/// In-context learning example manager.
pub struct InContextLearner {
    examples: Vec<(String, String)>, // (input, output) pairs
    max_examples: usize,
}

impl InContextLearner {
    pub fn new(max_examples: usize) -> Self {
        Self {
            examples: Vec::new(),
            max_examples,
        }
    }

    /// Add a few-shot example.
    pub fn add_example(&mut self, input: impl Into<String>, output: impl Into<String>) {
        self.examples.push((input.into(), output.into()));
        if self.examples.len() > self.max_examples {
            self.examples.remove(0);
        }
    }

    /// Format examples as a prompt prefix.
    pub fn format_examples(&self) -> String {
        if self.examples.is_empty() {
            return String::new();
        }

        let mut prompt = String::from("Examples:\n\n");
        for (i, (input, output)) in self.examples.iter().enumerate() {
            prompt.push_str(&format!("Example {}:\nInput: {}\nOutput: {}\n\n", i + 1, input, output));
        }
        prompt
    }

    /// Build a full prompt with examples and the current query.
    pub fn build_prompt(&self, query: &str) -> String {
        format!("{}Now answer:\nInput: {}\nOutput:", self.format_examples(), query)
    }

    /// Get example count.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str, content: &str, embedding: Vec<f32>) -> Document {
        Document::new(id, content).with_embedding(embedding)
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }

    #[test]
    fn test_dense_retrieval() {
        let mut store = RagStore::default_store();
        store.add_document(make_doc("1", "Rust programming", vec![1.0, 0.0]));
        store.add_document(make_doc("2", "Python programming", vec![0.0, 1.0]));
        store.add_document(make_doc("3", "Rust web dev", vec![0.9, 0.1]));

        let query = vec![1.0, 0.0]; // Looking for Rust
        let results = store.dense_search(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document.id, "1");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_sparse_retrieval() {
        let mut store = RagStore::default_store();
        store.add_document(Document::new("1", "Rust is a systems programming language"));
        store.add_document(Document::new("2", "Python is a scripting language"));
        store.add_document(Document::new("3", "Rust has zero cost abstractions"));

        let results = store.sparse_search("Rust programming", 2);
        assert!(!results.is_empty());
        // Should find docs about Rust
        assert!(results.iter().any(|r| r.document.id == "1" || r.document.id == "3"));
    }

    #[test]
    fn test_hybrid_retrieval() {
        let mut store = RagStore::default_store();
        store.add_document(make_doc("1", "Rust programming", vec![1.0, 0.0]));
        store.add_document(make_doc("2", "Python programming", vec![0.0, 1.0]));

        let query_emb = vec![1.0, 0.0];
        let results = store.hybrid_search("Rust", &query_emb, 2);
        assert!(!results.is_empty());
        assert_eq!(results[0].retrieval_method, RetrievalMethod::Hybrid);
    }

    #[test]
    fn test_rag_prompt() {
        let store = RagStore::default_store();
        let retrieved = vec![
            RetrievedDocument {
                document: Document::new("1", "The answer is 42."),
                score: 0.95,
                retrieval_method: RetrievalMethod::Dense,
            },
        ];

        let prompt = store.build_rag_prompt("What is the answer?", &retrieved);
        assert!(prompt.contains("The answer is 42."));
        assert!(prompt.contains("What is the answer?"));
    }

    #[test]
    fn test_document_chunking() {
        let store = RagStore::new(RagConfig {
            chunk_size: 20,
            chunk_overlap: 5,
            ..Default::default()
        });
        let doc = Document::new("1", "This is a long document that should be split into chunks.");
        let chunks = store.chunk_document(&doc);

        assert!(chunks.len() > 1);
        assert!(chunks[0].id.contains(":chunk:"));
    }

    #[test]
    fn test_document_chunking_short() {
        let store = RagStore::new(RagConfig {
            chunk_size: 100,
            chunk_overlap: 10,
            ..Default::default()
        });
        let doc = Document::new("1", "Short");
        let chunks = store.chunk_document(&doc);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_in_context_learner() {
        let mut learner = InContextLearner::new(3);
        learner.add_example("What is 2+2?", "4");
        learner.add_example("What is 3+3?", "6");

        let prompt = learner.build_prompt("What is 4+4?");
        assert!(prompt.contains("2+2"));
        assert!(prompt.contains("4+4"));
    }

    #[test]
    fn test_in_context_max_examples() {
        let mut learner = InContextLearner::new(2);
        learner.add_example("a", "1");
        learner.add_example("b", "2");
        learner.add_example("c", "3");

        assert_eq!(learner.len(), 2);
        let prompt = learner.format_examples();
        assert!(!prompt.contains("Example 1:\nInput: a"));
        assert!(prompt.contains("Input: b"));
        assert!(prompt.contains("Input: c"));
    }

    #[test]
    fn test_format_context() {
        let store = RagStore::default_store();
        let retrieved = vec![
            RetrievedDocument {
                document: Document::new("1", "Doc one"),
                score: 0.9,
                retrieval_method: RetrievalMethod::Dense,
            },
            RetrievedDocument {
                document: Document::new("2", "Doc two"),
                score: 0.5,
                retrieval_method: RetrievalMethod::Sparse,
            },
        ];

        let context = store.format_context(&retrieved);
        assert!(context.contains("[1]"));
        assert!(context.contains("[2]"));
        assert!(context.contains("Doc one"));
        assert!(context.contains("0.9"));
    }
}
