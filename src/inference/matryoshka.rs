//! Matryoshka Representation Learning — elastic nested embeddings.
//!
//! Implements the key idea from Kusupati et al., 2022:
//! "Matryoshka Representation Learning: Flexible Representations at Any Granularity"
//!
//! Embeddings are trained so that any prefix dimension [:d] is a valid representation.
//! This enables elastic RAG: truncate embeddings to match device capability.

use crate::device::profile::DeviceProfile;

/// Dimensions supported for Matryoshka truncation.
pub const DEFAULT_MATRYOSHKA_DIMS: [usize; 5] = [32, 64, 128, 256, 768];

/// Elastic RAG store using Matryoshka embeddings.
///
/// Stores full-dimensional embeddings but retrieves using only the first `query_dim`
/// dimensions, enabling adaptive dimensionality based on device capability.
pub struct ElasticRagStore {
    /// Stored embeddings (full dimension).
    embeddings: Vec<Vec<f32>>,
    /// Document IDs corresponding to embeddings.
    doc_ids: Vec<String>,
    /// Current query dimension (can be changed at runtime).
    query_dim: usize,
    /// Nesting dimensions supported.
    matryoshka_dims: Vec<usize>,
    /// Full embedding dimension.
    full_dim: usize,
}

impl ElasticRagStore {
    /// Create a new elastic store.
    pub fn new(full_dim: usize, matryoshka_dims: Vec<usize>) -> Self {
        Self {
            embeddings: Vec::new(),
            doc_ids: Vec::new(),
            query_dim: full_dim,
            matryoshka_dims,
            full_dim,
        }
    }

    /// Create with default Matryoshka dimensions.
    pub fn default_store() -> Self {
        Self::new(768, DEFAULT_MATRYOSHKA_DIMS.to_vec())
    }

    /// Set query dimension based on device profile.
    pub fn set_query_dim_for_profile(&mut self, profile: &DeviceProfile) {
        self.query_dim = match profile {
            DeviceProfile::Integrated => 64,
            DeviceProfile::LowEnd => 128,
            DeviceProfile::MidRange => 256,
            DeviceProfile::HighEnd => self.matryoshka_dims.last().copied().unwrap_or(self.full_dim),
        };
        tracing::info!("Matryoshka: set query_dim={} for profile {:?}", self.query_dim, profile);
    }

    /// Set query dimension explicitly.
    pub fn set_query_dim(&mut self, dim: usize) {
        assert!(dim <= self.full_dim, "query_dim {} exceeds full_dim {}", dim, self.full_dim);
        self.query_dim = dim;
    }

    /// Add a document embedding (full dimension).
    pub fn add(&mut self, doc_id: impl Into<String>, embedding: Vec<f32>) {
        assert_eq!(embedding.len(), self.full_dim, "embedding dimension mismatch");
        self.embeddings.push(embedding);
        self.doc_ids.push(doc_id.into());
    }

    /// Search for top-k nearest documents using truncated Matryoshka embeddings.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        let q_dim = self.query_dim.min(query.len());
        let q = &query[..q_dim];

        // Compute query norm
        let q_norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);

        let mut scores: Vec<(usize, f32)> = self.embeddings.iter().enumerate().map(|(i, emb)| {
            let e = &emb[..q_dim];
            let e_norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            let dot: f32 = q.iter().zip(e.iter()).map(|(a, b)| a * b).sum();
            (i, dot / (q_norm * e_norm))
        }).collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        scores.into_iter()
            .map(|(i, score)| (self.doc_ids[i].clone(), score))
            .collect()
    }

    /// Get the current query dimension.
    pub fn query_dim(&self) -> usize {
        self.query_dim
    }

    /// Get the full embedding dimension.
    pub fn full_dim(&self) -> usize {
        self.full_dim
    }

    /// Number of stored documents.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Compute the storage savings ratio from using truncated embeddings.
    pub fn compression_ratio(&self) -> f32 {
        self.full_dim as f32 / self.query_dim as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elastic_store_basic() {
        let mut store = ElasticRagStore::new(8, vec![2, 4, 8]);
        store.add("doc1", vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        store.add("doc2", vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_elastic_search_full_dim() {
        let mut store = ElasticRagStore::new(4, vec![2, 4]);
        store.add("a", vec![1.0, 0.0, 1.0, 0.0]);
        store.add("b", vec![0.0, 1.0, 0.0, 1.0]);
        store.set_query_dim(4);

        let query = vec![1.0, 0.0, 1.0, 0.0];
        let results = store.search(&query, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "a");
    }

    #[test]
    fn test_elastic_search_truncated() {
        let mut store = ElasticRagStore::new(4, vec![2, 4]);
        // doc a: [1,0,*,*] — high sim to [1,0] in dim 2
        store.add("a", vec![1.0, 0.0, 0.0, 0.0]);
        // doc b: [0,1,*,*] — low sim to [1,0] in dim 2
        store.add("b", vec![0.0, 1.0, 1.0, 1.0]);
        store.set_query_dim(2);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = store.search(&query, 2);
        assert_eq!(results[0].0, "a"); // a should rank first in dim 2
    }

    #[test]
    fn test_compression_ratio() {
        let mut store = ElasticRagStore::new(768, vec![32, 64, 128, 256, 768]);
        store.set_query_dim(64);
        assert!((store.compression_ratio() - 12.0).abs() < 0.01);
    }

    #[test]
    fn test_default_dims() {
        assert_eq!(DEFAULT_MATRYOSHKA_DIMS.len(), 5);
        assert_eq!(DEFAULT_MATRYOSHKA_DIMS[0], 32);
        assert_eq!(DEFAULT_MATRYOSHKA_DIMS[4], 768);
    }
}
