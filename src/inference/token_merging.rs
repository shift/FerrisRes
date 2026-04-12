//! Token Merging (ToMe) — training-free visual token reduction.
//!
//! Implements the bipartite soft matching algorithm from:
//! Bolya et al., "Token Merging: Your ViT But Faster", ICLR 2023
//!
//! At each transformer layer, finds the top-r most similar token pairs using
//! bipartite matching on key vectors, merges them (weighted average), and
//! proceeds with fewer tokens. **No retraining required.**
//!
//! # Algorithm
//! 1. Split tokens into two sets (even/odd indices)
//! 2. Compute pairwise cosine similarity between sets
//! 3. Greedy match: pick the most similar pair, remove both, repeat r times
//! 4. Merge matched pairs via weighted average based on token_size

/// Configuration for Token Merging.
#[derive(Debug, Clone)]
pub struct ToMeConfig {
    /// Number of token pairs to merge per layer.
    /// Higher values = more compression but more quality loss.
    /// Typical: 8–32 for ViT-B/16.
    pub merge_tokens: usize,
    /// Whether merging is enabled.
    pub enabled: bool,
}

impl Default for ToMeConfig {
    fn default() -> Self {
        Self {
            merge_tokens: 8,
            enabled: false,
        }
    }
}

impl ToMeConfig {
    /// Create a new config that merges `r` tokens per layer.
    pub fn new(r: usize) -> Self {
        Self {
            merge_tokens: r,
            enabled: true,
        }
    }
}

/// Tracks how many original tokens each merged token represents.
/// Used for size-weighted attention after merging.
#[derive(Debug, Clone)]
pub struct TokenSizes {
    /// Size[i] = number of original tokens represented by token i.
    sizes: Vec<f32>,
}

impl TokenSizes {
    /// Create uniform sizes (1.0 per token).
    pub fn uniform(n: usize) -> Self {
        Self {
            sizes: vec![1.0; n],
        }
    }

    /// Get sizes slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.sizes
    }

    /// Get mutable sizes slice.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.sizes
    }

    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.sizes.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }

    /// Get size at index.
    pub fn get(&self, idx: usize) -> f32 {
        self.sizes[idx]
    }
}

/// Result of a bipartite match.
#[derive(Debug, Clone, Copy)]
struct Match {
    /// Index in set A.
    a: usize,
    /// Index in set B.
    b: usize,
    /// Similarity score.
    score: f32,
}

/// Token Merging engine. CPU-side bipartite soft matching.
///
/// For typical visual token counts (N ≤ 500), CPU-side matching is fast enough.
/// For larger sequences, a GPU kernel variant can be added later.
pub struct ToMeMerger {
    config: ToMeConfig,
}

impl ToMeMerger {
    /// Create a new merger with the given config.
    pub fn new(config: ToMeConfig) -> Self {
        Self { config }
    }

    /// Create a disabled merger (no-op).
    pub fn disabled() -> Self {
        Self {
            config: ToMeConfig::default(),
        }
    }

    /// Get the config.
    pub fn config(&self) -> &ToMeConfig {
        &self.config
    }

    /// Check if merging is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Merge tokens using bipartite soft matching.
    ///
    /// # Arguments
    /// * `tokens` - Token representations, shape [N × D], row-major
    /// * `sizes` - Token sizes (how many original tokens each represents)
    /// * `n_tokens` - Number of tokens (N)
    /// * `dim` - Hidden dimension (D)
    ///
    /// # Returns
    /// (merged_tokens, merged_sizes) with N - r tokens.
    pub fn merge(
        &self,
        tokens: &[f32],
        sizes: &TokenSizes,
        n_tokens: usize,
        dim: usize,
    ) -> (Vec<f32>, TokenSizes) {
        if !self.config.enabled || self.config.merge_tokens == 0 || n_tokens <= 2 {
            return (tokens.to_vec(), sizes.clone());
        }

        let r = self.config.merge_tokens.min(n_tokens / 2);

        // Split tokens into two sets: even indices (A) and odd indices (B)
        let set_a_indices: Vec<usize> = (0..n_tokens).step_by(2).collect();
        let set_b_indices: Vec<usize> = (1..n_tokens).step_by(2).collect();

        // Compute pairwise cosine similarity between A and B
        let matches = bipartite_match(
            tokens, dim,
            &set_a_indices, &set_b_indices,
            r,
        );

        // Build merge map: for each matched pair, merge B into A
        let mut merged_into = vec![None; n_tokens]; // merged_into[b] = Some(a)
        for m in &matches {
            merged_into[m.b] = Some(m.a);
        }

        // Collect surviving tokens (unmatched A + unmatched B + merged pairs)
        let mut result_tokens = Vec::new();
        let mut result_sizes = Vec::new();
        let mut merged_a_written = vec![false; n_tokens];

        // First pass: handle matched pairs
        for m in &matches {
            let a = m.a;
            let b = m.b;
            let size_a = sizes.get(a);
            let size_b = sizes.get(b);
            let total_size = size_a + size_b;

            // Weighted average
            for d in 0..dim {
                let val_a = tokens[a * dim + d] * size_a;
                let val_b = tokens[b * dim + d] * size_b;
                result_tokens.push((val_a + val_b) / total_size);
            }
            result_sizes.push(total_size);
            merged_a_written[a] = true;
        }

        // Second pass: add unmatched tokens from A
        for &a in &set_a_indices {
            if !merged_a_written[a] {
                for d in 0..dim {
                    result_tokens.push(tokens[a * dim + d]);
                }
                result_sizes.push(sizes.get(a));
            }
        }

        // Third pass: add unmatched tokens from B
        for &b in &set_b_indices {
            if merged_into[b].is_none() {
                for d in 0..dim {
                    result_tokens.push(tokens[b * dim + d]);
                }
                result_sizes.push(sizes.get(b));
            }
        }

        tracing::debug!(
            "ToMe: {} tokens → {} tokens (merged {} pairs)",
            n_tokens,
            result_sizes.len(),
            r,
        );

        (
            result_tokens,
            TokenSizes { sizes: result_sizes },
        )
    }

    /// Merge keys specifically for attention (uses key vectors for similarity).
    /// Returns merged keys and updated token sizes.
    pub fn merge_keys(
        &self,
        keys: &[f32],    // [N, D_head]
        values: &[f32],  // [N, D]
        sizes: &TokenSizes,
        n_tokens: usize,
        d_head: usize,
        d_model: usize,
    ) -> (Vec<f32>, Vec<f32>, TokenSizes) {
        if !self.config.enabled || self.config.merge_tokens == 0 || n_tokens <= 2 {
            return (keys.to_vec(), values.to_vec(), sizes.clone());
        }

        let r = self.config.merge_tokens.min(n_tokens / 2);

        let set_a_indices: Vec<usize> = (0..n_tokens).step_by(2).collect();
        let set_b_indices: Vec<usize> = (1..n_tokens).step_by(2).collect();

        let matches = bipartite_match(
            keys, d_head,
            &set_a_indices, &set_b_indices,
            r,
        );

        let mut merged_into = vec![None; n_tokens];
        for m in &matches {
            merged_into[m.b] = Some(m.a);
        }

        let mut result_keys = Vec::new();
        let mut result_values = Vec::new();
        let mut result_sizes = Vec::new();
        let mut merged_a_written = vec![false; n_tokens];

        // Matched pairs
        for m in &matches {
            let a = m.a;
            let b = m.b;
            let size_a = sizes.get(a);
            let size_b = sizes.get(b);
            let total = size_a + size_b;

            for d in 0..d_head {
                result_keys.push((keys[a * d_head + d] * size_a + keys[b * d_head + d] * size_b) / total);
            }
            for d in 0..d_model {
                result_values.push((values[a * d_model + d] * size_a + values[b * d_model + d] * size_b) / total);
            }
            result_sizes.push(total);
            merged_a_written[a] = true;
        }

        // Unmatched A
        for &a in &set_a_indices {
            if !merged_a_written[a] {
                for d in 0..d_head { result_keys.push(keys[a * d_head + d]); }
                for d in 0..d_model { result_values.push(values[a * d_model + d]); }
                result_sizes.push(sizes.get(a));
            }
        }

        // Unmatched B
        for &b in &set_b_indices {
            if merged_into[b].is_none() {
                for d in 0..d_head { result_keys.push(keys[b * d_head + d]); }
                for d in 0..d_model { result_values.push(values[b * d_model + d]); }
                result_sizes.push(sizes.get(b));
            }
        }

        tracing::debug!(
            "ToMe merge_keys: {} → {} tokens ({} pairs merged)",
            n_tokens, result_sizes.len(), r,
        );

        (result_keys, result_values, TokenSizes { sizes: result_sizes })
    }
}

/// Bipartite soft matching: greedy algorithm.
///
/// Splits tokens into sets A and B by index, computes cosine similarity
/// between all A-B pairs, greedily selects the top-r pairs.
fn bipartite_match(
    tokens: &[f32],
    dim: usize,
    set_a: &[usize],
    set_b: &[usize],
    r: usize,
) -> Vec<Match> {
    let n_a = set_a.len();
    let n_b = set_b.len();

    // Pre-compute norms
    let norms: Vec<f32> = (0..set_a.len() + set_b.len())
        .map(|i| {
            let idx = if i < n_a { set_a[i] } else { set_b[i - n_a] };
            let mut sq = 0.0f32;
            for d in 0..dim {
                sq += tokens[idx * dim + d] * tokens[idx * dim + d];
            }
            sq.sqrt().max(1e-8)
        })
        .collect();

    // Compute all pairwise similarities
    let mut similarities: Vec<Match> = Vec::with_capacity(n_a * n_b);
    for (i, &a) in set_a.iter().enumerate() {
        for (j, &b) in set_b.iter().enumerate() {
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += tokens[a * dim + d] * tokens[b * dim + d];
            }
            let sim = dot / (norms[i] * norms[n_a + j]);
            similarities.push(Match { a, b, score: sim });
        }
    }

    // Sort by similarity descending
    similarities.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy match: take top-r non-conflicting pairs
    let mut used_a = vec![false; set_a.iter().max().unwrap_or(&0) + 1];
    let mut used_b = vec![false; set_b.iter().max().unwrap_or(&0) + 1];
    let mut result = Vec::with_capacity(r);

    for m in &similarities {
        if result.len() >= r {
            break;
        }
        if !used_a[m.a] && !used_b[m.b] {
            result.push(*m);
            used_a[m.a] = true;
            used_b[m.b] = true;
        }
    }

    result
}

/// Compute cosine similarity between two vectors.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    dot / (norm_a.sqrt().max(1e-8) * norm_b.sqrt().max(1e-8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tome_disabled() {
        let merger = ToMeMerger::disabled();
        assert!(!merger.is_enabled());

        let tokens = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sizes = TokenSizes::uniform(3);
        let (merged, merged_sizes) = merger.merge(&tokens, &sizes, 3, 2);
        assert_eq!(merged, tokens);
        assert_eq!(merged_sizes.len(), 3);
    }

    #[test]
    fn test_tome_merge_reduces_count() {
        let config = ToMeConfig::new(2);
        let merger = ToMeMerger::new(config);

        // 6 tokens, dim=4, merge 2 pairs → 4 tokens
        let tokens = vec![1.0; 24]; // 6 × 4
        let sizes = TokenSizes::uniform(6);
        let (merged, merged_sizes) = merger.merge(&tokens, &sizes, 6, 4);
        assert_eq!(merged_sizes.len(), 4); // 6 - 2 = 4
        assert_eq!(merged.len(), 4 * 4);
    }

    #[test]
    fn test_tome_merge_preserves_similar() {
        let config = ToMeConfig::new(1);
        let merger = ToMeMerger::new(config);

        // 4 tokens dim=2: A0=[1,0], A1=[0,1], B0=[0.99,0], B1=[0,0.99]
        // A0 and B0 should match (high similarity)
        let tokens = vec![
            1.0, 0.0,   // A0 (index 0)
            0.99, 0.0,  // B0 (index 1)
            0.0, 1.0,   // A1 (index 2)
            0.0, 0.99,  // B1 (index 3)
        ];
        let sizes = TokenSizes::uniform(4);
        let (merged, merged_sizes) = merger.merge(&tokens, &sizes, 4, 2);

        assert_eq!(merged_sizes.len(), 3); // 4 - 1 = 3
        // The merged token should be the average of A0 and B0
        // which are both ~[1, 0]
        assert_eq!(merged.len(), 3 * 2);
        let merged_0 = &merged[0..2];
        assert!((merged_0[0] - 0.995).abs() < 0.01);
        assert!(merged_0[1].abs() < 0.01);
    }

    #[test]
    fn test_tome_merge_too_few_tokens() {
        let config = ToMeConfig::new(5);
        let merger = ToMeMerger::new(config);

        // 2 tokens: guard catches n_tokens <= 2, returns unchanged
        let tokens = vec![1.0, 2.0];
        let sizes = TokenSizes::uniform(2);
        let (merged, merged_sizes) = merger.merge(&tokens, &sizes, 2, 1);
        assert_eq!(merged_sizes.len(), 2); // unchanged — too few to merge
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_tome_merge_keys() {
        let config = ToMeConfig::new(1);
        let merger = ToMeMerger::new(config);

        let keys = vec![1.0, 0.0, 0.99, 0.0, 0.0, 1.0, 0.0, 0.99]; // 4 × d_head=2
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 4 × d_model=3
        let sizes = TokenSizes::uniform(4);

        let (mk, mv, ms) = merger.merge_keys(&keys, &values, &sizes, 4, 2, 3);
        assert_eq!(ms.len(), 3); // 4 - 1 = 3
        assert_eq!(mk.len(), 3 * 2); // 3 tokens × 2 dim
        assert_eq!(mv.len(), 3 * 3); // 3 tokens × 3 dim
    }

    #[test]
    fn test_tome_token_sizes_weighted() {
        let config = ToMeConfig::new(1);
        let merger = ToMeMerger::new(config);

        // After first merge, sizes should reflect the merge
        let tokens = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let sizes = TokenSizes::uniform(4);
        let (_, merged_sizes) = merger.merge(&tokens, &sizes, 4, 2);

        // One pair merged: one size should be 2.0, others 1.0
        assert!(merged_sizes.as_slice().iter().any(|&s| (s - 2.0).abs() < 0.01));
    }

    #[test]
    fn test_cosine_sim() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_sim(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_sim(&a, &c).abs() < 0.001);
    }

    #[test]
    fn test_bipartite_match_basic() {
        // 4 tokens: A0=[1,0], B0=[1,0] (perfect match), A1=[0,1], B1=[0,1]
        let tokens = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let set_a = vec![0, 2];
        let set_b = vec![1, 3];
        let matches = bipartite_match(&tokens, 2, &set_a, &set_b, 2);
        assert_eq!(matches.len(), 2);
        // All matches should have high similarity
        for m in &matches {
            assert!(m.score > 0.99);
        }
    }
}
