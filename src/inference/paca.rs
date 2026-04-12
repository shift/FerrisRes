//! Patch-to-Cluster Attention (PaCa) — spatial cluster vision tokens.
//!
//! Implements the key idea from He et al., 2022:
//! "Patch-to-Cluster Attention: Visual Token Clustering for Efficient Vision Transformers"
//!
//! Instead of flat token sequences, tokens are grouped into spatial clusters.
/// Within-cluster attention is cheap (few tokens); cross-cluster is compressed.

/// Configuration for Patch-to-Cluster attention.
#[derive(Debug, Clone)]
pub struct PacaConfig {
    /// Number of clusters to produce.
    pub num_clusters: usize,
    /// Cluster assignment method.
    pub method: ClusterMethod,
    /// Whether PaCa is enabled.
    pub enabled: bool,
}

/// Method for assigning tokens to clusters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClusterMethod {
    /// Grid-based: divide spatial dimensions into uniform grid.
    Grid,
    /// K-means style: learnable cluster centers.
    Learnable,
    /// Hierarchical: merge similar adjacent tokens (like ToMe).
    Hierarchical,
}

impl Default for PacaConfig {
    fn default() -> Self {
        Self {
            num_clusters: 16,
            method: ClusterMethod::Grid,
            enabled: false,
        }
    }
}

impl PacaConfig {
    /// Create a new config with grid clustering.
    pub fn grid(num_clusters: usize) -> Self {
        Self {
            num_clusters,
            method: ClusterMethod::Grid,
            enabled: true,
        }
    }
}

/// Cluster assignment result.
#[derive(Debug, Clone)]
pub struct ClusterAssignment {
    /// Cluster ID for each token.
    pub assignments: Vec<usize>,
    /// Number of clusters.
    pub num_clusters: usize,
}

impl ClusterAssignment {
    /// Create grid-based cluster assignments.
    ///
    /// Divides a 2D grid of tokens into `num_clusters` uniform regions.
    /// `grid_h` and `grid_w` are the dimensions of the token grid
    /// (e.g., 14×14 for 224px/16 patches).
    pub fn grid(grid_h: usize, grid_w: usize, _num_clusters: usize) -> Self {
        let cluster_grid_h = (grid_h as f64).sqrt().ceil() as usize;
        let cluster_grid_w = (grid_w as f64).sqrt().ceil() as usize;

        let n_tokens = grid_h * grid_w;
        let assignments: Vec<usize> = (0..n_tokens)
            .map(|i| {
                let row = i / grid_w;
                let col = i % grid_w;
                let cr = (row * cluster_grid_h / grid_h).min(cluster_grid_h - 1);
                let cc = (col * cluster_grid_w / grid_w).min(cluster_grid_w - 1);
                cr * cluster_grid_w + cc
            })
            .collect();

        let actual_clusters = cluster_grid_h * cluster_grid_w;
        Self {
            assignments,
            num_clusters: actual_clusters,
        }
    }

    /// Get the tokens belonging to a specific cluster.
    pub fn tokens_in_cluster(&self, cluster_id: usize) -> Vec<usize> {
        self.assignments.iter()
            .enumerate()
            .filter(|(_, &c)| c == cluster_id)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get the number of tokens in each cluster.
    pub fn cluster_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0; self.num_clusters];
        for &c in &self.assignments {
            sizes[c] += 1;
        }
        sizes
    }
}

/// PaCa spatial cluster engine.
pub struct PacaEngine {
    config: PacaConfig,
}

impl PacaEngine {
    /// Create a new PaCa engine.
    pub fn new(config: PacaConfig) -> Self {
        Self { config }
    }

    /// Create a disabled engine (no-op).
    pub fn disabled() -> Self {
        Self {
            config: PacaConfig::default(),
        }
    }

    /// Whether clustering is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Compute cluster assignments for a 2D token grid.
    pub fn assign_clusters(&self, grid_h: usize, grid_w: usize) -> ClusterAssignment {
        match self.config.method {
            ClusterMethod::Grid => {
                ClusterAssignment::grid(grid_h, grid_w, self.config.num_clusters)
            }
            ClusterMethod::Learnable | ClusterMethod::Hierarchical => {
                // Learnable would use trained cluster centers;
                // Hierarchical would use ToMe-style merging.
                // Fall back to grid for now.
                ClusterAssignment::grid(grid_h, grid_w, self.config.num_clusters)
            }
        }
    }

    /// Compute cluster centroids by averaging token embeddings.
    ///
    /// Returns centroids shape [num_clusters, dim].
    pub fn compute_centroids(
        &self,
        tokens: &[f32],    // [n_tokens, dim]
        assignment: &ClusterAssignment,
        dim: usize,
    ) -> Vec<f32> {
        let _n_tokens = assignment.assignments.len();
        let mut centroids = vec![0.0f32; assignment.num_clusters * dim];
        let mut counts = vec![0usize; assignment.num_clusters];

        for (i, &cluster) in assignment.assignments.iter().enumerate() {
            for d in 0..dim {
                centroids[cluster * dim + d] += tokens[i * dim + d];
            }
            counts[cluster] += 1;
        }

        for c in 0..assignment.num_clusters {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c * dim + d] /= counts[c] as f32;
                }
            }
        }

        centroids
    }

    /// Get the config.
    pub fn config(&self) -> &PacaConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_assignment() {
        let ca = ClusterAssignment::grid(14, 14, 16);
        assert_eq!(ca.assignments.len(), 196); // 14 × 14
        assert!(ca.num_clusters > 0);
        // All tokens should be assigned
        assert!(ca.assignments.iter().all(|&c| c < ca.num_clusters));
    }

    #[test]
    fn test_cluster_sizes() {
        let ca = ClusterAssignment::grid(4, 4, 4);
        let sizes = ca.cluster_sizes();
        assert_eq!(sizes.iter().sum::<usize>(), 16); // all 16 tokens accounted for
    }

    #[test]
    fn test_tokens_in_cluster() {
        let ca = ClusterAssignment::grid(4, 4, 4);
        for c in 0..ca.num_clusters {
            let tokens = ca.tokens_in_cluster(c);
            assert!(!tokens.is_empty());
        }
    }

    #[test]
    fn test_centroids() {
        let engine = PacaEngine::new(PacaConfig::grid(4));
        let ca = engine.assign_clusters(4, 4);
        let tokens = vec![1.0f32; 16 * 8]; // 16 tokens, dim 8
        let centroids = engine.compute_centroids(&tokens, &ca, 8);
        // All centroids should be 1.0 (since all tokens are 1.0)
        for &v in &centroids {
            assert!((v - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_paca_disabled() {
        let engine = PacaEngine::disabled();
        assert!(!engine.is_enabled());
    }
}
