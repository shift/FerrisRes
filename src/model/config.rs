use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockAttnResConfig {
    pub hidden_dim: usize,
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_layers: usize,
    pub include_embedding: bool,
    pub attention_heads: usize,
    pub intermediate_dim: usize,
}

impl BlockAttnResConfig {
    pub fn new(hidden_dim: usize) -> Self {
        let num_blocks = 8;
        let block_size = 8;
        Self {
            hidden_dim,
            num_blocks,
            block_size,
            num_layers: num_blocks * block_size,
            include_embedding: true,
            attention_heads: 8,
            intermediate_dim: 4 * hidden_dim,
        }
    }

    pub fn total_layers(&self) -> usize {
        self.num_blocks * self.block_size
    }
}
