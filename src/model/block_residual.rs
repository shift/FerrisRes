//! Token-local depth-block residual state shared by batch, training and decode.
use crate::model::cpu_block_attn_res::CpuBlockAttnResModel;

pub(crate) struct BlockResidualState {
    // Every snapshot preserves the token axis: [seq, hidden_dim].
    snapshots: Vec<Vec<f32>>,
    partial_sum: Vec<f32>,
    layer_count: usize,
}

impl BlockResidualState {
    pub(crate) fn new(embedding: &[f32]) -> Self {
        Self {
            snapshots: vec![embedding.to_vec()],
            partial_sum: vec![0.0; embedding.len()],
            layer_count: 0,
        }
    }

    pub(crate) fn apply_layer(
        &mut self,
        model: &CpuBlockAttnResModel,
        hidden: &mut [f32],
        layer_idx: usize,
    ) {
        assert_eq!(self.partial_sum.len(), hidden.len());
        for (sum, value) in self.partial_sum.iter_mut().zip(hidden.iter()) {
            *sum += value;
        }
        self.layer_count += 1;
        if !model.is_block_boundary(layer_idx) { return; }

        // Average over depth only, including unequal-size configured blocks.
        let mut snapshot = std::mem::replace(&mut self.partial_sum, vec![0.0; hidden.len()]);
        for value in &mut snapshot { *value /= self.layer_count as f32; }
        self.snapshots.push(snapshot);
        self.layer_count = 0;
        let residual = model.inter_block_attention(hidden, &self.snapshots, hidden.len() / model.hidden_dim);
        for (value, addition) in hidden.iter_mut().zip(residual) { *value += addition; }
    }
}
