//! Distributed tensor parallelism for multi-GPU training and inference.
//!
//! Implements:
//! - Tensor parallel: split weight matrices across N GPUs
//! - Pipeline parallel: assign layers to different GPUs with micro-batching
//! - Communication primitives: all-reduce, all-gather, scatter
//! - Device coordinator: assign model shards to available devices

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// DeviceId — identifies a compute device
// ---------------------------------------------------------------------------

/// Unique identifier for a compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(pub usize);

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gpu:{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// TensorParallelConfig
// ---------------------------------------------------------------------------

/// Configuration for tensor parallelism.
#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    /// Number of devices for tensor parallelism.
    pub world_size: usize,
    /// Rank of this device (0..world_size).
    pub rank: usize,
    /// Whether to split attention heads across devices.
    pub split_heads: bool,
    /// Whether to split FFN across devices.
    pub split_ffn: bool,
}

impl TensorParallelConfig {
    pub fn new(world_size: usize, rank: usize) -> Self {
        Self {
            world_size,
            rank,
            split_heads: true,
            split_ffn: true,
        }
    }

    /// Single device (no parallelism).
    pub fn single() -> Self {
        Self::new(1, 0)
    }

    /// Split dimension for a given total dimension.
    pub fn split_dim(&self, total: usize) -> usize {
        (total + self.world_size - 1) / self.world_size
    }

    /// Start index for this rank's shard.
    pub fn shard_start(&self, total: usize) -> usize {
        self.rank * self.split_dim(total)
    }

    /// End index for this rank's shard.
    pub fn shard_end(&self, total: usize) -> usize {
        let end = (self.rank + 1) * self.split_dim(total);
        end.min(total)
    }

    /// Shard size for this rank.
    pub fn shard_size(&self, total: usize) -> usize {
        self.shard_end(total) - self.shard_start(total)
    }

    /// Whether this is the master rank.
    pub fn is_master(&self) -> bool {
        self.rank == 0
    }

    /// Number of attention heads for this rank.
    pub fn local_heads(&self, total_heads: usize) -> usize {
        if self.split_heads {
            self.shard_size(total_heads)
        } else {
            total_heads
        }
    }

    /// FFN hidden dimension for this rank.
    pub fn local_ffn_dim(&self, total_dim: usize) -> usize {
        if self.split_ffn {
            self.shard_size(total_dim)
        } else {
            total_dim
        }
    }
}

// ---------------------------------------------------------------------------
// PipelineParallelConfig
// ---------------------------------------------------------------------------

/// Configuration for pipeline parallelism.
#[derive(Debug, Clone)]
pub struct PipelineParallelConfig {
    /// Number of pipeline stages.
    pub num_stages: usize,
    /// Stage index for this device.
    pub stage_id: usize,
    /// Total number of layers.
    pub num_layers: usize,
    /// Number of micro-batches for pipelining.
    pub num_micro_batches: usize,
}

impl PipelineParallelConfig {
    pub fn new(num_stages: usize, stage_id: usize, num_layers: usize) -> Self {
        Self {
            num_stages,
            stage_id,
            num_layers,
            num_micro_batches: num_stages,
        }
    }

    /// Layers assigned to this stage.
    pub fn stage_layers(&self) -> std::ops::Range<usize> {
        let layers_per_stage = self.num_layers / self.num_stages;
        let start = self.stage_id * layers_per_stage;
        let end = if self.stage_id == self.num_stages - 1 {
            self.num_layers
        } else {
            (self.stage_id + 1) * layers_per_stage
        };
        start..end
    }

    /// Number of layers in this stage.
    pub fn stage_layer_count(&self) -> usize {
        self.stage_layers().end - self.stage_layers().start
    }

    /// Whether this is the first stage.
    pub fn is_first_stage(&self) -> bool {
        self.stage_id == 0
    }

    /// Whether this is the last stage.
    pub fn is_last_stage(&self) -> bool {
        self.stage_id == self.num_stages - 1
    }

    /// Micro-batch size.
    pub fn micro_batch_size(&self, total_batch: usize) -> usize {
        (total_batch + self.num_micro_batches - 1) / self.num_micro_batches
    }
}

// ---------------------------------------------------------------------------
// CommunicationPrimitives — simulated collective operations
// ---------------------------------------------------------------------------

/// Simulated all-reduce operation (sum).
pub fn all_reduce_sum(local_data: &mut [f32], world_size: usize) {
    // In a real distributed system, this would communicate across devices
    // Here we simulate by scaling (as if all ranks had the same data)
    for v in local_data.iter_mut() {
        *v *= world_size as f32;
    }
}

/// Simulated all-reduce mean.
pub fn all_reduce_mean(_local_data: &mut [f32], _world_size: usize) {
    // In practice, sum across ranks then divide by world_size
    // Simulated: data stays the same (already the local contribution)
}

/// Scatter a buffer across ranks: returns the shard for a given rank.
pub fn scatter(data: &[f32], world_size: usize, rank: usize) -> Vec<f32> {
    let shard_size = data.len() / world_size;
    let start = rank * shard_size;
    let end = if rank == world_size - 1 {
        data.len()
    } else {
        (rank + 1) * shard_size
    };
    data[start..end].to_vec()
}

/// Gather shards from all ranks into a single buffer.
pub fn gather(shards: &[Vec<f32>]) -> Vec<f32> {
    shards.iter().flat_map(|s| s.iter().copied()).collect()
}

/// All-gather: each rank contributes its shard and receives the full buffer.
pub fn all_gather(local_shard: &[f32], world_size: usize) -> Vec<f32> {
    // Simulated: replicate the local shard across all ranks
    let mut result = Vec::with_capacity(local_shard.len() * world_size);
    for _ in 0..world_size {
        result.extend_from_slice(local_shard);
    }
    result
}

// ---------------------------------------------------------------------------
// DeviceCoordinator — assigns model shards to devices
// ---------------------------------------------------------------------------

/// A device in the cluster.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub id: DeviceId,
    pub name: String,
    pub memory_mb: usize,
    pub compute_capability: f32,
}

/// Assignment of model shards to devices.
#[derive(Debug, Clone)]
pub struct ShardAssignment {
    /// Device ID → tensor parallel rank.
    pub tp_ranks: HashMap<DeviceId, usize>,
    /// Device ID → pipeline stage.
    pub pp_stages: HashMap<DeviceId, usize>,
    /// Tensor parallelism degree.
    pub tp_degree: usize,
    /// Pipeline parallelism degree.
    pub pp_degree: usize,
}

impl ShardAssignment {
    pub fn new(tp_degree: usize, pp_degree: usize) -> Self {
        Self {
            tp_ranks: HashMap::new(),
            pp_stages: HashMap::new(),
            tp_degree,
            pp_degree,
        }
    }

    /// Total number of devices needed.
    pub fn total_devices(&self) -> usize {
        self.tp_degree * self.pp_degree
    }

    /// Assign a device to a TP rank and PP stage.
    pub fn assign(&mut self, device: DeviceId, tp_rank: usize, pp_stage: usize) {
        self.tp_ranks.insert(device, tp_rank);
        self.pp_stages.insert(device, pp_stage);
    }

    /// Get the TP config for a device.
    pub fn tp_config(&self, device: DeviceId) -> Option<TensorParallelConfig> {
        self.tp_ranks.get(&device).map(|&rank| {
            TensorParallelConfig::new(self.tp_degree, rank)
        })
    }

    /// Get the PP config for a device.
    pub fn pp_config(&self, device: DeviceId, num_layers: usize) -> Option<PipelineParallelConfig> {
        self.pp_stages.get(&device).map(|&stage| {
            PipelineParallelConfig::new(self.pp_degree, stage, num_layers)
        })
    }
}

/// Coordinates device assignment for distributed training.
pub struct DeviceCoordinator {
    devices: Vec<DeviceInfo>,
}

impl DeviceCoordinator {
    pub fn new(devices: Vec<DeviceInfo>) -> Self {
        Self { devices }
    }

    /// Auto-assign devices based on TP and PP degrees.
    pub fn auto_assign(&self, tp_degree: usize, pp_degree: usize) -> Option<ShardAssignment> {
        let needed = tp_degree * pp_degree;
        if self.devices.len() < needed {
            return None;
        }

        let mut assignment = ShardAssignment::new(tp_degree, pp_degree);

        for (i, device) in self.devices.iter().take(needed).enumerate() {
            let pp_stage = i / tp_degree;
            let tp_rank = i % tp_degree;
            assignment.assign(device.id, tp_rank, pp_stage);
        }

        Some(assignment)
    }

    /// Number of available devices.
    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }

    /// Get device info.
    pub fn device(&self, id: DeviceId) -> Option<&DeviceInfo> {
        self.devices.iter().find(|d| d.id == id)
    }

    /// Total memory across all devices.
    pub fn total_memory_mb(&self) -> usize {
        self.devices.iter().map(|d| d.memory_mb).sum()
    }
}

// ---------------------------------------------------------------------------
// WeightSharder — splits weights across devices
// ---------------------------------------------------------------------------

/// Splits weight matrices for tensor parallelism.
pub struct WeightSharder;

impl WeightSharder {
    /// Split a weight matrix along rows.
    /// Weight shape: [out × in] → [out/world_size × in] per rank.
    pub fn split_rows(weight: &[f32], rows: usize, cols: usize, world_size: usize) -> Vec<Vec<f32>> {
        let rows_per_rank = (rows + world_size - 1) / world_size;
        let mut shards = Vec::with_capacity(world_size);

        for rank in 0..world_size {
            let start = rank * rows_per_rank;
            let end = (start + rows_per_rank).min(rows);
            let mut shard = Vec::with_capacity((end - start) * cols);
            for r in start..end {
                shard.extend_from_slice(&weight[r * cols..(r + 1) * cols]);
            }
            shards.push(shard);
        }

        shards
    }

    /// Split a weight matrix along columns.
    /// Weight shape: [out × in] → [out × in/world_size] per rank.
    pub fn split_cols(weight: &[f32], rows: usize, cols: usize, world_size: usize) -> Vec<Vec<f32>> {
        let cols_per_rank = (cols + world_size - 1) / world_size;
        let mut shards = Vec::with_capacity(world_size);

        for rank in 0..world_size {
            let start = rank * cols_per_rank;
            let end = (start + cols_per_rank).min(cols);
            let mut shard = Vec::with_capacity(rows * (end - start));
            for r in 0..rows {
                shard.extend_from_slice(&weight[r * cols + start..r * cols + end]);
            }
            shards.push(shard);
        }

        shards
    }

    /// Reconstruct a weight matrix from row-split shards.
    pub fn reconstruct_rows(shards: &[Vec<f32>], _cols: usize) -> Vec<f32> {
        let mut full = Vec::new();
        for shard in shards {
            full.extend_from_slice(shard);
        }
        full
    }

    /// Reconstruct a weight matrix from column-split shards.
    pub fn reconstruct_cols(shards: &[Vec<f32>], total_cols: usize, rows: usize) -> Vec<f32> {
        let world_size = shards.len();
        let cols_per_rank = (total_cols + world_size - 1) / world_size;
        let mut full = vec![0.0f32; rows * total_cols];

        for (rank, shard) in shards.iter().enumerate() {
            let start = rank * cols_per_rank;
            let end = (start + cols_per_rank).min(total_cols);
            let local_cols = end - start;
            for r in 0..rows {
                for c in 0..local_cols {
                    full[r * total_cols + start + c] = shard[r * local_cols + c];
                }
            }
        }

        full
    }
}

// ---------------------------------------------------------------------------
// PipelineSchedule — schedule for pipeline parallelism
// ---------------------------------------------------------------------------

/// A pipeline schedule step.
#[derive(Debug, Clone)]
pub struct PipelineStep {
    /// Micro-batch index.
    pub micro_batch: usize,
    /// Stage index.
    pub stage: usize,
    /// Whether this is a forward pass.
    pub is_forward: bool,
}

/// Generates a pipeline schedule (simple GPipe-style).
pub fn gpipe_schedule(num_stages: usize, num_micro_batches: usize) -> Vec<PipelineStep> {
    let mut schedule = Vec::new();

    // Forward passes: staggered by stage
    for mb in 0..num_micro_batches {
        for stage in 0..num_stages {
            schedule.push(PipelineStep {
                micro_batch: mb,
                stage,
                is_forward: true,
            });
        }
    }

    // Backward passes: reverse order
    for mb in (0..num_micro_batches).rev() {
        for stage in (0..num_stages).rev() {
            schedule.push(PipelineStep {
                micro_batch: mb,
                stage,
                is_forward: false,
            });
        }
    }

    schedule
}

/// One-forward-one-backward (1F1B) schedule for better memory efficiency.
pub fn one_f_one_b_schedule(num_stages: usize, num_micro_batches: usize) -> Vec<PipelineStep> {
    let mut schedule = Vec::new();
    let warmup = num_stages - 1;

    // Warmup: forward passes to fill the pipeline
    for mb in 0..warmup.min(num_micro_batches) {
        for stage in 0..=mb {
            schedule.push(PipelineStep {
                micro_batch: mb - stage,
                stage,
                is_forward: true,
            });
        }
    }

    // Steady state: 1 forward + 1 backward per step
    for mb in warmup..num_micro_batches {
        // Forward
        for stage in 0..num_stages {
            schedule.push(PipelineStep {
                micro_batch: mb - stage,
                stage,
                is_forward: true,
            });
        }
        // Backward (oldest micro-batch)
        if mb >= warmup {
            for stage in (0..num_stages).rev() {
                schedule.push(PipelineStep {
                    micro_batch: mb - warmup,
                    stage,
                    is_forward: false,
                });
            }
        }
    }

    // Cooldown: remaining backward passes
    for mb in (num_micro_batches - warmup)..num_micro_batches {
        for stage in (0..num_stages).rev() {
            schedule.push(PipelineStep {
                micro_batch: mb,
                stage,
                is_forward: false,
            });
        }
    }

    schedule
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_parallel_config() {
        let config = TensorParallelConfig::new(4, 1);
        assert_eq!(config.world_size, 4);
        assert_eq!(config.rank, 1);
        assert!(!config.is_master());
        assert_eq!(config.split_dim(128), 32);
        assert_eq!(config.shard_start(128), 32);
        assert_eq!(config.shard_end(128), 64);
        assert_eq!(config.shard_size(128), 32);
    }

    #[test]
    fn test_tensor_parallel_uneven() {
        let config = TensorParallelConfig::new(3, 0);
        assert_eq!(config.shard_size(10), 4); // ceil(10/3) = 4
        let config1 = TensorParallelConfig::new(3, 2);
        assert_eq!(config1.shard_start(10), 8);
        assert_eq!(config1.shard_end(10), 10);
        assert_eq!(config1.shard_size(10), 2);
    }

    #[test]
    fn test_tensor_parallel_heads() {
        let config = TensorParallelConfig::new(4, 0);
        assert_eq!(config.local_heads(32), 8);
        let config_single = TensorParallelConfig::single();
        assert_eq!(config_single.local_heads(32), 32);
    }

    #[test]
    fn test_tensor_parallel_ffn() {
        let config = TensorParallelConfig::new(2, 0);
        assert_eq!(config.local_ffn_dim(4096), 2048);
    }

    #[test]
    fn test_pipeline_parallel_config() {
        let config = PipelineParallelConfig::new(4, 1, 24);
        let layers = config.stage_layers();
        assert_eq!(layers, 6..12);
        assert_eq!(config.stage_layer_count(), 6);
        assert!(!config.is_first_stage());
        assert!(!config.is_last_stage());
    }

    #[test]
    fn test_pipeline_parallel_last_stage() {
        let config = PipelineParallelConfig::new(4, 3, 24);
        let layers = config.stage_layers();
        assert_eq!(layers, 18..24);
        assert!(config.is_last_stage());
    }

    #[test]
    fn test_pipeline_micro_batch() {
        let config = PipelineParallelConfig::new(4, 0, 24);
        assert_eq!(config.micro_batch_size(32), 8);
    }

    #[test]
    fn test_scatter_gather() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let s0 = scatter(&data, 3, 0);
        let s1 = scatter(&data, 3, 1);
        let s2 = scatter(&data, 3, 2);
        assert_eq!(s0.len(), 4);
        assert_eq!(s1.len(), 4);
        assert_eq!(s2.len(), 4);

        let gathered = gather(&[s0, s1, s2]);
        assert_eq!(gathered.len(), 12);
        for i in 0..12 {
            assert!((gathered[i] - i as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_all_reduce_sum() {
        let mut data = vec![1.0, 2.0, 3.0];
        all_reduce_sum(&mut data, 4);
        assert!((data[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_all_gather() {
        let shard = vec![1.0, 2.0];
        let gathered = all_gather(&shard, 3);
        assert_eq!(gathered.len(), 6);
    }

    #[test]
    fn test_weight_sharder_rows() {
        // 4×3 matrix, split across 2 ranks
        let weight: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let shards = WeightSharder::split_rows(&weight, 4, 3, 2);
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].len(), 6); // 2 rows × 3 cols
        assert_eq!(shards[1].len(), 6);

        let reconstructed = WeightSharder::reconstruct_rows(&shards, 3);
        for i in 0..12 {
            assert!((reconstructed[i] - weight[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_weight_sharder_cols() {
        // 3×4 matrix, split across 2 ranks
        let weight: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let shards = WeightSharder::split_cols(&weight, 3, 4, 2);
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].len(), 6); // 3 rows × 2 cols
        assert_eq!(shards[1].len(), 6);

        let reconstructed = WeightSharder::reconstruct_cols(&shards, 4, 3);
        for i in 0..12 {
            assert!((reconstructed[i] - weight[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_weight_sharder_uneven() {
        let weight: Vec<f32> = (0..10).map(|i| i as f32).collect();
        // 5×2 matrix, split rows across 3 ranks
        let shards = WeightSharder::split_rows(&weight, 5, 2, 3);
        assert_eq!(shards.len(), 3);
        assert_eq!(shards[0].len(), 4); // 2 rows
        assert_eq!(shards[1].len(), 4); // 2 rows
        assert_eq!(shards[2].len(), 2); // 1 row
    }

    #[test]
    fn test_device_coordinator() {
        let devices: Vec<DeviceInfo> = (0..8).map(|i| DeviceInfo {
            id: DeviceId(i),
            name: format!("GPU {}", i),
            memory_mb: 8192,
            compute_capability: 7.5,
        }).collect();
        let coord = DeviceCoordinator::new(devices);
        assert_eq!(coord.num_devices(), 8);
        assert_eq!(coord.total_memory_mb(), 8 * 8192);
    }

    #[test]
    fn test_auto_assign() {
        let devices: Vec<DeviceInfo> = (0..4).map(|i| DeviceInfo {
            id: DeviceId(i),
            name: format!("GPU {}", i),
            memory_mb: 8192,
            compute_capability: 7.5,
        }).collect();
        let coord = DeviceCoordinator::new(devices);
        let assignment = coord.auto_assign(2, 2).unwrap();
        assert_eq!(assignment.tp_degree, 2);
        assert_eq!(assignment.pp_degree, 2);

        // GPU 0: tp_rank=0, pp_stage=0
        let tp0 = assignment.tp_config(DeviceId(0)).unwrap();
        assert_eq!(tp0.rank, 0);
        let pp0 = assignment.pp_config(DeviceId(0), 24).unwrap();
        assert_eq!(pp0.stage_id, 0);

        // GPU 2: tp_rank=0, pp_stage=1
        let tp2 = assignment.tp_config(DeviceId(2)).unwrap();
        assert_eq!(tp2.rank, 0);
        let pp2 = assignment.pp_config(DeviceId(2), 24).unwrap();
        assert_eq!(pp2.stage_id, 1);
    }

    #[test]
    fn test_auto_assign_not_enough() {
        let devices: Vec<DeviceInfo> = (0..2).map(|i| DeviceInfo {
            id: DeviceId(i),
            name: format!("GPU {}", i),
            memory_mb: 8192,
            compute_capability: 7.5,
        }).collect();
        let coord = DeviceCoordinator::new(devices);
        assert!(coord.auto_assign(2, 2).is_none());
    }

    #[test]
    fn test_gpipe_schedule() {
        let schedule = gpipe_schedule(2, 3);
        // 3 micro-batches × 2 stages forward + 3 × 2 backward = 12 steps
        assert_eq!(schedule.len(), 12);
        assert!(schedule[0].is_forward);
        assert!(!schedule.last().unwrap().is_forward);
    }

    #[test]
    fn test_1f1b_schedule() {
        let schedule = one_f_one_b_schedule(2, 4);
        // Should have both forward and backward passes
        let fwd = schedule.iter().filter(|s| s.is_forward).count();
        let bwd = schedule.iter().filter(|s| !s.is_forward).count();
        assert!(fwd > 0);
        assert!(bwd > 0);
    }

    #[test]
    fn test_single_device() {
        let config = TensorParallelConfig::single();
        assert_eq!(config.world_size, 1);
        assert!(config.is_master());
        assert_eq!(config.shard_size(128), 128);
    }
}
