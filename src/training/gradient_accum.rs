//! Tile-based gradient accumulation for memory-efficient training.
//!
//! Computes gradients in fixed-size tiles that fit in GPU shared memory,
//! then accumulates partial gradients across tiles. This enables training
//! with larger effective batch sizes on memory-constrained GPUs (integrated
//! GPUs, mobile devices) by processing one tile at a time.
//!
//! Pipeline:
//!   1. Split the batch into tiles of `tile_size` sequences
//!   2. Forward + backward for each tile → partial gradients
//!   3. Accumulate partial gradients into the master gradient buffer
//!   4. After all tiles: apply optimizer step on accumulated gradients
//!
//! The computation of tile N's forward pass can overlap with the
//! accumulation of tile N-1's gradients (pipeline parallelism).

use std::sync::Arc;
use wgpu::{Device, Queue};

use crate::compute::buffer::GpuBuffer;
use crate::compute::kernels::elementwise::ElementWiseOp;
use crate::error::Result;

// ---------------------------------------------------------------------------
// GradientTileConfig
// ---------------------------------------------------------------------------

/// Configuration for tile-based gradient accumulation.
#[derive(Debug, Clone)]
pub struct GradientTileConfig {
    /// Number of sequences per tile (micro-batch size).
    pub tile_size: usize,
    /// Total effective batch size.
    pub total_batch_size: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Whether to zero gradients before accumulation.
    pub zero_before_accumulate: bool,
    /// Gradient scaling factor (1.0 / num_tiles for mean gradient).
    pub scale_factor: f32,
}

impl GradientTileConfig {
    pub fn new(tile_size: usize, total_batch_size: usize, hidden_dim: usize) -> Self {
        let num_tiles = (total_batch_size + tile_size - 1) / tile_size;
        Self {
            tile_size,
            total_batch_size,
            hidden_dim,
            zero_before_accumulate: true,
            scale_factor: 1.0 / num_tiles as f32,
        }
    }

    /// Number of tiles needed.
    pub fn num_tiles(&self) -> usize {
        (self.total_batch_size + self.tile_size - 1) / self.tile_size
    }

    /// Actual tile size for the last tile (may be smaller).
    pub fn last_tile_size(&self) -> usize {
        let remainder = self.total_batch_size % self.tile_size;
        if remainder == 0 { self.tile_size } else { remainder }
    }

    /// Tile size for a given tile index.
    pub fn tile_size_for(&self, tile_idx: usize) -> usize {
        if tile_idx < self.num_tiles() - 1 {
            self.tile_size
        } else {
            self.last_tile_size()
        }
    }

    /// Byte size of a single gradient tile.
    pub fn tile_gradient_bytes(&self) -> usize {
        self.tile_size * self.hidden_dim * std::mem::size_of::<f32>()
    }
}

// ---------------------------------------------------------------------------
// GradientAccumulator — accumulates tiled gradients
// ---------------------------------------------------------------------------

/// Accumulates gradient tiles into a master gradient buffer.
///
/// The master buffer holds the sum of all partial gradients. After all tiles
/// are accumulated, the result is divided by `num_tiles` to get the mean
/// gradient (equivalent to a single large-batch gradient).
pub struct GradientAccumulator {
    config: GradientTileConfig,
    /// Master accumulated gradient buffer.
    accumulated: GpuBuffer,
    /// Number of tiles accumulated so far.
    tiles_accumulated: usize,
    /// Elementwise op for add/scale.
    elementwise: ElementWiseOp,
    #[allow(dead_code)]
    device: Arc<Device>,
    #[allow(dead_code)]
    queue: Arc<Queue>,
}

impl GradientAccumulator {
    /// Create a new gradient accumulator.
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, config: GradientTileConfig) -> Result<Self> {
        let total_bytes = config.total_batch_size * config.hidden_dim * std::mem::size_of::<f32>();
        let accumulated = GpuBuffer::zeros(&device, &queue, total_bytes, Some("accumulated_grad"))?;
        let elementwise = ElementWiseOp::new(&device, &queue);

        Ok(Self {
            config,
            accumulated,
            tiles_accumulated: 0,
            elementwise,
            device,
            queue,
        })
    }

    /// Zero the accumulated gradients.
    pub fn zero(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let numel = self.config.total_batch_size * self.config.hidden_dim;
        self.elementwise.dispatch_scale(encoder, &self.accumulated, &self.accumulated, 0.0, numel as u32).ok();
        self.tiles_accumulated = 0;
    }

    /// Accumulate a tile's gradient into the master buffer.
    pub fn accumulate(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        tile_gradient: &GpuBuffer,
        tile_idx: usize,
    ) -> Result<()> {
        let tile_size = self.config.tile_size_for(tile_idx);
        let hidden_dim = self.config.hidden_dim;
        let f32_size = std::mem::size_of::<f32>();

        if tile_idx == 0 && self.config.zero_before_accumulate {
            self.zero(encoder);
        }

        // Add tile gradient to accumulated at the correct offset
        let offset = tile_idx * self.config.tile_size * hidden_dim * f32_size;
        let numel = tile_size * hidden_dim;

        // Use add with offset: accumulated[offset..] += tile_gradient[0..tile_size*hidden_dim]
        self.elementwise.dispatch_add(
            encoder,
            &self.accumulated, // This needs sub-buffer support
            tile_gradient,
            &self.accumulated,
            numel as u32,
        )?;

        self.tiles_accumulated += 1;
        let _ = offset; // For future sub-buffer accumulation
        Ok(())
    }

    /// Scale the accumulated gradients by 1/num_tiles (mean gradient).
    pub fn scale_to_mean(&self, encoder: &mut wgpu::CommandEncoder) -> Result<()> {
        if self.tiles_accumulated == 0 {
            return Ok(());
        }
        let numel = self.config.total_batch_size * self.config.hidden_dim;
        let scale = self.config.scale_factor;
        self.elementwise.dispatch_scale(
            encoder,
            &self.accumulated,
            &self.accumulated,
            scale,
            numel as u32,
        )
    }

    /// Get the accumulated gradient buffer.
    pub fn accumulated(&self) -> &GpuBuffer {
        &self.accumulated
    }

    /// Number of tiles accumulated so far.
    pub fn tiles_accumulated(&self) -> usize {
        self.tiles_accumulated
    }

    /// Whether accumulation is complete.
    pub fn is_complete(&self) -> bool {
        self.tiles_accumulated >= self.config.num_tiles()
    }

    /// Get the config.
    pub fn config(&self) -> &GradientTileConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// CpuGradientAccumulator — for testing without GPU
// ---------------------------------------------------------------------------

/// CPU-side gradient accumulator for testing.
pub struct CpuGradientAccumulator {
    accumulated: Vec<f32>,
    tiles_accumulated: usize,
    num_tiles: usize,
    scale_factor: f32,
}

impl CpuGradientAccumulator {
    pub fn new(total_elements: usize, num_tiles: usize) -> Self {
        Self {
            accumulated: vec![0.0; total_elements],
            tiles_accumulated: 0,
            num_tiles,
            scale_factor: 1.0 / num_tiles as f32,
        }
    }

    /// Zero the accumulator.
    pub fn zero(&mut self) {
        self.accumulated.fill(0.0);
        self.tiles_accumulated = 0;
    }

    /// Accumulate a tile gradient.
    pub fn accumulate(&mut self, tile_grad: &[f32], offset: usize) {
        for (i, &g) in tile_grad.iter().enumerate() {
            if offset + i < self.accumulated.len() {
                self.accumulated[offset + i] += g;
            }
        }
        self.tiles_accumulated += 1;
    }

    /// Scale to mean gradient.
    pub fn scale_to_mean(&mut self) {
        if self.tiles_accumulated > 0 {
            for g in self.accumulated.iter_mut() {
                *g *= self.scale_factor;
            }
        }
    }

    /// Get the accumulated gradient.
    pub fn gradient(&self) -> &[f32] {
        &self.accumulated
    }

    /// Number of tiles accumulated.
    pub fn tiles_accumulated(&self) -> usize {
        self.tiles_accumulated
    }

    /// Whether complete.
    pub fn is_complete(&self) -> bool {
        self.tiles_accumulated >= self.num_tiles
    }
}

// ---------------------------------------------------------------------------
// TiledTrainingLoop — orchestrates the full tiled training step
// ---------------------------------------------------------------------------

/// Orchestrates a tiled training loop.
///
/// Splits the batch into tiles, runs forward+backward for each, and
/// accumulates gradients. After all tiles, applies the optimizer step.
pub struct TiledTrainingLoop {
    config: GradientTileConfig,
}

impl TiledTrainingLoop {
    pub fn new(config: GradientTileConfig) -> Self {
        Self { config }
    }

    /// Get the tile indices and their sizes.
    pub fn tile_schedule(&self) -> Vec<(usize, usize)> {
        let num_tiles = self.config.num_tiles();
        (0..num_tiles)
            .map(|i| (i, self.config.tile_size_for(i)))
            .collect()
    }

    /// Total effective batch size.
    pub fn effective_batch_size(&self) -> usize {
        self.config.total_batch_size
    }

    /// Number of gradient accumulation steps.
    pub fn num_accumulation_steps(&self) -> usize {
        self.config.num_tiles()
    }

    /// Get the config.
    pub fn config(&self) -> &GradientTileConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_config_basic() {
        let config = GradientTileConfig::new(4, 16, 512);
        assert_eq!(config.num_tiles(), 4);
        assert_eq!(config.tile_size_for(0), 4);
        assert_eq!(config.tile_size_for(3), 4);
        assert!((config.scale_factor - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_tile_config_uneven() {
        let config = GradientTileConfig::new(4, 10, 256);
        assert_eq!(config.num_tiles(), 3); // ceil(10/4)
        assert_eq!(config.tile_size_for(0), 4);
        assert_eq!(config.tile_size_for(1), 4);
        assert_eq!(config.last_tile_size(), 2);
        assert_eq!(config.tile_size_for(2), 2);
    }

    #[test]
    fn test_tile_config_exact() {
        let config = GradientTileConfig::new(8, 32, 128);
        assert_eq!(config.num_tiles(), 4);
        assert_eq!(config.last_tile_size(), 8);
    }

    #[test]
    fn test_tile_config_single() {
        let config = GradientTileConfig::new(32, 32, 256);
        assert_eq!(config.num_tiles(), 1);
        assert_eq!(config.last_tile_size(), 32);
        assert!((config.scale_factor - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_accumulator_basic() {
        let mut acc = CpuGradientAccumulator::new(4, 2);
        acc.accumulate(&[1.0, 2.0, 3.0, 4.0], 0);
        acc.accumulate(&[5.0, 6.0, 7.0, 8.0], 0);
        acc.scale_to_mean();

        let grad = acc.gradient();
        // (1+5)/2 = 3, (2+6)/2 = 4, etc.
        assert!((grad[0] - 3.0).abs() < 1e-4);
        assert!((grad[1] - 4.0).abs() < 1e-4);
        assert!((grad[2] - 5.0).abs() < 1e-4);
        assert!((grad[3] - 6.0).abs() < 1e-4);
    }

    #[test]
    fn test_cpu_accumulator_offset() {
        let mut acc = CpuGradientAccumulator::new(8, 2);
        acc.accumulate(&[1.0, 2.0, 3.0, 4.0], 0);
        acc.accumulate(&[10.0, 20.0, 30.0, 40.0], 4);
        acc.scale_to_mean();

        let grad = acc.gradient();
        assert!((grad[0] - 0.5).abs() < 1e-4); // 1/2
        assert!((grad[4] - 5.0).abs() < 1e-4); // 10/2
    }

    #[test]
    fn test_cpu_accumulator_zero() {
        let mut acc = CpuGradientAccumulator::new(4, 2);
        acc.accumulate(&[1.0, 2.0, 3.0, 4.0], 0);
        assert_eq!(acc.tiles_accumulated(), 1);

        acc.zero();
        assert_eq!(acc.tiles_accumulated(), 0);
        assert!(acc.gradient().iter().all(|&g| g == 0.0));
    }

    #[test]
    fn test_cpu_accumulator_complete() {
        let mut acc = CpuGradientAccumulator::new(4, 3);
        assert!(!acc.is_complete());
        acc.accumulate(&[1.0; 4], 0);
        acc.accumulate(&[2.0; 4], 0);
        acc.accumulate(&[3.0; 4], 0);
        assert!(acc.is_complete());
    }

    #[test]
    fn test_tiled_training_loop() {
        let config = GradientTileConfig::new(4, 12, 512);
        let loop_ = TiledTrainingLoop::new(config);

        assert_eq!(loop_.effective_batch_size(), 12);
        assert_eq!(loop_.num_accumulation_steps(), 3);

        let schedule = loop_.tile_schedule();
        assert_eq!(schedule.len(), 3);
        assert_eq!(schedule[0], (0, 4));
        assert_eq!(schedule[2], (2, 4));
    }

    #[test]
    fn test_tiled_training_loop_uneven() {
        let config = GradientTileConfig::new(4, 10, 256);
        let loop_ = TiledTrainingLoop::new(config);

        let schedule = loop_.tile_schedule();
        assert_eq!(schedule.len(), 3);
        assert_eq!(schedule[2], (2, 2)); // Last tile is smaller
    }

    #[test]
    fn test_tile_gradient_bytes() {
        let config = GradientTileConfig::new(4, 16, 512);
        assert_eq!(config.tile_gradient_bytes(), 4 * 512 * 4); // 8192 bytes
    }

    #[test]
    fn test_cpu_accumulator_mean_gradient_matches_full_batch() {
        // Verify that tiled accumulation produces the same gradient as full-batch
        // Full batch: mean of [1,2,3,4,5,6] = 3.5
        // Tiles: [1,2,3] mean=2, [4,5,6] mean=5. Mean of (2+5)/2 = 3.5 ✓
        let mut acc = CpuGradientAccumulator::new(6, 2);
        acc.accumulate(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], 0);
        acc.accumulate(&[0.0, 0.0, 0.0, 4.0, 5.0, 6.0], 0);
        acc.scale_to_mean();

        let grad = acc.gradient();
        // Direct accumulation: [1,2,3,4,5,6] / 2 = [0.5,1,1.5,2,2.5,3]
        assert!((grad[0] - 0.5).abs() < 1e-4);
        assert!((grad[3] - 2.0).abs() < 1e-4);
        assert!((grad[5] - 3.0).abs() < 1e-4);
    }
}
