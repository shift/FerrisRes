pub mod optimizer;
pub mod optimizer_scale;
pub mod optimizer_adamem;
pub mod checkpointing;
pub mod cpu_offload;
pub mod async_offload;
pub mod lora;
pub mod gradient_accum;
pub mod partial_backprop;
pub mod qlora;
pub mod tool_triggered_lora;
pub mod backward;
pub mod shadow_weights;
pub mod bit_moe;

pub use optimizer::{SgdOptimizer, AdamOptimizer, CrossEntropyLoss, WeightOptimizer, OptimizerHint, optimizer_for_profile};
pub use optimizer_scale::ScaleOptimizer;
pub use optimizer_adamem::AdaMeMOptimizer;
pub use tool_triggered_lora::{ToolTriggeredLora, ToolTriggeredLoraConfig, StackedAdapter, FisherDiagonal, LearningEvent, ToolTriggeredLoraStats};
pub use checkpointing::CheckpointStore;
pub use cpu_offload::CpuGradientBuffer;
pub use async_offload::AsyncGradientOffload;
pub use lora::{LoraConfig, LoraLayer, LoraManager};
use std::fmt;
use std::sync::Arc;
use wgpu::Device;

use crate::compute::{BorrowedBufferPool, BlockCache, DeviceMemoryPhase};
use crate::device::profile::DeviceProfile;
use crate::error::Result;

/// Controls the granularity at which activation checkpointing is applied
/// during the forward pass.
///
/// - `PerBlock`     — save hidden_states at each BlockAttnRes block boundary.
///                    Best for low-VRAM devices (e.g. DeviceProfile::Integrated).
/// - `PerLayer`     — finer-grained: save after every sub-layer (norm, proj, …).
///                    Requires more recompute work during backward.
/// - `PerAttention` — save only at attention sub-module boundaries.
/// - `None`         — disabled; all intermediate activations stay live (default
///                    for MidRange / HighEnd devices).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointGranularity {
    PerLayer,
    PerBlock,
    PerAttention,
    None,
}

pub struct TrainingState {
    pub epoch: u32,
    pub step: u32,
    pub total_loss: f32,
    pub best_loss: f32,
    /// Borrowed-buffer pool — only active for [`DeviceProfile::Integrated`].
    /// Manages the lifecycle of KV-cache buffers reused for gradient scratch
    /// space, avoiding double allocation on shared DRAM.
    borrowed_pool: BorrowedBufferPool,
    /// Device reference needed to reconstruct the KV cache after training.
    device: Option<Arc<Device>>,
    /// Async gradient offload engine — only set when the device profile
    /// uses [`ComputeMode::CpuOffload`] (i.e. [`DeviceProfile::Integrated`]).
    /// Routes gradient buffers through a multi-staged staging pool to overlap
    /// GPU→CPU transfer with CPU-side processing.
    pub async_offload: Option<AsyncGradientOffload>,
}

impl TrainingState {
    pub fn new() -> Self {
        // Default construction without a known profile — pool is disabled.
        Self::with_profile(&DeviceProfile::HighEnd, None)
    }

    /// Construct a `TrainingState` for a specific device profile.
    ///
    /// For [`DeviceProfile::Integrated`] the borrowed-buffer pool is enabled;
    /// for all other profiles it is disabled (zero overhead).
    pub fn with_profile(profile: &DeviceProfile, device: Option<Arc<Device>>) -> Self {
        Self {
            epoch: 0,
            step: 0,
            total_loss: 0.0,
            best_loss: f32::INFINITY,
            borrowed_pool: BorrowedBufferPool::new(profile),
            device,
            async_offload: None,
        }
    }

    /// Construct a `TrainingState` with async gradient offload enabled.
    ///
    /// For [`DeviceProfile::Integrated`] (CpuOffload mode) this initialises
    /// the async offload engine with the appropriate buffering depth.
    pub fn with_async_offload(
        profile: &DeviceProfile,
        device: Arc<Device>,
        grad_size: usize,
        mem_usage_pct: f32,
    ) -> Result<Self> {
        let depth =
            AsyncGradientOffload::buffering_depth_for_profile(*profile, mem_usage_pct);
        let offload = AsyncGradientOffload::new(Arc::clone(&device), grad_size, depth)?;
        Ok(Self {
            epoch: 0,
            step: 0,
            total_loss: 0.0,
            best_loss: f32::INFINITY,
            borrowed_pool: BorrowedBufferPool::new(profile),
            device: Some(device),
            async_offload: Some(offload),
        })
    }

    pub fn record_loss(&mut self, loss: f32) {
        self.total_loss += loss;
        if loss < self.best_loss {
            self.best_loss = loss;
        }
        tracing::debug!(event = "trainingstate_recorded_loss_best_loss", "TrainingState: recorded loss={:.6} best_loss={:.6}", loss, self.best_loss);
    }

    pub fn next_step(&mut self) {
        self.step += 1;
        tracing::debug!(event = "trainingstate_step", "TrainingState: step -> {}", self.step);
    }

    pub fn next_epoch(&mut self) {
        self.epoch += 1;
        self.step = 0;
        self.total_loss = 0.0;
        tracing::info!(event = "trainingstate_epoch", "TrainingState: epoch -> {}", self.epoch);
    }

    pub fn avg_loss(&self) -> f32 {
        if self.step == 0 {
            return 0.0;
        }
        self.total_loss / self.step as f32
    }

    pub fn summary(&self) -> String {
        format!(
            "epoch={} step={} avg_loss={:.6} best_loss={:.6}",
            self.epoch,
            self.step,
            self.avg_loss(),
            self.best_loss,
        )
    }

    /// Returns `true` if this training state has async gradient offload enabled.
    pub fn has_async_offload(&self) -> bool {
        self.async_offload.is_some()
    }

    /// Transition to training phase for an integrated-GPU device.
    ///
    /// Hands the KV-cache buffer to the borrowed-buffer pool and returns the
    /// raw [`GpuBuffer`] slice that the backward pass should write gradients
    /// into.  For non-integrated profiles this is a cheap no-op that drops the
    /// provided cache and returns an empty `Vec`.
    ///
    /// Call this **before** the backward pass.
    pub fn begin_training_step(
        &mut self,
        kv_cache: BlockCache,
    ) -> Vec<crate::compute::GpuBuffer> {
        if self.borrowed_pool.is_enabled() {
            tracing::debug!(
                "TrainingState: transitioning to Training phase (borrowed-buffer strategy active)"
            );
        }
        self.borrowed_pool.transition_to_training(kv_cache)
    }

    /// Transition back to inference phase after an optimizer step.
    ///
    /// Reconstructs the [`BlockCache`] from the returned gradient buffers.
    /// For non-integrated profiles allocates a fresh cache.
    ///
    /// Call this **after** the optimizer step.
    pub fn end_training_step(
        &mut self,
        gradient_bufs: Vec<crate::compute::GpuBuffer>,
        hidden_dim: usize,
        cache_capacity: usize,
    ) -> Result<BlockCache> {
        let device = self
            .device
            .clone()
            .ok_or_else(|| crate::error::FerrisResError::Device(
                "TrainingState::end_training_step called without a device".into(),
            ))?;
        if self.borrowed_pool.is_enabled() {
            tracing::debug!(
                "TrainingState: restoring KV cache (borrowed-buffer strategy)"
            );
        }
        self.borrowed_pool
            .transition_to_inference(gradient_bufs, device, hidden_dim, cache_capacity)
    }

    /// Returns the current memory phase of the borrowed-buffer pool.
    pub fn memory_phase(&self) -> DeviceMemoryPhase {
        self.borrowed_pool.phase()
    }
}

impl fmt::Display for TrainingState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TrainingConfig {
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f32,
    pub gradient_clip_norm: f32,
    pub log_every_n_steps: u32,
    pub checkpoint_every_n_epochs: u32,
    /// Activation checkpoint granularity.
    /// When `None`, all intermediate activations are kept live in VRAM.
    /// For `DeviceProfile::Integrated`, the training loop auto-promotes this
    /// to `PerBlock` if it is still `None` at training start.
    pub checkpoint_granularity: CheckpointGranularity,
    /// Number of micro-batches to accumulate gradients over before calling the
    /// optimizer step.  Defaults to 1 (no accumulation).  For
    /// `DeviceProfile::Integrated` / `CpuOffload` mode the training loop should
    /// set this to >= 4 to amortise the GPU↔CPU transfer overhead.
    pub gradient_accumulation_steps: u32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 32,
            learning_rate: 1e-3,
            gradient_clip_norm: 1.0,
            log_every_n_steps: 10,
            checkpoint_every_n_epochs: 1,
            checkpoint_granularity: CheckpointGranularity::None,
            gradient_accumulation_steps: 1,
        }
    }
}

impl TrainingConfig {
    pub fn new(epochs: u32, batch_size: u32, learning_rate: f32) -> Self {
        Self {
            epochs,
            batch_size,
            learning_rate,
            ..Default::default()
        }
    }

    /// Apply device-profile-aware defaults.
    ///
    /// For `DeviceProfile::Integrated`, auto-promotes `checkpoint_granularity`
    /// from `None` to `PerBlock` to reduce peak VRAM usage.  Callers that
    /// have already set an explicit non-None value are not overridden.
    ///
    /// Also sets `gradient_accumulation_steps` to 4 for `Integrated` devices
    /// when it has not been explicitly configured (i.e., is still 1), as this
    /// amortises GPU↔CPU transfer overhead in CpuOffload mode.
    pub fn apply_device_profile(&mut self, profile: crate::device::profile::DeviceProfile) {
        if profile == crate::device::profile::DeviceProfile::Integrated
            && self.checkpoint_granularity == CheckpointGranularity::None
        {
            self.checkpoint_granularity = CheckpointGranularity::PerBlock;
            tracing::info!(
                "TrainingConfig: auto-enabled PerBlock checkpointing for DeviceProfile::Integrated"
            );
        }
        if profile == crate::device::profile::DeviceProfile::Integrated
            && self.gradient_accumulation_steps == 1
        {
            self.gradient_accumulation_steps = 4;
            tracing::info!(
                "TrainingConfig: auto-set gradient_accumulation_steps=4 for DeviceProfile::Integrated (CpuOffload mode)"
            );
        }
    }
}
