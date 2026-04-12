//! Vision encoder for Phase 7 multimodal tokenization.
//!
//! Supports two patch embedding backends:
//!   1. **Explicit im2col** — legacy path via `Im2ColOp` + `MatMulOp`.
//!   2. **Implicit GEMM** (default) — fused via `FusedPatchEmbedOp`, zero
//!      intermediate buffer.
//!
//! Optional **Token Merging (ToMe)** can be enabled to reduce the visual token
//! count before feeding into the text transformer. This is controlled by
//! `VisionConfig::tome_r`.
//!
//! Task: b3f74a12, c9e2d541 — see papers_research/ for details.

use std::sync::Arc;
use wgpu::{Device, Queue};
use crate::compute::GpuBuffer;
use crate::compute::kernels::fused_patch_embed::FusedPatchEmbedOp;
use crate::compute::kernels::im2col::Im2ColOp;
use crate::compute::kernels::matmul::MatMulOp;
use crate::compute::kernels::tome_merge::{TomeMergeOp, bipartite_match};
use crate::error::Result;

/// Configuration for the vision encoder.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Image height in pixels (must be divisible by patch_size).
    pub image_height: u32,
    /// Image width in pixels (must be divisible by patch_size).
    pub image_width: u32,
    /// Number of image channels (typically 3 for RGB).
    pub channels: u32,
    /// Patch side length in pixels.
    pub patch_size: u32,
    /// Projection output dimensionality (the token embedding dim).
    pub embed_dim: u32,
    /// Use the fused Implicit GEMM kernel instead of explicit im2col + matmul.
    pub use_implicit_gemm: bool,
    /// Number of token pairs to merge per layer via ToMe (0 = disabled).
    pub tome_r: u32,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            image_height: 224,
            image_width: 224,
            channels: 3,
            patch_size: 16,
            embed_dim: 768,
            use_implicit_gemm: true,
            tome_r: 0,
        }
    }
}

impl VisionConfig {
    /// Number of patches the encoder will produce.
    pub fn n_patches(&self) -> u32 {
        (self.image_height / self.patch_size) * (self.image_width / self.patch_size)
    }

    /// Inner dimension of the patch projection (P*P*C).
    pub fn patch_inner_dim(&self) -> u32 {
        self.patch_size * self.patch_size * self.channels
    }
}

/// Vision encoder — converts raw image tensors into token embeddings suitable
/// for the BlockAttnRes text transformer.
///
/// The encoder owns its projection weight (and optional bias) as GPU buffers
/// and dispatches either the fused Implicit GEMM kernel or the legacy
/// im2col+matmul pipeline.
pub struct VisionEncoder {
    config: VisionConfig,
    /// Projection weight `[P*P*C, embed_dim]`.
    weight: GpuBuffer,
    /// Optional projection bias `[embed_dim]`.
    bias: Option<GpuBuffer>,
    /// Fused Implicit GEMM kernel (used when `use_implicit_gemm = true`).
    fused_op: FusedPatchEmbedOp,
    /// Legacy im2col kernel.
    im2col_op: Im2ColOp,
    /// Legacy matmul kernel.
    matmul_op: MatMulOp,
    /// ToMe merge kernel.
    tome_op: TomeMergeOp,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl VisionEncoder {
    /// Create a new vision encoder with random (zero-initialized) weights.
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, config: VisionConfig) -> Result<Self> {
        let inner_dim = config.patch_inner_dim() as usize;
        let embed_dim = config.embed_dim as usize;

        let weight_bytes = inner_dim * embed_dim * std::mem::size_of::<f32>();
        let weight = GpuBuffer::zeros(&device, &queue, weight_bytes, Some("VisionEncoder Weight"))?;

        let bias = None; // No bias by default; can be loaded from a pre-trained checkpoint

        let fused_op = FusedPatchEmbedOp::new(&device, &queue);
        let im2col_op = Im2ColOp::new(&device, &queue);
        let matmul_op = MatMulOp::new(&device, &queue);
        let tome_op = TomeMergeOp::new(&device, &queue);

        tracing::info!(
            "VisionEncoder created: {}x{}x{} patch_size={} embed_dim={} n_patches={} implicit_gemm={} tome_r={}",
            config.image_height, config.image_width, config.channels,
            config.patch_size, config.embed_dim, config.n_patches(),
            config.use_implicit_gemm, config.tome_r,
        );

        Ok(Self {
            config,
            weight,
            bias,
            fused_op,
            im2col_op,
            matmul_op,
            tome_op,
            device,
            queue,
        })
    }

    /// Encode an image into visual token embeddings.
    ///
    /// # Arguments
    /// * `encoder`  – wgpu command encoder.
    /// * `image`    – Raw image tensor in HWC layout `[H, W, C]`.
    /// * `output`   – Output buffer. Must be sized for `[n_patches * embed_dim]` f32s
    ///   (or fewer if ToMe is enabled and this is a subsequent layer).
    ///
    /// # Returns
    /// The number of tokens written to `output` (equals `n_patches` if ToMe is
    /// disabled, or `n_patches - tome_r` if enabled).
    pub fn encode(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        image: &GpuBuffer,
        output: &GpuBuffer,
    ) -> Result<usize> {
        let n_patches = self.config.n_patches();

        if self.config.use_implicit_gemm {
            self.fused_op.dispatch(
                cmd_encoder,
                image,
                &self.weight,
                self.bias.as_ref(),
                output,
                self.config.image_height,
                self.config.image_width,
                self.config.channels,
                self.config.patch_size,
                self.config.embed_dim,
            )?;
        } else {
            // Legacy path: im2col → matmul
            let inner_dim = self.config.patch_inner_dim();
            let patch_buf_size = (n_patches * inner_dim) as usize * std::mem::size_of::<f32>();
            let patch_buf = GpuBuffer::zeros(
                &self.device, &self.queue, patch_buf_size, Some("im2col patches"),
            )?;

            self.im2col_op.dispatch_im2col(
                cmd_encoder,
                image,
                &patch_buf,
                self.config.image_height,
                self.config.image_width,
                self.config.channels,
                self.config.patch_size,
            )?;

            self.matmul_op.dispatch(
                cmd_encoder,
                &patch_buf,  // [N_patches, P*P*C]
                &self.weight, // [P*P*C, D]
                output,       // [N_patches, D]
                n_patches,
                inner_dim,
                self.config.embed_dim,
            )?;
        }

        Ok(n_patches as usize)
    }

    /// Encode with optional ToMe token merging.
    ///
    /// If `tome_r > 0`, performs CPU-side bipartite matching on key vectors
    /// (read back from GPU), then dispatches the GPU merge kernel.
    ///
    /// The caller must provide:
    /// * `key_buffer` – Key vectors `[n_tokens, dim]` on GPU (readable).
    /// * `token_embeddings` – Token embeddings `[n_tokens, dim]` on GPU.
    /// * `merged_output` – Output buffer sized for `[(n_tokens - tome_r), dim]`.
    /// * `merged_sizes` – Output buffer sized for `[(n_tokens - tome_r)]` f32s.
    ///
    /// Returns `(n_out, pair_a, pair_b)` for downstream KV cache management.
    pub fn encode_with_tome(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        image: &GpuBuffer,
        patch_output: &GpuBuffer,
        key_buffer: &GpuBuffer,
        token_sizes: &GpuBuffer,
        merged_output: &GpuBuffer,
        merged_sizes: &GpuBuffer,
    ) -> Result<(usize, Vec<u32>, Vec<u32>)> {
        // Step 1: Encode image → patch tokens
        let n_patches = self.encode(cmd_encoder, image, patch_output)?;

        if self.config.tome_r == 0 {
            // No merging — copy tokens directly
            return Ok((n_patches, Vec::new(), Vec::new()));
        }

        // Step 2: Read back key vectors for CPU-side matching
        // (For typical n_patches ≤ 500 this is acceptable latency)
        let keys_cpu = {
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tome key staging"),
                size: key_buffer.size() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("tome key readback") }
            );
            enc.copy_buffer_to_buffer(key_buffer.buffer(), 0, &staging, 0, key_buffer.size() as u64);
            self.queue.submit(std::iter::once(enc.finish()));

            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            rx.recv().ok().unwrap().ok().unwrap();
            let data = slice.get_mapped_range();
            let keys: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging.unmap();
            keys
        };

        // Step 3: Bipartite matching on CPU
        let dim = self.config.embed_dim as usize;
        let match_result = bipartite_match(&keys_cpu, n_patches, dim, self.config.tome_r as usize);

        // Step 4: Upload pair assignments and dispatch merge kernel
        let pair_a_bytes = match_result.pair_a.len() * std::mem::size_of::<u32>();
        let pair_b_bytes = match_result.pair_b.len() * std::mem::size_of::<u32>();

        let pair_a_buf = GpuBuffer::zeros(&self.device, &self.queue, pair_a_bytes, Some("tome pair_a"))?;
        let pair_b_buf = GpuBuffer::zeros(&self.device, &self.queue, pair_b_bytes, Some("tome pair_b"))?;

        self.queue.write_buffer(pair_a_buf.buffer(), 0, bytemuck::cast_slice(&match_result.pair_a));
        self.queue.write_buffer(pair_b_buf.buffer(), 0, bytemuck::cast_slice(&match_result.pair_b));

        self.tome_op.dispatch(
            cmd_encoder,
            patch_output,
            token_sizes,
            &pair_a_buf,
            &pair_b_buf,
            merged_output,
            merged_sizes,
            match_result.n_out as u32,
            self.config.embed_dim,
        )?;

        Ok((match_result.n_out, match_result.pair_a, match_result.pair_b))
    }

    /// Get the configuration.
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// Get the projection weight buffer (for loading pre-trained weights).
    pub fn weight_buffer(&self) -> &GpuBuffer {
        &self.weight
    }

    /// Get the bias buffer if present (for loading pre-trained weights).
    pub fn bias_buffer(&self) -> Option<&GpuBuffer> {
        self.bias.as_ref()
    }
}
