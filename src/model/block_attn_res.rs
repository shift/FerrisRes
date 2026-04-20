use std::cell::RefCell;
use std::sync::Arc;
use wgpu::{Device, Queue};
use crate::compute::GpuBuffer;
use crate::compute::kernels::matmul::MatMulOp;
use crate::compute::kernels::rmsnorm::RmsNormOp;
use crate::compute::kernels::softmax::SoftmaxOp;
use crate::compute::kernels::elementwise::ElementWiseOp;
use crate::compute::kernels::rope::RopeOp;
use crate::compute::kernels::flash_decode::FlashDecodeOp;
use crate::compute::kernels::causal_mask::CausalMaskOp;
use crate::compute::kernels::prefill_attn::PrefillAttnOp;
use crate::model::config::BlockAttnResConfig;
use crate::model::linear::Linear;
use crate::model::moe_linear::MoELinear;
use crate::inference::kv_cache::LayerKVCache;
use crate::training::{CheckpointGranularity, CheckpointStore};
use crate::error::Result;

const HEAD_WEIGHT_MUL_WGSL: &str = r#"
    struct Params {
        batch_size: u32,
        num_heads: u32,
        head_dim: u32,
    }

    @group(0) @binding(0) var<storage, read> weights: array<f32>;
    @group(0) @binding(1) var<storage, read> values: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;
    @group(0) @binding(3) var<uniform> p: Params;

    @compute @workgroup_size(256)
    fn head_weight_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x;
        let total = p.batch_size * p.num_heads * p.head_dim;
        if (idx >= total) {
            return;
        }

        let b = idx / (p.num_heads * p.head_dim);
        let remainder = idx % (p.num_heads * p.head_dim);
        let h = remainder / p.head_dim;

        let w = weights[b * p.num_heads + h];
        output[idx] = w * values[idx];
    }
    "#;

struct HeadWeightMulOp {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl HeadWeightMulOp {
    fn new(device: &Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Head Weight Mul Shader"),
            source: wgpu::ShaderSource::Wgsl(HEAD_WEIGHT_MUL_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Head Weight Mul BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Head Weight Mul Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Head Weight Mul Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("head_weight_mul"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self { pipeline, bind_group_layout }
    }
}

pub struct BlockAttnResLayer {
    layer_number: usize,
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_dim: usize,

    // Attention
    attn_norm: RmsNormOp,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    /// Per-head Q RMSNorm (Gemma 4 feature 3)
    q_norm: RmsNormOp,
    /// Per-head K RMSNorm (Gemma 4 feature 3)
    k_norm: RmsNormOp,
    /// Per-head V RMSNorm (Gemma 4 feature 4)
    v_norm: RmsNormOp,
    /// Post-attention RMSNorm (Gemma 4 feature 1)
    post_attn_norm: RmsNormOp,

    // FFN
    ff_norm: RmsNormOp,
    ff_up: Option<Linear>,
    ff_gate: Option<Linear>,
    ff_down: Option<Linear>,
    moe_linear: Option<RefCell<MoELinear>>,
    /// Post-FFN RMSNorm (Gemma 4 feature 2)
    post_ffn_norm: RmsNormOp,

    // Layer scalar (Gemma 4 feature 5)
    layer_scalar: f32,

    // PLE (Gemma 4 feature 6)
    ple_input_gate: Option<Linear>,
    ple_projection: Option<Linear>,
    ple_post_norm: Option<RmsNormOp>,

    // RoPE parameters (Gemma 4 feature 8)
    rope_theta: f64,
    partial_rotary_factor: f32,

    // Activation (Gemma 4 feature 10)
    use_gelu: bool,

    // KV sharing (Gemma 4 feature 7)
    kv_shared: bool,

    #[allow(dead_code)]
    pseudo_query: GpuBuffer,
    #[allow(dead_code)]
    attn_res_proj: Linear,
    attn_res_norm: RmsNormOp,

    elementwise: ElementWiseOp,
    matmul: MatMulOp,
    head_weight_mul: HeadWeightMulOp,
    softmax: SoftmaxOp,
    rope: RopeOp,
    #[allow(dead_code)]
    causal_mask: CausalMaskOp,
    flash_decode: FlashDecodeOp,
    prefill_attn: PrefillAttnOp,

    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl BlockAttnResLayer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: &BlockAttnResConfig,
        layer_number: usize,
    ) -> Result<Self> {
        Self::new_with_params(
            device, queue, config, layer_number,
            1,      // num_kv_heads
            config.hidden_dim / config.attention_heads, // head_dim
            config.intermediate_dim,
            10000.0, // rope_theta
            1.0,     // partial_rotary_factor
            false,   // use_gelu
            false,   // kv_shared
        )
    }

    /// Create a layer with per-layer Gemma 4 parameters.
    pub fn new_with_params(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: &BlockAttnResConfig,
        layer_number: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
        rope_theta: f64,
        partial_rotary_factor: f32,
        use_gelu: bool,
        kv_shared: bool,
    ) -> Result<Self> {
        let num_heads = config.attention_heads;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        tracing::info!(
            "Creating BlockAttnResLayer {}: hidden={} q_dim={} kv_dim={} intermediate={} heads={} kv_heads={} head_dim={} theta={} rotary={} gelu={} kv_shared={}",
            layer_number, config.hidden_dim, q_dim, kv_dim, intermediate_dim,
            num_heads, num_kv_heads, head_dim, rope_theta, partial_rotary_factor, use_gelu, kv_shared
        );

        let attn_norm = RmsNormOp::new(&device)?;
        let ff_norm = RmsNormOp::new(&device)?;
        let attn_res_norm = RmsNormOp::new(&device)?;

        // Per-head Q/K/V norms (Gemma 4 features 3-4)
        let q_norm = RmsNormOp::new(&device)?;
        let k_norm = RmsNormOp::new(&device)?;
        let v_norm = RmsNormOp::new(&device)?;

        // Post-attention and post-FFN norms (Gemma 4 features 1-2)
        let post_attn_norm = RmsNormOp::new(&device)?;
        let post_ffn_norm = RmsNormOp::new(&device)?;

        let q_proj = Linear::new(
            Arc::clone(&device), Arc::clone(&queue),
            config.hidden_dim, q_dim, false,
        )?;
        let k_proj = Linear::new(
            Arc::clone(&device), Arc::clone(&queue),
            config.hidden_dim, kv_dim, false,
        )?;
        let v_proj = Linear::new(
            Arc::clone(&device), Arc::clone(&queue),
            config.hidden_dim, kv_dim, false,
        )?;
        let out_proj = Linear::new(
            Arc::clone(&device), Arc::clone(&queue),
            q_dim, config.hidden_dim, false,
        )?;

        let (ff_up, ff_gate, ff_down, moe_linear) = if config.use_moe {
            let moe = MoELinear::new(
                &device, &queue,
                config.hidden_dim, intermediate_dim,
                config.num_experts, config.top_k,
            )?;
            (None, None, None, Some(RefCell::new(moe)))
        } else {
            let gate = Linear::new(
                Arc::clone(&device), Arc::clone(&queue),
                config.hidden_dim, intermediate_dim, false,
            )?;
            let up = Linear::new(
                Arc::clone(&device), Arc::clone(&queue),
                config.hidden_dim, intermediate_dim, false,
            )?;
            let down = Linear::new(
                Arc::clone(&device), Arc::clone(&queue),
                intermediate_dim, config.hidden_dim, false,
            )?;
            (Some(up), Some(gate), Some(down), None)
        };

        let pseudo_query_bytes = config.hidden_dim * std::mem::size_of::<f32>();
        let pseudo_query = GpuBuffer::zeros(&device, &queue, pseudo_query_bytes, Some("Pseudo Query"))?;

        let attn_res_proj = Linear::new(
            Arc::clone(&device), Arc::clone(&queue),
            config.hidden_dim, 2 * config.hidden_dim, false,
        )?;

        let elementwise = ElementWiseOp::new(&device, &queue);
        let softmax = SoftmaxOp::new(&device, &queue)?;
        let matmul = MatMulOp::new(&device, &queue);
        let head_weight_mul = HeadWeightMulOp::new(&device);
        let rope = RopeOp::new(&device)?;
        let causal_mask = CausalMaskOp::new(&device)?;
        let flash_decode = FlashDecodeOp::new(&device, &queue)?;
        let prefill_attn = PrefillAttnOp::new(&device)?;

        tracing::info!(
            "BlockAttnResLayer {} created: heads={} kv_heads={} head_dim={}",
            layer_number, num_heads, num_kv_heads, head_dim
        );

        Ok(Self {
            layer_number,
            hidden_dim: config.hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,

            // Attention
            attn_norm,
            q_proj, k_proj, v_proj, out_proj,
            q_norm, k_norm, v_norm,
            post_attn_norm,

            // FFN
            ff_norm,
            ff_up, ff_gate, ff_down,
            moe_linear,
            post_ffn_norm,

            // Layer scalar
            layer_scalar: 1.0,

            // PLE — none by default, set via setter
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,

            // RoPE
            rope_theta,
            partial_rotary_factor,

            // Activation
            use_gelu,

            // KV sharing
            kv_shared,

            pseudo_query,
            attn_res_proj,
            attn_res_norm,
            elementwise,
            matmul,
            head_weight_mul,
            softmax,
            rope,
            causal_mask,
            flash_decode,
            prefill_attn,
            device,
            queue,
        })
    }

    // --- Setters for per-layer weights (for Gemma 4 weight loading) ---

    /// Set the layer scalar (feature 5).
    pub fn set_layer_scalar(&mut self, scalar: f32) {
        self.layer_scalar = scalar;
    }

    /// Set PLE weights (feature 6).
    pub fn set_ple(
        &mut self,
        gate: Linear,
        projection: Linear,
        post_norm: RmsNormOp,
    ) {
        self.ple_input_gate = Some(gate);
        self.ple_projection = Some(projection);
        self.ple_post_norm = Some(post_norm);
    }

    /// Get per-layer RoPE parameters.
    pub fn rope_params(&self) -> (f64, f32) {
        (self.rope_theta, self.partial_rotary_factor)
    }

    /// Whether this layer uses shared KV.
    pub fn is_kv_shared(&self) -> bool {
        self.kv_shared
    }

    pub fn forward_intra_block(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuBuffer,
        output: &GpuBuffer,
        partial_sum: &GpuBuffer,
        batch_size: u32,
        checkpoint_granularity: CheckpointGranularity,
        checkpoint_store: Option<&mut CheckpointStore>,
    ) -> Result<()> {
        tracing::debug!(
            "BlockAttnResLayer::forward_intra_block layer={} batch={} heads={}",
            self.layer_number, batch_size, self.num_heads
        );

        if checkpoint_granularity == CheckpointGranularity::PerBlock {
            if let Some(store) = checkpoint_store {
                store.save(encoder, self.layer_number, "hidden_states_input", hidden_states)?;
            }
        }

        let hidden_dim = self.hidden_dim as u32;
        let intermediate_dim = self.intermediate_dim as u32;
        let numel = batch_size * hidden_dim;

        // --- Self-Attention ---

        // 1. Pre-attention RMSNorm
        let normed = GpuBuffer::new(
            &self.device,
            numel as usize * std::mem::size_of::<f32>(),
            Some("intra_normed"),
        )?;
        self.attn_norm.dispatch(
            &self.device, &self.queue, encoder,
            hidden_states, &normed, batch_size, hidden_dim,
        )?;

        // 2. Q, K, V projections (with per-layer dims)
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let q_buf = GpuBuffer::new(
            &self.device,
            batch_size as usize * q_dim * std::mem::size_of::<f32>(),
            Some("intra_q"),
        )?;
        let k_buf = GpuBuffer::new(
            &self.device,
            batch_size as usize * kv_dim * std::mem::size_of::<f32>(),
            Some("intra_k"),
        )?;
        let v_buf = GpuBuffer::new(
            &self.device,
            batch_size as usize * kv_dim * std::mem::size_of::<f32>(),
            Some("intra_v"),
        )?;
        self.q_proj.forward(encoder, &normed, &q_buf, batch_size)?;
        self.k_proj.forward(encoder, &normed, &k_buf, batch_size)?;
        self.v_proj.forward(encoder, &normed, &v_buf, batch_size)?;

        // 3. Per-head Q/K/V RMSNorm (Gemma 4 features 3-4)
        let q_normed = GpuBuffer::new(
            &self.device, q_buf.size() as usize, Some("intra_q_normed"),
        )?;
        let k_normed = GpuBuffer::new(
            &self.device, k_buf.size() as usize, Some("intra_k_normed"),
        )?;
        let v_normed = GpuBuffer::new(
            &self.device, v_buf.size() as usize, Some("intra_v_normed"),
        )?;
        self.q_norm.dispatch(
            &self.device, &self.queue, encoder,
            &q_buf, &q_normed, batch_size, q_dim as u32,
        )?;
        self.k_norm.dispatch(
            &self.device, &self.queue, encoder,
            &k_buf, &k_normed, batch_size, kv_dim as u32,
        )?;
        self.v_norm.dispatch(
            &self.device, &self.queue, encoder,
            &v_buf, &v_normed, batch_size, kv_dim as u32,
        )?;

        // 4. Scaled dot-product attention (seq_len=1 intra-block)
        let head_dim = self.head_dim as u32;
        let num_heads = self.num_heads as u32;

        let scores = GpuBuffer::new(
            &self.device,
            batch_size as usize * self.num_heads * std::mem::size_of::<f32>(),
            Some("intra_attn_scores"),
        )?;
        self.matmul.dispatch(
            encoder, &q_normed, &k_normed, &scores,
            batch_size * num_heads, head_dim, 1u32,
        )?;

        let scale_factor = 1.0f32 / (self.head_dim as f32).sqrt();
        self.elementwise.dispatch_scale(encoder, &scores, &scores, scale_factor, batch_size * num_heads)?;

        let attn_weights = GpuBuffer::new(
            &self.device,
            batch_size as usize * self.num_heads * std::mem::size_of::<f32>(),
            Some("intra_attn_weights"),
        )?;
        self.softmax.dispatch(encoder, &scores, &attn_weights, batch_size, num_heads)?;

        let attn_out = GpuBuffer::new(
            &self.device,
            batch_size as usize * q_dim * std::mem::size_of::<f32>(),
            Some("intra_attn_out"),
        )?;
        self.dispatch_head_weight_mul(
            encoder, &attn_weights, &v_normed, &attn_out,
            batch_size, num_heads, head_dim,
        )?;

        // 5. Output projection
        let proj_out = GpuBuffer::new(
            &self.device,
            numel as usize * std::mem::size_of::<f32>(),
            Some("intra_proj_out"),
        )?;
        self.out_proj.forward(encoder, &attn_out, &proj_out, batch_size)?;

        // 6. Post-attention RMSNorm + layer scalar + residual (features 1, 5)
        let post_attn = GpuBuffer::new(
            &self.device, proj_out.size() as usize, Some("intra_post_attn"),
        )?;
        self.post_attn_norm.dispatch(
            &self.device, &self.queue, encoder,
            &proj_out, &post_attn, batch_size, hidden_dim,
        )?;
        if (self.layer_scalar - 1.0).abs() > 1e-8 {
            self.elementwise.dispatch_scale(encoder, &post_attn, &post_attn, self.layer_scalar, numel)?;
        }
        self.elementwise.dispatch_add(encoder, hidden_states, &post_attn, output, numel)?;

        // --- Feed-Forward Network ---

        // 7. Pre-FFN RMSNorm
        let ff_normed = GpuBuffer::new(
            &self.device,
            numel as usize * std::mem::size_of::<f32>(),
            Some("intra_ff_normed"),
        )?;
        self.ff_norm.dispatch(
            &self.device, &self.queue, encoder,
            output, &ff_normed, batch_size, hidden_dim,
        )?;

        // 8. FFN with gated activation (feature 10: GELU vs ReLU)
        if let Some(ref moe) = self.moe_linear {
            let mut moe_ref = moe.borrow_mut();
            let ff_out = moe_ref.forward(encoder, &ff_normed, batch_size as usize)?;

            let post_ffn = GpuBuffer::new(
                &self.device, ff_out.size() as usize, Some("intra_post_ffn_moe"),
            )?;
            self.post_ffn_norm.dispatch(
                &self.device, &self.queue, encoder,
                ff_out, &post_ffn, batch_size, hidden_dim,
            )?;
            if (self.layer_scalar - 1.0).abs() > 1e-8 {
                self.elementwise.dispatch_scale(encoder, &post_ffn, &post_ffn, self.layer_scalar, numel)?;
            }
            self.elementwise.dispatch_add(encoder, output, &post_ffn, output, numel)?;
        } else {
            let ff_gate_buf = GpuBuffer::new(
                &self.device,
                batch_size as usize * self.intermediate_dim * std::mem::size_of::<f32>(),
                Some("intra_ff_gate"),
            )?;
            let ff_up_buf = GpuBuffer::new(
                &self.device,
                batch_size as usize * self.intermediate_dim * std::mem::size_of::<f32>(),
                Some("intra_ff_up"),
            )?;

            self.ff_gate.as_ref().unwrap().forward(encoder, &ff_normed, &ff_gate_buf, batch_size)?;
            self.ff_up.as_ref().unwrap().forward(encoder, &ff_normed, &ff_up_buf, batch_size)?;

            // Activation: act(up)
            if self.use_gelu {
                self.elementwise.dispatch_gelu(
                    encoder, &ff_up_buf, &ff_up_buf, batch_size * intermediate_dim,
                )?;
            } else {
                self.elementwise.dispatch_relu(
                    encoder, &ff_up_buf, &ff_up_buf, batch_size * intermediate_dim,
                )?;
            }
            // Gated: gate * act(up)
            self.elementwise.dispatch_mul(
                encoder, &ff_gate_buf, &ff_up_buf, &ff_up_buf, batch_size * intermediate_dim,
            )?;

            let ff_out = GpuBuffer::new(
                &self.device,
                numel as usize * std::mem::size_of::<f32>(),
                Some("intra_ff_out"),
            )?;
            self.ff_down.as_ref().unwrap().forward(encoder, &ff_up_buf, &ff_out, batch_size)?;

            // Post-FFN RMSNorm + layer scalar + residual (features 2, 5)
            let post_ffn = GpuBuffer::new(
                &self.device, ff_out.size() as usize, Some("intra_post_ffn"),
            )?;
            self.post_ffn_norm.dispatch(
                &self.device, &self.queue, encoder,
                &ff_out, &post_ffn, batch_size, hidden_dim,
            )?;
            if (self.layer_scalar - 1.0).abs() > 1e-8 {
                self.elementwise.dispatch_scale(encoder, &post_ffn, &post_ffn, self.layer_scalar, numel)?;
            }
            self.elementwise.dispatch_add(encoder, output, &post_ffn, output, numel)?;
        }

        self.elementwise.dispatch_add(encoder, partial_sum, output, partial_sum, numel)?;

        tracing::debug!(
            "BlockAttnResLayer::forward_intra_block layer={} complete",
            self.layer_number
        );

        Ok(())
    }

    pub fn forward_decode_token(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuBuffer,
        kv_cache: &LayerKVCache,
    ) -> Result<GpuBuffer> {
        self.forward_decode_token_with_pos(encoder, hidden_states, kv_cache, None)
    }

    /// Decode a single token with optional position override for YaRN/StreamingLLM.
    /// When `effective_pos` is `Some(pos)`, uses that position for RoPE instead of
    /// the KV cache length. This enables context extension methods to remap
    /// positions without modifying the cache structure.
    pub fn forward_decode_token_with_pos(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuBuffer,
        kv_cache: &LayerKVCache,
        effective_pos: Option<u32>,
    ) -> Result<GpuBuffer> {
        let hidden_dim = self.hidden_dim;
        let num_heads = self.num_heads as u32;
        let head_dim = self.head_dim as u32;
        let intermediate_dim = self.intermediate_dim as u32;
        let f32_size = std::mem::size_of::<f32>();

        let normed = GpuBuffer::new(
            &self.device,
            hidden_dim * f32_size,
            Some("decode_normed"),
        )?;
        self.attn_norm.dispatch(
            &self.device,
            &self.queue,
            encoder,
            hidden_states,
            &normed,
            1u32,
            hidden_dim as u32,
        )?;

        let q_buf = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_q"))?;
        let k_buf = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_k"))?;
        let v_buf = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_v"))?;
        self.q_proj.forward(encoder, &normed, &q_buf, 1u32)?;
        self.k_proj.forward(encoder, &normed, &k_buf, 1u32)?;
        self.v_proj.forward(encoder, &normed, &v_buf, 1u32)?;

        let pos = effective_pos.unwrap_or_else(|| kv_cache.current_len());

        let rope_q = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_rope_q"))?;
        let rope_k = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_rope_k"))?;
        self.rope.dispatch_with_offset(encoder, &q_buf, &rope_q, 1u32, num_heads, head_dim, pos)?;
        self.rope.dispatch_with_offset(encoder, &k_buf, &rope_k, 1u32, num_heads, head_dim, pos)?;

        let new_len = kv_cache.update(encoder, &rope_k, &v_buf)?;

        let attn_out = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_attn_out"))?;
        self.flash_decode.dispatch(
            encoder,
            &rope_q,
            kv_cache.key_buffer(),
            kv_cache.value_buffer(),
            &attn_out,
            new_len,
            num_heads,
            head_dim,
        )?;

        let proj_out = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_proj_out"))?;
        self.out_proj.forward(encoder, &attn_out, &proj_out, 1u32)?;

        let residual1 = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_residual1"))?;
        self.elementwise.dispatch_add(encoder, hidden_states, &proj_out, &residual1, hidden_dim as u32)?;

        let ff_normed = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_ff_normed"))?;
        self.ff_norm.dispatch(
            &self.device,
            &self.queue,
            encoder,
            &residual1,
            &ff_normed,
            1u32,
            hidden_dim as u32,
        )?;

        let output = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_output"))?;

        if let Some(ref moe) = self.moe_linear {
            let mut moe_ref = moe.borrow_mut();
            let ff_out = moe_ref.forward(encoder, &ff_normed, 1)?;
            self.elementwise.dispatch_add(encoder, &residual1, ff_out, &output, hidden_dim as u32)?;
        } else {
            let ff_hidden = GpuBuffer::new(
                &self.device,
                self.intermediate_dim * f32_size,
                Some("decode_ff_hidden"),
            )?;
            self.ff_up.as_ref().unwrap().forward(encoder, &ff_normed, &ff_hidden, 1u32)?;

            let ff_relu = GpuBuffer::new(
                &self.device,
                self.intermediate_dim * f32_size,
                Some("decode_ff_relu"),
            )?;
            self.elementwise.dispatch_relu(encoder, &ff_hidden, &ff_relu, intermediate_dim)?;

            let ff_out = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_ff_out"))?;
            self.ff_down.as_ref().unwrap().forward(encoder, &ff_relu, &ff_out, 1u32)?;

            self.elementwise.dispatch_add(encoder, &residual1, &ff_out, &output, hidden_dim as u32)?;
        }

        Ok(output)
    }

    /// Optimized decode: project K directly into KV cache, apply RoPE in-place.
    ///
    /// Eliminates the per-step `copy_buffer_to_buffer` by writing the K
    /// projection directly into the cache slot and applying RoPE in-place.
    /// The V projection still uses the regular path since V doesn't need RoPE.
    ///
    /// Callers should prefer this over `forward_decode_token_with_pos` for
    /// single-token decode steps.
    pub fn forward_decode_token_direct(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuBuffer,
        kv_cache: &LayerKVCache,
        effective_pos: Option<u32>,
    ) -> Result<GpuBuffer> {
        let hidden_dim = self.hidden_dim;
        let num_heads = self.num_heads as u32;
        let head_dim = self.head_dim as u32;
        let intermediate_dim = self.intermediate_dim as u32;
        let f32_size = std::mem::size_of::<f32>();

        // RMSNorm on input
        let normed = GpuBuffer::new(
            &self.device,
            hidden_dim * f32_size,
            Some("decode_direct_normed"),
        )?;
        self.attn_norm.dispatch(
            &self.device,
            &self.queue,
            encoder,
            hidden_states,
            &normed,
            1u32,
            hidden_dim as u32,
        )?;

        // Q projection → temp buffer (Q is consumed by flash_decode, not cached)
        let q_buf = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_direct_q"))?;
        self.q_proj.forward(encoder, &normed, &q_buf, 1u32)?;

        // Compute the cache slot offset
        let pos = effective_pos.unwrap_or_else(|| kv_cache.current_len());
        let per_pos_bytes = num_heads as u64 * head_dim as u64 * f32_size as u64;
        let slot_offset = kv_cache.current_len() as u64 * per_pos_bytes;
        let row_bytes = hidden_dim as u64 * f32_size as u64;

        // K projection → directly into KV cache slot
        self.k_proj.forward_into_offset(
            encoder,
            &normed,
            kv_cache.key_buffer().buffer(),
            slot_offset,
            row_bytes,
            1u32,
        )?;

        // RoPE in-place on the cache slot we just wrote
        self.rope.dispatch_inplace_at_offset(
            encoder,
            kv_cache.key_buffer().buffer(),
            slot_offset,
            row_bytes,
            1u32,
            num_heads,
            head_dim,
            pos,
        )?;

        // V projection → temp buffer (V is appended to cache via update)
        let v_buf = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_direct_v"))?;
        self.v_proj.forward(encoder, &normed, &v_buf, 1u32)?;

        // RoPE on Q (Q needs RoPE but goes to flash_decode, not cache)
        let rope_q = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_direct_rope_q"))?;
        self.rope.dispatch_with_offset(encoder, &q_buf, &rope_q, 1u32, num_heads, head_dim, pos)?;

        // V: copy into cache (no RoPE needed for V)
        // K is already in-place via direct projection, so we only copy V
        // and manually bump the counter.
        let v_slot_offset = kv_cache.current_len() as u64 * per_pos_bytes;
        let v_copy_size = (v_buf.size() as u64).min(per_pos_bytes);
        encoder.copy_buffer_to_buffer(
            v_buf.buffer(),
            0,
            kv_cache.value_buffer().buffer(),
            v_slot_offset,
            v_copy_size,
        );
        kv_cache.increment_len();
        let new_len = kv_cache.current_len();

        // Flash decode attention
        let attn_out = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_direct_attn_out"))?;
        self.flash_decode.dispatch(
            encoder,
            &rope_q,
            kv_cache.key_buffer(),
            kv_cache.value_buffer(),
            &attn_out,
            new_len,
            num_heads,
            head_dim,
        )?;

        // Output projection + residual + FFN (same as regular decode)
        let proj_out = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_direct_proj"))?;
        self.out_proj.forward(encoder, &attn_out, &proj_out, 1u32)?;

        let residual1 = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_direct_res1"))?;
        self.elementwise.dispatch_add(encoder, hidden_states, &proj_out, &residual1, hidden_dim as u32)?;

        let output = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_direct_output"))?;

        let ff_normed = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_direct_ff_norm"))?;
        self.ff_norm.dispatch(
            &self.device,
            &self.queue,
            encoder,
            &residual1,
            &ff_normed,
            1u32,
            hidden_dim as u32,
        )?;

        if let Some(ref moe) = self.moe_linear {
            let mut moe_ref = moe.borrow_mut();
            let ff_out = moe_ref.forward(encoder, &ff_normed, 1)?;
            self.elementwise.dispatch_add(encoder, &residual1, ff_out, &output, hidden_dim as u32)?;
        } else {
            let ff_hidden = GpuBuffer::new(
                &self.device,
                self.intermediate_dim * f32_size,
                Some("decode_direct_ff_hidden"),
            )?;
            self.ff_up.as_ref().unwrap().forward(encoder, &ff_normed, &ff_hidden, 1u32)?;

            let ff_relu = GpuBuffer::new(
                &self.device,
                self.intermediate_dim * f32_size,
                Some("decode_direct_ff_relu"),
            )?;
            self.elementwise.dispatch_relu(encoder, &ff_hidden, &ff_relu, intermediate_dim)?;

            let ff_out = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_direct_ff_out"))?;
            self.ff_down.as_ref().unwrap().forward(encoder, &ff_relu, &ff_out, 1u32)?;

            self.elementwise.dispatch_add(encoder, &residual1, &ff_out, &output, hidden_dim as u32)?;
        }

        Ok(output)
    }

    pub fn forward_prefill(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuBuffer,
        kv_cache: &LayerKVCache,
        seq_len: u32,
    ) -> Result<GpuBuffer> {
        let hidden_dim = self.hidden_dim;
        let num_heads = self.num_heads as u32;
        let head_dim = self.head_dim as u32;
        let intermediate_dim = self.intermediate_dim as u32;
        let f32_size = std::mem::size_of::<f32>();
        let numel = seq_len * hidden_dim as u32;

        let normed = GpuBuffer::new(
            &self.device,
            seq_len as usize * hidden_dim * f32_size,
            Some("prefill_normed"),
        )?;
        self.attn_norm.dispatch(
            &self.device,
            &self.queue,
            encoder,
            hidden_states,
            &normed,
            seq_len,
            hidden_dim as u32,
        )?;

        let q_buf = GpuBuffer::new(&self.device, seq_len as usize * hidden_dim * f32_size, Some("prefill_q"))?;
        let k_buf = GpuBuffer::new(&self.device, seq_len as usize * hidden_dim * f32_size, Some("prefill_k"))?;
        let v_buf = GpuBuffer::new(&self.device, seq_len as usize * hidden_dim * f32_size, Some("prefill_v"))?;
        self.q_proj.forward(encoder, &normed, &q_buf, seq_len)?;
        self.k_proj.forward(encoder, &normed, &k_buf, seq_len)?;
        self.v_proj.forward(encoder, &normed, &v_buf, seq_len)?;

        let pos = kv_cache.current_len();

        let rope_q = GpuBuffer::new(&self.device, seq_len as usize * hidden_dim * f32_size, Some("prefill_rope_q"))?;
        let rope_k = GpuBuffer::new(&self.device, seq_len as usize * hidden_dim * f32_size, Some("prefill_rope_k"))?;
        self.rope.dispatch_with_offset(encoder, &q_buf, &rope_q, seq_len, num_heads, head_dim, pos)?;
        self.rope.dispatch_with_offset(encoder, &k_buf, &rope_k, seq_len, num_heads, head_dim, pos)?;

        let _new_len = kv_cache.update_batch(encoder, &rope_k, &v_buf, seq_len)?;

        let attn_out = GpuBuffer::new(
            &self.device,
            seq_len as usize * hidden_dim * f32_size,
            Some("prefill_attn_out"),
        )?;
        // rope_q and rope_k are both [seq_len, num_heads, head_dim].
        // v_buf is [seq_len, num_heads, head_dim].
        // PrefillAttnOp correctly handles this layout: each thread owns
        // one (query_pos, head) pair, indexes k/v as
        //   s * num_heads * head_dim + h * head_dim + d
        // and applies causal masking + online softmax internally.
        self.prefill_attn.dispatch(
            encoder,
            &rope_q,
            &rope_k,
            &v_buf,
            &attn_out,
            seq_len,
            num_heads,
            head_dim,
        )?;

        let proj_out = GpuBuffer::new(&self.device, seq_len as usize * hidden_dim * f32_size, Some("prefill_proj_out"))?;
        self.out_proj.forward(encoder, &attn_out, &proj_out, seq_len)?;

        let residual1 = GpuBuffer::new(&self.device, seq_len as usize * hidden_dim * f32_size, Some("prefill_residual1"))?;
        self.elementwise.dispatch_add(encoder, hidden_states, &proj_out, &residual1, numel)?;

        let ff_normed = GpuBuffer::new(&self.device, seq_len as usize * hidden_dim * f32_size, Some("prefill_ff_normed"))?;
        self.ff_norm.dispatch(
            &self.device,
            &self.queue,
            encoder,
            &residual1,
            &ff_normed,
            seq_len,
            hidden_dim as u32,
        )?;

        let output = GpuBuffer::new(&self.device, seq_len as usize * hidden_dim * f32_size, Some("prefill_output"))?;

        if let Some(ref moe) = self.moe_linear {
            let mut moe_ref = moe.borrow_mut();
            let ff_out = moe_ref.forward(encoder, &ff_normed, seq_len as usize)?;
            self.elementwise.dispatch_add(encoder, &residual1, ff_out, &output, numel)?;
        } else {
            let ff_hidden = GpuBuffer::new(
                &self.device,
                seq_len as usize * self.intermediate_dim * f32_size,
                Some("prefill_ff_hidden"),
            )?;
            self.ff_up.as_ref().unwrap().forward(encoder, &ff_normed, &ff_hidden, seq_len)?;

            let ff_relu = GpuBuffer::new(
                &self.device,
                seq_len as usize * self.intermediate_dim * f32_size,
                Some("prefill_ff_relu"),
            )?;
            self.elementwise.dispatch_relu(encoder, &ff_hidden, &ff_relu, seq_len * intermediate_dim)?;

            let ff_out = GpuBuffer::new(&self.device, seq_len as usize * hidden_dim * f32_size, Some("prefill_ff_out"))?;
            self.ff_down.as_ref().unwrap().forward(encoder, &ff_relu, &ff_out, seq_len)?;

            self.elementwise.dispatch_add(encoder, &residual1, &ff_out, &output, numel)?;
        }

        Ok(output)
    }

    fn dispatch_head_weight_mul(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        weights: &GpuBuffer,
        values: &GpuBuffer,
        output: &GpuBuffer,
        batch_size: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        let params_data: [u32; 3] = [batch_size, num_heads, head_dim];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Head Weight Mul Params"),
            size: 12,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        params_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&params_data));
        params_buffer.unmap();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Head Weight Mul Bind Group"),
            layout: &self.head_weight_mul.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: values.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let total_elements = batch_size * num_heads * head_dim;
        let workgroup_count = (total_elements + 255) / 256;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Head Weight Mul Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.head_weight_mul.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);

        drop(pass);

        Ok(())
    }

    pub fn forward_inter_block(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        block_reps: &GpuBuffer,
        partial_sum: &GpuBuffer,
        attn_output: &GpuBuffer,
        num_blocks: u32,
        batch_size: u32,
    ) -> Result<()> {
        tracing::debug!(
            "BlockAttnResLayer::forward_inter_block layer={} num_blocks={} batch={}",
            self.layer_number, num_blocks, batch_size
        );

        let hidden_dim = self.hidden_dim as u32;
        let total_entries = num_blocks + 1;

        let normed_keys = GpuBuffer::new(
            &self.device,
            total_entries as usize * self.hidden_dim * std::mem::size_of::<f32>(),
            Some("inter_normed_keys"),
        )?;
        self.attn_res_norm.dispatch(
            &self.device,
            &self.queue,
            encoder,
            block_reps,
            &normed_keys,
            total_entries,
            hidden_dim,
        )?;

        let scale_factor = 1.0f32 / (self.hidden_dim as f32).sqrt();

        let scores = GpuBuffer::new(
            &self.device,
            batch_size as usize * total_entries as usize * std::mem::size_of::<f32>(),
            Some("inter_scores"),
        )?;

        self.matmul.dispatch(
            encoder,
            partial_sum,
            &normed_keys,
            &scores,
            batch_size,
            hidden_dim,
            total_entries,
        )?;

        self.elementwise.dispatch_scale(
            encoder,
            &scores,
            &scores,
            scale_factor,
            batch_size * total_entries,
        )?;

        let softmax_out = GpuBuffer::new(
            &self.device,
            batch_size as usize * total_entries as usize * std::mem::size_of::<f32>(),
            Some("inter_softmax"),
        )?;
        self.softmax.dispatch(encoder, &scores, &softmax_out, batch_size, total_entries)?;

        self.matmul.dispatch(
            encoder,
            &softmax_out,
            &normed_keys,
            attn_output,
            batch_size,
            total_entries,
            hidden_dim,
        )?;

        tracing::debug!(
            "BlockAttnResLayer::forward_inter_block layer={} complete",
            self.layer_number
        );

        Ok(())
    }
}
