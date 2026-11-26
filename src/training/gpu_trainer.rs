//! GPU training path for the f32 reference transformer.
//!
//! The model is the same architecture as
//! [`crate::training::f32_trainer::F32Model`]: pre-norm Gemma-style layers with
//! per-head Q/K RMSNorm, RoPE, causal GQA attention (scale 1), SwiGLU, a
//! post-attention/post-FFN norm before each residual, and one `layer_scalar`
//! per layer. Parameters live in `GpuBuffer`s, the forward pass records a
//! [`ComputationGraph`] tape, [`BackwardPass`] fills parameter gradients, and
//! [`AdamOptimizer`] updates in place. `tests/gpu_train_parity.rs` checks the
//! whole path against the CPU oracle; it is the only claim this module makes.
//!
//! Block AttnRes boundaries and LoRA are deliberately absent: the block path
//! has a separate open gradient defect.

use std::collections::BTreeMap;
use std::sync::Arc;

use wgpu::{Device, Queue};

use crate::autodiff::gpu_ops::TrainOps;
use crate::autodiff::{BackwardPass, ComputationGraph, NodeId};
use crate::compute::kernels::elementwise::ElementWiseOp;
use crate::compute::kernels::matmul::MatMulOp;
use crate::compute::GpuBuffer;
use crate::error::{FerrisResError, Result};
use crate::training::f32_trainer::{F32Config, F32Model};
use crate::training::optimizer::AdamOptimizer;

/// Reads a buffer back as f32. The controller needs logits and checkpoints on
/// the CPU; no activation leaves the device during a training step.
pub fn read_f32(device: &Device, queue: &Queue, buf: &GpuBuffer, len: usize) -> Result<Vec<f32>> {
    let bytes = len * std::mem::size_of::<f32>();
    let mut out = vec![0u8; bytes];
    buf.read(device, queue, &mut out)?;
    Ok(bytemuck::cast_slice(&out).to_vec())
}

pub struct GpuModel {
    pub config: F32Config,
    device: Arc<Device>,
    queue: Arc<Queue>,
    graph: ComputationGraph,
    ops: TrainOps,
    matmul: MatMulOp,
    elementwise: ElementWiseOp,
    backward: BackwardPass,
    /// One buffer per parameter, in `F32Model::tensors` order. The graph holds
    /// a second handle to the same allocation, so updating either updates both.
    weights: BTreeMap<String, GpuBuffer>,
    param_nodes: BTreeMap<String, NodeId>,
    param_order: Vec<String>,
    /// Gradient accumulation across `batch_size` windows.
    grad_acc: BTreeMap<String, GpuBuffer>,
    /// Node ids from the most recent forward pass, keyed by a stable name.
    /// Tests read them back to localise a parity failure to one op.
    trace: BTreeMap<String, NodeId>,
    pub adam: AdamOptimizer,
}

impl GpuModel {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: F32Config,
        seed: u64,
    ) -> Result<Self> {
        let model = F32Model::new_random(config, seed);
        Self::from_f32_model(device, queue, &model)
    }

    /// Upload a CPU model's parameters. Both paths therefore start from the
    /// exact same tensor values, which is what the parity gate needs.
    pub fn from_f32_model(
        device: Arc<Device>,
        queue: Arc<Queue>,
        model: &F32Model,
    ) -> Result<Self> {
        if model.block_config != crate::training::f32_trainer::F32BlockConfig::plain() {
            return Err(FerrisResError::Unsupported(
                "GPU trainer supports the plain transformer only; block boundaries have a separate gradient defect".to_string(),
            ));
        }
        let ops = TrainOps::new(Arc::clone(&device), Arc::clone(&queue));
        let matmul = MatMulOp::new(&device, &queue);
        let elementwise = ElementWiseOp::new(&device, &queue);
        let backward = BackwardPass::new(Arc::clone(&device), Arc::clone(&queue));
        let mut graph = ComputationGraph::new(Arc::clone(&device), Arc::clone(&queue));

        let mut weights = BTreeMap::new();
        let mut param_nodes = BTreeMap::new();
        let mut param_order = Vec::new();
        let mut adam = AdamOptimizer::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            0.001,
            0.9,
            0.999,
            1e-8,
        );

        for (name, values) in model.tensors() {
            let buf = GpuBuffer::new_device_local(
                &device,
                &queue,
                bytemuck::cast_slice(values),
                Some(&format!("param_{}", name)),
            )?;
            let numel = values.len();
            let id = graph.add_parameter(&name, buf.clone())?;
            adam.register_param(&name, numel)?;
            param_nodes.insert(name.clone(), id);
            param_order.push(name.clone());
            weights.insert(name, buf);
        }

        Ok(Self {
            config: model.config,
            device,
            queue,
            graph,
            ops,
            matmul,
            elementwise,
            backward,
            weights,
            param_nodes,
            param_order,
            grad_acc: BTreeMap::new(),
            trace: BTreeMap::new(),
            adam,
        })
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    pub fn param_numel(&self, name: &str) -> usize {
        self.weights[name].size() / std::mem::size_of::<f32>()
    }

    /// Copy every parameter back into `F32Model::tensors` order.
    pub fn to_f32_model(&self) -> Result<F32Model> {
        let mut model = F32Model::new_random(self.config, 0);
        {
            let mut tensors = model.tensors_mut();
            for (name, values) in tensors.iter_mut() {
                let gpu = &self.weights[name];
                let cpu = read_f32(
                    &self.device,
                    &self.queue,
                    gpu,
                    gpu.size() / std::mem::size_of::<f32>(),
                )?;
                values.copy_from_slice(&cpu);
            }
        }
        Ok(model)
    }

    pub fn read_param(&self, name: &str) -> Result<Vec<f32>> {
        let buf = &self.weights[name];
        read_f32(
            &self.device,
            &self.queue,
            buf,
            buf.size() / std::mem::size_of::<f32>(),
        )
    }

    /// Read a parameter's gradient buffer. Used by the parity and
    /// finite-difference tests; training itself never reads gradients back.
    pub fn read_grad(&self, name: &str) -> Result<Vec<f32>> {
        let node = self.graph.get_node(self.param_nodes[name]).unwrap();
        read_f32(
            &self.device,
            &self.queue,
            node.grad(),
            node.grad().size() / std::mem::size_of::<f32>(),
        )
    }

    fn alloc(&self, numel: usize, label: &str) -> Result<GpuBuffer> {
        GpuBuffer::zeros(
            &self.device,
            &self.queue,
            numel * std::mem::size_of::<f32>(),
            Some(label),
        )
    }

    fn node_buf(&self, id: NodeId) -> GpuBuffer {
        self.graph.get_node(id).expect("graph node").buf().clone()
    }

    fn param_id(&self, name: &str) -> NodeId {
        self.param_nodes[name]
    }

    /// Wrap an add: forward dispatch, then record. `a` and `b` have `numel`
    /// elements; the result is a new node.
    fn add(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        a: NodeId,
        b: NodeId,
        numel: usize,
        label: &str,
    ) -> Result<NodeId> {
        let out = self.alloc(numel, label)?;
        self.elementwise.dispatch_add(
            enc,
            &self.node_buf(a),
            &self.node_buf(b),
            &out,
            numel as u32,
        )?;
        self.graph.record_add(a, b, out, numel as u32)
    }

    fn matmul(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        input: NodeId,
        weight: &str,
        m: usize,
        k: usize,
        n: usize,
        label: &str,
    ) -> Result<NodeId> {
        let out = self.alloc(m * n, label)?;
        self.matmul.dispatch(
            enc,
            &self.node_buf(input),
            &self.weights[weight],
            &out,
            m as u32,
            k as u32,
            n as u32,
        )?;
        let wid = self.param_id(weight);
        self.graph.record_matmul(input, wid, out, m as u32, k as u32, n as u32)
    }

    /// `out = rmsnorm(x) * weight`, recorded as the two ops the backward pass
    /// knows how to differentiate.
    fn rmsnorm_weighted(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        x: NodeId,
        rows: usize,
        dim: usize,
        weight: &str,
        label: &str,
    ) -> Result<NodeId> {
        let eps = self.config.eps;
        let normed = self.alloc(rows * dim, label)?;
        self.ops
            .rmsnorm_forward(enc, &self.node_buf(x), &normed, rows as u32, dim as u32, eps)?;
        let n = self.graph.record_rmsnorm_no_scale(
            x,
            normed,
            rows as u32,
            dim as u32,
            eps,
        )?;
        let out = self.alloc(rows * dim, label)?;
        self.ops.mul_broadcast_forward(
            enc,
            &self.node_buf(n),
            &self.weights[weight],
            &out,
            rows as u32,
            dim as u32,
        )?;
        let wid = self.param_id(weight);
        self.graph
            .record_mul_broadcast(n, wid, out, rows as u32, dim as u32)
    }

    /// Record one forward pass. Returns `(logits_node, loss_node)`; the logits
    /// node's buffer can be read back for the loss value. The tape is rebuilt
    /// every step while parameter nodes stay put.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<(NodeId, NodeId)> {
        let cfg = self.config;
        let seq = tokens.len();
        if seq == 0 || seq > cfg.max_seq {
            return Err(FerrisResError::Shape(format!(
                "sequence length {} outside 1..={}",
                seq, cfg.max_seq
            )));
        }
        let h = cfg.hidden;
        let heads = cfg.num_heads;
        let kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let inter = cfg.intermediate;
        let eps = cfg.eps;

        self.graph.clear_activations();
        self.trace.clear();

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_train_forward"),
            });

        // Embedding: gather then scale by sqrt(hidden), as the CPU oracle does.
        let ids_buf = GpuBuffer::new_device_local(
            &self.device,
            &self.queue,
            bytemuck::cast_slice(tokens),
            Some("token_ids"),
        )?;
        let ids = self.graph.add_input("token_ids", ids_buf)?;
        let emb = self.alloc(seq * h, "embed")?;
        self.ops.embedding_forward(
            &mut enc,
            &self.node_buf(ids),
            &self.weights["embed"],
            &emb,
            seq as u32,
            cfg.vocab as u32,
            h as u32,
        )?;
        let embed_w = self.param_id("embed");
        let emb = self.graph.record_embedding_gather(
            ids,
            embed_w,
            emb,
            cfg.vocab as u32,
            h as u32,
        )?;
        let sqrt_h = (h as f32).sqrt();
        let scaled = self.alloc(seq * h, "embed_scaled")?;
        self.elementwise
            .dispatch_scale(&mut enc, &self.node_buf(emb), &scaled, sqrt_h, (seq * h) as u32)?;
        let mut x = self.graph.record_scale(emb, scaled, sqrt_h, (seq * h) as u32)?;
        self.trace.insert("embed_scaled".to_string(), x);

        for layer in 0..cfg.num_layers {
            let n = |suffix: &str| format!("layer{}.{}", layer, suffix);
            let attn_norm = self.rmsnorm_weighted(&mut enc, x, seq, h, &n("attn_norm"), "h_attn")?;
            self.trace.insert(format!("L{layer}.h_attn"), attn_norm);
            let q_proj = self.matmul(&mut enc, attn_norm, &n("q"), seq, h, q_dim, "q_proj")?;
            self.trace.insert(format!("L{layer}.q_proj"), q_proj);
            let k_proj = self.matmul(&mut enc, attn_norm, &n("k"), seq, h, kv_dim, "k_proj")?;
            let v_raw = self.matmul(&mut enc, attn_norm, &n("v"), seq, h, kv_dim, "v_raw")?;

            // Per-head V norm without a learned scale.
            let v_norm = self.alloc(seq * kv_dim, "v_norm")?;
            self.ops.rmsnorm_forward(
                &mut enc,
                &self.node_buf(v_raw),
                &v_norm,
                (seq * kv_heads) as u32,
                head_dim as u32,
                eps,
            )?;
            let v = self.graph.record_rmsnorm_no_scale(
                v_raw,
                v_norm,
                (seq * kv_heads) as u32,
                head_dim as u32,
                eps,
            )?;
            self.trace.insert(format!("L{layer}.v_norm"), v);

            let q_normed = self.rmsnorm_weighted(
                &mut enc,
                q_proj,
                seq * heads,
                head_dim,
                &n("q_norm"),
                "q_normed",
            )?;
            let k_normed = self.rmsnorm_weighted(
                &mut enc,
                k_proj,
                seq * kv_heads,
                head_dim,
                &n("k_norm"),
                "k_normed",
            )?;

            let q_rope = self.alloc(seq * q_dim, "q_rope")?;
            self.ops.rope_forward(
                &mut enc,
                &self.node_buf(q_normed),
                &q_rope,
                seq as u32,
                heads as u32,
                head_dim as u32,
                cfg.rope_theta,
                0,
            )?;
            let q = self.graph.record_rope(
                q_normed,
                q_rope,
                seq as u32,
                heads as u32,
                head_dim as u32,
                cfg.rope_theta,
                0,
            )?;
            self.trace.insert(format!("L{layer}.q_rope"), q);

            let k_rope = self.alloc(seq * kv_dim, "k_rope")?;
            self.ops.rope_forward(
                &mut enc,
                &self.node_buf(k_normed),
                &k_rope,
                seq as u32,
                kv_heads as u32,
                head_dim as u32,
                cfg.rope_theta,
                0,
            )?;
            let k = self.graph.record_rope(
                k_normed,
                k_rope,
                seq as u32,
                kv_heads as u32,
                head_dim as u32,
                cfg.rope_theta,
                0,
            )?;
            self.trace.insert(format!("L{layer}.k_rope"), k);

            // GQA: expand each KV head for its query group, then run one
            // causal attention op over the expanded layout.
            let k_exp = self.alloc(seq * q_dim, "k_exp")?;
            self.ops.gqa_expand_forward(
                &mut enc,
                &self.node_buf(k),
                &k_exp,
                seq as u32,
                heads as u32,
                kv_heads as u32,
                head_dim as u32,
            )?;
            let k_exp = self.graph.record_gqa_expand(
                k,
                k_exp,
                seq as u32,
                heads as u32,
                kv_heads as u32,
                head_dim as u32,
            )?;

            let v_exp = self.alloc(seq * q_dim, "v_exp")?;
            self.ops.gqa_expand_forward(
                &mut enc,
                &self.node_buf(v),
                &v_exp,
                seq as u32,
                heads as u32,
                kv_heads as u32,
                head_dim as u32,
            )?;
            let v_exp = self.graph.record_gqa_expand(
                v,
                v_exp,
                seq as u32,
                heads as u32,
                kv_heads as u32,
                head_dim as u32,
            )?;

            let attn = self.alloc(seq * q_dim, "attn")?;
            self.ops.attention_forward(
                &mut enc,
                &self.node_buf(q),
                &self.node_buf(k_exp),
                &self.node_buf(v_exp),
                &attn,
                seq as u32,
                heads as u32,
                head_dim as u32,
                1.0,
            )?;
            let attn = self.graph.record_attention(
                q,
                k_exp,
                v_exp,
                attn,
                seq as u32,
                heads as u32,
                head_dim as u32,
                1.0,
            )?;

            self.trace.insert(format!("L{layer}.attn"), attn);
            let attn_out = self.matmul(&mut enc, attn, &n("o"), seq, q_dim, h, "attn_out")?;
            self.trace.insert(format!("L{layer}.attn_out"), attn_out);
            let post_attn =
                self.rmsnorm_weighted(&mut enc, attn_out, seq, h, &n("post_attn_norm"), "post_attn")?;
            self.trace.insert(format!("L{layer}.post_attn"), post_attn);
            let x_mid = self.add(&mut enc, x, post_attn, seq * h, "x_mid")?;
            self.trace.insert(format!("L{layer}.x_mid"), x_mid);

            let h_ffn = self.rmsnorm_weighted(&mut enc, x_mid, seq, h, &n("pre_ffn_norm"), "h_ffn")?;
            self.trace.insert(format!("L{layer}.h_ffn"), h_ffn);
            let gate = self.matmul(&mut enc, h_ffn, &n("gate"), seq, h, inter, "gate")?;
            let up = self.matmul(&mut enc, h_ffn, &n("up"), seq, h, inter, "up")?;
            let act = self.alloc(seq * inter, "act")?;
            self.ops.silu_mul_forward(
                &mut enc,
                &self.node_buf(gate),
                &self.node_buf(up),
                &act,
                (seq * inter) as u32,
            )?;
            let act = self.graph.record_silu_mul(gate, up, act, (seq * inter) as u32)?;
            self.trace.insert(format!("L{layer}.gate"), gate);
            self.trace.insert(format!("L{layer}.act"), act);
            let ffn_out = self.matmul(&mut enc, act, &n("down"), seq, inter, h, "ffn_out")?;
            self.trace.insert(format!("L{layer}.ffn_out"), ffn_out);
            let post_ffn =
                self.rmsnorm_weighted(&mut enc, ffn_out, seq, h, &n("post_ffn_norm"), "post_ffn")?;
            let summed = self.add(&mut enc, x_mid, post_ffn, seq * h, "x_sum")?;

            let scaled = self.alloc(seq * h, "layer_out")?;
            let scalar_id = self.param_id(&n("layer_scalar"));
            self.ops.layer_scale_forward(
                &mut enc,
                &self.node_buf(summed),
                &self.node_buf(scalar_id),
                &scaled,
                (seq * h) as u32,
            )?;
            x = self.graph.record_layer_scale(
                summed,
                scalar_id,
                scaled,
                (seq * h) as u32,
            )?;
            self.trace.insert(format!("L{layer}.x_out"), x);
        }

        let h_final = self.rmsnorm_weighted(&mut enc, x, seq, h, "final_norm", "h_final")?;
        self.trace.insert("h_final".to_string(), h_final);
        let logits = self.matmul(&mut enc, h_final, "lm_head", seq, h, cfg.vocab, "logits")?;

        // Next-token targets: row t predicts tokens[t + 1]. The loss op divides
        // by `batch_size`, so pass seq - 1 rows to match the CPU loss exactly.
        let targets: Vec<u32> = tokens[1..].to_vec();
        let target_buf = GpuBuffer::new_device_local(
            &self.device,
            &self.queue,
            bytemuck::cast_slice(&targets),
            Some("targets"),
        )?;
        let targets_id = self.graph.add_input("targets", target_buf)?;
        let loss_buf = self.alloc(1, "loss")?;
        let loss = self.graph.record_loss(
            logits,
            targets_id,
            loss_buf,
            (seq - 1) as u32,
            cfg.vocab as u32,
        )?;

        self.queue.submit(std::iter::once(enc.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        Ok((logits, loss))
    }

    pub fn node_id(&self, name: &str) -> Option<NodeId> {
        self.trace.get(name).copied()
    }

    pub fn read_logits(&self, logits: NodeId, len: usize) -> Result<Vec<f32>> {
        let buf = self.node_buf(logits);
        read_f32(&self.device, &self.queue, &buf, len)
    }

    /// Run the recorded backward pass. Parameter gradients land in each
    /// parameter node's `grad` buffer, zeroed first by the pass.
    pub fn backward(&mut self, loss: NodeId) -> Result<()> {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_train_backward"),
            });
        self.backward.run(&mut self.graph, &mut enc, loss)?;
        self.queue.submit(std::iter::once(enc.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        Ok(())
    }

    fn ensure_accumulators(&mut self) -> Result<()> {
        if !self.grad_acc.is_empty() {
            return Ok(());
        }
        for name in &self.param_order {
            let numel = self.param_numel(name);
            self.grad_acc.insert(name.clone(), self.alloc(numel, "grad_acc")?);
        }
        Ok(())
    }

    /// Add this step's parameter gradients into the accumulation buffers.
    pub fn accumulate_grads(&mut self) -> Result<()> {
        self.ensure_accumulators()?;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_train_accumulate"),
            });
        for name in &self.param_order {
            let grad = self.graph.get_node(self.param_nodes[name]).unwrap().grad().clone();
            let numel = self.param_numel(name);
            self.elementwise.dispatch_add(
                &mut enc,
                &self.grad_acc[name],
                &grad,
                &self.grad_acc[name],
                numel as u32,
            )?;
        }
        self.queue.submit(std::iter::once(enc.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        Ok(())
    }

    pub fn reset_accumulators(&mut self) -> Result<()> {
        for buf in self.grad_acc.values() {
            let zeros = GpuBuffer::zeros(&self.device, &self.queue, buf.size(), Some("grad_acc_zero"))?;
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gpu_train_reset_acc"),
                });
            enc.copy_buffer_to_buffer(zeros.buffer(), 0, buf.buffer(), 0, buf.size() as u64);
            self.queue.submit(std::iter::once(enc.finish()));
        }
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        Ok(())
    }

    /// One Adam step using the accumulated gradients scaled by `1 / windows`.
    pub fn step_accumulated(&mut self, lr: f32, windows: u32) -> Result<()> {
        self.ensure_accumulators()?;
        self.adam.set_learning_rate(lr);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_train_step"),
            });
        let scale = 1.0 / windows.max(1) as f32;
        for name in &self.param_order {
            let numel = self.param_numel(name);
            self.elementwise
                .dispatch_scale(&mut enc, &self.grad_acc[name], &self.grad_acc[name], scale, numel as u32)?;
        }
        // Split the fields so the Adam borrow and the parameter/gradient
        // borrows do not overlap through `self`.
        let GpuModel {
            adam,
            weights,
            param_order,
            grad_acc,
            ..
        } = self;
        // The step must use the **accumulated, scaled** gradient. Reading
        // `graph.get_node(node).grad()` here takes only the last window's raw
        // gradient and discards the other windows and the 1/windows scale, which
        // makes an effective learning rate of lr * windows. That diverges for any
        // batch size above 1 while a single-window run looks perfect.
        let items: Vec<(&str, &GpuBuffer, &GpuBuffer, u32)> = param_order
            .iter()
            .map(|name| {
                let numel = (weights[name].size() / std::mem::size_of::<f32>()) as u32;
                (name.as_str(), &weights[name], &grad_acc[name], numel)
            })
            .collect();
        adam.step_all(&mut enc, &items)?;
        self.queue.submit(std::iter::once(enc.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        Ok(())
    }
}
