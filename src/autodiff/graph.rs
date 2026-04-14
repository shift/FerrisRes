use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Device, Queue};
use crate::compute::GpuBuffer;
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone)]
pub enum NodeKind {
    MatMul {
        m: u32,
        k: u32,
        n: u32,
    },
    Add {
        numel: u32,
    },
    Scale {
        scale: f32,
        numel: u32,
    },
    RmsNorm {
        hidden_dim: u32,
        eps: f32,
    },
    Softmax {
        rows: u32,
        cols: u32,
    },
    ReLU {
        numel: u32,
    },
    Linear {
        in_features: u32,
        out_features: u32,
        has_bias: bool,
    },
    Embedding {
        vocab_size: u32,
        hidden_dim: u32,
    },
    Loss {
        batch_size: u32,
        vocab_size: u32,
    },
    /// Block Summary cross-attention: compresses block of tokens into summary.
    /// Forward: Q=queries×W_q, K=tokens×W_k, V=tokens×W_v, attn=softmax(QK^T/√d)V, out=attn×W_o.
    /// Backward: propagates gradients to queries, W_q, W_k, W_v, W_o, and bridge_weight.
    BlockSummaryCrossAttn {
        num_queries: u32,
        hidden_dim: u32,
        block_size: u32,
    },
    Parameter {
        name: String,
        numel: usize,
    },
    Input {
        name: String,
        numel: usize,
    },
}

pub struct TapeEntry {
    op: NodeKind,
    inputs: Vec<NodeId>,
    output_id: NodeId,
}

impl TapeEntry {
    pub fn output_id(&self) -> NodeId {
        self.output_id
    }

    pub fn op(&self) -> &NodeKind {
        &self.op
    }

    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }
}

pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    pub buf: GpuBuffer,
    pub grad: GpuBuffer,
    pub is_leaf: bool,
    pub requires_grad: bool,
}

impl Node {
    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn buf(&self) -> &GpuBuffer {
        &self.buf
    }

    pub fn grad(&self) -> &GpuBuffer {
        &self.grad
    }
}

pub struct ComputationGraph {
    nodes: HashMap<NodeId, Node>,
    tape: Vec<TapeEntry>,
    next_id: usize,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl ComputationGraph {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        tracing::info!(event = "creating_computationgraph", "Creating ComputationGraph");
        Self {
            nodes: HashMap::new(),
            tape: Vec::new(),
            next_id: 0,
            device,
            queue,
        }
    }

    fn alloc_id(&mut self) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        id
    }

    pub fn add_parameter(&mut self, name: &str, buf: GpuBuffer) -> Result<NodeId> {
        let numel = buf.size() / std::mem::size_of::<f32>();
        let id = self.alloc_id();
        let grad = GpuBuffer::zeros(&self.device, &self.queue, buf.size(), Some(&format!("grad_param_{}", name)))?;

        let kind = NodeKind::Parameter {
            name: name.to_string(),
            numel,
        };

        tracing::debug!(event = "graph_added_parameter_id_numel", "Graph: added parameter '{}' id={:?} numel={}", name, id, numel);

        self.nodes.insert(id, Node {
            id,
            kind,
            buf,
            grad,
            is_leaf: true,
            requires_grad: true,
        });

        Ok(id)
    }

    pub fn add_input(&mut self, name: &str, buf: GpuBuffer) -> Result<NodeId> {
        let numel = buf.size() / std::mem::size_of::<f32>();
        let id = self.alloc_id();
        let grad = GpuBuffer::zeros(&self.device, &self.queue, buf.size(), Some(&format!("grad_input_{}", name)))?;

        let kind = NodeKind::Input {
            name: name.to_string(),
            numel,
        };

        tracing::debug!(event = "graph_added_input_id_numel", "Graph: added input '{}' id={:?} numel={}", name, id, numel);

        self.nodes.insert(id, Node {
            id,
            kind,
            buf,
            grad,
            is_leaf: true,
            requires_grad: false,
        });

        Ok(id)
    }

    pub fn record_matmul(
        &mut self,
        a_id: NodeId,
        b_id: NodeId,
        output_buf: GpuBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<NodeId> {
        let id = self.alloc_id();
        let grad = GpuBuffer::zeros(&self.device, &self.queue, output_buf.size(), Some("grad_matmul"))?;

        let op = NodeKind::MatMul { m, k, n };
        tracing::debug!(event = "graph_record_matmul_id_m_k", "Graph: record matmul id={:?} M={} K={} N={}", id, m, k, n);

        self.nodes.insert(id, Node {
            id,
            kind: op.clone(),
            buf: output_buf,
            grad,
            is_leaf: false,
            requires_grad: true,
        });

        self.tape.push(TapeEntry {
            op,
            inputs: vec![a_id, b_id],
            output_id: id,
        });

        Ok(id)
    }

    pub fn record_add(
        &mut self,
        a_id: NodeId,
        b_id: NodeId,
        output_buf: GpuBuffer,
        numel: u32,
    ) -> Result<NodeId> {
        let id = self.alloc_id();
        let grad = GpuBuffer::zeros(&self.device, &self.queue, output_buf.size(), Some("grad_add"))?;

        let op = NodeKind::Add { numel };
        tracing::debug!(event = "graph_record_add_id_numel", "Graph: record add id={:?} numel={}", id, numel);

        self.nodes.insert(id, Node {
            id,
            kind: op.clone(),
            buf: output_buf,
            grad,
            is_leaf: false,
            requires_grad: true,
        });

        self.tape.push(TapeEntry {
            op,
            inputs: vec![a_id, b_id],
            output_id: id,
        });

        Ok(id)
    }

    pub fn record_scale(
        &mut self,
        input_id: NodeId,
        output_buf: GpuBuffer,
        scale: f32,
        numel: u32,
    ) -> Result<NodeId> {
        let id = self.alloc_id();
        let grad = GpuBuffer::zeros(&self.device, &self.queue, output_buf.size(), Some("grad_scale"))?;

        let op = NodeKind::Scale { scale, numel };
        tracing::debug!(event = "graph_record_scale_id_scale_numel", "Graph: record scale id={:?} scale={} numel={}", id, scale, numel);

        self.nodes.insert(id, Node {
            id,
            kind: op.clone(),
            buf: output_buf,
            grad,
            is_leaf: false,
            requires_grad: true,
        });

        self.tape.push(TapeEntry {
            op,
            inputs: vec![input_id],
            output_id: id,
        });

        Ok(id)
    }

    pub fn record_relu(
        &mut self,
        input_id: NodeId,
        output_buf: GpuBuffer,
        numel: u32,
    ) -> Result<NodeId> {
        let id = self.alloc_id();
        let grad = GpuBuffer::zeros(&self.device, &self.queue, output_buf.size(), Some("grad_relu"))?;

        let op = NodeKind::ReLU { numel };
        tracing::debug!(event = "graph_record_relu_id_numel", "Graph: record relu id={:?} numel={}", id, numel);

        self.nodes.insert(id, Node {
            id,
            kind: op.clone(),
            buf: output_buf,
            grad,
            is_leaf: false,
            requires_grad: true,
        });

        self.tape.push(TapeEntry {
            op,
            inputs: vec![input_id],
            output_id: id,
        });

        Ok(id)
    }

    pub fn record_rmsnorm(
        &mut self,
        input_id: NodeId,
        output_buf: GpuBuffer,
        hidden_dim: u32,
        eps: f32,
    ) -> Result<NodeId> {
        let id = self.alloc_id();
        let grad = GpuBuffer::zeros(&self.device, &self.queue, output_buf.size(), Some("grad_rmsnorm"))?;

        let op = NodeKind::RmsNorm { hidden_dim, eps };
        tracing::debug!(event = "graph_record_rmsnorm_id_hidden_dim", "Graph: record rmsnorm id={:?} hidden_dim={} eps={}", id, hidden_dim, eps);

        self.nodes.insert(id, Node {
            id,
            kind: op.clone(),
            buf: output_buf,
            grad,
            is_leaf: false,
            requires_grad: true,
        });

        self.tape.push(TapeEntry {
            op,
            inputs: vec![input_id],
            output_id: id,
        });

        Ok(id)
    }

    pub fn record_softmax(
        &mut self,
        input_id: NodeId,
        output_buf: GpuBuffer,
        rows: u32,
        cols: u32,
    ) -> Result<NodeId> {
        let id = self.alloc_id();
        let grad = GpuBuffer::zeros(&self.device, &self.queue, output_buf.size(), Some("grad_softmax"))?;

        let op = NodeKind::Softmax { rows, cols };
        tracing::debug!(event = "graph_record_softmax_id_rows_cols", "Graph: record softmax id={:?} rows={} cols={}", id, rows, cols);

        self.nodes.insert(id, Node {
            id,
            kind: op.clone(),
            buf: output_buf,
            grad,
            is_leaf: false,
            requires_grad: true,
        });

        self.tape.push(TapeEntry {
            op,
            inputs: vec![input_id],
            output_id: id,
        });

        Ok(id)
    }

    pub fn record_linear(
        &mut self,
        input_id: NodeId,
        weight_id: NodeId,
        bias_id: Option<NodeId>,
        output_buf: GpuBuffer,
        in_features: u32,
        out_features: u32,
    ) -> Result<NodeId> {
        let id = self.alloc_id();
        let grad = GpuBuffer::zeros(&self.device, &self.queue, output_buf.size(), Some("grad_linear"))?;

        let mut inputs = vec![input_id, weight_id];
        if let Some(bid) = bias_id {
            inputs.push(bid);
        }

        let has_bias = bias_id.is_some();
        let op = NodeKind::Linear { in_features, out_features, has_bias };
        tracing::debug!(event = "graph_record_linear_id_in_out", "Graph: record linear id={:?} in={} out={} has_bias={}", id, in_features, out_features, has_bias);

        self.nodes.insert(id, Node {
            id,
            kind: op.clone(),
            buf: output_buf,
            grad,
            is_leaf: false,
            requires_grad: true,
        });

        self.tape.push(TapeEntry {
            op,
            inputs,
            output_id: id,
        });

        Ok(id)
    }

    pub fn record_embedding(
        &mut self,
        token_ids_id: NodeId,
        output_buf: GpuBuffer,
        vocab_size: u32,
        hidden_dim: u32,
    ) -> Result<NodeId> {
        let id = self.alloc_id();
        let grad = GpuBuffer::zeros(&self.device, &self.queue, output_buf.size(), Some("grad_embedding"))?;

        let op = NodeKind::Embedding { vocab_size, hidden_dim };
        tracing::debug!(event = "graph_record_embedding_id_vocab_dim", "Graph: record embedding id={:?} vocab={} dim={}", id, vocab_size, hidden_dim);

        self.nodes.insert(id, Node {
            id,
            kind: op.clone(),
            buf: output_buf,
            grad,
            is_leaf: false,
            requires_grad: true,
        });

        self.tape.push(TapeEntry {
            op,
            inputs: vec![token_ids_id],
            output_id: id,
        });

        Ok(id)
    }

    pub fn record_loss(
        &mut self,
        logits_id: NodeId,
        targets_id: NodeId,
        output_buf: GpuBuffer,
        batch_size: u32,
        vocab_size: u32,
    ) -> Result<NodeId> {
        let id = self.alloc_id();
        let grad = GpuBuffer::zeros(&self.device, &self.queue, output_buf.size(), Some("grad_loss"))?;

        let op = NodeKind::Loss { batch_size, vocab_size };
        tracing::debug!(event = "graph_record_loss_id_batch_vocab", "Graph: record loss id={:?} batch={} vocab={}", id, batch_size, vocab_size);

        self.nodes.insert(id, Node {
            id,
            kind: op.clone(),
            buf: output_buf,
            grad,
            is_leaf: false,
            requires_grad: true,
        });

        self.tape.push(TapeEntry {
            op,
            inputs: vec![logits_id, targets_id],
            output_id: id,
        });

        Ok(id)
    }

    /// Record a Block Summary cross-attention operation.
    ///
    /// Takes block tokens and summary queries as input, produces
    /// compressed summary as output. During backward, computes gradients
    /// for queries, key/value projections, and bridge weight.
    ///
    /// Inputs: [0] = block_tokens [block_size × hidden_dim],
    ///         [1] = summary_queries [num_queries × hidden_dim]
    /// Output: summary [num_queries × hidden_dim]
    pub fn record_block_summary(
        &mut self,
        block_tokens_id: NodeId,
        queries_id: NodeId,
        num_queries: u32,
        hidden_dim: u32,
        block_size: u32,
    ) -> Result<NodeId> {
        let numel = num_queries as usize * hidden_dim as usize;
        let buf = GpuBuffer::zeros(
            &self.device, &self.queue,
            numel * std::mem::size_of::<f32>(),
            Some("block_summary_output"),
        )?;

        let id = NodeId(self.next_id);
        self.next_id += 1;

        let kind = NodeKind::BlockSummaryCrossAttn { num_queries, hidden_dim, block_size };
        self.nodes.insert(id, Node {
            id,
            kind,
            buf,
            grad: GpuBuffer::zeros(&self.device, &self.queue, 0, Some("empty_grad"))?,
            is_leaf: false,
            requires_grad: false,
        });

        self.tape.push(TapeEntry {
            op: NodeKind::BlockSummaryCrossAttn { num_queries, hidden_dim, block_size },
            inputs: vec![block_tokens_id, queries_id],
            output_id: id,
        });

        Ok(id)
    }

    pub fn tape(&self) -> &[TapeEntry] {
        &self.tape
    }

    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    pub fn clear(&mut self) {
        tracing::debug!(event = "graph_clearing_tape_and_nodes", "Graph: clearing tape and nodes");
        self.tape.clear();
        self.nodes.clear();
        self.next_id = 0;
    }

    pub fn zero_all_grads(&self, encoder: &mut wgpu::CommandEncoder) -> Result<()> {
        for (_id, node) in &self.nodes {
            if node.requires_grad {
                self.zero_buffer(encoder, &node.grad)?;
            }
        }
        Ok(())
    }

    fn zero_buffer(&self, encoder: &mut wgpu::CommandEncoder, buf: &GpuBuffer) -> Result<()> {
        let zero_buf = GpuBuffer::zeros(&self.device, &self.queue, buf.size(), Some("zero_temp"))?;
        encoder.copy_buffer_to_buffer(zero_buf.buffer(), 0, buf.buffer(), 0, buf.size() as u64);
        Ok(())
    }
}
