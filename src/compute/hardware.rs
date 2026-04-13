//! Hardware abstraction for heterogeneous compute: cloud GPU, ANE/NPU, RDMA/DirectGPU.
//!
//! This module provides:
//! - Cloud GPU training orchestration (remote workers, fault tolerance, cost scheduling)
//! - Multi-NPU support (Apple Neural Engine, op placement, unified memory)
//! - RDMA/DirectGPU communication (NVLink, RoCE, TCP fallback)

use std::collections::HashMap;
use std::time::Duration;

// ===========================================================================
// Part 1: Cloud GPU Training Orchestration
// ===========================================================================

/// Status of a cloud worker node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkerStatus {
    Idle,
    Busy,
    Failed,
    Connecting,
    ShuttingDown,
}

/// A cloud GPU worker node.
#[derive(Debug, Clone)]
pub struct WorkerNode {
    pub id: String,
    pub endpoint: String,
    pub gpu_type: String,
    pub gpu_count: usize,
    pub memory_gb: usize,
    pub status: WorkerStatus,
    pub cost_per_hour: f32,
    pub is_spot: bool,
    pub assigned_shard: Option<usize>,
}

impl WorkerNode {
    pub fn new(id: &str, endpoint: &str, gpu_type: &str, gpu_count: usize, memory_gb: usize) -> Self {
        Self {
            id: id.to_string(),
            endpoint: endpoint.to_string(),
            gpu_type: gpu_type.to_string(),
            gpu_count,
            memory_gb,
            status: WorkerStatus::Idle,
            cost_per_hour: 0.0,
            is_spot: false,
            assigned_shard: None,
        }
    }

    /// Estimated cost for a given training duration.
    pub fn estimated_cost(&self, hours: f32) -> f32 {
        self.cost_per_hour * hours * self.gpu_count as f32
    }

    /// Whether this worker is available.
    pub fn is_available(&self) -> bool {
        matches!(self.status, WorkerStatus::Idle)
    }

    /// Whether this worker is healthy.
    pub fn is_healthy(&self) -> bool {
        !matches!(self.status, WorkerStatus::Failed | WorkerStatus::ShuttingDown)
    }
}

/// Cloud training configuration.
#[derive(Debug, Clone)]
pub struct CloudTrainingConfig {
    /// Maximum number of workers.
    pub max_workers: usize,
    /// Minimum number of workers (below this, training pauses).
    pub min_workers: usize,
    /// Maximum cost per hour.
    pub max_cost_per_hour: f32,
    /// Whether to prefer spot instances.
    pub prefer_spot: bool,
    /// Heartbeat timeout.
    pub heartbeat_timeout: Duration,
    /// Whether to auto-scale workers.
    pub auto_scale: bool,
}

impl Default for CloudTrainingConfig {
    fn default() -> Self {
        Self {
            max_workers: 8,
            min_workers: 1,
            max_cost_per_hour: 100.0,
            prefer_spot: true,
            heartbeat_timeout: Duration::from_secs(30),
            auto_scale: true,
        }
    }
}

/// Cloud training orchestrator.
pub struct CloudTrainingOrchestrator {
    config: CloudTrainingConfig,
    workers: HashMap<String, WorkerNode>,
    /// Current training step.
    step: usize,
    /// Gradient aggregation buffer.
    gradient_buffer: Vec<f32>,
    /// Number of gradient shards received.
    shards_received: usize,
}

impl CloudTrainingOrchestrator {
    pub fn new(config: CloudTrainingConfig) -> Self {
        Self {
            config,
            workers: HashMap::new(),
            step: 0,
            gradient_buffer: Vec::new(),
            shards_received: 0,
        }
    }

    /// Register a worker.
    pub fn register_worker(&mut self, worker: WorkerNode) {
        self.workers.insert(worker.id.clone(), worker);
    }

    /// Remove a worker.
    pub fn remove_worker(&mut self, id: &str) {
        self.workers.remove(id);
    }

    /// Assign model shards to available workers.
    pub fn assign_shards(&mut self, num_shards: usize) {
        let available: Vec<String> = self.workers.iter()
            .filter(|(_, w)| w.is_available())
            .map(|(id, _)| id.clone())
            .collect();

        for (shard_idx, worker_id) in available.iter().cycle().take(num_shards).enumerate() {
            if let Some(worker) = self.workers.get_mut(worker_id) {
                worker.assigned_shard = Some(shard_idx);
                worker.status = WorkerStatus::Busy;
            }
        }
    }

    /// Receive a gradient shard from a worker.
    pub fn receive_gradient_shard(&mut self, worker_id: &str, shard: &[f32], offset: usize) {
        if offset + shard.len() > self.gradient_buffer.len() {
            self.gradient_buffer.resize(offset + shard.len(), 0.0);
        }
        for (i, &g) in shard.iter().enumerate() {
            self.gradient_buffer[offset + i] += g;
        }
        self.shards_received += 1;

        // Mark worker as idle after sending gradient
        if let Some(worker) = self.workers.get_mut(worker_id) {
            worker.status = WorkerStatus::Idle;
        }
    }

    /// Whether all gradient shards have been received.
    pub fn gradients_complete(&self) -> bool {
        let active_workers = self.workers.values()
            .filter(|w| w.assigned_shard.is_some())
            .count();
        active_workers > 0 && self.shards_received >= active_workers
    }

    /// Get the aggregated gradients and advance to next step.
    pub fn take_gradients(&mut self) -> Vec<f32> {
        let grads = std::mem::take(&mut self.gradient_buffer);
        self.shards_received = 0;
        self.step += 1;
        grads
    }

    /// Detect failed workers (heartbeat timeout).
    pub fn detect_failures(&mut self) -> Vec<String> {
        let failed: Vec<String> = self.workers.iter()
            .filter(|(_, w)| !w.is_healthy())
            .map(|(id, _)| id.clone())
            .collect();

        for id in &failed {
            if let Some(w) = self.workers.get_mut(id) {
                w.status = WorkerStatus::Failed;
            }
        }
        failed
    }

    /// Redistribute work from failed workers.
    pub fn redistribute_failed(&mut self) {
        let failed_shards: Vec<usize> = self.workers.iter()
            .filter(|(_, w)| matches!(w.status, WorkerStatus::Failed))
            .filter_map(|(_, w)| w.assigned_shard)
            .collect();

        // Clear failed assignments
        for w in self.workers.values_mut() {
            if matches!(w.status, WorkerStatus::Failed) {
                w.assigned_shard = None;
            }
        }

        // Reassign to healthy workers
        let healthy: Vec<String> = self.workers.iter()
            .filter(|(_, w)| w.is_available())
            .map(|(id, _)| id.clone())
            .collect();

        for (i, shard) in failed_shards.into_iter().enumerate() {
            if let Some(worker_id) = healthy.get(i % healthy.len()) {
                if let Some(w) = self.workers.get_mut(worker_id) {
                    w.assigned_shard = Some(shard);
                    w.status = WorkerStatus::Busy;
                }
            }
        }
    }

    /// Get cost-optimal workers for a given budget.
    pub fn cost_optimal_workers(&self, budget_per_hour: f32) -> Vec<&WorkerNode> {
        let mut workers: Vec<&WorkerNode> = self.workers.values()
            .filter(|w| w.is_healthy())
            .collect();

        // Prefer spot instances if configured
        if self.config.prefer_spot {
            workers.sort_by(|a, b| {
                let a_priority = (a.is_spot as i32, a.cost_per_hour as i64);
                let b_priority = (b.is_spot as i32, b.cost_per_hour as i64);
                b_priority.cmp(&a_priority) // Higher priority first
            });
        } else {
            workers.sort_by(|a, b| a.cost_per_hour.partial_cmp(&b.cost_per_hour).unwrap());
        }

        // Select workers within budget
        let mut total_cost = 0.0;
        workers.into_iter().take_while(|w| {
            total_cost += w.cost_per_hour;
            total_cost <= budget_per_hour
        }).collect()
    }

    /// Number of active workers.
    pub fn active_workers(&self) -> usize {
        self.workers.values().filter(|w| w.is_healthy()).count()
    }

    /// Current training step.
    pub fn step(&self) -> usize {
        self.step
    }
}

// ===========================================================================
// Part 2: Apple Neural Engine (ANE) / Multi-NPU Support
// ===========================================================================

/// Type of compute accelerator.
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum AcceleratorType {
    /// Discrete or integrated GPU (Vulkan/Metal/DX12).
    GPU,
    /// Apple Neural Engine.
    ANE,
    /// Intel Gaudi.
    Gaudi,
    /// Google TPU.
    TPU,
    /// Generic NPU.
    NPU,
}

/// Capabilities of an accelerator.
#[derive(Debug, Clone)]
pub struct AcceleratorCapabilities {
    pub acc_type: AcceleratorType,
    pub supports_matmul: bool,
    pub supports_conv: bool,
    pub supports_batch_norm: bool,
    pub supports_attention: bool,
    pub supports_f16: bool,
    pub supports_bf16: bool,
    pub supports_int8: bool,
    pub memory_bandwidth_gbps: f32,
    pub peak_tflops: f32,
    pub unified_memory: bool,
}

impl AcceleratorCapabilities {
    pub fn gpu() -> Self {
        Self {
            acc_type: AcceleratorType::GPU,
            supports_matmul: true,
            supports_conv: true,
            supports_batch_norm: true,
            supports_attention: true,
            supports_f16: true,
            supports_bf16: true,
            supports_int8: true,
            memory_bandwidth_gbps: 500.0,
            peak_tflops: 10.0,
            unified_memory: false,
        }
    }

    pub fn ane() -> Self {
        Self {
            acc_type: AcceleratorType::ANE,
            supports_matmul: true,
            supports_conv: true,
            supports_batch_norm: true,
            supports_attention: false, // ANE doesn't support custom attention
            supports_f16: true,
            supports_bf16: false,
            supports_int8: true,
            memory_bandwidth_gbps: 100.0,
            peak_tflops: 11.0,
            unified_memory: true,
        }
    }

    /// Whether this accelerator can execute a given operation.
    pub fn can_execute(&self, op: &HardwareOp) -> bool {
        match op {
            HardwareOp::MatMul => self.supports_matmul,
            HardwareOp::Conv => self.supports_conv,
            HardwareOp::BatchNorm => self.supports_batch_norm,
            HardwareOp::Attention => self.supports_attention,
            HardwareOp::Activation => true,
            HardwareOp::Softmax => true,
            HardwareOp::LayerNorm => true,
            HardwareOp::Embedding => true,
        }
    }
}

/// Operation types for hardware placement.
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum HardwareOp {
    MatMul,
    Conv,
    BatchNorm,
    Attention,
    Activation,
    Softmax,
    LayerNorm,
    Embedding,
}

/// Decision for where to place an operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlacementDecision {
    /// Execute on GPU.
    GPU,
    /// Execute on ANE/NPU.
    ANE,
    /// Execute on CPU.
    CPU,
}

/// Automatic operation placement based on accelerator capabilities.
pub struct OpPlacer {
    gpu_caps: AcceleratorCapabilities,
    ane_caps: Option<AcceleratorCapabilities>,
}

impl OpPlacer {
    pub fn new(gpu_caps: AcceleratorCapabilities, ane_caps: Option<AcceleratorCapabilities>) -> Self {
        Self { gpu_caps, ane_caps }
    }

    /// Default placer with GPU + optional ANE.
    pub fn default_with_ane(has_ane: bool) -> Self {
        let ane_caps = if has_ane {
            Some(AcceleratorCapabilities::ane())
        } else {
            None
        };
        Self::new(AcceleratorCapabilities::gpu(), ane_caps)
    }

    /// Decide placement for an operation.
    pub fn place(&self, op: HardwareOp) -> PlacementDecision {
        // Prefer ANE for simple ops if available
        if let Some(ref ane) = self.ane_caps {
            if ane.can_execute(&op) && !matches!(op, HardwareOp::Attention | HardwareOp::MatMul) {
                // Route batch_norm, activation, layer_norm to ANE
                return PlacementDecision::ANE;
            }
        }

        // GPU for compute-heavy ops
        if self.gpu_caps.can_execute(&op) {
            return PlacementDecision::GPU;
        }

        PlacementDecision::CPU
    }

    /// Get the full placement plan for a transformer layer.
    pub fn transformer_layer_plan(&self) -> Vec<(HardwareOp, PlacementDecision)> {
        let ops = [
            HardwareOp::Embedding,
            HardwareOp::LayerNorm,
            HardwareOp::MatMul,  // QKV projection
            HardwareOp::Attention,
            HardwareOp::MatMul,  // Output projection
            HardwareOp::LayerNorm,
            HardwareOp::MatMul,  // FFN up
            HardwareOp::Activation,
            HardwareOp::MatMul,  // FFN down
        ];
        ops.iter().map(|&op| (op, self.place(op))).collect()
    }
}

/// Represents a shared buffer between GPU and ANE on Apple Silicon.
pub struct UnifiedMemoryBuffer {
    data: Vec<f32>,
    /// Whether the buffer is currently owned by ANE.
    ane_owned: bool,
    /// Whether the buffer is currently owned by GPU.
    gpu_owned: bool,
}

impl UnifiedMemoryBuffer {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0.0; size],
            ane_owned: false,
            gpu_owned: false,
        }
    }

    /// Get the buffer for GPU access (zero-copy on Apple Silicon).
    pub fn gpu_access(&mut self) -> &mut [f32] {
        self.gpu_owned = true;
        self.ane_owned = false;
        &mut self.data
    }

    /// Get the buffer for ANE access (zero-copy on Apple Silicon).
    pub fn ane_access(&mut self) -> &mut [f32] {
        self.ane_owned = true;
        self.gpu_owned = false;
        &mut self.data
    }

    /// Size of the buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Read-only access.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

// ===========================================================================
// Part 3: RDMA/DirectGPU Multi-Node Communication
// ===========================================================================

/// Transport type for inter-node communication.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransportType {
    /// NVLink/NVSwitch direct GPU-to-GPU.
    NVLink,
    /// RDMA over Converged Ethernet.
    RoCE,
    /// InfiniBand.
    InfiniBand,
    /// TCP fallback.
    TCP,
}

/// Configuration for RDMA communication.
#[derive(Debug, Clone)]
pub struct RdmaConfig {
    pub transport: TransportType,
    /// Maximum transfer size in bytes.
    pub max_transfer_size: usize,
    /// Number of send/receive queue pairs.
    pub num_qp: usize,
    /// Whether to use GPU-direct RDMA.
    pub gpu_direct: bool,
    /// Connection timeout.
    pub timeout: Duration,
}

impl Default for RdmaConfig {
    fn default() -> Self {
        Self {
            transport: TransportType::TCP,
            max_transfer_size: 16 * 1024 * 1024, // 16 MB
            num_qp: 4,
            gpu_direct: false,
            timeout: Duration::from_secs(5),
        }
    }
}

impl RdmaConfig {
    pub fn nvlink() -> Self {
        Self {
            transport: TransportType::NVLink,
            gpu_direct: true,
            max_transfer_size: 64 * 1024 * 1024,
            ..Default::default()
        }
    }

    pub fn roce() -> Self {
        Self {
            transport: TransportType::RoCE,
            gpu_direct: true,
            ..Default::default()
        }
    }

    pub fn tcp_fallback() -> Self {
        Self::default()
    }

    /// Bandwidth estimate in GB/s.
    pub fn estimated_bandwidth(&self) -> f32 {
        match self.transport {
            TransportType::NVLink => 300.0, // NVLink 4.0: ~300 GB/s
            TransportType::RoCE => 100.0,
            TransportType::InfiniBand => 200.0,
            TransportType::TCP => 10.0,
        }
    }

    /// Latency estimate in microseconds.
    pub fn estimated_latency_us(&self) -> f32 {
        match self.transport {
            TransportType::NVLink => 1.0,
            TransportType::RoCE => 5.0,
            TransportType::InfiniBand => 2.0,
            TransportType::TCP => 100.0,
        }
    }
}

/// A remote node in the RDMA cluster.
#[derive(Debug, Clone)]
pub struct RemoteNode {
    pub id: String,
    pub address: String,
    pub port: u16,
    pub transport: TransportType,
    pub connected: bool,
}

impl RemoteNode {
    pub fn new(id: &str, address: &str, port: u16, transport: TransportType) -> Self {
        Self {
            id: id.to_string(),
            address: address.to_string(),
            port,
            transport,
            connected: false,
        }
    }
}

/// RDMA communication channel (simulated).
pub struct RdmaChannel {
    config: RdmaConfig,
    local_node: String,
    remote_nodes: HashMap<String, RemoteNode>,
    /// Pending send operations.
    pending_sends: usize,
    /// Pending receive operations.
    pending_receives: usize,
}

impl RdmaChannel {
    pub fn new(config: RdmaConfig, local_node: String) -> Self {
        Self {
            config,
            local_node,
            remote_nodes: HashMap::new(),
            pending_sends: 0,
            pending_receives: 0,
        }
    }

    /// Add a remote node.
    pub fn add_remote(&mut self, node: RemoteNode) {
        self.remote_nodes.insert(node.id.clone(), node);
    }

    /// Connect to a remote node.
    pub fn connect(&mut self, node_id: &str) -> bool {
        if let Some(node) = self.remote_nodes.get_mut(node_id) {
            node.connected = true;
            true
        } else {
            false
        }
    }

    /// Send data to a remote node.
    pub fn send(&mut self, node_id: &str, data: &[f32], _offset: usize) -> Result<usize, String> {
        let node = self.remote_nodes.get(node_id)
            .ok_or_else(|| format!("Unknown node: {}", node_id))?;

        if !node.connected {
            return Err(format!("Node {} not connected", node_id));
        }

        // Simulated send: just count bytes
        self.pending_sends += 1;
        Ok(data.len() * std::mem::size_of::<f32>())
    }

    /// Receive data from a remote node.
    pub fn receive(&mut self, _node_id: &str, buffer: &mut [f32]) -> Result<usize, String> {
        // Simulated receive
        self.pending_receives += 1;
        Ok(buffer.len())
    }

    /// Perform all-reduce across all connected nodes (simulated).
    pub fn all_reduce(&mut self, data: &mut [f32]) {
        let n = self.remote_nodes.values().filter(|n| n.connected).count() + 1;
        // Simulate: just average
        let scale = 1.0 / n as f32;
        for v in data.iter_mut() {
            *v *= scale;
        }
    }

    /// Broadcast data from root to all nodes.
    pub fn broadcast(&mut self, data: &[f32], _root: usize) -> usize {
        let num_remotes = self.remote_nodes.values().filter(|n| n.connected).count();
        data.len() * num_remotes
    }

    /// Number of connected nodes.
    pub fn connected_count(&self) -> usize {
        self.remote_nodes.values().filter(|n| n.connected).count()
    }

    /// Get config.
    pub fn config(&self) -> &RdmaConfig {
        &self.config
    }

    /// Local node ID.
    pub fn local_node(&self) -> &str {
        &self.local_node
    }

    /// Pending operations.
    pub fn pending_ops(&self) -> usize {
        self.pending_sends + self.pending_receives
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Cloud GPU Tests --

    #[test]
    fn test_worker_node() {
        let w = WorkerNode::new("w0", "10.0.0.1:8080", "A100", 4, 80);
        assert_eq!(w.gpu_count, 4);
        assert!(w.is_available());
        assert!(w.is_healthy());
        assert!(w.assigned_shard.is_none());
    }

    #[test]
    fn test_worker_cost() {
        let mut w = WorkerNode::new("w0", "10.0.0.1:8080", "A100", 4, 80);
        w.cost_per_hour = 2.50;
        assert!((w.estimated_cost(10.0) - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_orchestrator_register() {
        let mut orch = CloudTrainingOrchestrator::new(CloudTrainingConfig::default());
        orch.register_worker(WorkerNode::new("w0", "addr", "GPU", 1, 16));
        assert_eq!(orch.active_workers(), 1);
    }

    #[test]
    fn test_orchestrator_assign_shards() {
        let mut orch = CloudTrainingOrchestrator::new(CloudTrainingConfig::default());
        for i in 0..4 {
            orch.register_worker(WorkerNode::new(&format!("w{}", i), "addr", "GPU", 1, 16));
        }
        orch.assign_shards(4);
        // All workers should be busy
        assert_eq!(orch.active_workers(), 4);
    }

    #[test]
    fn test_orchestrator_gradient_aggregation() {
        let mut orch = CloudTrainingOrchestrator::new(CloudTrainingConfig::default());
        orch.register_worker(WorkerNode::new("w0", "addr", "GPU", 1, 16));
        orch.register_worker(WorkerNode::new("w1", "addr", "GPU", 1, 16));
        orch.assign_shards(2);

        orch.receive_gradient_shard("w0", &[1.0, 2.0, 3.0], 0);
        assert!(!orch.gradients_complete());

        orch.receive_gradient_shard("w1", &[4.0, 5.0, 6.0], 3);
        assert!(orch.gradients_complete());

        let grads = orch.take_gradients();
        assert_eq!(grads.len(), 6);
        assert_eq!(orch.step(), 1);
    }

    #[test]
    fn test_orchestrator_fault_tolerance() {
        let mut orch = CloudTrainingOrchestrator::new(CloudTrainingConfig::default());
        let mut w = WorkerNode::new("w0", "addr", "GPU", 1, 16);
        w.status = WorkerStatus::Failed;
        w.assigned_shard = Some(0);
        orch.register_worker(w);

        let mut w1 = WorkerNode::new("w1", "addr", "GPU", 1, 16);
        w1.status = WorkerStatus::Idle;
        orch.register_worker(w1);

        orch.redistribute_failed();
        // w1 should now have shard 0
        let w1_ref = orch.workers.get("w1").unwrap();
        assert_eq!(w1_ref.assigned_shard, Some(0));
    }

    #[test]
    fn test_orchestrator_cost_optimal() {
        let mut orch = CloudTrainingOrchestrator::new(CloudTrainingConfig {
            prefer_spot: true,
            max_cost_per_hour: 100.0,
            ..CloudTrainingConfig::default()
        });

        let mut w0 = WorkerNode::new("w0", "addr", "A100", 1, 40);
        w0.cost_per_hour = 5.0;
        w0.is_spot = false;
        orch.register_worker(w0);

        let mut w1 = WorkerNode::new("w1", "addr", "A100", 1, 40);
        w1.cost_per_hour = 2.0;
        w1.is_spot = true;
        orch.register_worker(w1);

        let optimal = orch.cost_optimal_workers(50.0);
        assert!(optimal.len() >= 1);
    }

    // -- ANE/NPU Tests --

    #[test]
    fn test_gpu_capabilities() {
        let gpu = AcceleratorCapabilities::gpu();
        assert!(gpu.supports_matmul);
        assert!(gpu.supports_attention);
        assert!(!gpu.unified_memory);
    }

    #[test]
    fn test_ane_capabilities() {
        let ane = AcceleratorCapabilities::ane();
        assert!(ane.supports_matmul);
        assert!(!ane.supports_attention);
        assert!(ane.unified_memory);
    }

    #[test]
    fn test_ane_op_placement() {
        let placer = OpPlacer::default_with_ane(true);

        // Attention should go to GPU
        assert_eq!(placer.place(HardwareOp::Attention), PlacementDecision::GPU);

        // MatMul should go to GPU (compute-heavy)
        assert_eq!(placer.place(HardwareOp::MatMul), PlacementDecision::GPU);

        // BatchNorm should go to ANE
        assert_eq!(placer.place(HardwareOp::BatchNorm), PlacementDecision::ANE);

        // Activation should go to ANE
        assert_eq!(placer.place(HardwareOp::Activation), PlacementDecision::ANE);
    }

    #[test]
    fn test_no_ane_placement() {
        let placer = OpPlacer::default_with_ane(false);
        assert_eq!(placer.place(HardwareOp::BatchNorm), PlacementDecision::GPU);
        assert_eq!(placer.place(HardwareOp::Activation), PlacementDecision::GPU);
    }

    #[test]
    fn test_transformer_layer_plan() {
        let placer = OpPlacer::default_with_ane(true);
        let plan = placer.transformer_layer_plan();
        assert_eq!(plan.len(), 9); // 9 ops in a transformer layer

        // Check that attention ops go to GPU
        let attn_ops: Vec<_> = plan.iter().filter(|(op, _)| *op == HardwareOp::Attention).collect();
        assert!(attn_ops.iter().all(|(_, d)| *d == PlacementDecision::GPU));
    }

    #[test]
    fn test_unified_memory_buffer() {
        let mut buf = UnifiedMemoryBuffer::new(100);
        assert_eq!(buf.len(), 100);
        assert!(!buf.is_empty());

        let gpu_data = buf.gpu_access();
        gpu_data[0] = 1.0;
        let _ = gpu_data;

        let ane_data = buf.ane_access();
        assert!((ane_data[0] - 1.0).abs() < 1e-5); // Same memory
    }

    // -- RDMA Tests --

    #[test]
    fn test_rdma_config() {
        let config = RdmaConfig::nvlink();
        assert_eq!(config.transport, TransportType::NVLink);
        assert!(config.gpu_direct);
        assert!(config.estimated_bandwidth() > 200.0);
        assert!(config.estimated_latency_us() < 5.0);
    }

    #[test]
    fn test_rdma_config_roce() {
        let config = RdmaConfig::roce();
        assert_eq!(config.transport, TransportType::RoCE);
        assert!(config.gpu_direct);
    }

    #[test]
    fn test_rdma_config_tcp() {
        let config = RdmaConfig::tcp_fallback();
        assert_eq!(config.transport, TransportType::TCP);
        assert!(!config.gpu_direct);
        assert!(config.estimated_bandwidth() < 20.0);
    }

    #[test]
    fn test_rdma_channel() {
        let config = RdmaConfig::tcp_fallback();
        let mut channel = RdmaChannel::new(config, "node0".to_string());
        assert_eq!(channel.local_node(), "node0");

        channel.add_remote(RemoteNode::new("node1", "10.0.0.2", 8080, TransportType::TCP));
        assert!(channel.connect("node1"));
        assert_eq!(channel.connected_count(), 1);
    }

    #[test]
    fn test_rdma_send_receive() {
        let config = RdmaConfig::tcp_fallback();
        let mut channel = RdmaChannel::new(config, "node0".to_string());
        channel.add_remote(RemoteNode::new("node1", "10.0.0.2", 8080, TransportType::TCP));
        channel.connect("node1");

        let data = vec![1.0, 2.0, 3.0];
        let sent = channel.send("node1", &data, 0).unwrap();
        assert_eq!(sent, 12); // 3 × 4 bytes

        let mut buf = [0.0f32; 3];
        let received = channel.receive("node1", &mut buf).unwrap();
        assert_eq!(received, 3);
    }

    #[test]
    fn test_rdma_send_not_connected() {
        let config = RdmaConfig::tcp_fallback();
        let mut channel = RdmaChannel::new(config, "node0".to_string());
        channel.add_remote(RemoteNode::new("node1", "10.0.0.2", 8080, TransportType::TCP));

        let result = channel.send("node1", &[1.0], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rdma_all_reduce() {
        let config = RdmaConfig::nvlink();
        let mut channel = RdmaChannel::new(config, "node0".to_string());
        channel.add_remote(RemoteNode::new("node1", "addr", 8080, TransportType::NVLink));
        channel.connect("node1");

        let mut data = vec![4.0, 8.0];
        channel.all_reduce(&mut data);
        // 2 nodes: 4.0/2 = 2.0, 8.0/2 = 4.0
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_rdma_broadcast() {
        let config = RdmaConfig::tcp_fallback();
        let mut channel = RdmaChannel::new(config, "node0".to_string());
        channel.add_remote(RemoteNode::new("node1", "addr", 8080, TransportType::TCP));
        channel.add_remote(RemoteNode::new("node2", "addr", 8080, TransportType::TCP));
        channel.connect("node1");
        channel.connect("node2");

        let bytes = channel.broadcast(&[1.0, 2.0, 3.0], 0);
        assert_eq!(bytes, 6); // 3 values × 2 remotes
    }
}
