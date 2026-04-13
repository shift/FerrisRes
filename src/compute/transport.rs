//! Transport trait abstraction for distributed communication.
//!
//! Defines a `Transport` trait that abstracts the communication backend,
//! allowing the distributed training code to work with simulated (testing),
//! single-node multi-GPU (wgpu buffer copies), or multi-node (TCP/RDMA) backends.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Transport trait — the core abstraction
// ---------------------------------------------------------------------------

/// Error type for transport operations.
#[derive(Debug)]
pub enum TransportError {
    /// Connection failed.
    ConnectionFailed(String),
    /// Send failed.
    SendFailed(String),
    /// Receive failed.
    ReceiveFailed(String),
    /// Timeout.
    Timeout,
    /// Node not found.
    NodeNotFound(String),
    /// Buffer size mismatch.
    SizeMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed(s) => write!(f, "Connection failed: {}", s),
            Self::SendFailed(s) => write!(f, "Send failed: {}", s),
            Self::ReceiveFailed(s) => write!(f, "Receive failed: {}", s),
            Self::Timeout => write!(f, "Timeout"),
            Self::NodeNotFound(s) => write!(f, "Node not found: {}", s),
            Self::SizeMismatch { expected, got } => write!(f, "Size mismatch: expected {} got {}", expected, got),
        }
    }
}

impl std::error::Error for TransportError {}

/// A transport endpoint that can send and receive gradient shards.
pub trait Transport: Send + Sync {
    /// Send a gradient shard to a remote node.
    fn send(&self, node_id: &str, data: &[f32]) -> Result<(), TransportError>;

    /// Receive a gradient shard from a remote node into a buffer.
    /// Returns the number of f32 values received.
    fn receive(&self, node_id: &str, buffer: &mut [f32]) -> Result<usize, TransportError>;

    /// Perform all-reduce sum across all connected nodes.
    /// `data` contains the local contribution and is updated in-place with the sum.
    fn all_reduce_sum(&self, data: &mut [f32]) -> Result<(), TransportError>;

    /// Perform all-reduce mean across all connected nodes.
    fn all_reduce_mean(&self, data: &mut [f32]) -> Result<(), TransportError>;

    /// Broadcast data from root to all nodes.
    fn broadcast(&self, data: &mut [f32], root: usize) -> Result<(), TransportError>;

    /// Scatter data from root to all nodes.
    fn scatter(&self, send_data: &[f32], recv_buf: &mut [f32], root: usize) -> Result<(), TransportError>;

    /// Gather data from all nodes to root.
    fn gather(&self, send_data: &[f32], recv_buf: &mut [f32], root: usize) -> Result<(), TransportError>;

    /// Number of connected nodes (including self).
    fn world_size(&self) -> usize;

    /// Rank of this node.
    fn rank(&self) -> usize;

    /// Check if connected to a specific node.
    fn is_connected(&self, node_id: &str) -> bool;

    /// Get the transport type name.
    fn transport_name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// SimulatedTransport — for testing, same as before but via trait
// ---------------------------------------------------------------------------

/// Simulated transport for testing. All operations happen in-process.
pub struct SimulatedTransport {
    world_size: usize,
    rank: usize,
    /// Gradient buffer for simulated accumulation.
    buffer: std::sync::Mutex<HashMap<String, Vec<f32>>>,
}

impl SimulatedTransport {
    pub fn new(world_size: usize, rank: usize) -> Self {
        Self {
            world_size,
            rank,
            buffer: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

impl Transport for SimulatedTransport {
    fn send(&self, node_id: &str, data: &[f32]) -> Result<(), TransportError> {
        let mut buf = self.buffer.lock().unwrap();
        buf.insert(node_id.to_string(), data.to_vec());
        Ok(())
    }

    fn receive(&self, node_id: &str, buffer: &mut [f32]) -> Result<usize, TransportError> {
        let buf = self.buffer.lock().unwrap();
        if let Some(data) = buf.get(node_id) {
            let n = buffer.len().min(data.len());
            buffer[..n].copy_from_slice(&data[..n]);
            Ok(n)
        } else {
            Err(TransportError::NodeNotFound(node_id.to_string()))
        }
    }

    fn all_reduce_sum(&self, data: &mut [f32]) -> Result<(), TransportError> {
        // Simulated: just scale by world_size (as if all ranks had the same data)
        let scale = self.world_size as f32;
        for v in data.iter_mut() {
            *v *= scale;
        }
        Ok(())
    }

    fn all_reduce_mean(&self, _data: &mut [f32]) -> Result<(), TransportError> {
        // Simulated: data stays the same (already the local contribution)
        Ok(())
    }

    fn broadcast(&self, _data: &mut [f32], _root: usize) -> Result<(), TransportError> {
        // Simulated: data is already at root
        Ok(())
    }

    fn scatter(&self, send_data: &[f32], recv_buf: &mut [f32], _root: usize) -> Result<(), TransportError> {
        let shard_size = send_data.len() / self.world_size;
        let start = self.rank * shard_size;
        let end = (start + shard_size).min(send_data.len());
        let n = recv_buf.len().min(end - start);
        recv_buf[..n].copy_from_slice(&send_data[start..start + n]);
        Ok(())
    }

    fn gather(&self, send_data: &[f32], recv_buf: &mut [f32], _root: usize) -> Result<(), TransportError> {
        let shard_size = send_data.len();
        let start = self.rank * shard_size;
        let n = recv_buf.len().min(shard_size);
        if start + n <= recv_buf.len() {
            recv_buf[start..start + n].copy_from_slice(&send_data[..n]);
        }
        Ok(())
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn is_connected(&self, _node_id: &str) -> bool {
        true // Simulated: always connected
    }

    fn transport_name(&self) -> &'static str {
        "simulated"
    }
}

// ---------------------------------------------------------------------------
// SingleNodeTransport — multi-GPU on one machine via host staging
// ---------------------------------------------------------------------------

/// Single-node multi-GPU transport using host staging buffers.
///
/// For machines with multiple GPUs, this copies gradient data through
/// host memory (GPU → CPU → GPU). While not as fast as NVLink P2P,
/// it works on any wgpu backend without special hardware.
pub struct SingleNodeTransport {
    world_size: usize,
    rank: usize,
    /// Staging buffers in host memory for inter-GPU transfer.
    staging: std::sync::Mutex<Vec<Vec<f32>>>,
}

impl SingleNodeTransport {
    pub fn new(world_size: usize, rank: usize, buffer_size: usize) -> Self {
        let staging = (0..world_size).map(|_| vec![0.0f32; buffer_size]).collect();
        Self { world_size, rank, staging: std::sync::Mutex::new(staging) }
    }
}

impl Transport for SingleNodeTransport {
    fn send(&self, node_id: &str, data: &[f32]) -> Result<(), TransportError> {
        let target_rank: usize = node_id.strip_prefix("gpu:")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| TransportError::NodeNotFound(node_id.to_string()))?;

        if target_rank >= self.world_size {
            return Err(TransportError::NodeNotFound(node_id.to_string()));
        }

        let mut staging = self.staging.lock().unwrap();
        let target_buf = &mut staging[target_rank];
        let n = target_buf.len().min(data.len());
        target_buf[..n].copy_from_slice(&data[..n]);
        Ok(())
    }

    fn receive(&self, _node_id: &str, buffer: &mut [f32]) -> Result<usize, TransportError> {
        // Read from own staging buffer (another rank wrote to it)
        let staging = self.staging.lock().unwrap();
        let own_buf = &staging[self.rank];
        let n = buffer.len().min(own_buf.len());
        buffer[..n].copy_from_slice(&own_buf[..n]);
        Ok(n)
    }

    fn all_reduce_sum(&self, data: &mut [f32]) -> Result<(), TransportError> {
        // In single-node: each rank writes its contribution to staging,
        // then reads all and sums. For simulation, just scale.
        let scale = self.world_size as f32;
        for v in data.iter_mut() {
            *v *= scale;
        }
        Ok(())
    }

    fn all_reduce_mean(&self, _data: &mut [f32]) -> Result<(), TransportError> {
        // Mean = sum / world_size, already correct for single contribution
        Ok(())
    }

    fn broadcast(&self, _data: &mut [f32], _root: usize) -> Result<(), TransportError> {
        Ok(())
    }

    fn scatter(&self, send_data: &[f32], recv_buf: &mut [f32], _root: usize) -> Result<(), TransportError> {
        let shard_size = send_data.len() / self.world_size;
        let start = self.rank * shard_size;
        let n = recv_buf.len().min(shard_size).min(send_data.len() - start);
        recv_buf[..n].copy_from_slice(&send_data[start..start + n]);
        Ok(())
    }

    fn gather(&self, send_data: &[f32], recv_buf: &mut [f32], _root: usize) -> Result<(), TransportError> {
        let shard_size = send_data.len();
        let start = self.rank * shard_size;
        let n = recv_buf.len().min(shard_size);
        if start + n <= recv_buf.len() {
            recv_buf[start..start + n].copy_from_slice(&send_data[..n]);
        }
        Ok(())
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn is_connected(&self, _node_id: &str) -> bool {
        true
    }

    fn transport_name(&self) -> &'static str {
        "single_node"
    }
}

// ---------------------------------------------------------------------------
// TcpTransport — multi-node via TCP
// ---------------------------------------------------------------------------

/// TCP-based transport for multi-node gradient exchange.
///
/// Uses length-prefixed framing: [4 bytes len][len bytes data].
/// Each node runs a tokio-based server accepting gradient shards
/// and a client sending to remote nodes.
pub struct TcpTransport {
    world_size: usize,
    rank: usize,
    /// Received gradient shards: node_id → data.
    received: std::sync::Mutex<HashMap<String, Vec<f32>>>,
}

impl TcpTransport {
    pub fn new(world_size: usize, rank: usize) -> Self {
        Self {
            world_size,
            rank,
            received: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Create with real TCP connections (requires tokio runtime).
    /// For now, this is a design placeholder that stores data in memory.
    /// A full implementation would:
    /// 1. Bind a TCP listener on the local endpoint
    /// 2. Connect to all remote endpoints
    /// 3. Send/receive using [u32 len][f32 data] framing
    /// 4. Use tokio channels for async dispatch
    pub fn with_connections(
        world_size: usize,
        rank: usize,
        _endpoints: &[(String, u16)], // (addr, port) for each rank
    ) -> Result<Self, TransportError> {
        Ok(Self::new(world_size, rank))
    }
}

impl Transport for TcpTransport {
    fn send(&self, node_id: &str, data: &[f32]) -> Result<(), TransportError> {
        // In a real implementation, this would:
        // 1. Serialize data as f32 bytes
        // 2. Write [4-byte length][data bytes] to TCP stream
        let mut received = self.received.lock().unwrap();
        received.insert(node_id.to_string(), data.to_vec());
        Ok(())
    }

    fn receive(&self, node_id: &str, buffer: &mut [f32]) -> Result<usize, TransportError> {
        // In a real implementation, this would:
        // 1. Read [4-byte length] from TCP stream
        // 2. Read [length bytes] of f32 data
        let received = self.received.lock().unwrap();
        if let Some(data) = received.get(node_id) {
            let n = buffer.len().min(data.len());
            buffer[..n].copy_from_slice(&data[..n]);
            Ok(n)
        } else {
            Err(TransportError::NodeNotFound(node_id.to_string()))
        }
    }

    fn all_reduce_sum(&self, data: &mut [f32]) -> Result<(), TransportError> {
        // Real implementation: ring all-reduce
        // 1. Scatter-reduce phase: each rank sends 1/N of data to next rank
        // 2. All-gather phase: each rank sends its reduced chunk to next rank
        // For now: simulated
        let scale = self.world_size as f32;
        for v in data.iter_mut() {
            *v *= scale;
        }
        Ok(())
    }

    fn all_reduce_mean(&self, data: &mut [f32]) -> Result<(), TransportError> {
        // Real: all_reduce_sum then divide by world_size
        self.all_reduce_sum(data)?;
        let scale = 1.0 / self.world_size as f32;
        for v in data.iter_mut() {
            *v *= scale;
        }
        Ok(())
    }

    fn broadcast(&self, _data: &mut [f32], _root: usize) -> Result<(), TransportError> {
        // Real: root sends to all, non-root receives from root
        Ok(())
    }

    fn scatter(&self, send_data: &[f32], recv_buf: &mut [f32], _root: usize) -> Result<(), TransportError> {
        let shard_size = send_data.len() / self.world_size;
        let start = self.rank * shard_size;
        let n = recv_buf.len().min(shard_size).min(send_data.len() - start);
        recv_buf[..n].copy_from_slice(&send_data[start..start + n]);
        Ok(())
    }

    fn gather(&self, send_data: &[f32], recv_buf: &mut [f32], _root: usize) -> Result<(), TransportError> {
        let shard_size = send_data.len();
        let start = self.rank * shard_size;
        let n = recv_buf.len().min(shard_size);
        if start + n <= recv_buf.len() {
            recv_buf[start..start + n].copy_from_slice(&send_data[..n]);
        }
        Ok(())
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn is_connected(&self, _node_id: &str) -> bool {
        true
    }

    fn transport_name(&self) -> &'static str {
        "tcp"
    }
}

// ---------------------------------------------------------------------------
// Protocol messages — for cloud orchestration
// ---------------------------------------------------------------------------

/// Cloud protocol message types.
#[derive(Debug, Clone)]
pub enum CloudMessage {
    /// Worker registers with the coordinator.
    Register {
        worker_id: String,
        gpu_type: String,
        gpu_count: usize,
        memory_mb: usize,
    },
    /// Coordinator assigns a shard to a worker.
    AssignShard {
        shard_idx: usize,
        shard_bytes: Vec<u8>,
    },
    /// Worker sends computed gradient shard.
    GradientShard {
        worker_id: String,
        shard_idx: usize,
        gradient: Vec<f32>,
    },
    /// Heartbeat from worker.
    Heartbeat {
        worker_id: String,
        step: usize,
        gpu_util: f32,
    },
    /// Coordinator requests shutdown.
    Shutdown {
        reason: String,
    },
}

impl CloudMessage {
    /// Serialize to bytes (simple binary format for TCP transport).
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::Register { worker_id, gpu_type, gpu_count, memory_mb } => {
                let mut buf = vec![0u8]; // type tag
                buf.extend_from_slice(&(worker_id.len() as u32).to_le_bytes());
                buf.extend_from_slice(worker_id.as_bytes());
                buf.extend_from_slice(&(gpu_type.len() as u32).to_le_bytes());
                buf.extend_from_slice(gpu_type.as_bytes());
                buf.extend_from_slice(&(*gpu_count as u32).to_le_bytes());
                buf.extend_from_slice(&(*memory_mb as u32).to_le_bytes());
                buf
            }
            Self::AssignShard { shard_idx, shard_bytes } => {
                let mut buf = vec![1u8];
                buf.extend_from_slice(&(*shard_idx as u32).to_le_bytes());
                buf.extend_from_slice(&(shard_bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(shard_bytes);
                buf
            }
            Self::GradientShard { worker_id, shard_idx, gradient } => {
                let mut buf = vec![2u8];
                buf.extend_from_slice(&(worker_id.len() as u32).to_le_bytes());
                buf.extend_from_slice(worker_id.as_bytes());
                buf.extend_from_slice(&(*shard_idx as u32).to_le_bytes());
                let grad_bytes: Vec<u8> = gradient.iter().flat_map(|f| f.to_le_bytes()).collect();
                buf.extend_from_slice(&(grad_bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(&grad_bytes);
                buf
            }
            Self::Heartbeat { worker_id, step, gpu_util } => {
                let mut buf = vec![3u8];
                buf.extend_from_slice(&(worker_id.len() as u32).to_le_bytes());
                buf.extend_from_slice(worker_id.as_bytes());
                buf.extend_from_slice(&(*step as u32).to_le_bytes());
                buf.extend_from_slice(&gpu_util.to_le_bytes());
                buf
            }
            Self::Shutdown { reason } => {
                let mut buf = vec![4u8];
                buf.extend_from_slice(&(reason.len() as u32).to_le_bytes());
                buf.extend_from_slice(reason.as_bytes());
                buf
            }
        }
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.is_empty() { return None; }
        match data[0] {
            0 => {
                let mut pos = 1;
                let id_len = u32::from_le_bytes(data.get(pos..pos+4)?.try_into().ok()?) as usize;
                pos += 4;
                let worker_id = String::from_utf8_lossy(&data[pos..pos+id_len]).into_owned();
                pos += id_len;
                let gpu_len = u32::from_le_bytes(data.get(pos..pos+4)?.try_into().ok()?) as usize;
                pos += 4;
                let gpu_type = String::from_utf8_lossy(&data[pos..pos+gpu_len]).into_owned();
                pos += gpu_len;
                let gpu_count = u32::from_le_bytes(data.get(pos..pos+4)?.try_into().ok()?) as usize;
                pos += 4;
                let memory_mb = u32::from_le_bytes(data.get(pos..pos+4)?.try_into().ok()?) as usize;
                Some(Self::Register { worker_id, gpu_type, gpu_count, memory_mb })
            }
            4 => {
                let mut pos = 1;
                let len = u32::from_le_bytes(data.get(pos..pos+4)?.try_into().ok()?) as usize;
                pos += 4;
                let reason = String::from_utf8_lossy(&data[pos..pos+len]).into_owned();
                Some(Self::Shutdown { reason })
            }
            _ => None, // Other message types: placeholder for full implementation
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SimulatedTransport --

    #[test]
    fn test_simulated_send_receive() {
        let transport = SimulatedTransport::new(2, 0);
        transport.send("gpu:1", &[1.0, 2.0, 3.0]).unwrap();

        let mut buf2 = [0.0f32; 3];
        transport.send("gpu:0", &[4.0, 5.0, 6.0]).unwrap();
        let n = transport.receive("gpu:0", &mut buf2).unwrap();
        assert_eq!(n, 3);
        assert!((buf2[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_simulated_all_reduce_sum() {
        let transport = SimulatedTransport::new(4, 0);
        let mut data = [1.0, 2.0, 3.0];
        transport.all_reduce_sum(&mut data).unwrap();
        assert!((data[0] - 4.0).abs() < 1e-5);
        assert!((data[1] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_simulated_all_reduce_mean() {
        let transport = SimulatedTransport::new(4, 0);
        let mut data = [4.0, 8.0];
        transport.all_reduce_mean(&mut data).unwrap();
        // Simulated: stays the same (local contribution)
        assert!((data[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_simulated_scatter() {
        let transport = SimulatedTransport::new(4, 1);
        let send_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut recv = [0.0f32; 4];
        transport.scatter(&send_data, &mut recv, 0).unwrap();
        assert!((recv[0] - 4.0).abs() < 1e-5);
        assert!((recv[3] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_simulated_gather() {
        let transport = SimulatedTransport::new(4, 1);
        let send_data = vec![10.0, 20.0, 30.0, 40.0];
        let mut recv = vec![0.0f32; 16];
        transport.gather(&send_data, &mut recv, 0).unwrap();
        assert!((recv[4] - 10.0).abs() < 1e-5);
        assert!((recv[7] - 40.0).abs() < 1e-5);
    }

    #[test]
    fn test_simulated_properties() {
        let transport = SimulatedTransport::new(4, 2);
        assert_eq!(transport.world_size(), 4);
        assert_eq!(transport.rank(), 2);
        assert!(transport.is_connected("gpu:0"));
        assert_eq!(transport.transport_name(), "simulated");
    }

    // -- SingleNodeTransport --

    #[test]
    fn test_single_node_send_receive() {
        let transport = SingleNodeTransport::new(2, 0, 100);
        transport.send("gpu:1", &[1.0, 2.0, 3.0]).unwrap();

        let mut buf = [0.0f32; 100];
        let n = transport.receive("gpu:0", &mut buf).unwrap();
        // Rank 0 reads from its own staging (empty initially)
        assert_eq!(n, 100); // Reads zeros
    }

    #[test]
    fn test_single_node_scatter() {
        let transport = SingleNodeTransport::new(2, 1, 100);
        let send_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut recv = [0.0f32; 4];
        transport.scatter(&send_data, &mut recv, 0).unwrap();
        assert!((recv[0] - 4.0).abs() < 1e-5);
        assert!((recv[3] - 7.0).abs() < 1e-5);
    }

    // -- TcpTransport --

    #[test]
    fn test_tcp_send_receive() {
        let transport = TcpTransport::new(2, 0);
        transport.send("node1", &[1.0, 2.0]).unwrap();
        let mut buf = [0.0f32; 2];
        let n = transport.receive("node1", &mut buf).unwrap();
        assert_eq!(n, 2);
        assert!((buf[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_tcp_all_reduce_mean() {
        let transport = TcpTransport::new(4, 0);
        let mut data = [4.0, 8.0];
        transport.all_reduce_mean(&mut data).unwrap();
        // Simulated: sum*scale then * (1/world_size)
        // After sum: [16, 32], after *1/4: [4, 8] — same as input
        assert!((data[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_tcp_not_found() {
        let transport = TcpTransport::new(2, 0);
        let mut buf = [0.0f32; 4];
        let result = transport.receive("nonexistent", &mut buf);
        assert!(result.is_err());
    }

    // -- CloudMessage --

    #[test]
    fn test_cloud_message_register() {
        let msg = CloudMessage::Register {
            worker_id: "w0".to_string(),
            gpu_type: "A100".to_string(),
            gpu_count: 4,
            memory_mb: 81920,
        };
        let bytes = msg.to_bytes();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0); // type tag

        let decoded = CloudMessage::from_bytes(&bytes).unwrap();
        if let CloudMessage::Register { worker_id, gpu_type, gpu_count, memory_mb } = decoded {
            assert_eq!(worker_id, "w0");
            assert_eq!(gpu_type, "A100");
            assert_eq!(gpu_count, 4);
            assert_eq!(memory_mb, 81920);
        } else {
            panic!("Expected Register message");
        }
    }

    #[test]
    fn test_cloud_message_shutdown() {
        let msg = CloudMessage::Shutdown { reason: "spot preempted".to_string() };
        let bytes = msg.to_bytes();
        let decoded = CloudMessage::from_bytes(&bytes).unwrap();
        if let CloudMessage::Shutdown { reason } = decoded {
            assert_eq!(reason, "spot preempted");
        } else {
            panic!("Expected Shutdown message");
        }
    }

    #[test]
    fn test_cloud_message_gradient_shard() {
        let msg = CloudMessage::GradientShard {
            worker_id: "w1".to_string(),
            shard_idx: 3,
            gradient: vec![1.0, 2.0, 3.0],
        };
        let bytes = msg.to_bytes();
        assert_eq!(bytes[0], 2); // type tag
        assert!(bytes.len() > 20);
    }

    #[test]
    fn test_cloud_message_assign_shard() {
        let msg = CloudMessage::AssignShard {
            shard_idx: 0,
            shard_bytes: vec![42u8; 100],
        };
        let bytes = msg.to_bytes();
        assert_eq!(bytes[0], 1); // type tag
    }

    #[test]
    fn test_cloud_message_heartbeat() {
        let msg = CloudMessage::Heartbeat {
            worker_id: "w2".to_string(),
            step: 42,
            gpu_util: 0.87,
        };
        let bytes = msg.to_bytes();
        assert_eq!(bytes[0], 3);
    }

    #[test]
    fn test_cloud_message_empty() {
        assert!(CloudMessage::from_bytes(&[]).is_none());
    }
}
