//! PagedAttention — multi-request KV cache with block-based memory management.
//!
//! Implements vLLM-style PagedAttention where the KV cache is managed as a pool
//! of fixed-size blocks rather than contiguous per-request buffers. This enables:
//! - **Memory efficiency**: no wasted padding for variable-length sequences
//! - **Prefix sharing**: common system prompts share KV blocks across requests
//! - **Copy-on-write**: forked requests (beam search, speculative) share blocks
//!   until they diverge
//!
//! Reference: Kwon et al., "Efficient Memory Management for Large Language Model
//! Serving with PagedAttention" (OSDI 2023)

use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size: 16 tokens per block.
pub const DEFAULT_BLOCK_SIZE: usize = 16;

// ---------------------------------------------------------------------------
// PhysicalBlock — a single GPU memory block (metadata only)
// ---------------------------------------------------------------------------

/// Metadata for a physical block in the memory pool.
#[derive(Debug)]
pub struct PhysicalBlockMeta {
    /// Reference count (for copy-on-write).
    pub ref_count: u32,
    /// Block ID in the pool.
    pub id: u32,
    /// Whether this block is currently allocated.
    pub allocated: bool,
}

// ---------------------------------------------------------------------------
// BlockAllocator — manages the physical block pool
// ---------------------------------------------------------------------------

/// Manages a pool of virtual blocks (metadata). GPU buffer creation is
/// separate — this handles only the allocation/free/refcounting logic.
pub struct BlockAllocator {
    /// Block metadata.
    blocks: Vec<PhysicalBlockMeta>,
    /// Free block indices.
    free_list: Vec<u32>,
    /// Block size in tokens.
    block_size: usize,
    /// Hidden dimension per token.
    hidden_dim: usize,
}

impl BlockAllocator {
    /// Create a new block allocator with the given pool size.
    /// Does not allocate GPU buffers — only metadata.
    pub fn new(num_blocks: usize, block_size: usize, hidden_dim: usize) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut free_list = Vec::with_capacity(num_blocks);

        for i in 0..num_blocks {
            blocks.push(PhysicalBlockMeta {
                ref_count: 0,
                id: i as u32,
                allocated: false,
            });
            free_list.push(i as u32);
        }

        Self {
            blocks,
            free_list,
            block_size,
            hidden_dim,
        }
    }

    /// Allocate a new physical block.
    pub fn allocate(&mut self) -> Option<u32> {
        if let Some(id) = self.free_list.pop() {
            let block = &mut self.blocks[id as usize];
            block.ref_count = 1;
            block.allocated = true;
            Some(id)
        } else {
            None
        }
    }

    /// Free a physical block (decrement ref count, return to pool if 0).
    pub fn free(&mut self, block_id: u32) {
        let block = &mut self.blocks[block_id as usize];
        if block.ref_count > 0 {
            block.ref_count -= 1;
        }
        if block.ref_count == 0 && block.allocated {
            block.allocated = false;
            self.free_list.push(block_id);
        }
    }

    /// Increase reference count (for copy-on-write sharing).
    pub fn add_ref(&mut self, block_id: u32) {
        self.blocks[block_id as usize].ref_count += 1;
    }

    /// Get reference count for a block.
    pub fn ref_count(&self, block_id: u32) -> u32 {
        self.blocks[block_id as usize].ref_count
    }

    /// Number of free blocks.
    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }

    /// Total number of blocks.
    pub fn total_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Block size in tokens.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Memory usage fraction (0.0–1.0).
    pub fn utilization(&self) -> f32 {
        if self.blocks.is_empty() {
            return 0.0;
        }
        1.0 - (self.free_list.len() as f32 / self.blocks.len() as f32)
    }

    /// Total memory used in bytes (blocks × block_size × hidden_dim × sizeof(f32) × 2 for K+V).
    pub fn memory_used_bytes(&self) -> usize {
        let used_blocks = self.blocks.len() - self.free_list.len();
        used_blocks * self.block_size * self.hidden_dim * std::mem::size_of::<f32>() * 2
    }
}

// ---------------------------------------------------------------------------
// BlockTable — maps logical positions to physical blocks
// ---------------------------------------------------------------------------

/// Maps logical KV positions (per-request) to physical block IDs.
///
/// For a request with `num_tokens` tokens and `block_size = 16`:
///   - Logical position 0-15 → physical block table[0]
///   - Logical position 16-31 → physical block table[1]
///   - etc.
pub struct BlockTable {
    /// Physical block IDs for this request.
    table: Vec<u32>,
    /// Number of tokens currently stored (may be < table.len() * block_size).
    num_tokens: usize,
    /// Block size.
    block_size: usize,
}

impl BlockTable {
    pub fn new(block_size: usize) -> Self {
        Self {
            table: Vec::new(),
            num_tokens: 0,
            block_size,
        }
    }

    /// Number of logical blocks needed.
    pub fn num_blocks_needed(&self) -> usize {
        (self.num_tokens + self.block_size - 1) / self.block_size
    }

    /// Number of blocks currently allocated.
    pub fn num_blocks_allocated(&self) -> usize {
        self.table.len()
    }

    /// Get physical block ID for a logical position.
    pub fn block_for_position(&self, pos: usize) -> Option<u32> {
        let block_idx = pos / self.block_size;
        self.table.get(block_idx).copied()
    }

    /// Append a physical block to the table.
    pub fn append_block(&mut self, block_id: u32) {
        self.table.push(block_id);
    }

    /// Record that `n` tokens were added.
    pub fn add_tokens(&mut self, n: usize) {
        self.num_tokens += n;
    }

    /// Get the number of tokens stored.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Get the block table (for GPU indirection).
    pub fn table(&self) -> &[u32] {
        &self.table
    }

    /// Clear the table (after freeing blocks).
    pub fn clear(&mut self) {
        self.table.clear();
        self.num_tokens = 0;
    }

    /// Slot offset within a block for a given token position.
    pub fn slot_in_block(&self, pos: usize) -> usize {
        pos % self.block_size
    }
}

// ---------------------------------------------------------------------------
// RequestState — per-request paged KV state
// ---------------------------------------------------------------------------

/// The KV cache state for a single inference request.
pub struct RequestState {
    /// Block tables per layer (one BlockTable per layer).
    layer_tables: Vec<BlockTable>,
    /// Request ID.
    request_id: u64,
    /// Block size.
    #[allow(dead_code)]
    block_size: usize,
    /// Number of tokens in the request so far.
    num_tokens: usize,
}

impl RequestState {
    pub fn new(request_id: u64, num_layers: usize, block_size: usize) -> Self {
        let layer_tables = (0..num_layers)
            .map(|_| BlockTable::new(block_size))
            .collect();
        Self {
            layer_tables,
            request_id,
            block_size,
            num_tokens: 0,
        }
    }

    /// Get the block table for a given layer.
    pub fn layer_table(&self, layer_idx: usize) -> &BlockTable {
        &self.layer_tables[layer_idx]
    }

    /// Get mutable block table for a given layer.
    pub fn layer_table_mut(&mut self, layer_idx: usize) -> &mut BlockTable {
        &mut self.layer_tables[layer_idx]
    }

    /// Number of tokens processed.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Record tokens added.
    pub fn add_tokens(&mut self, n: usize) {
        self.num_tokens += n;
    }

    /// Request ID.
    pub fn request_id(&self) -> u64 {
        self.request_id
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layer_tables.len()
    }
}

// ---------------------------------------------------------------------------
// PrefixSharing — share KV blocks for common prefixes
// ---------------------------------------------------------------------------

/// Manages prefix sharing across requests.
///
/// When multiple requests share a common system prompt, their KV cache
/// blocks for the prompt can be shared (copy-on-write). When a request
/// diverges (generates different tokens), it copies its block.
pub struct PrefixSharingManager {
    /// Hash of token sequence → block IDs per layer.
    prefix_blocks: HashMap<Vec<u32>, Vec<Vec<u32>>>,
    /// Block allocator reference.
    allocator: Arc<std::sync::Mutex<BlockAllocator>>,
    /// Block size.
    #[allow(dead_code)]
    block_size: usize,
}

impl PrefixSharingManager {
    pub fn new(allocator: Arc<std::sync::Mutex<BlockAllocator>>, block_size: usize) -> Self {
        Self {
            prefix_blocks: HashMap::new(),
            allocator,
            block_size,
        }
    }

    /// Register a prefix and its allocated blocks.
    pub fn register_prefix(&mut self, prefix_tokens: &[u32], layer_blocks: Vec<Vec<u32>>) {
        self.prefix_blocks.insert(prefix_tokens.to_vec(), layer_blocks);
    }

    /// Look up a prefix and return shared block IDs (incrementing ref counts).
    pub fn lookup_prefix(&self, prefix_tokens: &[u32]) -> Option<Vec<Vec<u32>>> {
        if let Some(blocks) = self.prefix_blocks.get(prefix_tokens) {
            let mut alloc = self.allocator.lock().unwrap();
            for layer_blocks in blocks {
                for &block_id in layer_blocks {
                    alloc.add_ref(block_id);
                }
            }
            Some(blocks.clone())
        } else {
            None
        }
    }

    /// Number of registered prefixes.
    pub fn num_prefixes(&self) -> usize {
        self.prefix_blocks.len()
    }

    /// Check if a prefix is registered.
    pub fn has_prefix(&self, tokens: &[u32]) -> bool {
        self.prefix_blocks.contains_key(tokens)
    }

    /// Remove a prefix registration.
    pub fn remove_prefix(&mut self, tokens: &[u32]) {
        self.prefix_blocks.remove(tokens);
    }
}

// ---------------------------------------------------------------------------
// PagedAttentionManager — top-level manager
// ---------------------------------------------------------------------------

/// Top-level manager for paged attention across all requests.
pub struct PagedAttentionManager {
    /// Active request states.
    requests: HashMap<u64, RequestState>,
    /// Block allocator (shared).
    allocator: Arc<std::sync::Mutex<BlockAllocator>>,
    /// Prefix sharing manager.
    prefix_manager: PrefixSharingManager,
    /// Block size.
    block_size: usize,
    /// Hidden dimension.
    #[allow(dead_code)]
    hidden_dim: usize,
    /// Number of layers.
    num_layers: usize,
    /// Next request ID.
    next_request_id: u64,
}

impl PagedAttentionManager {
    pub fn new(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        hidden_dim: usize,
    ) -> Self {
        let allocator = BlockAllocator::new(num_blocks, block_size, hidden_dim);
        let allocator = Arc::new(std::sync::Mutex::new(allocator));

        let prefix_manager = PrefixSharingManager::new(Arc::clone(&allocator), block_size);

        Self {
            requests: HashMap::new(),
            allocator,
            prefix_manager,
            block_size,
            hidden_dim,
            num_layers,
            next_request_id: 0,
        }
    }

    /// Create a new request with pre-allocated blocks for prefill.
    pub fn create_request(&mut self, prefill_len: usize) -> u64 {
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        // Calculate number of layers from first request
        // (we derive it from prefix manager or a default)
        let num_layers = self.num_layers;

        let mut state = RequestState::new(request_id, num_layers, self.block_size);

        // Allocate blocks for prefill
        let num_blocks_needed = (prefill_len + self.block_size - 1) / self.block_size;
        let mut alloc = self.allocator.lock().unwrap();
        for layer_idx in 0..num_layers {
            let table = state.layer_table_mut(layer_idx);
            for _ in 0..num_blocks_needed {
                if let Some(block_id) = alloc.allocate() {
                    table.append_block(block_id);
                }
            }
            table.add_tokens(prefill_len);
        }
        state.add_tokens(prefill_len);

        self.requests.insert(request_id, state);
        request_id
    }

    /// Create a new request with a specific number of layers.
    pub fn create_request_with_layers(&mut self, prefill_len: usize, num_layers: usize) -> u64 {
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let mut state = RequestState::new(request_id, num_layers, self.block_size);
        let num_blocks_needed = (prefill_len + self.block_size - 1) / self.block_size;
        let mut alloc = self.allocator.lock().unwrap();
        for layer_idx in 0..num_layers {
            let table = state.layer_table_mut(layer_idx);
            for _ in 0..num_blocks_needed {
                if let Some(block_id) = alloc.allocate() {
                    table.append_block(block_id);
                }
            }
            table.add_tokens(prefill_len);
        }
        state.add_tokens(prefill_len);

        self.requests.insert(request_id, state);
        request_id
    }

    /// Fork a request (for beam search / speculative decoding).
    /// The forked request shares all existing blocks (copy-on-write).
    pub fn fork_request(&mut self, parent_id: u64) -> Option<u64> {
        let parent = self.requests.get(&parent_id)?;
        let child_id = self.next_request_id;
        self.next_request_id += 1;

        let num_layers = parent.num_layers();
        let mut child = RequestState::new(child_id, num_layers, self.block_size);
        child.num_tokens = parent.num_tokens;

        let mut alloc = self.allocator.lock().unwrap();
        for layer_idx in 0..num_layers {
            let parent_table = parent.layer_table(layer_idx);
            let child_table = child.layer_table_mut(layer_idx);
            for &block_id in parent_table.table() {
                alloc.add_ref(block_id);
                child_table.append_block(block_id);
            }
            child_table.add_tokens(parent_table.num_tokens());
        }

        self.requests.insert(child_id, child);
        Some(child_id)
    }

    /// Free a request (release all blocks).
    pub fn free_request(&mut self, request_id: u64) {
        if let Some(state) = self.requests.remove(&request_id) {
            let mut alloc = self.allocator.lock().unwrap();
            for table in &state.layer_tables {
                for &block_id in table.table() {
                    alloc.free(block_id);
                }
            }
        }
    }

    /// Get a request state.
    pub fn get_request(&self, request_id: u64) -> Option<&RequestState> {
        self.requests.get(&request_id)
    }

    /// Get mutable request state.
    pub fn get_request_mut(&mut self, request_id: u64) -> Option<&mut RequestState> {
        self.requests.get_mut(&request_id)
    }

    /// Number of active requests.
    pub fn num_requests(&self) -> usize {
        self.requests.len()
    }

    /// Memory utilization (0.0–1.0).
    pub fn utilization(&self) -> f32 {
        self.allocator.lock().unwrap().utilization()
    }

    /// Number of free blocks.
    pub fn num_free_blocks(&self) -> usize {
        self.allocator.lock().unwrap().num_free()
    }

    /// Block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Access the prefix sharing manager.
    pub fn prefix_manager(&self) -> &PrefixSharingManager {
        &self.prefix_manager
    }

    /// Mutable access to prefix sharing manager.
    pub fn prefix_manager_mut(&mut self) -> &mut PrefixSharingManager {
        &mut self.prefix_manager
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_allocator_basic() {
        let mut alloc = BlockAllocator::new(10, 16, 64);
        assert_eq!(alloc.num_free(), 10);

        let b0 = alloc.allocate();
        assert!(b0.is_some());
        assert_eq!(alloc.num_free(), 9);

        let b1 = alloc.allocate();
        assert!(b1.is_some());
        assert_eq!(alloc.num_free(), 8);

        alloc.free(b0.unwrap());
        assert_eq!(alloc.num_free(), 9);
    }

    #[test]
    fn test_block_allocator_exhaustion() {
        let mut alloc = BlockAllocator::new(2, 16, 64);
        let b0 = alloc.allocate();
        let b1 = alloc.allocate();
        let b2 = alloc.allocate();

        assert!(b0.is_some());
        assert!(b1.is_some());
        assert!(b2.is_none()); // pool exhausted
    }

    #[test]
    fn test_block_allocator_ref_counting() {
        let mut alloc = BlockAllocator::new(4, 16, 64);
        let b0 = alloc.allocate().unwrap();
        assert_eq!(alloc.ref_count(b0), 1);

        alloc.add_ref(b0);
        assert_eq!(alloc.ref_count(b0), 2);

        alloc.free(b0);
        assert_eq!(alloc.ref_count(b0), 1);
        assert_eq!(alloc.num_free(), 3); // not freed yet

        alloc.free(b0);
        assert_eq!(alloc.ref_count(b0), 0);
        assert_eq!(alloc.num_free(), 4); // now freed
    }

    #[test]
    fn test_block_table() {
        let mut table = BlockTable::new(16);
        assert_eq!(table.num_blocks_needed(), 0);

        table.append_block(0);
        table.append_block(1);
        table.add_tokens(25);
        assert_eq!(table.num_blocks_needed(), 2);
        assert_eq!(table.num_tokens(), 25);
        assert_eq!(table.block_for_position(0), Some(0));
        assert_eq!(table.block_for_position(15), Some(0));
        assert_eq!(table.block_for_position(16), Some(1));
        assert_eq!(table.block_for_position(31), Some(1));
        assert_eq!(table.slot_in_block(5), 5);
        assert_eq!(table.slot_in_block(16), 0);
    }

    #[test]
    fn test_request_state() {
        let state = RequestState::new(42, 4, 16);
        assert_eq!(state.request_id(), 42);
        assert_eq!(state.num_tokens(), 0);
        assert_eq!(state.num_layers(), 4);
        assert_eq!(state.layer_table(0).num_blocks_needed(), 0);
    }

    #[test]
    fn test_prefix_sharing() {
        let alloc = BlockAllocator::new(10, 16, 64);
        let alloc = Arc::new(std::sync::Mutex::new(alloc));
        let mut mgr = PrefixSharingManager::new(Arc::clone(&alloc), 16);

        let prefix = vec![1u32, 2, 3, 4];
        let blocks = vec![vec![0u32], vec![1u32]]; // 2 layers, 1 block each
        mgr.register_prefix(&prefix, blocks);

        assert!(mgr.has_prefix(&prefix));
        assert!(!mgr.has_prefix(&[5, 6, 7]));
        assert_eq!(mgr.num_prefixes(), 1);

        let shared = mgr.lookup_prefix(&prefix);
        assert!(shared.is_some());
        assert_eq!(shared.unwrap()[0], vec![0]);

        // Ref counts should be incremented (from 0 to 1 by lookup)
        {
            let a = alloc.lock().unwrap();
            assert_eq!(a.ref_count(0), 1); // lookup added 1
            assert_eq!(a.ref_count(1), 1);
        }
    }

    #[test]
    fn test_prefix_remove() {
        let alloc = BlockAllocator::new(10, 16, 64);
        let alloc = Arc::new(std::sync::Mutex::new(alloc));
        let mut mgr = PrefixSharingManager::new(alloc, 16);

        let prefix = vec![1u32, 2, 3];
        mgr.register_prefix(&prefix, vec![vec![0u32]]);
        mgr.remove_prefix(&prefix);
        assert!(!mgr.has_prefix(&prefix));
    }

    #[test]
    fn test_paged_manager_create_and_free() {
        let mut mgr = PagedAttentionManager::new(4, 100, 16, 64);

        // Create request with 32-token prefill → needs 2 blocks per layer
        let req_id = mgr.create_request_with_layers(32, 4);
        assert_eq!(mgr.num_requests(), 1);

        let state = mgr.get_request(req_id).unwrap();
        assert_eq!(state.num_tokens(), 32);
        // 2 blocks per layer × 4 layers = 8 blocks used
        assert_eq!(mgr.num_free_blocks(), 92);

        mgr.free_request(req_id);
        assert_eq!(mgr.num_requests(), 0);
        assert_eq!(mgr.num_free_blocks(), 100);
    }

    #[test]
    fn test_paged_manager_fork() {
        let mut mgr = PagedAttentionManager::new(2, 100, 16, 64);

        let parent_id = mgr.create_request_with_layers(16, 2);
        assert_eq!(mgr.num_free_blocks(), 98); // 2 blocks (1 per layer)

        let child_id = mgr.fork_request(parent_id).unwrap();
        assert_eq!(mgr.num_requests(), 2);
        // Shared blocks: ref_count=2 but not freed, so still 98 free
        assert_eq!(mgr.num_free_blocks(), 98);

        // Free parent: ref_count drops to 1, blocks still held
        mgr.free_request(parent_id);
        assert_eq!(mgr.num_free_blocks(), 98);

        // Free child: ref_count drops to 0, blocks freed
        mgr.free_request(child_id);
        assert_eq!(mgr.num_free_blocks(), 100);
    }

    #[test]
    fn test_utilization() {
        let mut mgr = PagedAttentionManager::new(2, 10, 16, 64);
        assert!((mgr.utilization() - 0.0).abs() < 0.01);

        mgr.create_request_with_layers(16, 2); // 1 block per layer = 2 blocks
        assert!((mgr.utilization() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_memory_used_bytes() {
        let alloc = BlockAllocator::new(10, 16, 64);
        assert_eq!(alloc.memory_used_bytes(), 0);
    }

    #[test]
    fn test_block_table_clear() {
        let mut table = BlockTable::new(16);
        table.append_block(0);
        table.add_tokens(10);
        assert_eq!(table.num_tokens(), 10);

        table.clear();
        assert_eq!(table.num_tokens(), 0);
        assert_eq!(table.num_blocks_allocated(), 0);
    }

    #[test]
    fn test_multiple_requests() {
        let mut mgr = PagedAttentionManager::new(2, 20, 16, 64);

        let r1 = mgr.create_request_with_layers(16, 2); // 2 blocks
        let r2 = mgr.create_request_with_layers(32, 2); // 4 blocks
        let r3 = mgr.create_request_with_layers(8, 2);  // 2 blocks

        assert_eq!(mgr.num_requests(), 3);
        assert_eq!(mgr.num_free_blocks(), 12); // 20 - 8

        mgr.free_request(r2);
        assert_eq!(mgr.num_free_blocks(), 16); // 12 + 4

        mgr.free_request(r1);
        mgr.free_request(r3);
        assert_eq!(mgr.num_free_blocks(), 20);
    }
}
