pub mod pipeline;
pub mod buffer;
pub mod memory;
pub mod kernels;
pub mod cache;

pub use pipeline::WgpuCompute;
pub use buffer::GpuBuffer;
pub use memory::{MemoryBudget, MemoryPool, TiledCompute, ComputeDispatcher, DeviceMemoryPhase, BorrowedBufferPool};
pub use cache::{BlockCache, PipelineStage, PipelineScheduler};
pub use kernels::softmax::SoftmaxOp;
pub use kernels::rmsnorm::RmsNormOp;
pub use kernels::elementwise::ElementWiseOp;
pub use kernels::matmul::MatMulOp;
pub use kernels::moe::{MoEGatingOp, MoEExpertOp};
