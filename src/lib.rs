pub mod error;
pub mod device;
pub mod tensor;
pub mod compute;
pub mod model;
pub mod inference;

pub use error::{FerrisResError, Result};
pub use device::{DeviceProfile, Capability};
pub use tensor::{GpuTensor, Tensor};
pub use compute::{GpuBuffer, WgpuCompute, MemoryBudget, MemoryPool, TiledCompute, ComputeDispatcher};
pub use model::{ModelShard, ShardManager, QuantizedBuffer, QuantDtype};
pub use inference::{TwoPhaseConfig, TwoPhaseInference, AutoregressiveGenerator};
