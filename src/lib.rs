pub mod error;
pub mod device;
pub mod tensor;
pub mod compute;

pub use error::{FerrisResError, Result};
pub use device::{DeviceProfile, Capability};
pub use tensor::Tensor;
pub use compute::{ComputePipeline, GpuBuffer, WgpuCompute};
