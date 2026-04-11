pub mod error;
pub mod device;
pub mod tensor;
pub mod compute;
pub mod model;
pub mod inference;
pub mod training;
pub mod autodiff;

pub use error::{FerrisResError, Result};
pub use device::{DeviceProfile, Capability};
pub use tensor::{GpuTensor, Tensor};
pub use compute::{GpuBuffer, WgpuCompute, MemoryBudget, MemoryPool, TiledCompute, ComputeDispatcher};
pub use model::{ModelShard, ShardManager, QuantizedBuffer, QuantDtype, TokenEmbedding, LMHead, SimpleTokenizer, BlockAttnResConfig, BlockAttnResModel};
pub use inference::{TwoPhaseConfig, TwoPhaseInference, AutoregressiveGenerator, KVCache, Sampler, GenerationState, LayerKVCache, ModelKVCache, TokenGenerator, PromptTemplateRegistry, TemplateFormat, LogitProcessor, LogitProcessorConfig};
pub use compute::turboquant::{TurboQuantConfig, TurboQuantEngine, TurboQuantError, OutlierChannelSplitter};
pub use training::{CrossEntropyLoss, AdamOptimizer, SgdOptimizer};
