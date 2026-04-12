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
pub use compute::{GpuBuffer, WgpuCompute, MemoryBudget, MemoryPool, TiledCompute, ComputeDispatcher, MatMulDoubleBufferOp, FusedPatchEmbedOp};
pub use model::{ModelShard, ShardManager, QuantizedBuffer, QuantDtype, TokenEmbedding, LMHead, SimpleTokenizer, BlockAttnResConfig, BlockAttnResModel, ImagePreprocessor, VisionConfig, VisionEncoder};
pub use model::dispatcher::{AnyModel, AnyModelConfig, ArchitectureHint, DetectedArchitecture};
#[allow(deprecated)]
pub use inference::{TwoPhaseConfig, TwoPhaseInference, AutoregressiveGenerator, KVCache, Sampler, GenerationState, LayerKVCache, ModelKVCache, TokenGenerator, GenerateConfig, PromptTemplateRegistry, TemplateFormat, LogitProcessor, LogitProcessorConfig, ToMeMerger, ToMeConfig, PacaEngine, PacaConfig};
pub use inference::unified_generator::{UnifiedTokenGenerator, UnifiedGenerateConfig};
pub use inference::rag::{ElasticRagStore, EmbedProfile};
pub use inference::sampling::{sample_argmax, sample_temperature, sample_top_k, sample_top_p};
pub use compute::turboquant::{TurboQuantConfig, TurboQuantEngine, TurboQuantError, OutlierChannelSplitter};
pub use training::{CrossEntropyLoss, AdamOptimizer, SgdOptimizer};
pub use training::lora::{LoraConfig, LoraLayer, LoraManager};
