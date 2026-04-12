pub mod kv_cache;
pub mod two_phase;
pub mod sampling;
pub mod generator;
pub mod logit_processors;
pub mod prompt_templates;
pub mod context_extension;
pub mod rag;
pub mod tool_search;
pub mod decs;
pub mod hull_kv_cache;
pub mod llm_computer;
pub mod token_merging;
pub mod matryoshka;
pub mod paca;

pub use kv_cache::{LayerKVCache, ModelKVCache};
#[allow(deprecated)]
pub use two_phase::{TwoPhaseConfig, TwoPhaseInference, AutoregressiveGenerator, KVCache, Sampler, GenerationState};
pub use generator::{TokenGenerator, GenerateConfig};
pub use logit_processors::{LogitProcessor, LogitProcessorConfig, TokenHistory};
pub use prompt_templates::{PromptTemplateRegistry, TemplateFormat, ChatMessage, Role};
pub use context_extension::{ContextExtensionConfig, ContextExtensionEngine, ExtensionMethod, YarnParams, AttentionSinkManager};
pub use rag::{RagStore, RagConfig, Document, RetrievedDocument, RetrievalMethod, InContextLearner, cosine_similarity};
pub use tool_search::{ToolRegistry, ToolSearchConfig, Tool, SelectedTool, ToolCall, ToolResult};
pub use decs::{DecsOptimizer, DecsConfig, ReasoningChain, ReasoningMetrics, StopReason};
pub use hull_kv_cache::{HullKVCache, Point2D, Attention2D};
pub use llm_computer::{LlmComputer, LlmComputerConfig, CalmInstruction, CalmOp, VmState};
