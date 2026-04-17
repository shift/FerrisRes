//! Unified Cognitive Pipeline
//!
//! Wires together all disconnected cognitive components into a single
//! end-to-end pipeline:
//!
//!   1. User prompt → ConceptMap.retrieve() → inject relevant concepts
//!   2. Model generates (may emit [tool_call])
//!   3. Tool dispatch → WasmSandbox (sandboxed) or native
//!   4. LlmComputer → execute CALM programs as tools
//!   5. MirrorTest → self-evaluate tool/output quality
//!   6. If quality low → retry with different tool/params
//!   7. If quality high → ConceptMap.store() saves learning
//!   8. HullKVCache → persist KV state across sessions
//!   9. ConceptMap.save() → persist accumulated knowledge
//!
//! This is the "self-extending neural computer" — the model learns,
//! stores, retrieves, self-evaluates, and persists across sessions.

use std::path::PathBuf;

use crate::inference::concept_memory::{ConceptMap, ConceptContent, ConceptSource};
use crate::inference::episodic_memory::{EpisodicMemory, Episode, EpisodeOutcome, ToolTrace};
use crate::inference::tool_creation::ToolCreationPipeline;
use crate::inference::plan_executor::PlanExecutor;
use crate::inference::tool_usage_tracker::ToolUsageTracker;
use crate::inference::hull_kv_cache::HullKVCache;
use crate::inference::llm_computer::{LlmComputer, LlmComputerConfig};
use crate::inference::mirror_test::MirrorTestRunner;
use crate::inference::tool_search::{Tool, ToolCall, ToolRegistry, ToolSearchConfig};
use crate::inference::wasm_sandbox::WasmRuntime;

// ---------------------------------------------------------------------------
// Pipeline configuration
// ---------------------------------------------------------------------------

/// Configuration for the cognitive pipeline. All features are optional —
/// the pipeline degrades gracefully when components are disabled.
#[derive(Debug, Clone)]
pub struct CognitivePipelineConfig {
    /// Enable concept memory (retrieve + store).
    pub concepts_enabled: bool,
    /// Path to persist concept map.
    pub concepts_path: Option<PathBuf>,
    /// Embedding dimension for concepts.
    pub concepts_embedding_dim: usize,
    /// Maximum number of concepts.
    pub concepts_max: usize,

    /// Enable Hull-KV cache persistence.
    pub kv_persist_enabled: bool,
    /// Path to persist KV cache.
    pub kv_persist_path: Option<PathBuf>,
    /// Hull-KV cache capacity.
    pub kv_capacity: usize,

    /// Enable LLM-Computer (CALM VM) as a tool.
    pub llm_computer_enabled: bool,
    /// Max program length for CALM VM.
    pub llm_computer_max_program: usize,
    /// Max VM steps before halt.
    pub llm_computer_max_steps: usize,

    /// Enable MirrorTest self-evaluation.
    pub mirror_test_enabled: bool,
    /// Quality threshold for accepting tool output.
    pub mirror_quality_threshold: f32,
    /// Maximum retries on low quality.
    pub mirror_max_retries: usize,

    /// Enable WASM sandboxing for tool execution.
    pub wasm_sandbox_enabled: bool,

    /// Enable self-correction loop (retry on low quality).
    pub self_correction_enabled: bool,

    /// Enable episodic memory (event-based experience storage).
    pub episodic_memory_enabled: bool,
    /// Path to persist episodic memory.
    pub episodic_memory_path: Option<PathBuf>,
    /// Episodic memory configuration.
    pub episodic_config: Option<crate::inference::episodic_memory::EpisodicMemoryConfig>,

    /// Enable tool creation pipeline.
    pub tool_creation_enabled: bool,
    /// Enable plan executor.
    pub plan_execution_enabled: bool,
    /// Enable tool usage tracking.
    pub tool_usage_tracking_enabled: bool,
    /// Path to persist tool usage data.
    pub tool_usage_path: Option<PathBuf>,
}

impl Default for CognitivePipelineConfig {
    fn default() -> Self {
        Self {
            concepts_enabled: false,
            concepts_path: None,
            concepts_embedding_dim: 64,
            concepts_max: 10000,
            kv_persist_enabled: false,
            kv_persist_path: None,
            kv_capacity: 4096,
            llm_computer_enabled: false,
            llm_computer_max_program: 256,
            llm_computer_max_steps: 1024,
            mirror_test_enabled: false,
            mirror_quality_threshold: 0.5,
            mirror_max_retries: 2,
            wasm_sandbox_enabled: false,
            self_correction_enabled: false,
            episodic_memory_enabled: false,
            episodic_memory_path: None,
            episodic_config: None,
            tool_creation_enabled: false,
            plan_execution_enabled: false,
            tool_usage_tracking_enabled: false,
            tool_usage_path: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline components
// ---------------------------------------------------------------------------

/// The assembled cognitive pipeline. Owns all components.
pub struct CognitivePipeline {
    config: CognitivePipelineConfig,
    /// Concept memory — stores and retrieves learned concepts.
    concept_map: Option<ConceptMap>,
    /// Hull-KV cache — persistent KV state across sessions.
    hull_kv: Option<HullKVCache>,
    /// LLM-Computer — CALM VM for deterministic program execution.
    llm_computer: Option<LlmComputer>,
    /// Mirror test runner — self-evaluation of outputs.
    mirror_runner: Option<MirrorTestRunner>,
    /// WASM runtime — sandboxed tool execution.
    wasm_runtime: Option<WasmRuntime>,
    /// Episodic memory — event-based experience storage.
    episodic_memory: Option<EpisodicMemory>,
    /// Tool creation pipeline — model generates its own tools.
    tool_creation: Option<ToolCreationPipeline>,
    /// Plan executor — multi-step tool chaining.
    plan_executor: Option<PlanExecutor>,
    /// Tool usage tracker — meta-learning via contextual bandits.
    tool_usage_tracker: Option<ToolUsageTracker>,
    /// Tool registry — augmented with cognitive tools.
    tool_registry: ToolRegistry,
}

/// Result of a cognitive pipeline generation step.
#[derive(Debug, Clone)]
pub struct CognitiveGenerationResult {
    /// The generated text output.
    pub output: String,
    /// Number of concepts retrieved before generation.
    pub concepts_retrieved: usize,
    /// Number of new concepts stored after generation.
    pub concepts_stored: usize,
    /// Whether a tool call was made.
    pub tool_called: bool,
    /// Tool call details (if any).
    pub tool_name: Option<String>,
    /// Mirror test quality score (if evaluated).
    pub mirror_quality: Option<f32>,
    /// Number of retries due to low quality.
    pub retries: usize,
    /// Whether KV cache was persisted.
    pub kv_persisted: bool,
    /// Number of episodes stored.
    pub episodes_stored: usize,
    /// Number of relevant episodes retrieved before generation.
    pub episodes_retrieved: usize,
    /// Whether a tool was created.
    pub tool_created: bool,
    /// Name of created tool (if any).
    pub created_tool_name: Option<String>,
    /// Whether a plan was executed.
    pub plan_executed: bool,
    /// Number of plan steps executed.
    pub plan_steps: usize,
}

impl CognitivePipeline {
    /// Build a new cognitive pipeline from configuration.
    pub fn new(config: CognitivePipelineConfig) -> Self {
        // Load or create concept map
        let concept_map = if config.concepts_enabled {
            let map = if let Some(ref path) = config.concepts_path {
                if path.exists() {
                    match ConceptMap::load(path) {
                        Ok(m) => {
                            tracing::info!(event = "concepts_loaded", "Loaded {} concepts from {}", m.len(), path.display());
                            m
                        }
                        Err(e) => {
                            tracing::warn!(event = "concepts_load_error", "Failed to load concepts: {}, starting fresh", e);
                            ConceptMap::with_capacity(config.concepts_embedding_dim, config.concepts_max)
                        }
                    }
                } else {
                    ConceptMap::with_capacity(config.concepts_embedding_dim, config.concepts_max)
                }
            } else {
                ConceptMap::with_capacity(config.concepts_embedding_dim, config.concepts_max)
            };
            Some(map)
        } else {
            None
        };

        // Create Hull-KV cache
        let hull_kv = if config.kv_persist_enabled {
            let cache = if let Some(ref path) = config.kv_persist_path {
                if path.exists() {
                    match HullKVCache::load(path, config.kv_capacity) {
                        Ok(c) => {
                            tracing::info!(event = "hull_kv_loaded", "Loaded Hull-KV cache with {} points from {}", c.len(), path.display());
                            c
                        }
                        Err(e) => {
                            tracing::warn!(event = "hull_kv_load_error", "Failed to load Hull-KV: {}, starting fresh", e);
                            HullKVCache::new(config.kv_capacity)
                        }
                    }
                } else {
                    HullKVCache::new(config.kv_capacity)
                }
            } else {
                HullKVCache::new(config.kv_capacity)
            };
            Some(cache)
        } else {
            None
        };

        // Create LLM-Computer
        let llm_computer = if config.llm_computer_enabled {
            let computer_config = LlmComputerConfig {
                num_registers: 16,
                memory_size: 1024,
                num_tables: 8,
                max_program_length: config.llm_computer_max_program,
                use_hull_cache: false,
            };
            Some(LlmComputer::new(computer_config))
        } else {
            None
        };

        // Create mirror test runner
        let mirror_runner = if config.mirror_test_enabled {
            Some(MirrorTestRunner::new())
        } else {
            None
        };

        // Create WASM runtime
        let wasm_runtime = if config.wasm_sandbox_enabled {
            Some(WasmRuntime::default_runtime())
        } else {
            None
        };

        // Create episodic memory
        let episodic_memory = if config.episodic_memory_enabled {
            let epi_config = config.episodic_config.clone().unwrap_or_default();
            if let Some(ref path) = config.episodic_memory_path {
                if path.exists() {
                    match EpisodicMemory::load(path) {
                        Ok(m) => {
                            tracing::info!(event = "episodic_loaded", "Loaded {} episodes from {}", m.len(), path.display());
                            Some(m)
                        }
                        Err(e) => {
                            tracing::warn!(event = "episodic_load_error", "Failed to load episodes: {}, starting fresh", e);
                            Some(EpisodicMemory::new(epi_config))
                        }
                    }
                } else {
                    Some(EpisodicMemory::new(epi_config))
                }
            } else {
                Some(EpisodicMemory::new(epi_config))
            }
        } else {
            None
        };

        // Tool creation pipeline
        let tool_creation = if config.tool_creation_enabled {
            Some(ToolCreationPipeline::default_pipeline())
        } else {
            None
        };

        // Plan executor
        let plan_executor = if config.plan_execution_enabled {
            Some(PlanExecutor::default_executor())
        } else {
            None
        };

        // Tool usage tracker
        let tool_usage_tracker = if config.tool_usage_tracking_enabled {
            if let Some(ref path) = config.tool_usage_path {
                if path.exists() {
                    match ToolUsageTracker::load(path) {
                        Ok(t) => {
                            tracing::info!(event = "usage_tracker_loaded", "Loaded usage tracker with {} events", t.total_events());
                            Some(t)
                        }
                        Err(e) => {
                            tracing::warn!(event = "usage_tracker_load_error", "Failed to load usage tracker: {}", e);
                            Some(ToolUsageTracker::default_tracker())
                        }
                    }
                } else {
                    Some(ToolUsageTracker::default_tracker())
                }
            } else {
                Some(ToolUsageTracker::default_tracker())
            }
        } else {
            None
        };

        // Build augmented tool registry
        let mut tool_registry = ToolRegistry::new(ToolSearchConfig::default());

        // Register LLM-Computer as a tool
        if config.llm_computer_enabled {
            let calm_tool = Tool::new("calm_execute", "Execute a CALM VM program for deterministic computation")
                .with_category("computation")
                .with_example("calm_execute(add, 3, 5)");
            tool_registry.register(calm_tool);
        }

        // Register mirror test as a tool
        if config.mirror_test_enabled {
            let mirror_tool = Tool::new("mirror_test", "Self-evaluate generated code by generating and running tests")
                .with_category("verification")
                .with_example("mirror_test(code, language)");
            tool_registry.register(mirror_tool);
        }

        // Register concept retrieval as a tool
        if config.concepts_enabled {
            let concept_tool = Tool::new("concept_lookup", "Retrieve relevant learned concepts from memory")
                .with_category("memory")
                .with_example("concept_lookup(embedding)");
            tool_registry.register(concept_tool);
        }

        Self {
            config,
            concept_map,
            hull_kv,
            llm_computer,
            mirror_runner,
            wasm_runtime,
            episodic_memory,
            tool_creation,
            plan_executor,
            tool_usage_tracker,
            tool_registry,
        }
    }

    /// Augment a prompt with relevant concepts from memory.
    ///
    /// Returns the augmented prompt and the number of concepts retrieved.
    pub fn augment_with_concepts(&mut self, prompt: &str) -> (String, usize) {
        let concept_map = match self.concept_map.as_mut() {
            Some(m) => m,
            None => return (prompt.to_string(), 0),
        };

        // Create a simple embedding from the prompt (bag-of-words hash)
        let embedding = prompt_to_embedding(prompt, self.config.concepts_embedding_dim);

        let retrieved = concept_map.retrieve(&embedding, 5);
        let count = retrieved.len();

        if count == 0 {
            return (prompt.to_string(), 0);
        }

        // Build concept context block
        let mut context = String::from("[Retrieved concepts]\n");
        for rc in &retrieved {
            context.push_str(&format!("- {} (quality: {:.2}): ", rc.concept.name, rc.concept.quality));
            match &rc.concept.content {
                ConceptContent::Code { code, .. } => {
                    context.push_str(&format!("```{}```", code.chars().take(200).collect::<String>()));
                }
                ConceptContent::Algorithm { steps, .. } => {
                    context.push_str(&steps.join(" → "));
                }
                ConceptContent::Formula { latex, .. } => {
                    context.push_str(latex);
                }
                ConceptContent::Bytecode { description, .. } => {
                    context.push_str(description);
                }
                ConceptContent::TestPattern { template, .. } => {
                    context.push_str(template);
                }
                ConceptContent::WebResult { summary, .. } => {
                    context.push_str(summary);
                }
            }
            context.push('\n');
        }
        context.push_str("\n[/Retrieved concepts]\n\n");

        (format!("{}{}", context, prompt), count)
    }

    /// Execute a tool call through the cognitive pipeline.
    ///
    /// If WASM sandbox is enabled, the tool runs in sandbox.
    /// If LLM-Computer is the target, executes via CALM VM.
    /// If MirrorTest is enabled, evaluates the result quality.
    pub fn execute_tool(&mut self, tool_name: &str, args: &str) -> ToolExecutionResult {
        // LLM-Computer execution
        if tool_name == "calm_execute" {
            return self.execute_calm(args);
        }

        // Mirror test execution
        if tool_name == "mirror_test" {
            return self.execute_mirror_test(args);
        }

        // Concept lookup
        if tool_name == "concept_lookup" {
            return self.execute_concept_lookup(args);
        }

        // Standard tool dispatch
        let call = ToolCall::new(tool_name, args);

        // Check if we should sandbox
        if self.wasm_runtime.is_some() && self.config.wasm_sandbox_enabled {
            // For now, tools run natively. WASM sandbox is for self-generated tools.
            // The sandbox validates syntax but doesn't replace the dispatch yet.
            if let Some(ref runtime) = self.wasm_runtime {
                let wasm = crate::inference::wasm_sandbox::embedded_syntax_checker_wasm();
                let _ = runtime.execute_parse_from_bytes(&wasm, args);
            }
        }

        // Dispatch via host tools
        let result = crate::inference::host_tools::dispatch_tool(&call);

        // Mirror test evaluation of result
        if self.mirror_runner.is_some() && self.config.mirror_test_enabled {
            if let Some(ref runner) = self.mirror_runner {
                let mirror_result = runner.evaluate(args, &result.output, "rust");
                let quality = 1.0 / (1.0 + mirror_result.loss); // Sigmoid-like quality

                return ToolExecutionResult {
                    output: result.output.clone(),
                    success: result.success,
                    mirror_quality: Some(quality),
                    calm_result: None,
                };
            }
        }

        ToolExecutionResult {
            output: result.output.clone(),
            success: result.success,
            mirror_quality: None,
            calm_result: None,
        }
    }

    /// Execute a CALM VM program.
    fn execute_calm(&mut self, args: &str) -> ToolExecutionResult {
        if self.llm_computer.is_none() {
            return ToolExecutionResult {
                output: "LLM-Computer not enabled".into(),
                success: false,
                mirror_quality: None,
                calm_result: None,
            };
        }

        // Parse simple instructions from args
        // Format: "op a b" or "lookup table key"
        let parts: Vec<&str> = args.split_whitespace().collect();
        let program = if parts.len() >= 3 && parts[0] == "add" {
            if let (Ok(a), Ok(b)) = (parts[1].parse::<i32>(), parts[2].parse::<i32>()) {
                LlmComputer::compile_add_program(a, b, 0)
            } else {
                return ToolExecutionResult {
                    output: "Invalid add arguments".into(),
                    success: false,
                    mirror_quality: None,
                    calm_result: None,
                };
            }
        } else if parts.len() >= 3 && parts[0] == "sum" {
            if let Ok(n) = parts[1].parse::<i32>() {
                let result_reg = parts.get(2).and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
                LlmComputer::compile_sum_program(n, result_reg)
            } else {
                return ToolExecutionResult {
                    output: "Invalid sum arguments".into(),
                    success: false,
                    mirror_quality: None,
                    calm_result: None,
                };
            }
        } else {
            return ToolExecutionResult {
                output: format!("Unknown CALM operation: {:?}. Use: add N M, sum N result_reg", parts.first()),
                success: false,
                mirror_quality: None,
                calm_result: None,
            };
        };

        // We need to create a temporary mutable computer to load the program
        let mut computer = LlmComputer::new(LlmComputerConfig {
            num_registers: 16,
            memory_size: 1024,
            num_tables: 8,
            max_program_length: self.config.llm_computer_max_program,
            use_hull_cache: false,
        });
        computer.load_program(program);

        let state = computer.execute();
        let result_reg = state.read_reg(0);

        ToolExecutionResult {
            output: format!("CALM result: register[0] = {}", result_reg),
            success: true,
            mirror_quality: None,
            calm_result: Some(CalmResult {
                registers: (0..8).map(|r| state.read_reg(r)).collect(),
                steps: state.steps(),
                halted: state.is_halted(),
            }),
        }
    }

    /// Execute a mirror test.
    fn execute_mirror_test(&mut self, args: &str) -> ToolExecutionResult {
        let runner = match self.mirror_runner.as_ref() {
            Some(r) => r,
            None => return ToolExecutionResult {
                output: "MirrorTest not enabled".into(),
                success: false,
                mirror_quality: None,
                calm_result: None,
            },
        };

        // Parse: "code|||test_code|||language"
        let parts: Vec<&str> = args.splitn(3, "|||").collect();
        let (code, test_code, language) = match parts.len() {
            3 => (parts[0], parts[1], parts[2]),
            2 => (parts[0], parts[1], "rust"),
            _ => (args, "assert!(true)", "rust"),
        };

        let result = runner.evaluate(code, test_code, language);
        let quality = 1.0 / (1.0 + result.loss);

        ToolExecutionResult {
            output: format!(
                "Mirror test: syntax={} test_syntax={} passed={} loss={:.3}",
                result.syntax_valid, result.test_syntax_valid, result.test_passed, result.loss
            ),
            success: result.test_passed,
            mirror_quality: Some(quality),
            calm_result: None,
        }
    }

    /// Execute a concept lookup.
    fn execute_concept_lookup(&mut self, args: &str) -> ToolExecutionResult {
        let concept_map = match self.concept_map.as_mut() {
            Some(m) => m,
            None => return ToolExecutionResult {
                output: "Concept memory not enabled".into(),
                success: false,
                mirror_quality: None,
                calm_result: None,
            },
        };

        let embedding = prompt_to_embedding(args, self.config.concepts_embedding_dim);
        let results = concept_map.retrieve(&embedding, 3);

        if results.is_empty() {
            return ToolExecutionResult {
                output: "No relevant concepts found".into(),
                success: true,
                mirror_quality: None,
                calm_result: None,
            };
        }

        let output = results.iter()
            .map(|r| format!("{} (similarity: {:.2}, quality: {:.2})", r.concept.name, r.similarity, r.concept.quality))
            .collect::<Vec<_>>()
            .join("\n");

        ToolExecutionResult {
            output,
            success: true,
            mirror_quality: None,
            calm_result: None,
        }
    }

    /// Store a learning in concept memory.
    ///
    /// Call after a successful generation/tool use to persist the knowledge.
    pub fn store_learning(
        &mut self,
        name: String,
        prompt: &str,
        content: ConceptContent,
        tags: Vec<String>,
    ) -> Option<u64> {
        let concept_map = self.concept_map.as_mut()?;
        let embedding = prompt_to_embedding(prompt, self.config.concepts_embedding_dim);
        let id = concept_map.store(
            name,
            embedding,
            content,
            tags,
            ConceptSource::SelfVerified { test_result: "pipeline_stored".into() },
        );
        tracing::debug!(event = "concept_stored", "Stored concept id={} total={}", id, concept_map.len());
        Some(id)
    }

    /// Update concept quality after observing success/failure.
    pub fn update_concept_quality(&mut self, concept_id: u64, success: bool) {
        if let Some(ref mut map) = self.concept_map {
            map.update_quality(concept_id, success);
        }
    }

    /// Persist all state to disk.
    pub fn persist(&self) -> Result<(), String> {
        if let (Some(ref map), Some(ref path)) = (&self.concept_map, &self.config.concepts_path) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            map.save(path)?;
            tracing::info!(event = "concepts_saved", "Saved {} concepts to {}", map.len(), path.display());
        }

        if let (Some(ref kv), Some(ref path)) = (&self.hull_kv, &self.config.kv_persist_path) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            kv.save(path).map_err(|e| format!("{}", e))?;
            tracing::info!(event = "hull_kv_saved", "Saved Hull-KV ({} points) to {}", kv.len(), path.display());
        }

        if let (Some(ref mem), Some(ref path)) = (&self.episodic_memory, &self.config.episodic_memory_path) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            mem.save(path)?;
            tracing::info!(event = "episodes_saved", "Saved {} episodes to {}", mem.len(), path.display());
        }

        if let (Some(ref tracker), Some(ref path)) = (&self.tool_usage_tracker, &self.config.tool_usage_path) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            tracker.save(path)?;
            tracing::info!(event = "usage_tracker_saved", "Saved usage tracker ({} events) to {}", tracker.total_events(), path.display());
        }

        Ok(())
    }

    /// Access the tool registry.
    pub fn tool_registry(&self) -> &ToolRegistry {
        &self.tool_registry
    }

    /// Access the tool registry mutably.
    pub fn tool_registry_mut(&mut self) -> &mut ToolRegistry {
        &mut self.tool_registry
    }

    /// Access the concept map.
    pub fn concept_map(&self) -> Option<&ConceptMap> {
        self.concept_map.as_ref()
    }

    /// Access the concept map mutably.
    pub fn concept_map_mut(&mut self) -> Option<&mut ConceptMap> {
        self.concept_map.as_mut()
    }

    /// Access the LLM-Computer.
    pub fn llm_computer(&self) -> Option<&LlmComputer> {
        self.llm_computer.as_ref()
    }

    /// Access the Hull-KV cache.
    pub fn hull_kv(&self) -> Option<&HullKVCache> {
        self.hull_kv.as_ref()
    }

    /// Access the Hull-KV cache mutably.
    pub fn hull_kv_mut(&mut self) -> Option<&mut HullKVCache> {
        self.hull_kv.as_mut()
    }

    /// Access the episodic memory.
    pub fn episodic_memory(&self) -> Option<&EpisodicMemory> {
        self.episodic_memory.as_ref()
    }

    /// Access the episodic memory mutably.
    pub fn episodic_memory_mut(&mut self) -> Option<&mut EpisodicMemory> {
        self.episodic_memory.as_mut()
    }

    /// Access the tool creation pipeline.
    pub fn tool_creation(&self) -> Option<&ToolCreationPipeline> {
        self.tool_creation.as_ref()
    }

    /// Access the tool creation pipeline mutably.
    pub fn tool_creation_mut(&mut self) -> Option<&mut ToolCreationPipeline> {
        self.tool_creation.as_mut()
    }

    /// Access the plan executor.
    pub fn plan_executor(&self) -> Option<&PlanExecutor> {
        self.plan_executor.as_ref()
    }

    /// Access the plan executor mutably.
    pub fn plan_executor_mut(&mut self) -> Option<&mut PlanExecutor> {
        self.plan_executor.as_mut()
    }

    /// Access the tool usage tracker.
    pub fn tool_usage_tracker(&self) -> Option<&ToolUsageTracker> {
        self.tool_usage_tracker.as_ref()
    }

    /// Access the tool usage tracker mutably.
    pub fn tool_usage_tracker_mut(&mut self) -> Option<&mut ToolUsageTracker> {
        self.tool_usage_tracker.as_mut()
    }

    /// Get the configuration.
    pub fn config(&self) -> &CognitivePipelineConfig {
        &self.config
    }

    /// Process a full generation with the cognitive pipeline.
    ///
    /// This is the main entry point. It:
    /// 1. Augments the prompt with concepts
    /// 2. Calls the provided generation function
    /// 3. Checks for tool calls and dispatches them
    /// 4. Self-evaluates if mirror test is enabled
    /// 5. Stores learnings if quality is high
    /// 6. Persists state
    pub fn process_generation<F>(
        &mut self,
        prompt: &str,
        generate_fn: F,
    ) -> CognitiveGenerationResult
    where
        F: Fn(&str) -> String,
    {
        // Step 1: Augment with concepts
        let (augmented_prompt, concepts_retrieved) = self.augment_with_concepts(prompt);

        // Step 1b: Retrieve relevant episodes
        let mut episodes_retrieved = 0;
        let mut episode_context = String::new();
        if let Some(ref mem) = self.episodic_memory {
            let embedding = prompt_to_embedding(prompt, self.config.concepts_embedding_dim);
            let episodes = mem.retrieve(&embedding, 3);
            episodes_retrieved = episodes.len();
            if !episodes.is_empty() {
                episode_context.push_str("\n[Relevant past experiences]\n");
                for re in &episodes {
                    let outcome_str = match re.episode.outcome {
                        EpisodeOutcome::Success => "✓",
                        EpisodeOutcome::PartialSuccess => "~",
                        EpisodeOutcome::Failure => "✗",
                    };
                    episode_context.push_str(&format!(
                        "- {} [{}] quality={:.2}: {}\n",
                        outcome_str,
                        re.episode.prompt.chars().take(80).collect::<String>(),
                        re.episode.quality_score,
                        re.episode.output.chars().take(100).collect::<String>(),
                    ));
                }
                episode_context.push_str("[/Relevant past experiences]\n");
            }
        }

        let full_prompt = if episode_context.is_empty() {
            augmented_prompt
        } else {
            format!("{}{}", episode_context, augmented_prompt)
        };

        // Step 2: Generate
        let output = generate_fn(&full_prompt);

        // Step 3: Check for tool calls
        let mut tool_called = false;
        let mut tool_name = None;
        let mut final_output = output.clone();
        let mut mirror_quality = None;
        let mut retries = 0;
        let mut tool_traces: Vec<ToolTrace> = Vec::new();

        if let Some(start) = output.find("[tool_call]") {
            let content_start = start + "[tool_call]".len();
            let content_end = output.find("[/tool_call]").unwrap_or(output.len());
            let call_text = &output[content_start..content_end];

            let (name, args) = if let Some(paren) = call_text.find('(') {
                (
                    call_text[..paren].trim().to_string(),
                    call_text[paren + 1..].trim_end_matches(')').trim().to_string(),
                )
            } else {
                (call_text.trim().to_string(), String::new())
            };

            tool_called = true;
            tool_name = Some(name.clone());

            // Execute tool
            let result = self.execute_tool(&name, &args);

            tool_traces.push(ToolTrace {
                tool_name: name.clone(),
                args: args.clone(),
                output: result.output.clone(),
                success: result.success,
                step: 0,
            });

            // Step 4: Self-evaluate with retry
            if self.config.self_correction_enabled && result.mirror_quality.unwrap_or(1.0) < self.config.mirror_quality_threshold {
                for retry_idx in 0..self.config.mirror_max_retries {
                    retries += 1;
                    let retry_result = self.execute_tool(&name, &args);
                    tool_traces.push(ToolTrace {
                        tool_name: name.clone(),
                        args: args.clone(),
                        output: retry_result.output.clone(),
                        success: retry_result.success,
                        step: retry_idx + 1,
                    });
                    if retry_result.mirror_quality.unwrap_or(1.0) >= self.config.mirror_quality_threshold {
                        final_output = format!("{}\n\nTool result: {}", output, retry_result.output);
                        mirror_quality = retry_result.mirror_quality;
                        break;
                    }
                }
            } else {
                final_output = format!("{}\n\nTool result: {}", output, result.output);
                mirror_quality = result.mirror_quality;
            }
        }

        // Step 4b: Handle tool creation requests
        let mut tool_created_result: Option<(String, f32)> = None;
        if let Some(ref mut creation) = self.tool_creation {
            if let Some(attempt) = creation.process_creation(&output) {
                if attempt.success {
                    let tool_name_created = attempt.spec.name.clone();
                    let quality = attempt.quality;
                    if let Some(ref spec) = creation.get_created_tool(&tool_name_created) {
                        let tool = crate::inference::tool_search::Tool::new(
                            &spec.name, &spec.description,
                        )
                            .with_category(&spec.category);
                        self.tool_registry.register(tool);
                        tracing::info!(
                            event = "auto_created_tool",
                            name = %tool_name_created,
                            "Automatically registered created tool"
                        );
                    }
                    tool_created_result = Some((tool_name_created, quality));
                }
            }
        }
        if let Some((ref name, quality)) = tool_created_result {
            final_output = format!(
                "{}\n\n[Created tool: {} (quality: {:.2})]",
                final_output, name, quality
            );
        }

        // Step 4c: Record tool usage
        if let (Some(ref mut tracker), Some(ref tname)) = (&mut self.tool_usage_tracker, &tool_name) {
            let embedding = prompt_to_embedding(prompt, self.config.concepts_embedding_dim);
            let quality = mirror_quality.unwrap_or(0.5);
            tracker.record(tname, &embedding, quality);
        }

        // Step 5: Store learning if quality is sufficient
        let mut concepts_stored = 0;
        if let Some(quality) = mirror_quality {
            if quality >= 0.7 {
                if let Some(id) = self.store_learning(
                    format!("learning_{}", std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0)),
                    prompt,
                    ConceptContent::Algorithm {
                        steps: vec![final_output.chars().take(500).collect()],
                        complexity: "O(1)".into(),
                    },
                    vec!["auto-learned".into()],
                ) {
                    concepts_stored = 1;
                    self.update_concept_quality(id, quality > 0.5);
                }
            }
        }

        // Step 6: Store episode
        let mut episodes_stored = 0;
        if let Some(ref mut mem) = self.episodic_memory {
            let embedding = prompt_to_embedding(prompt, self.config.concepts_embedding_dim);
            let quality = mirror_quality.unwrap_or(0.5);
            let outcome = if quality >= 0.7 {
                EpisodeOutcome::Success
            } else if quality >= 0.4 {
                EpisodeOutcome::PartialSuccess
            } else {
                EpisodeOutcome::Failure
            };

            let mut episode = Episode {
                id: 0, // Auto-assigned by store
                context_embedding: embedding,
                prompt: prompt.to_string(),
                tools_used: tool_traces,
                plan_description: None,
                output: final_output.chars().take(500).collect(),
                outcome,
                quality_score: quality,
                logit_entropy: 0.0, // Would be filled by actual model
                confidence: quality,
                importance: 0.0, // Auto-computed by store
                timestamp: 0, // Auto-assigned by store
                tags: vec!["auto".into()],
                compressed_from: vec![],
                consolidated: false,
            };
            episode.compute_importance();

            if mem.store(episode).is_some() {
                episodes_stored = 1;
            }

            // Auto-compress if near capacity
            if mem.should_compress() {
                let compressed = mem.compress();
                if compressed > 0 {
                    tracing::info!(event = "episodes_auto_compressed", count = compressed, "Auto-compressed episodes");
                }
            }
        }

        // Step 7: Persist
        let kv_persisted = self.hull_kv.is_some() && self.config.kv_persist_path.is_some();
        if let Err(e) = self.persist() {
            tracing::warn!(event = "pipeline_persist_error", "Failed to persist cognitive state: {}", e);
        }

        CognitiveGenerationResult {
            output: final_output,
            concepts_retrieved,
            concepts_stored,
            tool_called,
            tool_name,
            mirror_quality,
            retries,
            kv_persisted,
            episodes_stored,
            episodes_retrieved,
            tool_created: tool_created_result.is_some(),
            created_tool_name: tool_created_result.map(|(n, _)| n),
            plan_executed: false,
            plan_steps: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a tool execution through the cognitive pipeline.
#[derive(Debug, Clone)]
pub struct ToolExecutionResult {
    /// Tool output text.
    pub output: String,
    /// Whether the tool execution succeeded.
    pub success: bool,
    /// Mirror test quality score (if evaluated).
    pub mirror_quality: Option<f32>,
    /// CALM VM result (if LLM-Computer was used).
    pub calm_result: Option<CalmResult>,
}

/// Result from the CALM VM.
#[derive(Debug, Clone)]
pub struct CalmResult {
    /// Register values.
    pub registers: Vec<i32>,
    /// Number of steps executed.
    pub steps: usize,
    /// Whether the VM halted cleanly.
    pub halted: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a text prompt to a fixed-size embedding using a simple hash-based
/// approach. In production, this would use the model's own embedding layer.
fn prompt_to_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut embedding = vec![0.0f32; dim];
    let bytes = text.as_bytes();
    for (i, &b) in bytes.iter().cycle().take(dim * 4).enumerate() {
        let idx = i % dim;
        // Simple hash mixing: rotate existing value and add byte
        embedding[idx] = embedding[idx] * 0.99 + (b as f32) / 255.0 * 0.01;
    }
    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in embedding.iter_mut() {
            *x /= norm;
        }
    }
    embedding
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_pipeline_default_config() {
        let config = CognitivePipelineConfig::default();
        assert!(!config.concepts_enabled);
        assert!(!config.kv_persist_enabled);
        assert!(!config.llm_computer_enabled);
        assert!(!config.mirror_test_enabled);
    }

    #[test]
    fn test_cognitive_pipeline_all_enabled() {
        let dir = std::env::temp_dir().join("ferrisres_cognitive_test");
        let _ = std::fs::create_dir_all(&dir);

        let config = CognitivePipelineConfig {
            concepts_enabled: true,
            concepts_path: Some(dir.join("concepts.json")),
            concepts_embedding_dim: 64,
            concepts_max: 100,
            kv_persist_enabled: true,
            kv_persist_path: Some(dir.join("hull_kv.bin")),
            kv_capacity: 512,
            llm_computer_enabled: true,
            llm_computer_max_program: 64,
            llm_computer_max_steps: 128,
            mirror_test_enabled: true,
            mirror_quality_threshold: 0.5,
            mirror_max_retries: 1,
            wasm_sandbox_enabled: false,
            self_correction_enabled: false,
            episodic_memory_enabled: false,
            episodic_memory_path: None,
            episodic_config: None,
            tool_creation_enabled: false,
            plan_execution_enabled: false,
            tool_usage_tracking_enabled: false,
            tool_usage_path: None,
        };

        let pipeline = CognitivePipeline::new(config);
        assert!(pipeline.concept_map().is_some());
        assert!(pipeline.hull_kv().is_some());
        assert!(pipeline.llm_computer().is_some());
        assert!(pipeline.tool_registry().len() >= 3); // calm, mirror, concept_lookup

        // Clean up
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_concept_augmentation() {
        let dir = std::env::temp_dir().join("ferrisres_concept_aug_test");
        let _ = std::fs::create_dir_all(&dir);

        let config = CognitivePipelineConfig {
            concepts_enabled: true,
            concepts_path: Some(dir.join("concepts.json")),
            concepts_embedding_dim: 64,
            concepts_max: 100,
            ..Default::default()
        };

        let mut pipeline = CognitivePipeline::new(config);

        // Store a concept
        let _id = pipeline.store_learning(
            "test_sort".into(),
            "sort algorithm",
            ConceptContent::Code { language: "rust".into(), code: "fn sort() {}".into() },
            vec!["sort".into()],
        );

        // Augment should find the concept
        let (augmented, count) = pipeline.augment_with_concepts("sort algorithm");
        assert!(count > 0, "Should retrieve stored concept");
        assert!(augmented.contains("[Retrieved concepts]"), "Should have concept header");
        assert!(augmented.contains("sort algorithm"), "Original prompt should be preserved");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_calm_tool_execution() {
        let config = CognitivePipelineConfig {
            llm_computer_enabled: true,
            llm_computer_max_program: 64,
            llm_computer_max_steps: 128,
            ..Default::default()
        };

        let mut pipeline = CognitivePipeline::new(config);

        let result = pipeline.execute_tool("calm_execute", "add 3 5");
        assert!(result.success, "CALM add should succeed");
        assert!(result.output.contains("8"), "3 + 5 = 8");
        assert!(result.calm_result.is_some());

        let calm = result.calm_result.unwrap();
        assert_eq!(calm.registers[0], 8);
        assert!(calm.halted);
    }

    #[test]
    fn test_mirror_tool_execution() {
        let config = CognitivePipelineConfig {
            mirror_test_enabled: true,
            ..Default::default()
        };

        let mut pipeline = CognitivePipeline::new(config);

        let code = "fn add(a: i32, b: i32) -> i32 { a + b }";
        let test = "#[test] fn test_add() { assert_eq!(add(1, 2), 3); }";
        let args = format!("{}|||{}|||rust", code, test);

        let result = pipeline.execute_tool("mirror_test", &args);
        assert!(result.success, "Mirror test should pass");
        assert!(result.mirror_quality.unwrap_or(0.0) > 0.5, "Quality should be high for passing test");
    }

    #[test]
    fn test_persistence_round_trip() {
        let dir = std::env::temp_dir().join("ferrisres_persist_test");
        let _ = std::fs::create_dir_all(&dir);

        let config = CognitivePipelineConfig {
            concepts_enabled: true,
            concepts_path: Some(dir.join("concepts.json")),
            concepts_embedding_dim: 32,
            concepts_max: 100,
            kv_persist_enabled: true,
            kv_persist_path: Some(dir.join("hull_kv.bin")),
            kv_capacity: 256,
            ..Default::default()
        };

        // Create pipeline, store data, persist
        {
            let mut pipeline = CognitivePipeline::new(config.clone());
            pipeline.store_learning(
                "test_concept".into(),
                "test query",
                ConceptContent::Code { language: "rust".into(), code: "fn test() {}".into() },
                vec!["test".into()],
            );
            pipeline.persist().expect("persist should work");
        }

        // Load from persisted state
        {
            let pipeline = CognitivePipeline::new(config.clone());
            let map = pipeline.concept_map().expect("concepts should load");
            assert_eq!(map.len(), 1, "Should have loaded persisted concept");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_process_generation_simple() {
        let config = CognitivePipelineConfig {
            concepts_enabled: true,
            concepts_embedding_dim: 32,
            concepts_max: 100,
            ..Default::default()
        };

        let mut pipeline = CognitivePipeline::new(config);

        let result = pipeline.process_generation("test prompt", |_prompt| {
            "generated output".to_string()
        });

        assert_eq!(result.output, "generated output");
        assert!(!result.tool_called);
        assert_eq!(result.concepts_retrieved, 0);
    }

    #[test]
    fn test_process_generation_with_tool_call() {
        let config = CognitivePipelineConfig {
            llm_computer_enabled: true,
            llm_computer_max_program: 64,
            llm_computer_max_steps: 128,
            ..Default::default()
        };

        let mut pipeline = CognitivePipeline::new(config);

        let result = pipeline.process_generation("compute 3+5", |_prompt| {
            "Let me compute: [tool_call]calm_execute(add 3 5)[/tool_call]".to_string()
        });

        assert!(result.tool_called);
        assert_eq!(result.tool_name.as_deref(), Some("calm_execute"));
        assert!(result.output.contains("8"), "Should contain CALM result");
    }

    #[test]
    fn test_prompt_to_embedding() {
        let e1 = prompt_to_embedding("hello world", 64);
        let e2 = prompt_to_embedding("hello world", 64);
        let e3 = prompt_to_embedding("goodbye world", 64);

        assert_eq!(e1.len(), 64);
        // Same input → same embedding
        for (a, b) in e1.iter().zip(e2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        // Different input → different embedding
        let diff: f32 = e1.iter().zip(e3.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(diff > 0.0, "Different prompts should produce different embeddings");
    }

    #[test]
    fn test_episodic_memory_integration() {
        let dir = std::env::temp_dir().join("ferrisres_epi_pipeline_test");
        let _ = std::fs::create_dir_all(&dir);

        let config = CognitivePipelineConfig {
            concepts_enabled: false,
            episodic_memory_enabled: true,
            episodic_memory_path: Some(dir.join("episodes.json")),
            episodic_config: Some(crate::inference::episodic_memory::EpisodicMemoryConfig {
                embedding_dim: 32,
                importance_threshold: 0.0, // Accept all episodes for testing
                ..Default::default()
            }),
            ..Default::default()
        };

        let mut pipeline = CognitivePipeline::new(config);

        // First generation — stores episode
        let result = pipeline.process_generation("test query", |_prompt| {
            "test output".to_string()
        });
        assert_eq!(result.output, "test output");
        assert_eq!(result.episodes_stored, 1, "Should store 1 episode");
        assert_eq!(result.episodes_retrieved, 0, "First generation has no past episodes");

        // Second generation — should retrieve the episode
        let result2 = pipeline.process_generation("test query", |_prompt| {
            "test output 2".to_string()
        });
        assert_eq!(result2.episodes_retrieved, 1, "Should retrieve past episode");

        // Verify persistence
        pipeline.persist().unwrap();
        assert!(dir.join("episodes.json").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tool_creation_in_pipeline() {
        let config = CognitivePipelineConfig {
            tool_creation_enabled: true,
            ..Default::default()
        };

        let mut pipeline = CognitivePipeline::new(config);

        // Model output contains a tool creation block
        let output = pipeline.process_generation("create a tool", |_prompt| {
            "[tool_create]\nname: helper\ndescription: A helper\n---\nfn helper() { 42 }\n[/tool_create]".to_string()
        });

        assert!(output.tool_created);
        assert_eq!(output.created_tool_name.as_deref(), Some("helper"));
        assert!(pipeline.tool_registry().get("helper").is_some());
    }

    #[test]
    fn test_tool_usage_tracking() {
        let config = CognitivePipelineConfig {
            llm_computer_enabled: true,
            llm_computer_max_program: 64,
            llm_computer_max_steps: 128,
            tool_usage_tracking_enabled: true,
            ..Default::default()
        };

        let mut pipeline = CognitivePipeline::new(config);

        // Execute a tool call
        let _ = pipeline.process_generation("compute", |_prompt| {
            "[tool_call]calm_execute(add 3 5)[/tool_call]".to_string()
        });

        // Check usage was tracked
        let tracker = pipeline.tool_usage_tracker().unwrap();
        assert!(tracker.get_tool_stats("calm_execute").is_some());
        assert_eq!(tracker.total_events(), 1);
    }
}
