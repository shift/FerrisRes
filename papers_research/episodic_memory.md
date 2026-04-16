# Research: Episodic Memory for Self-Extending Models

## Status: Theory — derived from architecture analysis

## Core Theory

Episodic memory is the missing layer between ConceptMap (semantic memory) and the cognitive pipeline. Where ConceptMap stores *what* (facts, tools, algorithms), episodic memory stores *experiences* — the what, how, and why of past interactions.

## 10 Principles

### 1. Event-Based, Not Token-Based
LLMs "remember" via context windows — that's short-term attention, not memory.

Episodic memory stores **events**, each containing:
- **what happened**: compressed semantic representation of the interaction
- **what was attempted**: tools used, reasoning steps, plan executed
- **what the outcome was**: success / failure / partial success with quality score
- **what uncertainty was present**: confidence estimates (logit entropy, MirrorTest score)
- **what should be learned**: error signals, new tools needed, concepts to refine

Biological analog: hippocampal encoding — experience → evaluation → consolidation.

### 2. Sparse, Not Dense
Dense memory (RNN hidden states) doesn't scale. Sparse memory stores only salient events.

Properties:
- Only important events stored (importance = surprise × uncertainty × outcome_magnitude)
- Retrieval by similarity, not recency
- Grows over time without catastrophic forgetting
- Analogous to hippocampal indexing + sparse distributed representations

### 3. Content-Based Retrieval
Retrieval must answer: "Have I seen a situation like this before?"

Retrieval modes:
- **Similar past problems**: Find episodes with similar prompt embeddings
- **Similar failures**: Find episodes where the same tools failed
- **Similar successes**: Find episodes where similar approach worked
- **Similar tool-use patterns**: Find episodes that used the same tool chain

Biological analog: pattern completion in hippocampal CA3.

### 4. Editable and Compressible
Memory must support:
- **Merge**: Combine similar episodes into generalized episode
- **Compress**: Replace N specific episodes with 1 abstracted episode
- **Refine**: Update stored lessons based on new evidence
- **Delete**: Remove irrelevant, harmful, or redundant episodes

Analogous to sleep consolidation: hippocampus (fast, specific) → cortex (slow, general).

### 5. Connected to Tools
Each episode stores:
- Which tools were used and how
- Whether they succeeded or failed
- Whether they should be refined
- The full tool execution trace

This enables:
- Improve tools based on accumulated experience
- Create new tools for recurring failure patterns
- Avoid tools that consistently fail in certain contexts
- Specialize tools for specific contexts

### 6. Supports Meta-Learning
Once enough episodes accumulate, the system can learn:
- When to create a new tool (pattern: similar failure N times)
- When to refine an existing tool (pattern: success rate declining)
- When to retrieve a memory (pattern: uncertainty > threshold)
- When to generalize across memories (pattern: N similar episodes)

### 7. Multi-Modal
Episodes can contain:
- Text (prompts, outputs)
- Tool traces (execution logs)
- Reasoning traces (plan steps)
- Internal states (confidence, entropy, MirrorTest scores)
- Domain modality data (EEG, LiDAR, SCADA streams)

### 8. Persistent Across Sessions
Episodes survive restart. Tools become long-term skills, episodes become long-term experience.

Persistence layers:
1. **Hot**: In-memory episode buffer (current session)
2. **Warm**: ConceptMap (consolidated episodes → concepts)
3. **Cold**: Disk-backed episode archive (full history)

### 9. Safe and Bounded
Safety constraints:
- Cannot overwrite core abilities (base weights frozen)
- Cannot corrupt memory (validation before storage)
- Cannot spiral into runaway self-modification (bounded retries)
- Rollback via checkpoints
- Sandbox for all tool execution

Biological analog: cortex learns slowly, hippocampus learns fast. Fast learning is temporary, slow learning is permanent.

### 10. Self-Generated Curriculum
Episodic memory enables self-directed learning:
- Identify gaps (high uncertainty, low retrieval quality)
- Generate practice tasks targeting gaps
- Execute through cognitive pipeline
- Store results as new episodes
- Repeat until gap closed

## Architecture: EpisodicMemory

```rust
struct Episode {
    id: EpisodeId,
    timestamp: u64,
    
    // What happened
    context_embedding: Vec<f32>,   // Compressed semantic representation
    prompt: String,
    domain: Option<String>,
    
    // What was attempted
    tools_used: Vec<ToolTrace>,
    plan_executed: Option<Vec<PlanStep>>,
    reasoning_trace: Vec<String>,
    
    // What the outcome was
    outcome: EpisodeOutcome,
    quality_score: f32,            // MirrorTest or logit entropy
    output_summary: String,
    
    // What uncertainty was present
    logit_entropy: f32,
    retrieval_distance: f32,       // How close were retrieved concepts
    confidence: f32,
    
    // What should be learned
    learning_signal: LearningSignal,
    concepts_created: Vec<ConceptId>,
    tools_created: Vec<String>,
    
    // Metadata
    session_id: String,
    importance: f32,               // surprise × uncertainty × outcome
    access_count: u32,
    last_accessed: u64,
}

enum EpisodeOutcome {
    Success,
    PartialSuccess(f32),
    Failure(String),
    Abandoned,
}

enum LearningSignal {
    None,
    NewToolNeeded(String),
    ToolRefinementNeeded(String),
    ConceptGap(String),
    PatternRecognized(String),
    ErrorToAvoid(String),
}

struct ToolTrace {
    tool_name: String,
    args: String,
    result_summary: String,
    execution_time_ms: u64,
    mirror_quality: Option<f32>,
}

struct EpisodicMemory {
    // In-memory buffer (hot)
    recent: VecDeque<Episode>,
    recent_capacity: usize,
    
    // Persistent store (warm/cold)
    store_path: PathBuf,
    index: EpisodeIndex,
    
    // Importance filter
    importance_threshold: f32,
    
    // Compression
    compression_policy: CompressionPolicy,
}

struct EpisodeIndex {
    // Embedding → episode IDs (for similarity search)
    embeddings: Vec<(Vec<f32>, EpisodeId)>,
    // Tag → episode IDs
    tag_index: HashMap<String, Vec<EpisodeId>>,
    // Tool → episode IDs
    tool_index: HashMap<String, Vec<EpisodeId>>,
    // Outcome → episode IDs
    outcome_index: HashMap<String, Vec<EpisodeId>>,
}

impl EpisodicMemory {
    /// Store a new episode if it passes the importance filter.
    fn store(&mut self, episode: Episode) -> Option<EpisodeId> {
        if episode.importance < self.importance_threshold {
            return None; // Not important enough to store
        }
        
        let id = episode.id;
        
        // Add to recent buffer
        if self.recent.len() >= self.recent_capacity {
            // Evict oldest from recent, potentially persist to disk
            if let Some(old) = self.recent.pop_front() {
                self.persist_episode(&old);
            }
        }
        self.recent.push_back(episode.clone());
        
        // Update indices
        self.index.embeddings.push((episode.context_embedding.clone(), id));
        for trace in &episode.tools_used {
            self.index.tool_index.entry(trace.tool_name.clone()).or_default().push(id);
        }
        
        Some(id)
    }
    
    /// Retrieve episodes similar to a query.
    fn retrieve(&self, query: &[f32], top_k: usize) -> Vec<&Episode> {
        let mut scored: Vec<(f32, &Episode)> = self.recent.iter()
            .map(|ep| {
                let sim = cosine_similarity(query, &ep.context_embedding);
                // Recency bias: slightly prefer recent episodes
                let recency = 1.0 / (1.0 + ep.age_in_hours() as f32 / 24.0);
                (sim * 0.8 + recency * 0.2, ep)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        scored.truncate(top_k);
        scored.into_iter().map(|(_, ep)| ep).collect()
    }
    
    /// Retrieve episodes where a specific tool was used.
    fn retrieve_by_tool(&self, tool_name: &str) -> Vec<&Episode> {
        self.index.tool_index.get(tool_name)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }
    
    /// Retrieve failure episodes for a given context.
    fn retrieve_failures(&self, query: &[f32], top_k: usize) -> Vec<&Episode> {
        self.retrieve(query, top_k * 3)
            .into_iter()
            .filter(|ep| matches!(ep.outcome, EpisodeOutcome::Failure(_)))
            .take(top_k)
            .collect()
    }
    
    /// Compress similar episodes into a generalized episode.
    fn compress(&mut self) -> usize {
        // Find clusters of similar episodes
        let clusters = self.find_clusters(0.85); // cosine > 0.85
        
        let mut compressed = 0;
        for cluster in clusters {
            if cluster.len() < 3 { continue; }
            
            // Create generalized episode
            let generalized = self.generalize_cluster(&cluster);
            
            // Remove individual episodes, store generalized one
            for ep_id in &cluster {
                self.remove(ep_id);
            }
            self.store(generalized);
            compressed += cluster.len() - 1;
        }
        compressed
    }
}

/// Compute importance score for an episode.
fn compute_importance(episode: &Episode) -> f32 {
    let surprise = 1.0 - episode.confidence;           // Low confidence = surprising
    let uncertainty = episode.logit_entropy;            // High entropy = uncertain
    let outcome_magnitude = match &episode.outcome {
        EpisodeOutcome::Success => 0.3,
        EpisodeOutcome::PartialSuccess(q) => 0.5 + q * 0.5,
        EpisodeOutcome::Failure(_) => 1.0,             // Failures are highly important
        EpisodeOutcome::Abandoned => 0.1,
    };
    surprise * uncertainty * outcome_magnitude
}
```

## Integration with Cognitive Pipeline

```
User prompt arrives
  │
  ├─ EpisodicMemory.retrieve(prompt_embedding, 5)
  │    → "Last time you saw something similar, you used tool X and it worked"
  │    → "You tried approach Y here before and it failed — try Z instead"
  │
  ├─ ConceptMap.retrieve(prompt_embedding, 5)
  │    → Relevant concepts (facts, tools, algorithms)
  │
  ├─ Augment prompt with episodes + concepts
  │
  ├─ Model generates
  │
  ├─ Tool dispatch → execution → MirrorTest
  │
  ├─ EpisodicMemory.store(episode)
  │    → Store what happened, what was tried, what the outcome was
  │
  ├─ If quality high: ConceptMap.store(learning)
  │    → Consolidate episode into semantic memory
  │
  └─ EpisodicMemory.compress() (periodic)
       → Merge similar episodes into generalizations
```

## Key References
1. Tulving (1972) — Episodic vs Semantic Memory distinction
2. O'Keefe & Nadel (1978) — Hippocampus as cognitive map
3. McClelland et al. (1995) — Complementary Learning Systems (hippocampus fast, cortex slow)
4. Kumaran & McClelland (2012) — Generalization through hippocampal replay
