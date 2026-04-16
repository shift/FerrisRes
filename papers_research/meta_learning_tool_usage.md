# Research: Meta-Learning Through Tool Usage

## Task ID: e668f7f3-bbe4-4e28-b8b8-dc84888df7ca

## Key Papers & Techniques

### 1. MAML (Model-Agnostic Meta-Learning)
- **Paper**: Finn et al. (2017) [arXiv:1703.03400]
- **Core idea**: Learn initial parameters θ that can quickly adapt to any new task with few gradient steps.
- **Meta-training**: Sample task T_i, compute θ'_i = θ - α∇L_{T_i}(θ), then update θ ← θ - β∇_θ L_{T_i}(θ'_i)
- **Application**: Train FerrisRes student model so it's predisposed to quick tool learning.

### 2. RL² (Fast Reinforcement Learning via Slow Reinforcement Learning)
- **Paper**: Duan et al. (2016) [arXiv:1611.02779]
- **Core idea**: Train an RNN on a distribution of RL tasks. The RNN's hidden state becomes a learned learning algorithm. After a few episodes on a new task, the RNN has adapted.
- **Application**: The LlmComputer's register/memory state can encode a "learning state" — after seeing a few examples of a tool being used, it adapts its usage strategy.

### 3. MLC (Meta-Learning for Composition)
- **Paper**: Lake et al. (2023) "Human-like concept learning through meta-learning" and related work
- **Core idea**: Meta-train on compositional tasks so the model learns to combine known components in novel ways.
- **Application**: Meta-train the model on tool composition tasks so it learns to chain tools effectively without explicit training on every possible chain.

### 4. Learned Optimization
- **Paper**: Andrychowicz et al. (2016) "Learning to learn by gradient descent by gradient descent" [arXiv:1606.04474]
- **Core idea**: Replace hand-designed optimizer with a learned optimizer (an RNN that outputs parameter updates).
- **Application**: The model could learn its own learning strategy — instead of fixed Adam, use a learned update rule for tool-triggered LoRA updates.

### 5. Usage History Tracking
- **Our approach**: Track (tool, context_embedding, result_quality) tuples over time.
- **Metrics**: Per-tool success rate, per-context-type success rate, time-decayed success rate.
- **Usage policy**: When selecting tools, weight by historical success in similar contexts. This is a simple form of contextual bandit learning.

### 6. Tool Usage as Multi-Armed Bandit
- **Framework**: Each tool is an "arm". Context = prompt embedding. Reward = MirrorTest quality score.
- **Algorithms**: LinUCB (contextual bandit), Thompson Sampling, EXP3 (adversarial).
- **Application**: Tool selection becomes a bandit problem. Model explores new tools (exploration) vs uses known-good tools (exploitation).

## Recommended Approach for FerrisRes

### Phase 1: Simple Usage Tracking
```rust
struct ToolUsageTracker {
    // Per-tool: (success_count, failure_count, last_used)
    stats: HashMap<String, ToolStats>,
    // Contextual: (context_embedding_bin → tool → success_rate)
    context_stats: HashMap<u64, HashMap<String, f32>>,
}

impl ToolUsageTracker {
    fn record(&mut self, tool: &str, context: &[f32], quality: f32) {
        // Update tool-level stats
        self.stats.entry(tool.into())
            .and_modify(|s| s.update(quality))
            .or_insert(ToolStats::new(quality));
        
        // Update context-level stats (quantized embedding)
        let bin = quantize_embedding(context);
        self.context_stats.entry(bin)
            .or_default()
            .entry(tool.into())
            .and_modify(|q| *q = 0.9 * *q + 0.1 * quality)
            .or_insert(quality);
    }
    
    fn best_tool_for_context(&self, context: &[f32], candidates: &[String]) -> Option<String> {
        let bin = quantize_embedding(context);
        if let Some(tool_scores) = self.context_stats.get(&bin) {
            candidates.iter()
                .filter_map(|t| tool_scores.get(t).map(|q| (t, *q)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(t, _)| t.clone())
        } else {
            // No context-specific data, use global stats
            candidates.iter()
                .filter_map(|t| self.stats.get(t).map(|s| (t, s.success_rate())))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(t, _)| t.clone())
        }
    }
}
```

### Phase 2: Contextual Bandit (LinUCB)
For each tool, maintain a linear model: quality ≈ β · context_embedding. Use UCB for exploration. Requires: d-dimensional context embedding, K tools.

### Phase 3: Transfer Learning
When the model gets good at using tool A in domain X, transfer usage knowledge to tool A in domain Y via embedding similarity.

## Key References
1. [arXiv:1703.03400] Finn 2017 - MAML
2. [arXiv:1611.02779] Duan 2016 - RL²
3. [arXiv:1606.04474] Andrychowicz 2016 - Learning to Learn
4. Li et al. 2010 - Contextual Bandit (LinUCB)
