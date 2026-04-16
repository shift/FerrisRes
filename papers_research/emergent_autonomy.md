# Research: Emergent Autonomy — Proactive Problem Solving

## Task ID: b8363afa-cf27-4c29-995d-a2316e82f8c3

## Key Papers & Techniques

### 1. Autonomous Web Agents
- **Paper**: Yao et al. (2022) "WebShop: Towards Scalable Real-World Web Interaction" [arXiv:2207.01206]
- **Paper**: Zhou et al. (2023) "WebArena: A Realistic Web Environment for Building Autonomous Agents" [arXiv:2307.13854]
- **Core pattern**: Agent observes environment state → decides action → executes → observes new state. Loop continues until goal reached or budget exhausted.
- **Application**: FerrisRes as autonomous agent — observes user context, decides when to proactively act (look up concepts, verify outputs, suggest improvements).

### 2. Proactive Assistants
- **Paper**: Krusiewicz et al. (2023) "Proactive Dialogue Agents" — agents that initiate interactions based on context, not just respond.
- **Key insight**: Proactive behavior requires: (1) situation assessment, (2) opportunity detection, (3) action selection, (4) timing judgment (when to act).
- **Application**: Between user interactions, model reviews its state: concept quality degrading? New tools needed? Uncertainty high on recent outputs? If so, take proactive action.

### 3. Bounded Autonomy / Human-in-the-Loop
- **Paper**: Wu et al. (2022) "AI Chains: Transparent and Controllable Human-AI Interaction" [CHI 2022]
- **Core idea**: Autonomous actions should be bounded by human-set constraints. Different autonomy levels:
  - Level 0: Fully reactive (user must prompt everything)
  - Level 1: Suggestive (model suggests, user confirms)
  - Level 2: Semi-autonomous (model acts, user reviews after)
  - Level 3: Fully autonomous (model acts independently, logs actions)
- **Application**: Cognitive pipeline autonomy_level parameter. Default: Level 1 (suggestive). User can increase via CLI flag.

### 4. Goal Maintenance
- **Paper**: Schank & Abelson (1977) "Scripts, Plans, Goals and Understanding"
- **Core idea**: Maintain a stack of active goals. Goals can be user-specified or self-generated. Goals have priority, deadline, and completion criteria.
- **Application**: GoalStack in cognitive pipeline:
  - User goals: from prompts
  - Self-generated goals: from uncertainty/learning progress
  - Maintenance goals: review concept quality, refine tools, compress memory

### 5. Initiative Signals
When should the model take initiative?
- **Post-generation review**: After generating output, if MirrorTest quality < threshold, proactively retry or flag uncertainty
- **Concept degradation**: If a concept's quality score drops, proactively re-verify it
- **Tool obsolescence**: If a tool's success rate drops, proactively refine it
- **Knowledge gaps**: If retrieval returns no concepts for a prompt, proactively store the interaction for future learning
- **Idle time**: Between interactions, run self-improvement cycles (concept compression, tool refinement, spaced repetition)

### 6. Safety Boundaries
- **Hard limits**: Model cannot modify base weights, delete concepts below quality threshold, create unlimited tools, or act without logging
- **Soft limits**: Model should ask user before: creating new tools, deleting concepts, spending >N compute on self-improvement
- **Rollback**: All autonomous actions logged with before/after state. Can be rolled back.

## Recommended Approach for FerrisRes

### ProactiveController Architecture

```rust
struct ProactiveController {
    // Current autonomy level (0-3)
    autonomy_level: AutonomyLevel,
    // Active goal stack
    goals: Vec<Goal>,
    // Action log (for rollback and review)
    action_log: Vec<ActionRecord>,
    // Initiative signal thresholds
    config: ProactiveConfig,
}

enum AutonomyLevel {
    Reactive,       // Only respond to prompts
    Suggestive,     // Suggest actions, user confirms
    SemiAutonomous, // Act, user reviews after
    FullyAutonomous, // Act independently
}

struct Goal {
    description: String,
    source: GoalSource,
    priority: f32,
    created_at: u64,
    deadline: Option<u64>,
    completion_criteria: String,
}

enum GoalSource {
    UserPrompt(String),
    SelfGenerated(String),  // e.g. "concept X quality dropping"
    Maintenance(String),    // e.g. "compress concepts"
}

struct ActionRecord {
    action: String,
    before_state: String,  // JSON snapshot
    after_state: String,
    goal_id: Option<String>,
    timestamp: u64,
    approved: bool,
}

impl ProactiveController {
    fn should_act(&self, pipeline_state: &CognitivePipeline) -> Option<Goal> {
        // Check initiative signals
        if let Some(goal) = self.check_concept_degradation(pipeline_state) {
            return Some(goal);
        }
        if let Some(goal) = self.check_tool_obsolescence(pipeline_state) {
            return Some(goal);
        }
        if let Some(goal) = self.check_knowledge_gaps(pipeline_state) {
            return Some(goal);
        }
        None
    }
    
    fn execute_proactive_action(&mut self, goal: Goal, pipeline: &mut CognitivePipeline) -> ActionRecord {
        let before = pipeline.snapshot_state();
        
        // Execute based on goal type
        let action = match &goal.source {
            GoalSource::Maintenance(what) => {
                match what.as_str() {
                    "compress_concepts" => pipeline.compress_concepts(),
                    "refine_tools" => pipeline.refine_low_quality_tools(),
                    "spaced_repetition" => pipeline.review_stale_concepts(),
                    _ => "unknown_maintenance".into(),
                }
            }
            GoalSource::SelfGenerated(why) => {
                format!("Self-initiated: {}", why)
            }
            _ => "no_action".into(),
        };
        
        let after = pipeline.snapshot_state();
        
        ActionRecord {
            action,
            before_state: before,
            after_state: after,
            goal_id: None,
            timestamp: now(),
            approved: self.autonomy_level != AutonomyLevel::Suggestive,
        }
    }
}
```

### Proactive Behaviors (ordered by priority)
1. **Post-generation verification**: Always verify output quality with MirrorTest (autonomy level 0+)
2. **Concept quality maintenance**: Re-verify concepts with declining quality (level 1+)
3. **Tool refinement**: Attempt to improve tools with <80% success rate (level 2+)
4. **Concept compression**: When ConceptMap >80% capacity, compress similar concepts (level 2+)
5. **Spaced repetition**: Review concepts not accessed in >N sessions (level 2+)
6. **Self-directed practice**: Generate practice problems for high-uncertainty areas (level 3)

## Key References
1. [arXiv:2207.01206] Yao 2022 - WebShop
2. [arXiv:2307.13854] Zhou 2023 - WebArena
3. Wu et al. 2022 - AI Chains (CHI)
4. Schank & Abelson 1977 - Scripts, Plans, Goals
