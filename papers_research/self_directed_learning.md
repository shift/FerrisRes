# Research: Self-Directed Learning & Intrinsic Motivation

## Task ID: 01228747-6fcd-4b84-af63-8ff6b6b6eded

## Key Papers & Techniques

### 1. Curiosity-Driven Learning (Intrinsic Curiosity Module)
- **Paper**: Pathak et al. (2017) "Curiosity-driven Exploration by Self-supervised Prediction" [arXiv:1705.05363]
- **Core idea**: Agent explores states where its prediction error is highest. ICM = forward model (predict next state from current state + action) + inverse model (predict action from state pair). Curiosity reward = forward model prediction error.
- **Application to FerrisRes**: Track the model's prediction error on each generation. When error is high (uncertainty), that's a learning opportunity. Generate practice problems in that area.

### 2. Intrinsic Motivation in Robotics
- **Paper**: Oudeyer & Kaplan (2009) "What is Intrinsic Motivation? A Typology of Computational Approaches" [Frontiers in Neurorobotics]
- **Three types of intrinsic motivation**:
  1. **Novelty-based**: Seek states/actions never experienced before
  2. **Competence-based (learning progress)**: Seek tasks where learning progress is maximal (not too easy, not too hard) — the "zone of proximal development"
  3. **Information-theoretic**: Maximize information gain or minimize uncertainty
- **Application**: Use competence-based motivation — the model practices tasks where it's improving fastest, not tasks it's already mastered or tasks far beyond its ability.

### 3. Self-Play (AlphaZero)
- **Paper**: Silver et al. (2018) "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" [Science 362(6419)]
- **Core idea**: Agent plays against itself. Each game generates training data. Over time, agent discovers strategies beyond human knowledge.
- **Application**: Model generates problems for itself, solves them, self-evaluates. Successful strategies become concepts.

### 4. Autotelic Learning (Open-Ended Learning)
- **Paper**: Forestier et al. (2022) "Autotelic Agents: Open-Ended Learning with Intrinsically Motivated Goal-Conditioned Reinforcement Learning" [arXiv:2206.03084]
- **Core idea**: Agent sets its own goals, pursues them, learns from success/failure. Goal space explored via intrinsic motivation.
- **Application**: Model generates its own practice goals: "solve a sorting problem", "write a parser for format X", "optimize this computation". Each goal is a self-created learning task.

### 5. Uncertainty Estimation for LLMs
- **Entropy of output distribution**: H(p) = -Σ p_i log(p_i) over logits. High entropy = model is uncertain.
- **Monte Carlo Dropout**: Run inference multiple times with dropout. Variance = uncertainty.
- **Semantic uncertainty**: Kuhn et al. (2023) "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation" [arXiv:2302.09664]
- **Application**: After generation, compute entropy of logits at each step. If average entropy > threshold, flag as "uncertain" → learning opportunity.

### 6. Boredom Signal (Competence Threshold)
- **Biological basis**: Dopamine response to novelty decreases with repetition (habituation). The brain signals "boredom" when a task is mastered.
- **Implementation**: When MirrorTest quality score stays above 0.95 for N consecutive uses of a concept/tool, mark it as "mastered". Stop generating practice problems for mastered concepts.

## Recommended Approach for FerrisRes

### IntrinsicMotivation Module

```rust
struct IntrinsicMotivation {
    // Per-concept uncertainty tracking
    uncertainty: HashMap<ConceptId, UncertaintyEstimate>,
    // Learning progress: improvement rate per concept
    learning_progress: HashMap<ConceptId, f32>,
    // Mastered concepts (boredom threshold)
    mastered: HashSet<ConceptId>,
    // Goal history (what was attempted)
    goal_history: Vec<GoalAttempt>,
    // Zone of proximal development threshold
    zpd_low: f32,  // Below this = too easy (boring)
    zpd_high: f32, // Above this = too hard (frustrating)
}

struct UncertaintyEstimate {
    entropy_avg: f32,       // Average logit entropy
    mirror_test_avg: f32,   // Average MirrorTest quality
    retrieval_distance: f32, // Average concept retrieval distance
    sample_count: usize,
}

struct GoalAttempt {
    goal: String,
    concept_ids: Vec<ConceptId>,
    difficulty: f32,
    outcome: GoalOutcome,
    timestamp: u64,
}

enum GoalOutcome {
    Success,
    PartialSuccess(f32),
    Failure,
    Abandoned,
}
```

### Self-Directed Learning Loop
```
1. ESTIMATE UNCERTAINTY
   - After each generation, compute logit entropy
   - Track per-concept uncertainty over time
   - Identify concepts with highest uncertainty

2. SELECT PRACTICE GOAL
   - Find concepts in Zone of Proximal Development:
     zpd_low < uncertainty < zpd_high
   - Generate a practice problem targeting that concept
   - Difficulty calibrated to current ability

3. EXECUTE PRACTICE
   - Run practice problem through cognitive pipeline
   - Model generates solution
   - MirrorTest evaluates
   - Store result in episodic memory

4. UPDATE LEARNING PROGRESS
   - Track improvement: Δ(quality) / Δ(time)
   - Concepts with high learning progress = keep practicing
   - Concepts with low/no learning progress = try different approach
   - Mastered concepts (quality > 0.95 sustained) = stop practicing

5. CURRICULUM ADJUSTMENT
   - Increase difficulty as mastery grows
   - Move to new concepts when current ones plateau
   - Revisit old concepts periodically (spaced repetition)
```

### Spaced Repetition
- **Paper**: Ebbinghaus (1885) forgetting curve
- **Leitner system**: Review intervals increase with each successful recall
- **Application**: Concepts that haven't been retrieved recently get priority for review. If review quality drops, move back to active practice.

## Key References
1. [arXiv:1705.05363] Pathak 2017 - ICM / Curiosity-driven Exploration
2. Oudeyer & Kaplan 2009 - Intrinsic Motivation Typology
3. Silver et al. 2018 - AlphaZero Self-Play
4. [arXiv:2206.03084] Forestier 2022 - Autotelic Agents
5. [arXiv:2302.09664] Kuhn 2023 - Semantic Uncertainty
6. Ebbinghaus 1885 - Forgetting Curve
