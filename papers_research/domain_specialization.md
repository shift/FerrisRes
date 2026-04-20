# Research: Domain Specialization — Detection and Specialized Retrieval

## Summary

Research into domain detection from prompts and specialized retrieval for FerrisRes. Currently all retrieval (concepts, tools, episodes) is embedding-based with no domain awareness. This research covers domain classification approaches, per-domain concept/tool/episode preferences, domain transfer (applying knowledge from domain A to domain B), and domain vocabulary.

## Existing Architecture

- **ConceptMemory**: Retrieval by cosine similarity on embeddings. No domain filtering.
- **ToolUsageTracker**: Per-tool quality tracking. `best_tool_for_context()` finds best tool by context tags, not domain.
- **EpisodicMemory**: Stores episodes with importance scoring. Retrieval by recency and quality. No domain filtering.
- **ToolSearch** (`tool_search.rs`): Retrieves tools by query. No domain awareness.
- **AbstractionEngine**: Clusters concepts. Clusters are domain-agnostic.
- **CognitivePipeline**: Orchestrates everything but has no domain concept.

## Approaches to Domain Detection

### Approach 1: Keyword-Based Domain Classifier

**Method**: Maintain a domain keyword dictionary. Match input prompt against keyword patterns.

```
Domain keywords:
  programming: ["function", "class", "compile", "debug", "api", "code", "rust", "python"]
  mathematics: ["equation", "integral", "derivative", "proof", "theorem", "matrix"]
  science: ["experiment", "hypothesis", "molecule", "reaction", "force", "energy"]
  medicine: ["patient", "diagnosis", "symptom", "treatment", "drug", "clinical"]
  finance: ["stock", "portfolio", "risk", "option", "bond", "trading"]
  law: ["contract", "statute", "liability", "compliance", "regulation"]
```

**Pros**: Fast, deterministic, zero model dependency, easy to extend.
**Cons**: Brittle, can't handle ambiguous prompts, requires manual keyword curation.

**Implementation**: Simple enough to be a first pass.

### Approach 2: Embedding Clustering

**Method**: Cluster all past prompts by embedding similarity. Assign cluster IDs as domains. New prompts are classified by nearest cluster centroid.

```
Cluster 1 (label: "programming"): centroid at [0.2, -0.5, 0.8, ...]
Cluster 2 (label: "mathematics"): centroid at [0.7, 0.1, -0.3, ...]
...
```

**Pros**: Automatic domain discovery, handles ambiguity via soft assignment.
**Cons**: Requires embedding model (FerrisRes doesn't have one), cluster quality varies, needs re-clustering over time.

**Implementation**: Feasible with FerrisRes's character n-gram embeddings (crude but functional).

### Approach 3: LLM-Based Domain Tagging

**Method**: Ask the model to tag its own prompt with a domain.

```
Prompt: "Classify this query's domain. Reply with exactly one word.
Query: 'How do I implement a red-black tree in Rust?'
Domain:"
```

**Pros**: Most accurate, handles novel domains, zero configuration.
**Cons**: Requires a model call (slow, costs tokens), model may hallucinate domains.

**Implementation**: Can be integrated into the existing `[plan]` parsing — add a `[domain: programming]` prefix.

### Approach 4: Tool-Usage History (Behavioral Detection)

**Method**: Infer domain from which tools are selected. If `code_run` and `wasm_parse` are used → programming. If `web_fetch` with API URLs → data retrieval.

**Pros**: Based on actual behavior, not text. More reliable than keyword guessing.
**Cons**: Only works after tool selection, not before. Can't pre-filter retrieval.

### Recommended: Hybrid — Keywords + LLM + Behavioral

```
DomainDetector:
  1. FAST PATH: Keyword scan (deterministic, <1ms)
     → If confidence > 0.8: use keyword domain
  2. LLM PATH: Ask model to tag domain (when generating plan)
     → Model emits [domain: X] before [plan]
  3. BEHAVIORAL PATH: After tool selection, update domain tracker
     → Confirm or override initial domain guess
```

## Data Structures

```rust
/// Known domains with their characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainProfile {
    /// Domain name (e.g., "programming", "mathematics").
    pub name: String,
    /// Keywords strongly associated with this domain.
    pub keywords: Vec<String>,
    /// Tools preferred in this domain (by historical success rate).
    pub preferred_tools: HashMap<String, f32>, // tool_name → success_rate
    /// Concept categories most relevant.
    pub relevant_concept_categories: Vec<String>,
    /// Episode types most relevant.
    pub relevant_episode_types: Vec<String>,
    /// Domain-specific vocabulary (terms with special meanings).
    pub vocabulary: HashMap<String, String>, // term → domain-specific definition
    /// Cross-domain transfer targets (domains that share knowledge).
    pub transfer_sources: Vec<String>,
    /// Number of interactions in this domain.
    pub interaction_count: u32,
    /// Average quality in this domain.
    pub avg_quality: f32,
}

/// Domain detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainDetection {
    /// Primary detected domain.
    pub domain: String,
    /// Confidence (0.0–1.0).
    pub confidence: f32,
    /// Secondary domain (if ambiguous).
    pub secondary: Option<(String, f32)>,
    /// Detection method used.
    pub method: DetectionMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectionMethod {
    /// Matched by keyword scan.
    Keyword { matched_terms: Vec<String> },
    /// Tagged by LLM during planning.
    LlmTagged,
    /// Inferred from tool usage.
    Behavioral { tools_used: Vec<String> },
    /// Default (no domain detected).
    Unknown,
}
```

## Per-Domain Retrieval Preferences

When a domain is detected, retrieval is biased:

1. **Concept retrieval**: Filter by `relevant_concept_categories`, then rank by cosine similarity. E.g., in "programming", prefer `ConceptContent::Code` over `ConceptContent::Formula`.

2. **Tool retrieval**: `best_tool_for_context()` already uses context tags. Extend to factor in `DomainProfile.preferred_tools` — if tool X has 90% success in domain Y and current domain is Y, boost X's score by 20%.

3. **Episode retrieval**: When recalling past experiences, prefer episodes from the same domain. E.g., in "medicine", prefer episodes where medical tools were used successfully.

```rust
impl DomainDetector {
    /// Detect domain from a prompt.
    fn detect(&self, prompt: &str) -> DomainDetection;
    
    /// Detect domain from tool usage.
    fn detect_from_tools(&self, tools: &[String]) -> DomainDetection;
    
    /// Update domain profile with interaction outcome.
    fn update_profile(&mut self, domain: &str, tool: &str, quality: f32);
    
    /// Get retrieval bias for detected domain.
    fn retrieval_bias(&self, domain: &str) -> DomainRetrievalBias;
    
    /// Check for cross-domain transfer opportunities.
    fn transfer_candidates(&self, domain: &str) -> Vec<TransferCandidate>;
}

pub struct DomainRetrievalBias {
    /// Boost factor for preferred tools.
    pub tool_boost: HashMap<String, f32>,
    /// Preferred concept categories.
    pub concept_categories: Vec<String>,
    /// Preferred episode types.
    pub episode_types: Vec<String>,
}
```

## Domain Transfer

**Concept**: Knowledge from domain A may help in domain B.

Example: "Debugging strategies" (programming) transfers to "diagnostic reasoning" (medicine).

**Transfer mechanism**:
1. When a concept is retrieved in domain B but was created in domain A, mark it as "transferred".
2. Track transferred concept quality separately.
3. If transferred concepts perform well (quality > 0.7), add domain A → B as a transfer path.
4. ProactiveController can suggest: "Your programming debugging strategies seem useful for medical diagnosis. Want to formalize this?"

**Implementation**:
```rust
struct TransferPath {
    source_domain: String,
    target_domain: String,
    /// How many concepts were transferred.
    concepts_transferred: u32,
    /// Average quality of transferred concepts.
    avg_transfer_quality: f32,
    /// Whether this transfer has been validated.
    validated: bool,
}
```

**Transfer validation**: Before relying on a transfer path, verify that concepts from domain A actually improve performance in domain B. Compare retrieval quality with and without cross-domain concepts.

## Domain Vocabulary

Each domain has terms with specialized meanings:

```rust
struct DomainVocabulary {
    domain: String,
    /// Term → domain-specific meaning.
    terms: HashMap<String, TermDefinition>,
}

struct TermDefinition {
    /// Domain-specific definition.
    definition: String,
    /// How this term's meaning differs from general use.
    nuance: String,
    /// Related terms in the same domain.
    related: Vec<String>,
    /// Confidence in this definition.
    confidence: f32,
}
```

Vocabulary is learned from episodes: if "bug" consistently appears in programming episodes with a specific meaning, add it to the programming domain vocabulary.

## Integration with Existing Modules

- **CognitivePipeline**: After `detect()`, store domain in pipeline context. All subsequent module calls use domain-aware retrieval.
- **ConceptMemory**: Add optional domain tag to concepts. Filter by domain during retrieval.
- **ToolUsageTracker**: Track per-domain tool quality (already has per-context tracking, domain is a context dimension).
- **EpisodicMemory**: Add domain tag to episodes. Prefer same-domain episodes in retrieval.
- **AbstractionEngine**: Cluster within domains first, then across domains for transfer.
- **IntrinsicMotivation**: Track per-domain uncertainty. If uncertainty is high in a domain, prioritize practice there.
- **ProactiveController**: Suggest domain exploration ("You've never tried a finance task — interested?").
- **SubgoalGenerator**: Decomposition can be domain-aware (programming decomposition is different from medical decomposition).

## Test Plan

1. `test_keyword_detection_programming` — code terms detected
2. `test_keyword_detection_math` — math terms detected
3. `test_keyword_confidence` — multiple matches → high confidence
4. `test_keyword_ambiguity` — mixed domain → secondary detected
5. `test_llm_tag_parsing` — `[domain: programming]` parsed
6. `test_behavioral_detection` — tool usage infers domain
7. `test_domain_profile_creation` — new domain profile created
8. `test_preferred_tools_update` — tool success updates domain profile
9. `test_retrieval_bias_tool_boost` — preferred tools boosted
10. `test_retrieval_bias_concept_filter` — domain concepts prioritized
11. `test_episode_domain_filter` — same-domain episodes preferred
12. `test_transfer_path_creation` — cross-domain concept use tracked
13. `test_transfer_quality_tracking` — transferred concept quality measured
14. `test_transfer_validation` — transfer path validated by comparison
15. `test_vocabulary_learning` — term meaning learned from episodes
16. `test_unknown_domain_handling` — no match → Unknown, no bias
17. `test_domain_persistence` — profiles saved/loaded
18. `test_multi_domain_prompt` — prompt with multiple domains
19. `test_domain_interaction_count` — tracked per-domain
20. `test_per_domain_uncertainty` — IntrinsicMotivation uses domain

## References

- Bordes, A. et al. (2017). "Learning End-to-End Goal-Oriented Dialog" — domain classification for dialog
- Liu, X. et al. (2021). "Domain Adaptation for NLP" — transfer across domains
- Kirkpatrick, J. et al. (2017). "Overcoming Catastrophic Forgetting in Neural Networks" — domain interference prevention
- Ramesh, A. et al. (2023). "Domain-Specific LLMs" — specialized retrieval per domain
- Bengio, Y. (2012). "Deep Learning of Representations for Unsupervised and Transfer Learning" — transfer learning foundations
