# Research: Self-Generated Abstractions & Concept Formation

## Task ID: e86f6dd7-5282-4d2c-802b-6593d82ba602

## Key Papers & Techniques

### 1. Concept Formation in Cognitive Science
- **Eleanor Rosch (1978)**: Prototype theory — categories are organized around prototypes (best examples). New items classified by similarity to prototype.
- **Application**: When ConceptMap has N similar concepts, compute the "prototype" (centroid embedding) and store as a generalized meta-concept. Replace individuals with the prototype.

### 2. Gentner's Structure-Mapping Theory (Analogical Reasoning)
- **Paper**: Gentner (1983) "Structure-Mapping: A Theoretical Framework for Analogy"
- **Core idea**: Analogy = structural alignment between source and target domains. Map relations, not surface features.
- **Application**: Detect when a tool from domain A has structural similarity to a problem in domain B. Transfer the tool via analogical mapping.

### 3. Chunking in SOAR
- **Paper**: Laird et al. (1986) "SOAR: The anatomy of a general intelligence"
- **Chunking mechanism**: When SOAR solves a problem via search, it creates a "chunk" (production rule) that summarizes the solution. Next time, the chunk fires immediately without search.
- **Application**: When the model solves a problem via multi-step tool chain + planning, compress the successful plan into a single "chunked" concept. Next time, the chunk fires directly.

### 4. Chunking in ACT-R
- **Paper**: Anderson (1996) "ACT: A simple theory of complex cognition"
- **Declarative memory**: Chunks have activation levels based on recency, frequency, and associative strength. Base-level activation: B_i = ln(Σ t_j^(-d)) where t_j = time since jth use.
- **Application**: ConceptMap quality scores already approximate this. Add base-level activation decay to concepts — frequently used, recent concepts get higher activation.

### 5. LED Synthesis (Program Abstraction)
- **Paper**: Ellis et al. (2021) "DreamCoder: Learning to Code by Writing Programs in Your Sleep" [arXiv:2006.07732]
- **Core idea**: Learn a library of reusable functions by abstraction:
  1. Solve problems with current library
  2. Find common sub-expressions across solutions
  3. Extract as new library functions (abstraction)
  4. Re-solve with expanded library (compression)
- **Application**: Scan accumulated tool code for shared patterns. Extract common subroutines as meta-tools. This is exactly "concept formation" for programs.

### 6. Compression as Intelligence
- **Hutter Prize / Chaitin-Kolmogorov complexity**: Intelligence ≈ compression. Better abstractions → shorter descriptions → better generalization.
- **Minimum Description Length (MDL)**: Among hypotheses that fit the data, prefer the shortest.
- **Application**: When forming abstractions, prefer the meta-concept that most compresses the set of specific concepts it replaces.

## Recommended Approach for FerrisRes

### Abstraction Engine Pipeline
```
1. SCAN: Periodically scan ConceptMap for clusters of similar concepts
   - Group by embedding similarity (cosine > 0.8)
   - Group by structural similarity (same code pattern, different constants)
   - Group by tag overlap

2. EXTRACT: For each cluster, compute the generalized meta-concept
   - Code: parameterize the varying parts (e.g., sort(arr, asc) → sort(arr, order))
   - Algorithm: find common steps, parameterize the differences
   - Embedding: centroid of cluster members

3. VALIDATE: MirrorTest the meta-concept
   - Generate test cases for the parameterized version
   - Run tests in WasmSandbox
   - If pass rate > threshold, accept abstraction

4. COMPRESS: Replace N specific concepts with 1 meta-concept
   - Store meta-concept in ConceptMap
   - Mark specific concepts as "subsumed by" the meta-concept
   - Free capacity in ConceptMap

5. PROPAGATE: Update tool references
   - Tools that referenced specific concepts now reference meta-concept
   - ToolRegistry updated with parameterized version
```

### Hierarchical Concept Structure
```rust
enum ConceptLevel {
    // Level 0: Raw observations
    Instance,
    // Level 1: Abstracted from instances  
    Pattern { source_instances: Vec<ConceptId> },
    // Level 2: Abstracted from patterns
    Principle { source_patterns: Vec<ConceptId> },
    // Level 3: Cross-domain abstractions
    MetaPrinciple { source_principles: Vec<ConceptId> },
}
```

### Cross-Domain Transfer Detection
When a new problem's embedding is similar to a concept from a different domain:
1. Retrieve the cross-domain concept
2. Check structural similarity (not just embedding similarity)
3. If structurally similar: propose as a solution template
4. Validate via MirrorTest
5. If successful: create a new concept tagged for the new domain

## Key References
1. Rosch 1978 - Prototype Theory
2. Gentner 1983 - Structure-Mapping Theory
3. Laird 1986 - SOAR Chunking
4. Anderson 1996 - ACT-R
5. [arXiv:2006.07732] Ellis 2021 - DreamCoder
6. Hutter 2004 - Universal Artificial Intelligence
