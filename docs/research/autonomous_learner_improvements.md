# **Analysis of Structural and Algorithmic Improvements for the FerrisRes Phase 9 Autonomous Learner Architecture**

The transition toward decentralized, on-device artificial intelligence necessitates a fundamental reimagining of the transformer architecture, shifting from high-precision, compute-intensive dense models to sparse, hardware-aware systems capable of autonomous refinement. The FerrisRes project, particularly the structural and training advancements within the phase9-autonomous-learner branch, represents a sophisticated synthesis of contemporary research in linear-time transformers, mixture-of-experts (MoE) efficiency, and sub-ternary weight quantization.1 The primary objective of this phase is to establish a native Block-MoE-Res engine that facilitates not only efficient inference on resource-constrained hardware like the Raspberry Pi 5 but also a continuous self-improvement loop through local interaction data.3 This report examines the twenty-four planned improvements, articulating their theoretical foundations, causal relationships, and broader implications for the field of edge general intelligence (EGI).5

## **Core Architecture: Hierarchical Reasoning and Block AttnRes**

The foundation of the phase9 architecture is the Block-MoE-Res framework, which fundamentally alters the way information aggregates across the depth of the network. Traditional transformer models rely on additive residual connections, where the hidden state at layer ![][image1] is defined as ![][image2].7 While effective for gradient propagation, this equal-weight stacking creates a dilution effect in deeper models—often termed PreNorm dilution—where the magnitude of hidden states grows monotonically, complicating the distribution of gradients and limiting the model's ability to selectively prioritize early-layer representations.7

### **The Transition to Depth-Wise Softmax Attention**

The proposed Block-MoE-Res architecture implements Attention Residuals (AttnRes) to replace fixed additive accumulation with learned softmax attention over preceding representations.8 In this paradigm, each layer ![][image1] learns to retrieve specific representations from earlier layers that are most relevant to its current functional objective.8 Mathematically, this is expressed as:

![][image3]  
where ![][image4] represents softmax attention weights computed from a learned pseudo-query ![][image5].8 This transition mirrors the evolution of sequence modeling, where attention replaced recurrent compression by allowing each position direct access to its history.9 In the context of FerrisRes, this "time-depth duality" enables the network to function as a depth-wise attention mechanism, significantly improving downstream performance on reasoning-intensive tasks.9

The specific implementation in the phase9 branch utilizes a 7-block, 5-layer configuration.11 Partitioning the 35 layers into blocks serves as a critical optimization for memory and communication overhead.8 While "Full AttnRes" requires attending over all preceding layers—leading to ![][image6] complexity—Block AttnRes summarizes each block into a single representation, reducing the footprint to ![][image7], where ![][image8] is the number of blocks.8 Within each block, standard additive residuals are maintained, while cross-block attention allows later blocks to reference earlier context with high specificity.7

### **Inter-Block Attention and Hierarchical Reasoning**

Inter-block attention facilitates hierarchical reasoning by enabling the model to capture multi-hop dependencies that emerge through repeated attention composition.12 Conventional models typically rely on sequential stacking to implicitly capture these connections; however, if an intermediate token is not selected in a sparse attention layer, the multi-hop dependency is lost.12 By performing a blockwise walk or inter-block attention, FerrisRes ensures that later blocks can ground their abstract representations in the fine-grained patterns processed in early blocks.12

| Architecture Component | Traditional Transformer | Block-MoE-Res (FerrisRes) |
| :---- | :---- | :---- |
| Residual Connection | Additive (![][image9]) | Softmax Attention (Learned) |
| Representation Flow | Sequential Compression | Hierarchical Retrieval |
| Memory Complexity | ![][image10] per layer | ![][image11] block summaries |
| Scaling Advantage | Baseline | Equivalent to 1.25x Compute 9 |
| Gradient Norms | Monotonic/Diluted | Bounded and Uniform 8 |

The integration of inter-block attention is complemented by the Dense-to-MoE conversion strategy.3 During the distillation phase, every dense feed-forward network (FFN) from a teacher model—such as the Gemma 4 31B Dense—is converted into a 4-expert MoE.3 This structural transformation increases the model's total parameter capacity while maintaining a fixed compute budget per token.3 The resulting sparse compute path is particularly suited for heterogeneous tasks where different experts can specialize in distinct domains or modalities.13

## **Semantic Anchoring through Per-Layer Embeddings (PLE)**

A secondary but vital architectural innovation is the adoption of Per-Layer Embeddings (PLE), matching the design of the Gemma 4 edge variants (E2B and E4B).14 Standard transformers compute a single embedding per token at the input layer, which is then carried through the entire network.14 PLE introduces additional, layer-specific embedding matrices that inject fresh conditioning signals into every decoder layer.14

### **Decoupling Parameter Storage from Active Computation**

The primary advantage of PLE in an autonomous learner framework is the decoupling of parameter capacity from inference compute cost.15 While these embedding matrices significantly increase the total parameter count—for example, the E2B model carries 5.1 billion parameters but only 2.3 billion "effective" parameters—they do not increase the number of floating-point operations.15 In practice, embeddings are processed as lookup operations rather than matrix multiplications.15

From an engineering perspective, this allows the system to store the massive PLE matrices in flash memory or system RAM and only retrieve the specific vectors associated with the tokens currently in the prompt.15 This "parameter skipping" technique reduces the VRAM load while providing each layer with richer, contextually differentiated representations.3 The PLE vectors act as "semantic anchors" that help deeper layers maintain spatial and semantic awareness, mitigating the drift that often occurs in very deep residual stacks.14

| Model Variant | Effective Parameters | Total Parameters | Active Memory Strategy |
| :---- | :---- | :---- | :---- |
| FerrisRes E2B | 2.3B | 5.1B | PLE Caching on Flash 17 |
| FerrisRes E4B | 4.5B | 8B | Multi-tier VRAM/RAM 16 |
| Core Transformer | Always Active | Fixed | High-speed VRAM |
| MoE Experts | Sparsely Active | Varies | Expert mmap Loading 15 |

The use of PLE also enhances the model's "knowledge storage capacity".18 Research into these embedding spaces suggests they exhibit a larger angular spread compared to the address vectors produced by standard FFNs, enabling more precise and disentangled knowledge attribution.18 This interpretability is critical for an autonomous learner, as it allows for "surgical" modifications—such as swapping a specific token's PLE at a specific layer—to steer the model's behavior without the need for full-model fine-tuning.18

## **Advanced Optimization and Training Paradigms**

The "autonomous learner" designation of the phase9 branch implies a capacity for local, privacy-preserving training on edge hardware.6 Traditional training requires adaptive optimizers like Adam, which maintain first- and second-order moments of the gradients, consuming significant memory—often 2 to 3 times the size of the model weights themselves.20 To address this, FerrisRes integrates two hardware-aware optimizers: SCALE and AdaMeM.20

### **SCALE: Memory-Efficient Pretraining on the Edge**

The SCALE (Stochastic Column-normAlized Last-layer momEntum) optimizer is specifically designed for environments with extreme memory constraints, such as the Raspberry Pi 5\.20 SCALE addresses the overhead of Adam by identifying two simple yet highly efficient techniques: column-wise gradient normalization and the application of first-order momentum only to the last-layer weights.20 By normalizing gradients along the output dimension, SCALE boosts the performance of simple SGD to match or exceed that of Adam while using only 35-45% of the total memory.20 For the FerrisRes autonomous learner, this reduces the optimizer state to approximately 12 MB, fitting comfortably within the limited resources of edge devices.11

### **AdaMeM: Low-Rank Momentum for High-Capability Hardware**

For devices with capable consumer GPUs, the system employs AdaMeM (Memory Efficient Momentum for Adafactor).21 AdaMeM improves upon the Adafactor optimizer—which typically lacks momentum to save space—by maintaining momentum within a low-rank subspace of the weights.21

The AdaMeM update process involves several key steps to ensure stability and efficiency 23:

1. **Gradient Projection**: Gradients are projected into a compact space using a projection matrix ![][image12], which is updated every 200 steps via singular value decomposition (SVD).23  
2. **Subspace Momentum**: Momentum is updated and preconditioned only for the projected, low-rank component of the gradient.22  
3. **Orthogonal Residuals**: The portion of the gradient outside the compact space is processed using a "One Sided Adafactor Preconditioner" without momentum.23 This ensures that the low-rank and residual updates remain orthogonal, preventing suboptimal convergence.23

This dual-track approach allows FerrisRes to achieve Adam-like convergence speeds while maintaining a fraction of the memory cost.22 The DeviceProfile Routing mechanism automatically selects between SCALE and AdaMeM based on the hardware's VRAM and compute profile, ensuring optimal training efficiency regardless of the deployment target.11

### **Multi-Objective Distillation and Stability**

Training the MoE student requires a sophisticated distillation loss that captures the nuances of the teacher's internal logic.24 The phase9 branch utilizes Multi-Objective Distillation, combining KL divergence on the output logits with MSE loss on the hidden states.11 This ensures that the student replicates not only the teacher's final predictions but also its intermediate reasoning patterns, which is essential for the effectiveness of the inter-block attention mechanism.8

To stabilize the MoE training, a MoE Load Balance Loss is implemented to prevent "router collapse," a state where the gating network sends all tokens to a single expert, rendering the other experts useless.26 This is further refined by the Straight-Through Estimator (STE), which allows the model to train LoRA/QLoRA adapters in high precision (FP32) while the base weights remain in their 1.58-bit ternary form during the forward pass.11 This "quantization-on-the-fly" ensures that the training process remains memory-feasible without sacrificing the learned performance of the adapters.28

| Optimizer Profile | State Memory | Hardware Target | Core Advantage |
| :---- | :---- | :---- | :---- |
| Adam | 182 MB | Data Center GPUs | Standard Benchmark |
| SCALE | 12 MB | Raspberry Pi / Mobile | Column-Normalization 20 |
| AdaMeM | 182 MB (Low-Rank) | Consumer GPUs (RTX) | Orthogonal Subspace Momentum 22 |
| DeviceProfile | 0 MB (Logic) | Heterogeneous | Automatic Optimization Selection 11 |

## **Quantization and Bit-Level Compression**

The FerrisRes project achieves its significant memory reduction through 1.58-bit ternary weights, an evolution of the BitNet architecture.29 Ternary quantization constrains weight values to the set ![][image13], which theoretically provides log2(3) ![][image14] 1.58 bits of information density per parameter.28

### **The Computational Advantage of Ternary Weights**

The primary innovation of BitNet b1.58 is the replacement of floating-point multiplications with simple integer additions and scaling.28 In the forward pass, a BitLinear layer computes:

![][image15]  
where ![][image16] is a scaling factor and ![][image17] represents 8-bit quantized activations.28 This transformation eliminates the bulk of the power-intensive floating-point operations (FLOPs), leading to models that are 5x smaller and up to 3x faster than their BF16 equivalents.11

Phase 9 integrates this with 2:4 semi-structured sparsity, a framework known as Sparse-BitNet.29 In a 2:4 pattern, at most two elements are non-zero out of every four consecutive weights.28 This semi-structured sparsity halves the compute requirements again and is particularly well-supported by modern GPU tensor cores.28 Research indicates that 1.58-bit BitNet is naturally "sparsity-friendly," exhibiting significantly smaller performance degradation than full-precision models at the same sparsity levels.28 This resilience allows the FerrisRes model to maintain high accuracy even when operating at an effective bit-width of approximately 1 bit per weight.11

### **TurboQuant and Active Memory Compression**

While weight quantization addresses the "cold storage" problem, the "hot path" bottleneck in LLM inference is the growth of the Key-Value (KV) cache.32 During generation, the KV cache grows dynamically with context length and batch size, often consuming more memory than the model weights in production workloads.32

FerrisRes addresses this with 3-bit TurboQuant, a vector quantization algorithm that achieves a 6x reduction in KV cache memory with zero measurable accuracy loss.32 TurboQuant operates in two stages:

1. **Random Orthogonal Rotation**: Input vectors are rotated using a fixed random orthogonal matrix ![][image18].35 This ensures that the coordinates follow a predictable Beta distribution.35  
2. **Lloyd-Max Codebook Quantization**: The rotated coordinates are mapped to indices in a 3-bit codebook that is MSE-optimal for the Beta distribution.35

This "training-free and data-oblivious" method allows the KV cache for a 100K-token workload to drop from 6 GB to roughly 1 GB, enabling long-context reasoning on consumer hardware like the Mac Mini M4 Pro.34

| Quantization Tier | Target Component | Precision | Rationale |
| :---- | :---- | :---- | :---- |
| BitNet b1.58 | FFN/Experts | 1.58-bit Ternary | Multiplier-free compute 29 |
| Sparse Ternary | Linear Layers | 2:4 Sparsity | 50% compute reduction 28 |
| NF4 Embeddings | Input/Output | 4-bit NF4 | Preservation of high-variance info 38 |
| TurboQuant | KV Cache | 3-bit Vector | 6x cache reduction 32 |
| Normalization | LayerNorm/RMSNorm | BF16 | Stability of distribution 11 |

The project utilizes Tiered Quantization to ensure that different parts of the model are compressed according to their sensitivity to information loss.11 While ternary weights are sufficient for the massive FFN blocks, NF4 (NormalFloat 4-bit) is used for embeddings and the language model head, where ternary quantization would destroy critical semantic information.11

## **Memory Management and I/O Orchestration**

The high memory requirements of MoE models, even when quantized, necessitate sophisticated I/O strategies when deploying on commodity hardware with limited VRAM.39 FerrisRes employs Expert Memory-Mapped (mmap) Loading, where each expert is stored in a separate file and only paged into VRAM when selected by the router.11

### **Prefetching and BuddyMoE Strategies**

To hide the latency introduced by paging experts from host storage to GPU memory over the PCIe bus, the system utilizes predictive prefetching.40 Heuristics identify upcoming expert requirements by analyzing currently computed internal representations—specifically the "quasi-hidden state" from the previous layer's post-attention residual.41 This allows the system to speculatively load the required experts in the background, overlapping I/O with current layer computation.41

In instances where prefetching fails—a "prefetch miss"—the system employs BuddyMoE.43 BuddyMoE identifies pairs of functional "buddy" experts by analyzing co-activation patterns during an offline calibration phase.43 When an expert is missing from VRAM, the system dynamically substitutes a functionally similar buddy expert that is already resident.43 This trades a marginal amount of accuracy for a significant reduction in tail latency, as it avoids a synchronous I/O stall at the critical path of inference.42

| I/O Scenario | Action Taken | Latency Impact | Accuracy Impact |
| :---- | :---- | :---- | :---- |
| Prefetch Hit | Background Load | \~0 ms | None |
| Prefetch Miss (Native) | Synchronous Load | 9-10 ms stall | None |
| BuddyMoE Hit | Expert Substitution | \~0 ms | Minimal/Imperceptible |
| BuddyMoE Miss | Sync Load/Substitution | \~0 ms | Marginal Loss 43 |

The Block Summary KV Compression further optimizes I/O by compressing older context into fixed-size block representations (\~3.5 MB total).11 This ensures that the context size does not grow indefinitely, maintaining a constant memory footprint for very long sequences.44

## **Inference Engine and GPU Optimization**

The phase9-autonomous-learner branch targets cross-platform deployment through WGSL GPU Kernels.11 By utilizing WebGPU, FerrisRes can execute high-performance inference on any platform supporting modern graphics APIs (D3D12, Metal, Vulkan), ranging from mobile devices to high-end workstations.45

### **WGSL Acceleration and DeviceProfile Dispatch**

The WGSL backend is optimized specifically for the non-square matrix configurations and subgroup operations required by sparse MoE models.45 Key optimizations include:

* **Vectorized FlashAttention**: A split-pipeline implementation for FLASH\_ATTN\_EXT that increases efficiency for long sequences.45  
* **LDS Load Optimization**: Restructuring activation loading in mmq (matrix-matrix-quantized) kernels to replace scalar operations with vectorized 128-bit loads.45  
* **Parameter Arena**: Replacing a pool of small parameter buffers with a single asynchronous arena, cycling through offset slots to reduce GPU command overhead.45

The DeviceProfile Dispatch system serves as a central orchestrator, routing compute decisions through the hardware's specific capabilities—such as the presence of subgroup matrix features on Intel GPUs or the availability of high-bandwidth memory (HBM) on NVIDIA hardware.45 This ensure that the model always uses the most efficient kernel and quantization format for the local environment.48

Context extension is handled by YaRN (Yet another RoPE extensioN), which allows the model to handle sequences 4x longer than its training length.11 YaRN scales the Rotary Position Embeddings (RoPE) frequencies selectively, preserving local patterns while interpolating global context, which is essential for maintaining coherence in the 256K-token windows supported by the larger Gemma 4 models.16

## **Autonomous Learning: The Self-Evolution Loop**

The "autonomous learner" concept represents the integration of architectural efficiency with a self-directed learning objective.4 An autonomous learner is defined as software that improves through direct experience, guided by intrinsic curiosity rather than human-labeled data.4

### **Intrinsic Rewards and Skill Libraries**

To achieve autonomy, FerrisRes implements a self-rewarding reinforcement learning (RL) framework.50 The model evaluates its own intermediate reasoning states to determine the quality of a trajectory.51 Correct responses often exhibit "high consistency"—where reasoning states converge toward the final answer—and "low volatility"—where they deviate minimally toward other candidates.51 This CoVo (Consistency and Volatility) mechanism allows the model to assign itself rewards, enabling a scalable pathway for learning without external supervision.51

| Level of Autonomy | Human Role | Agent Role | FerrisRes Target |
| :---- | :---- | :---- | :---- |
| Level 1 | Supervisor | Assistant | Basic Automation |
| Level 2 | Operator | Partner | Intelligent Interaction |
| Level 3 | Manager | Specialist | Autonomous Problem Solving |
| Level 4 | Strategic Lead | Autonomous Learner | Self-Directed Evolution 4 |

The system also builds and maintains a Skill Library (SkillKB), which distills raw experience into a three-tiered hierarchy of strategic plans, functional skills, and atomic skills.52 This hierarchical experience representation allows the model to transfer "reusable wisdom" between tasks, improving execution efficiency over time.52 For instance, if an agent fails a task, it can trigger an evolution phase using Curriculum Learning (CL) or Genetic Algorithms (GA) to generate new tools and refine its reasoning.52

### **Active Learning and Personalized Training**

The on-device active learning framework (FDAL) ensures that the model selectively queries or trains on the most informative samples.24 A task-aware sampler network identifies instances that represent high uncertainty or distribution drift, prioritizing them for annotation or self-supervised updates.24 This joint optimization strategy ensures that the limited compute and energy budget of an edge device is focused on the data most likely to improve model utility.24

The autonomous learner thus shifts from a developer-owned, static model to a user-centric, adaptable intelligence.54 It evolves in response to human input and environment interaction, fostering a dynamic relationship where the model's parameters and skills remain in a state of persistent, real-time update.54 This localized intelligence preserves privacy, reduces latency, and enhances operational robustness in environments with limited cloud connectivity.6

## **Implementation Strategy and Actionable Trajectory**

The improvements planned for the FerrisRes phase9-autonomous-learner branch are not isolated features but an integrated stack designed for symbiotic efficiency. The transition from additive residuals to Block AttnRes provides the structural stability required for deep specialization, while the 1.58-bit ternary quantization and Sparse-BitNet provide the computational room for this complexity on edge hardware.

### **Synergy of Architectures**

The architectural synergy can be summarized as follows:

* **Structural Duality**: The model uses AttnRes to choose what to "remember" from depth and inter-block attention to ground those memories.10  
* **Compute-Memory Balancing**: The use of PLE on flash and MoE experts on mmap files allows for a model with tens of billions of parameters to operate in a memory space of less than 2 GB.3  
* **Self-Improving Optimizers**: The DeviceProfile routing ensures that whether the device is a mobile phone using SCALE or a workstation using AdaMeM, the autonomous learning loop remains active and efficient.11

The future outlook for this framework involves "Elastic Inference," where the model can dynamically switch between different active parameter counts (e.g., E2B vs E4B paths) based on real-time device load and task complexity.56 This maturation from static inference to a bustling, adaptive "city of agents" marks the beginning of the agentic era, where localized intelligence is defined by its capacity to "feel the bruise" of reality and understand its environment through direct experience.4

#### **Works cited**

1. shift/FerrisRes: Rust-native AI inference & training engine ... \- GitHub, accessed on April 20, 2026, [https://github.com/shift/FerrisRes](https://github.com/shift/FerrisRes)  
2. autodiff · GitHub Topics · GitHub, accessed on April 20, 2026, [https://github.com/topics/autodiff?l=rust\&o=asc\&s=forks](https://github.com/topics/autodiff?l=rust&o=asc&s=forks)  
3. Google's Gemma 4 Is the Most Architecturally Interesting Open Model Released This Year. Here's the Full Breakdown. | by Ari Vance \- Towards AI, accessed on April 20, 2026, [https://pub.towardsai.net/googles-gemma-4-is-the-most-architecturally-interesting-open-model-released-this-year-b245a406cd6a](https://pub.towardsai.net/googles-gemma-4-is-the-most-architecturally-interesting-open-model-released-this-year-b245a406cd6a)  
4. Raising AI Agents in Open Worlds — A 7-Minute Primer on Autonomous Learning | by Bran Kop, Engineer @Conformal, Founder of aiHQ | Stackademic, accessed on April 20, 2026, [https://blog.stackademic.com/raising-ai-agents-in-open-worlds-a-7-minute-primer-on-autonomous-learning-a37ddc1a372b](https://blog.stackademic.com/raising-ai-agents-in-open-worlds-a-7-minute-primer-on-autonomous-learning-a37ddc1a372b)  
5. Toward Edge General Intelligence with Multiple-Large Language Model (Multi-LLM): Architecture, Trust, and Orchestration \- arXiv, accessed on April 20, 2026, [https://arxiv.org/html/2507.00672v1](https://arxiv.org/html/2507.00672v1)  
6. on-device small language models for autonomous agent systems: design principles, efficiency, and real-world constraints \- ResearchGate, accessed on April 20, 2026, [https://www.researchgate.net/publication/400095374\_ON-DEVICE\_SMALL\_LANGUAGE\_MODELS\_FOR\_AUTONOMOUS\_AGENT\_SYSTEMS\_DESIGN\_PRINCIPLES\_EFFICIENCY\_AND\_REAL-WORLD\_CONSTRAINTS](https://www.researchgate.net/publication/400095374_ON-DEVICE_SMALL_LANGUAGE_MODELS_FOR_AUTONOMOUS_AGENT_SYSTEMS_DESIGN_PRINCIPLES_EFFICIENCY_AND_REAL-WORLD_CONSTRAINTS)  
7. Attention Residuals by Kimi AI: A Clear Explanation \- Data Science Dojo, accessed on April 20, 2026, [https://datasciencedojo.com/blog/attention-residuals-kimi-ai-explained/](https://datasciencedojo.com/blog/attention-residuals-kimi-ai-explained/)  
8. Attention Residuals \- arXiv, accessed on April 20, 2026, [https://arxiv.org/pdf/2603.15031](https://arxiv.org/pdf/2603.15031)  
9. Attention Residuals Explained: Rethinking Transformer Depth \- DataCamp, accessed on April 20, 2026, [https://www.datacamp.com/blog/attention-residuals-explained](https://www.datacamp.com/blog/attention-residuals-explained)  
10. Attention Residuals, Explained Simply: What If Residual Connections Could Finally Choose What to Remember? | by Varun Nathan \- Towards AI, accessed on April 20, 2026, [https://pub.towardsai.net/attention-residuals-explained-simply-what-if-residual-connections-could-finally-choose-what-to-35df67a22077](https://pub.towardsai.net/attention-residuals-explained-simply-what-if-residual-connections-could-finally-choose-what-to-35df67a22077)  
11. accessed on January 1, 1970, [https://github.com/shift/FerrisRes/tree/feature/phase9-autonomous-learner](https://github.com/shift/FerrisRes/tree/feature/phase9-autonomous-learner)  
12. Scout Before You Attend: Sketch-and-Walk Sparse Attention for Efficient LLM Inference, accessed on April 20, 2026, [https://arxiv.org/html/2602.07397v1](https://arxiv.org/html/2602.07397v1)  
13. Understanding Mixture of Experts (MoE) Neural Networks \- IntuitionLabs, accessed on April 20, 2026, [https://intuitionlabs.ai/articles/mixture-of-experts-moe-models](https://intuitionlabs.ai/articles/mixture-of-experts-moe-models)  
14. Gemma 4: What Computer Vision Engineers Actually Need to Know | Datature Blog, accessed on April 20, 2026, [https://datature.io/blog/gemma-4-what-computer-vision-engineers-actually-need-to-know](https://datature.io/blog/gemma-4-what-computer-vision-engineers-actually-need-to-know)  
15. Per-Layer Embeddings: A simple explanation of the magic behind the small Gemma 4 models : r/LocalLLaMA \- Reddit, accessed on April 20, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1sd5utm/perlayer\_embeddings\_a\_simple\_explanation\_of\_the/](https://www.reddit.com/r/LocalLLaMA/comments/1sd5utm/perlayer_embeddings_a_simple_explanation_of_the/)  
16. Gemma 4 Is Not Just Another Open Model : Here's What's Actually Different Under the Hood, accessed on April 20, 2026, [https://dev523.medium.com/gemma-4-is-not-just-another-open-model-heres-what-s-actually-different-under-the-hood-4e06f265f648](https://dev523.medium.com/gemma-4-is-not-just-another-open-model-heres-what-s-actually-different-under-the-hood-4e06f265f648)  
17. Gemma 3n model overview | Google AI for Developers, accessed on April 20, 2026, [https://ai.google.dev/gemma/docs/gemma-3n](https://ai.google.dev/gemma/docs/gemma-3n)  
18. STEM: Scaling Transformers with Embedding Modules \- arXiv, accessed on April 20, 2026, [https://arxiv.org/html/2601.10639](https://arxiv.org/html/2601.10639)  
19. On-Device Collaborative Language Modeling via a Mixture of Generalists and Specialists, accessed on April 20, 2026, [https://icml.cc/virtual/2025/poster/45925](https://icml.cc/virtual/2025/poster/45925)  
20. Memory-Efficient LLM Pretraining via Minimalist Optimizer Design | OpenReview, accessed on April 20, 2026, [https://openreview.net/forum?id=gyXfJUcR72](https://openreview.net/forum?id=gyXfJUcR72)  
21. AdaMeM: Memory Efficient Momentum for Adafactor \- ICML 2026, accessed on April 20, 2026, [https://icml.cc/virtual/2024/37197](https://icml.cc/virtual/2024/37197)  
22. AdaMeM: Memory Efficient Momentum for Adafactor \- OpenReview, accessed on April 20, 2026, [https://openreview.net/forum?id=fZqMVTz7K5](https://openreview.net/forum?id=fZqMVTz7K5)  
23. Memory Efficient Momentum for Adafactor \- AdaMeM \- OpenReview, accessed on April 20, 2026, [https://openreview.net/pdf?id=fZqMVTz7K5](https://openreview.net/pdf?id=fZqMVTz7K5)  
24. FDAL: Leveraging Feature Distillation for Efficient and Task-Aware Active Learning \- CVF Open Access, accessed on April 20, 2026, [https://openaccess.thecvf.com/content/ICCV2025W/ECLR/papers/Gaire\_FDAL\_Leveraging\_Feature\_Distillation\_for\_Efficient\_and\_Task-Aware\_Active\_Learning\_ICCVW\_2025\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2025W/ECLR/papers/Gaire_FDAL_Leveraging_Feature_Distillation_for_Efficient_and_Task-Aware_Active_Learning_ICCVW_2025_paper.pdf)  
25. Model Distillation: Teacher-Student Training Guide \- Label Your Data, accessed on April 20, 2026, [https://labelyourdata.com/articles/machine-learning/model-distillation](https://labelyourdata.com/articles/machine-learning/model-distillation)  
26. Scalable Training of Mixture-of-Experts Models with Megatron Core Technical Report \- arXiv, accessed on April 20, 2026, [https://arxiv.org/html/2603.07685v2](https://arxiv.org/html/2603.07685v2)  
27. Grouter: Decoupling Routing from Representation for Accelerated MoE Training \- arXiv, accessed on April 20, 2026, [https://arxiv.org/html/2603.06626v1](https://arxiv.org/html/2603.06626v1)  
28. Sparse-BitNet: 1.58-bit LLMs are Naturally Friendly to Semi-Structured Sparsity, accessed on April 20, 2026, [https://www.researchgate.net/publication/401600725\_Sparse-BitNet\_158-bit\_LLMs\_are\_Naturally\_Friendly\_to\_Semi-Structured\_Sparsity](https://www.researchgate.net/publication/401600725_Sparse-BitNet_158-bit_LLMs_are_Naturally_Friendly_to_Semi-Structured_Sparsity)  
29. Sparse-BitNet: 1.58-bit LLMs are Naturally Friendly to Semi-Structured Sparsity \- arXiv, accessed on April 20, 2026, [https://arxiv.org/pdf/2603.05168](https://arxiv.org/pdf/2603.05168)  
30. Bitnet.cpp: Efficient Edge Inference for Ternary LLMs \- arXiv, accessed on April 20, 2026, [https://arxiv.org/html/2502.11880v1](https://arxiv.org/html/2502.11880v1)  
31. Sparse-BitNet: 1.58-bit LLMs are Naturally Friendly to Semi-Structured Sparsity \- arXiv, accessed on April 20, 2026, [https://arxiv.org/html/2603.05168v1](https://arxiv.org/html/2603.05168v1)  
32. Google TurboQuant: 2026 LLM Compression Guide | Articles \- O-mega.ai, accessed on April 20, 2026, [https://o-mega.ai/articles/google-turboquant-the-2026-llm-compression-guide](https://o-mega.ai/articles/google-turboquant-the-2026-llm-compression-guide)  
33. Has anyone implemented Google's TurboQuant paper yet? : r/LocalLLaMA \- Reddit, accessed on April 20, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1s3ffzo/has\_anyone\_implemented\_googles\_turboquant\_paper/](https://www.reddit.com/r/LocalLLaMA/comments/1s3ffzo/has_anyone_implemented_googles_turboquant_paper/)  
34. What Is Google TurboQuant? The KV Cache Compression That Crashed Memory Chip Stocks | MindStudio, accessed on April 20, 2026, [https://www.mindstudio.ai/blog/what-is-google-turboquant-kv-cache-compression](https://www.mindstudio.ai/blog/what-is-google-turboquant-kv-cache-compression)  
35. TurboQuant KV Cache Compression — Working Implementation Ready for Review · Issue \#1509 · ikawrakow/ik\_llama.cpp \- GitHub, accessed on April 20, 2026, [https://github.com/ikawrakow/ik\_llama.cpp/issues/1509](https://github.com/ikawrakow/ik_llama.cpp/issues/1509)  
36. TurboQuant: Near-optimal KV cache quantization for LLM inference (3-bit keys, 2-bit values) with Triton kernels \+ vLLM integration \- GitHub, accessed on April 20, 2026, [https://github.com/0xSero/turboquant](https://github.com/0xSero/turboquant)  
37. TurboQuant Changes the Economics of Local AI Inference \- Medium, accessed on April 20, 2026, [https://medium.com/@michael.hannecke/googles-turboquant-changes-the-economics-of-local-ai-inference-acce5839014d](https://medium.com/@michael.hannecke/googles-turboquant-changes-the-economics-of-local-ai-inference-acce5839014d)  
38. MoEITS: A Green AI approach for simplifying MoE-LLMs \- arXiv, accessed on April 20, 2026, [https://arxiv.org/html/2604.10603v1](https://arxiv.org/html/2604.10603v1)  
39. Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading \- Hao Wang, accessed on April 20, 2026, [http://www.wanghao.in/paper/EuroSys26\_FineMoE.pdf](http://www.wanghao.in/paper/EuroSys26_FineMoE.pdf)  
40. PreScope: Unleashing the Power of Prefetching for Resource-Constrained MoE Inference, accessed on April 20, 2026, [https://arxiv.org/html/2509.23638v1](https://arxiv.org/html/2509.23638v1)  
41. Speculating Experts Accelerates Inference for Mixture-of-Experts \- arXiv, accessed on April 20, 2026, [https://arxiv.org/html/2603.19289v1](https://arxiv.org/html/2603.19289v1)  
42. \[Literature Review\] MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts \- Moonlight, accessed on April 20, 2026, [https://www.themoonlight.io/en/review/moe-speq-speculative-quantized-decoding-with-proactive-expert-prefetching-and-offloading-for-mixture-of-experts](https://www.themoonlight.io/en/review/moe-speq-speculative-quantized-decoding-with-proactive-expert-prefetching-and-offloading-for-mixture-of-experts)  
43. BuddyMoE: Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference \- arXiv, accessed on April 20, 2026, [https://arxiv.org/html/2511.10054v1](https://arxiv.org/html/2511.10054v1)  
44. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models, accessed on April 20, 2026, [https://www.researchgate.net/publication/401463261\_H2O\_Heavy-Hitter\_Oracle\_for\_Efficient\_Generative\_Inference\_of\_Large\_Language\_Models](https://www.researchgate.net/publication/401463261_H2O_Heavy-Hitter_Oracle_for_Efficient_Generative_Inference_of_Large_Language_Models)  
45. llama-cpp-pydist \- PyPI, accessed on April 20, 2026, [https://pypi.org/project/llama-cpp-pydist/](https://pypi.org/project/llama-cpp-pydist/)  
46. GPU Web 2024 10 F2F \- GitHub, accessed on April 20, 2026, [https://github.com/gpuweb/gpuweb/wiki/GPU-Web-2024-10-F2F](https://github.com/gpuweb/gpuweb/wiki/GPU-Web-2024-10-F2F)  
47. Optimize Gemma 3 Inference: vLLM on GKE 🏎️ \- Medium, accessed on April 20, 2026, [https://medium.com/google-cloud/optimize-gemma-3-inference-vllm-on-gke-c071a08f7c78](https://medium.com/google-cloud/optimize-gemma-3-inference-vllm-on-gke-c071a08f7c78)  
48. Edge intelligence unleashed: a survey on deploying large language models in resource-constrained environments \- Academy of Cognitive and Natural Sciences, accessed on April 20, 2026, [https://acnsci.org/journal/index.php/jec/article/download/1000/932/4487](https://acnsci.org/journal/index.php/jec/article/download/1000/932/4487)  
49. Gemma 4 model card | Google AI for Developers, accessed on April 20, 2026, [https://ai.google.dev/gemma/docs/core/model\_card\_4](https://ai.google.dev/gemma/docs/core/model_card_4)  
50. Daily Papers \- Hugging Face, accessed on April 20, 2026, [https://huggingface.co/papers?q=Skill-integrated%20Reward](https://huggingface.co/papers?q=Skill-integrated+Reward)  
51. Consistent Paths Lead to Truth: Self-Rewarding Reinforcement Learning for LLM Reasoning, accessed on April 20, 2026, [https://neurips.cc/virtual/2025/poster/117063](https://neurips.cc/virtual/2025/poster/117063)  
52. Daily Papers \- Hugging Face, accessed on April 20, 2026, [https://huggingface.co/papers?q=self-evolving%20paradigms](https://huggingface.co/papers?q=self-evolving+paradigms)  
53. Towards Reliable and Trustworthy Pipelines for MLOps and LLMOps \- PolyPublie, accessed on April 20, 2026, [https://publications.polymtl.ca/64774/1/2025\_AbbassiAltafAllah.pdf](https://publications.polymtl.ca/64774/1/2025_AbbassiAltafAllah.pdf)  
54. Online Training of Large Language Models: Learn while Chatting \- arXiv, accessed on April 20, 2026, [https://arxiv.org/html/2403.04790v1](https://arxiv.org/html/2403.04790v1)  
55. The Pluralistic Future of AI: A Comprehensive Analysis of Decentralized Large Language Models \- Preprints.org, accessed on April 20, 2026, [https://www.preprints.org/manuscript/202509.0756](https://www.preprints.org/manuscript/202509.0756)  
56. Gemma 3N Model Architecture: A Complete Technical Deep Dive \- Nageswarara Rao Vutla, accessed on April 20, 2026, [https://nageswararaovutla7.medium.com/gemma-3n-model-architecture-a-complete-technical-deep-dive-d20a3c85bc50](https://nageswararaovutla7.medium.com/gemma-3n-model-architecture-a-complete-technical-deep-dive-d20a3c85bc50)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAYAAAAcCAYAAABCgc61AAAAfElEQVR4XmNgoBkQA2JNdEF1IP4PxN3oEiAAkghBFwQBkIQAuiATEJ9HFwQBUyCeii4IAvlAHIMuCAIrgFgRXVCVAWIxBkhgwCExkwGHxCUgfgPE+kCsAxMUYoCo3gDEi4DYBSYhApUA+eEGELPCJEBgLRC/AuJIZMGRCQB7UxTSXRyQRAAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKAAAAAaCAYAAAAwnlc+AAAE70lEQVR4Xu2aachtUxjHH0N0M5Z7zbf3uiG5Zpk+yEEZMpXhhkuKUMqsKPFFmUKGEknKkPiCDCkf7kuIzMosPsgQColC4vlZe92zzv/s9+yz9znv3dvb+tW/+57/s849z17POmuvtfYxy2QymUyfLV0r1OwgM65lai5w9nVd6NpOAwWLXR+59tKAtVvXq13rqKmsdv3k+sf1psS6xEGuty3keZvEFjLnuz533e/6TmKwoesV1znib2qhtm3WlcF3vZpKz3WHhUTvGgx1ii0s5EqeKwdDC5o/XSe5PrZw7cqjrhfVdNazfn+1Wdc/XIeqqZAgiZ6ugQ5Cnlur2SG4HR6m5gRwvVtZGGQvSQyIH6VmQtt1ZeZ+y7WuBlLet5DoUg10DAYea50uw+B7RM2GULQf1BReU0Nou647uf5ynaWByGkWkry7eL2/hXVFFyFH8qUwO7v2Hgx3giNtegPwPNetaiZc6jpAzQT6KtZ1ewu1bQPW7AzCjTUA91gYgFdZWLTeaWFTcl/aqCN86Frl+sL1hOtxC4v0LsHt8GE1a8Da7UcLNVHtmrQDPmd98VKobazrGxZq20ZdT7WQ/3EagLi4/cV1cuE9U3i7xEY1OMJ1TQ1dFt5WCUcQ5PSO9Wc+Ov+rNS26wdE2vRnwOdc+aia8roZAbWNd43FI07pOwn4WPvdKDQABpkcGTuSBwj8k8Z620K6Kw214kI3S5eFtlTDzkdNS8fHSs6aY52OJV4cz1KjJNAfgNzZ6OfSlGgnxC5vWFeaqa9P+OtPCkc8oWAfyuTdqAAickryOic8mHrABwG8LNh/6+Vz8deLFPA8WvwpubwdavU3ORa4XRJy5fV3iI5YN43K8DV9vCrfqUfF7bThObWfFa9pf21oYyLxXJwWFtR/tHtIA/O3aPHnNjETjKxIPmBl+E29tETuJdUzKkza8sI55jlobjeITNWoyrRnwWhseQMr3aiRwHdQ2hdrOVdem/UWOM2oKDFDaxQ3RGvhGvCoeaywaLy9eLyr+5Rv1bPH3KG628P5x9a1VnBFZ//Z7U+LxiIlDWtjM+puRcfOci64MQGZLrnkUH6hREO9iZbWNdWX2praT9hefM6OmsKeFdjdogNGvJg155AV7JD63JnZUbRBvJxQ3ckHhAc9Jby/+njTPOrfgMqY1ANlc6QymlD0BAeqqBd+t8IC6Mhhh0v7i/6y6BfcstNPHhf/tovRs5j3Xu67nXZcUHhfE0UzVTDUfMLuR/Kz4nAP+7HrZwgVGNM9ehZRP1ajJNAbgMRauueqZ97FWfrRBXX+3wdpuYKG2aV1B+4t1cG+EFPJcpqbAcRHPs4coG7kUvOfaMfE4T2K9RawNdrfyX4Iw9fPoKyXmyS9ImvCZGjWZxgDkdIDC9sRXNnLdoqaFupb9AoZ+SesKk/YXee6gpsBaVdeetWC65gznbA10kJinHj+MS+k3tQac2+myZlwYHMxUT1nYSVf+lMlCO97TlEn7iwG4XM0EDuY5i9xEA3VgDcaiuHQb3TFinuMULyUeqdChq10PDobnnSWuXy3Mejz/vXggOjcsQ85VswZN+6tn/f5iz8DfbAwV+nIqv8bh9HySb9ra4v+Sp7KNhU0HRyXsXjnnG4cTLRxY84uZJsx3f7Gm1mVSZgHCScAJarYMvzPNZDKZTCaTyWQymfb5FwaPKm/nx21TAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAABNCAYAAAAb+jifAAAFjUlEQVR4Xu3dW4j1VRkH4NezUoKoeCDJBBVPmdKFXZh+iXZARY0UxLIQz6bhTamgKCoeSEUsTVHxHCLkEQy8sUgMRBGVSjoIlhgpaHlVIrre1t7u/6yZ75t9mL1nvpnngR+z11p79hyuXtZa/7UiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFaNl0s+bjsX8eW2AwCA6bmg5O22cwMeLXm/7QQAYHreKzmv7VzEC20HAADTk8uh+7Wdi1CwAQDM0IOd19uXrFtPuhRsAAAzsmnJmZ32LjG/UOunS8EGADAjp5ZsUvJQO7AIBRsAwIwcVHJ3yZHtwAY8U/Jh7+tPmjEAAAAAAAAAAAAAAAAAAKbpwpJLlygAAEzBIVGvpMoc2oxtyDYlp0U9CqT//ZfMeQcAAEumX3D9tWS7ZmwYZ5f8oeSNdgAAgKXx+ZL3YlC4jeveksfaTgCAjdGWJV+Leh3USnFSDAq2/ZuxUTzfdgAAbKx+FZPNZk3Dn2JQtN3ajAEArDn/iJVXsB1c8t+YfGkUAGBVyILop23nCrBJDAq235dsPncYAGDtyILoWyWnl/yoGVtub8WgaLuiGZvUnlGP/zi+HVghcm/huZ3250q+22kDAGvEXlGLoQd77U1L/jkY/r+tS/Zu+rruKHlmiIxbcJ0Tg6LtM83YOLYq+ajke712Pnjxl6iffXvJ0b3+SW3bdjS+GuufNfx2yRYlF5d8p9f3m7A8DABrUh44m4VAFmV9WczkcmTff0oe6LRbB5SsGyJZHI6rX7DlAxJZVE4iH2JoZxLzs88s+WPMLaJui3ow7yiOKnm5ZN92oNH/exaSS8Dp2agzgSm/dgu2nBE9sdMGAFape0q+2WnvVvJSp53eKTmv6Zu1N2NQtP24GRvFEVE/I2fVurLvzpJrm/48wLdbzA7rgrZjAfkzf9t29pxfsnvMn1H7W+d1Lhef1WkDAKvQF2J+QZAzPlnUdP2raS+XG0ueajtH9EbM/5tTvxgcxz0xd0YyZwDz8N9J/S7m/k55i8PPO20AYA04JeYWBDtGPUpjs5h7LVR/f9v6/CLm71dbKJM8ibpT1CunPtsOjCiLz4UKs+x7sdP+UsmVMdydpseUnNBpHxQL/4xR5We822nnQcBfiVocXhT1NgcAYJXLAujDTvuqqEVCbpZ/pdeXxcd+n75jeeSMVRYu+7QDY/h61D16fXuU3Ffy96jLrhdGvV0h96Hl393fP3Z4zN+T100+qHBd1OXTR2LxIncYL8SgYMvfY4fe6/xZ+UDC/b02ALCKtXuosiB6v+S5qEdKpGH2Yk3bDSWHtZ0TyCcwHy+5LGpxdWzJcSUfRC2S+nvWru59TesWSe4DzPfn9+ZyaD7AMKk8PPjVklui7iPsyhnQbzR9AMAa9XrJD2L5ioOcAcyz0mYtj9z4X9SnY3dpxlq5HNo9yy2XXbt72qbh1yUHllzfDgAAa0/uT8ult2kXIAv5YtQnN5dDFml/juEOqm3/P/l9d3fa03B51GXXSff0AQCM7bWoy5PjyiXdvCMVAIApyZsHdm07R/DvsFwIADAVec3VOEdi5PLpyTE4T617RAcAAEsoi608MPbSIfN01CM5+oVaP3lTAAAAS+yHUa+GygN2fxb1Ivb7om6uzxsYnox6AG8Wadl+OOrhsXf13n9TyTVRD7wFAAAAAAAAAAAAAABWqS2jXqY+ip3bDgAAVpYb2w4AAFYWBRsAwIxcFbX4eqLXvjnq+WsLpeuGpg0AwJScEPXi9jPagUWYYQMAmKFflmzTe/39mH8lVT9dCjYAgBnZouT4GP0+UAUbAMAM7dt2AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAbg08AOeUGL0npqB8AAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACQAAAAaCAYAAADfcP5FAAABc0lEQVR4Xu2VTSsFYRTH/15S3spCXm6Wyk4kO0VWNnwC6W5Y+gBSIhayULIhG8LGElF21+Jma2NjjWwoWViI/+k8t3kckztzMavnV7+ac87cmTNzn/MMEAgEAqii/bTeFrKmmq7QB3pM7+kebXL1TlrjjjNBmrijfS4eoB9038XXtMsdZ8I7omZKFKBN5eiFqU3QXpNLwzx9oY+2IFzSaZsk69CGtmitqfWYuBJu6JxNdkBv2m0L0DUltTdbSEmzTSC675AtyGuXQhyL0NqGl5MpXKO7Xi4J7SaepK/4/ubRCP0vLTJ1h9CGlrz8OJ2F/kaaS8oO9JolZBmcevEX5MRnuk2XaZEOu5o080Q3XSy00DEvbqUjZczTWzoFRR500B3H0gA9QRarfY0yZf7TzUD3JJm0pJzRNi+Wh5Rrjnq5ijmHrr1VW/gB2f19TqCDJIPzaxboAaJdvBxxU3ZFj6BfgD8hzWekziagY28nLxAI/DufCjw5QrfyNe8AAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAbCAYAAACeA7ShAAABHUlEQVR4Xu2QP0uCURTGT9mfTyAN0tTiFA6F0SR9hoagLQKF+gIh0dTg4iS0CBYIRQ3NETQ26ZSbW0g1SktEQz1P976Xxys5ODS9P/jBec55Pd57zVJS/o8cvIDvcFv6NdiDsz4f+N5EbuAS/IZt6X/AF8mn8E7yGJvwEu6aW1aXGfO15Ay88vUMfDD3TYDL1uCtHxRlxsyrKVWpef2h5MAn7Ec9LitEvS2pORs5WQKb+rhz8FVy0puXfAifJQe4bEMyT9CUTB6j/Ab3ot4vXFbxNR/6HnbMnYbswIavSd7cb1akF+AVB/DE3JIz+AW7sAWP4UL42mzf/rhiwiJc95IsLJk7Rcy5uT9ZjvpT8QTL8CgeTMuqufdNSZnEDwRrMxZc90w4AAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAaCAYAAAD8K6+QAAADD0lEQVR4Xu2XWaiNURTHlylDKIQMD1eGpCjz3L1mUgoPbpLiQVGU8iLTzfBgyJAyFEoejIWEB3IfJFFShhTqShSR4ZES/7+1v3PWWWd/55zOde6Dzq/+3bP/a+1v2N/ee+0rUuX/YwRU580U5kCPoA4+8A8ZGf5ug07bQKmw02voErQFeggtzcnI5x00yJuB7dAtp5tQa5uUAnOuQx+gJuPdyWSUQA10HloHdTH+cug3dFTiD9MV2uhNwzDosug13kJroTE5GYWphRqgU8YbDE0y7VTGQd+g5z4QOCH6YEt8AOyGOnvTcU+0P2dAOWyCVjrvvmvnMQD6CH0Pv2Nw1PhgT53fCfrivBg/RftP9YESuQ0Ncd4viQ90Bt6Qc7itDxj6ieZR041/TUp7MfZr9GYBOM04XduJbkzs7zkM/ZACGxY77fSmgzdJXmyG8T9DV0w7jVLukbBQ9AtxTb2Azkr8xZK1bwc6w0TRYLGFuEayL9YneD1Ce3+SlAJHlHmzfSDCCtHcVqG9IbRjLzZF1Oez5bEH+iqFpyG5IHoRTtmEocHbarwY00TXGNdjIVg3eb3HxksGr8l4CcNFY5t9gNyFrnrTsU/0AlxL3Y0/M/irjRejUbSWxRglOpVrJPtl+pr4juD5HZH0FI0d8wFyHLrhTQd3S16A9c2SrLtCNYxlgAt8lg8EuI4WBfFaz3LDfwsx/Vjx5y7J2F4fIOuhV940cIqy8znJzvsElgbGWMfSmCeaYwt+wmjoveg9OGjMO2ninLoclKbQrjUxMkG0D2tcHvzsDNY5n/CUcUb0CNTRxQhPHOzLr54G1zBzPAOhN9DB0OZOyLxdSQJYHLzkXLjMxMhc0Xi98zNw2/wkOnoNQS+hB9D8TFacA6Kj6uENi4nnUTsL6kSfg1PrItRf9EX5LGx76D3xpqe36DznOY4jMzk3nEpa8SyX9tBY5/WS+NmSu3mxjats2kj+Maul4C5drIQ0iwWi23VLwzJUcfj/ld81K8kqqJs3KwHr1RHJHrcqySFovDerVKlSpVn8AagAsDRA8RZcAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAaCAYAAADmF08eAAADVElEQVR4Xu2XWaiNURTHl3nIEMoslyg8yIMID/caygNJSsZ0UxJFKS/IGIlEHpCizFMeSJHkhZIxD3gwPMgQIkOeJPH/23ufb5317X3OTfee++D86t/5vv/6xv3ttfY6IlX+P0ZCddZMMBV6ALW3gSZkKNTb67mJNYij0AvoPLQeugvNLzoizxtoiDU9W6BrSmOKwzJWxYJKMRt6DP2GFntvItS/cEQZaqCz0Eqos/IXibvoAail8gNdoDXWVIyATou7BnWuOCxtoCs+xnvMKA7nGCBupvH4wcq/qraTcJS/Qk9swHNI3IXn2ADYAXWypuEmdFyyl7XUQ5etWYJ20Evj/RKXQkkGQR+gb347Rq24B3xk/I7QZ+PF4CBylvA+sRfll1xlzRJMgo4Z76S452tl/AK88XuotQ0o+kn2NXiTwCVp2Iuu9r/Mo5/QAhUjHIAWxrPUiSs85Ibka0JPcc+30fgFGNxqTcNoyV50svI/QRfUfgoWnMAZ6LraJ8zhFF3Ffb094orQdnHTNMZTyV/7L+PEPfx4GzAsl+xF+3ivh9/fHQ5KwGKlZ0tIg1HKW6a2Nd2g25LNmg7QO4lPf8KixniOndAXKT1tCStlmOKBYd7boLwY060h7ryDan+42tacEnfsXOXt9V4MzhamRm51YDW8aE3DLnEX5qh2V/4U76e+RuCONSQrSuugJSYWCJX+sPLYkPyAXilPEwYht6ZyVMuVdVZjnsz1VRPyttQayoH5bk2wTdy5zCmu3THYafGYecpjfaB3RHka5jLjTJciWNJLtU+c0jyRU8JWRS5FjHEdTTFTXN5YBoorKDw/NQ0/iovpJW+z99gRsXK3VTHCVSBaqPqKO7HO+ITznIs8WzIWAQtHjefqXLPsEzc9Y3DweP4zG/DcFxcP/TMHne0oPfa5seaGqWjX+gJs8Th6b6FNXrw5c2ta4ag4LPnMGUv4UkH8sjHuSbxYEXY/J6Bb4uoEvyKLFl+QU1S3qYRVnPdiL5ykFzQLWgEthCYUh5Ow5UpNvcaCvW2t2mcK6XU5wBryWsqvIP8E263kVKkwD6G11mxM+G+jxprNAFNIL39NAiurrcqVhH8ullqzKeDftP2StYeVhPlarg2tUqVKleblDwP7vPMOQQtHAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAaCAYAAAC6nQw6AAABCklEQVR4Xu2SMUuCURSGX4Ka+wVmIIRzTU26ipMuDkFiIk06CP4JwUEXHXRRaW0QHBr9A4Jr6dLS2NBc7/Xcj3vv0a9284EH7j3v4fDd717g31CmL565IAWuvSyyEHRYLmiGtuk3XQQpcErzdA7Je/Qs6FAM6RjSfKsyQ5k+6+I+NjQJGTQJoy19WtdFTYUO7PoTMqzhYiToh7ePxRzrzq47kEFLF6NEn7x9LGvIsQwpyCBj1ta69NGuY7mEDPKZQQZFX7GiaRfvZ0SrqnZCXyHD7iG3+Sdv9EoXSRPuBh9UtoP5H++6aDmnX5Bhpu9XanSqix7mSZj3FcsN3M1EtoIOR1EXjhw5bH4ARKU2OqtPSToAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD8AAAAWCAYAAAB3/EQhAAABXUlEQVR4Xu2WzytEURiGX8lCYcdKkZ0SOwsWhi02otlZsUNkLUliRdkoGwt7O0Wy8CtRbKX8B1byB/B+fd9tznyNzCxGnes89dSc9+1OfTP3nHuBROLfskUvnEJXhVzstv7Q5dOWR0U//aRf9IoWLW+mBbpr3QIdtE4YpUfWndD2oIuKF+gQK74gU9CuwRdkgh77MDauoQNu+oKsQrs2lzfSO9rp8ujIbl//Lx7QPesGXPdKW1wWJdm+PguyPvpMO6wbCzphzq2jZR064EOQndNh6O0t3UzQFSyvhbUandXL6s8ydMA3W4+jfAu80/lg/RR8rhY/3G/+2fCT0OHFJrpTXuOeblv3SHvK67gZQml4ee77g+wU+lKzAf0RckUvSsPfuk6QLSCPNdkWra6rluz7q/VSL6s/2Yku3rhO2Id24b7PFR/Q530lluiID/PEIn5+P5dzIJFIJBIx8g1M6GCT8hDgJwAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAWCAYAAABKbiVHAAAB5klEQVR4Xu2WzyttURTHFxEvKaW8njdgICOSHwNSyMNfwIBkoIQSSQYGDMQEGRgbKJFeKSVlYqgoSSkjUvIrE0mIEt9l78u66+1zbdftmfjUp+75rnVP69yz9zmX6JvoiIclsApmh1WCGdWBBxswRYeSP/AAzsFZeAurwzr+pQMm6FDwUweWEbiiQyYXLsFT+EPkafAR9opMUg7vdWhJIvO9M12w8AUc6rAS3sAtmKFqzDp8gnkqj4N7cFHlDPfzhXGdPwfRJQ/4F7mEF/CXLAjayJxwQeW1Ni9SuWSCIg8j78JL4z6ZqwyihkwfG7r/ifAOztjjIN4b5hUegBv7dUHRQm/DpNuMdxsfd4aaAvAepoJMY74uKKbI9J2IrNFmdSJzMUaew0zCIx062CVzwlWRddusTGQuvH8Z3iXzOnTAJ+OdJtfVuM1zRObCe5hp+FeHCt7OfDJexJIBm/PaiYT3MH1wW4eCVLgJh3UBtJN7SI33ML/JNBbrAkiGa2Tqrm3fQKZWrwsK72GYVnhFZiEPwUF4DJdhqehzwU9XuaglPIBLfo9FJJPMFfaQuXWF4eVAeDdeU+SX5H+DXyUPsEkXPkCWDj4DPxD5ORQt/P2YwX+OdnToSQE812Es4IfgR2nWwZfzDNkqakgLdKt3AAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACoAAAAWCAYAAAC2ew6NAAACWUlEQVR4Xu2WS6hOURiGP/fCwCXX1GGAXIpSLiWdksgluY0UpRgYiAExOkwNXAaUgRIlJHKJYqhIKbcRMUIRuc0MxPv61ur/9rvXPk6dk/6Bp57+f7/fXutfe+211/7N/vNv6Qfnwk44rVoqsho+hEO00AuWwUcaRs7CV/AiPA2/w/WVM+q8gR0aJrrg3eC8atmWhFp2baqxbY0p5oPbBYeHnN9/wSMhi4yEezUMzIKXzPug56plGwXXpdoxuBIOSrWB+aTMQvOZe66FxHnzjlZpARy16oWVeGA+QPbxE06uVJ0LGiicyQ/wizXfPl4lf4Q/GBkKv0lW4jMcAT+a93O4Wv7Ddg0UNnxvhWkOTLfWrVsQ8pvwUzhuYnf6nGQ+o+wnMlOOi7DRQQ2FTmsNlMskw0FeC8dNxAfoitUH+tfZXGTeaL4WhH3WGuiwlI1Lx00PWWY0HBCOl5q3mxEy7jTdwrXC9dNfC8It885fhmx2yrpCVoJPtPIUHg/Hb8P3IvfgZQ0LcEDsLM8mWZHynSEr8VgD84eQbbl298AT1XKdU/C6hsIE8063Sc61yvyA5JGx8KuGCbZ9Bm/ATVKrwat5oWFgMLwDz0hOppr/WGmryWw0H0iJvObpGKnVmGh+4mItmD8A+Y2S3xQRvlFY411p4iTcr2Hiqnn7bt/nkS3mm/07823qEHwN78PlrdOKcG390NCqs0X5wlD4x+eJ+S7QY8bDDebv+c3m21ZPWGM+kLaHSyJuWW0N/5J1aNiu3DZfc33NHA16C//m8QnnvtlX7IBbNWxbfgPOAX2uKs5OoAAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAbCAYAAACjkdXHAAAA1ElEQVR4XmNgGBYgBYh348E7gLgLrhoNKAOxAxAvBuL/QHwGygdhJyCuB+J3QKwBVo0DbGaAaC5HlwCCKiA+ii4IA0xA/J4BotkKTQ4EQF4DyTGjS4CALgNEEuQ8FjQ5EGgH4jfogjAA828jugQDxN8gOV50CRi4zwBR4IYkJgjEsQwQ12xBEkcBcgwQjej4ExAvBGI/hFJMEMcAUXwRXYIYMIsBonkyugQx4DoDRHMIugQxgCwng+LTkAGieRsQC6NK4wYyDJihC8JLkRWNglFAfwAA7xA1IZglGrYAAAAASUVORK5CYII=>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGAAAAAWCAYAAAA/45nkAAADL0lEQVR4Xu2ZS6hNURzGP9cj7/ejDLhCiSJ5i7qEgWKgmFwZKGREZCQiUUopZSDklYmBJEJMRB7JayCvUBKKKKKM+H/+Z3N855x919pnn3MNzq++buf7r7XPWut/1mtfoMF/y2RTi6mb+A3apodpkJqhHDf9VDNnhqmRgemmPmrWmd6mMWoW0Qofy9MaqEQTvMJHDeREX9MO0zcNRDDKdNm0x/TEdA7pg1ArVplemg5rQHiGiB90d3jhFxrIAQ4W9R4RDRImwX8c2wqfu5jOIJ/2djVNVLMMa0wPTE/h/WgrASwb3F9OaRZ+roEcGFD4ywELbpBwxfQdPlgJs5D9ecXMNZ1QswwjC9qIsATcRUT7OEgszOzWirOIaFARK+D1rorfqeBXux/MR1gCEjbBv/eIBoRbiOjvEHjhxxrIkawJ2AKvd1ED8D2FM6Ea5pmOqZlCaAJuIqK/zfDC98TPk6xL0G54vQsaML6aFqoZyQLEzYDQJeg6AvrL6cvsfzJth09rhV5LhCqRdQbw9FRpBnwxLVIzheEobe8GeHLVp6aglGQGHBVf6Qifvfvhz+JJsIQl8IftNfWTWEIz/PgXqp6/a5WSNQHr4PUuaQC+BE1TM4X1KG3vHdObMj7FWauEJoDwMsuyPAEuk9gfppoemj6bxkksT7ImYDm83jXxmwo+TybVELsEJQloa98YaHoN36M6SKyEOfCHntJAjmRNwAh4vUfi88pPn4mohlol4BAi+tsfXrhWpyC+I+EGz+8YKrHkBEadl1jCWNMH08oij5ug3tz3wZ8zU/w0YhNwEP4dPOWMl1gxUcdQbhAsXIt7QDK4qqWFeGfTAfj1Pu1my7We1/vFppPwS+Psf0r4pYrPWC1+GqEJWIvSPlBMejluIyIBtbwJh8LN+5WaZeBGOgOeuHLshL82CCU0AbFwcw9OQC944bRfYK3hQOSxB/HGPEHNFLiM7FIzB+4jIgGEJ6EfataJrfBjZrUb6mD4MtTecEXh3eqdBtLgtOZRtBV/X6DVCw5a5n9kFFH8sq694N52w/QWPquj4GkjuYTw/XuDOEbDT2ebUfli26BBA/wCZnPTiolYG7AAAAAASUVORK5CYII=>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAAqElEQVR4Xu3PPQ4BQRyG8b9CISQqWlrRU0k2Cp3CBRQqDiES4QCugFDpRIKjKEXnDjwys8k7kzgA2Sf5FfvOfmTNsn6sCm4YoYspHljrTVJbL2bo60BFrLBA3W8FLNMb0rbIxaPvhJc4h8dmtXiIaiJBx75/xA7YYIxqdKa19GKCow7mXnBHL9oTlHTYo6yDL48nLpjjau6/gwbxIDWwM/fgxzA8zvr33tn1GNLEzIQfAAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,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>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAbCAYAAACqenW9AAAApElEQVR4XmNgGAX0BglA/B+Ku5DE2YD4HJQGgzwGiKJjQPwLyg6HysUDcRWUzaAKxD+BuAzKlwDibQwQjSBwCoiFoWyGmQwIhTDAA8QvgFgOiCcgS+gCMTuyABRMAuJYIFZGl8AGkoB4GbogLuAAxB/QBXEBJyBejC6IC1QAcTm6IC6wG4iD0QWxAW4GSOQooIljBTVAPBFdEBcAhbEBuuAoQAYAHYcZKT1EKVMAAAAASUVORK5CYII=>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAAAxklEQVR4XmNgGJTgKBD/h+J7QDwJiBuQFSCDjUAcDMR9DBDFIE2MKCroAliAWBlNjB+NDwYghWeA+BYQ7wViWSDeBMTngNgPSR2DExDPQeLfBuKvQMzFAPHkdiQ5Bjcg1oSyhRkgCrZB+QVALAhlY4BQBojicnQJbKCHAaLYAl0CGcCccRKIPzJAPAwCxkBsAGWDwWIGiGkw9x5GkgN5zhSJz1AFxOeBuB2IJYD4CBBvAOLLQMyNpA4O5NH4OkBsiyY2CsgDALHYIs9fuZGRAAAAAElFTkSuQmCC>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAAe0lEQVR4XmNgGEgQCMS7ScR+YJ1AYA7EtUA8FYj/A/E8KB8Zg8RAciAMUmcK1okExBkgks7oEgwQMZhmaTQ5MBjVjAroo9kFXYKBBM2u6BIMJGh2Q5dgIEKzMQNEMhZdAggSGBCaDZElcpEkkPFkqDy6OAhnQOVGwdAAAPGKRjj8JhwbAAAAAElFTkSuQmCC>