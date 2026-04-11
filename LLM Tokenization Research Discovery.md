# **Structural Evolution of Textual Representation: A 2026 Comprehensive Analysis of Learnable, Dynamic, and Token-Free Architectures in Large Language Models**

The landscape of large language model (LLM) architecture has reached a pivotal juncture in 2026, characterized by the systematic dismantling of the traditional tokenization pipeline. For over a decade, the field relied on static, heuristic-based compression methods such as Byte-Pair Encoding (BPE) and WordPiece to bridge the gap between human language and neural computation.1 However, contemporary evidence suggests that these hardcoded preprocessing steps have become a primary bottleneck, limiting semantic reasoning, cross-lingual equity, and the efficiency of scaling laws.3 The latest research has shifted away from artisanal design toward end-to-end learnable representation layers, entropy-based dynamic patching, and token-free byte-level modeling.1

## **The Theoretical Complexity and Information-Theoretic Limits of Static Tokenization**

The year 2026 has witnessed a rigorous re-examination of the mathematical foundations of tokenization. Prior to this period, tokenization was largely viewed as a trivial string-matching problem. However, recent findings presented at ICLR 2026 have fundamentally altered this perception by proving that tokenization is NP-complete even under the most restrictive conditions.7 While previous proofs relied on unbounded alphabets, new research demonstrates that finding an optimal vocabulary is NP-complete for alphabets as small as two characters.7 This discovery implies that traditional greedy algorithms like BPE are mathematically incapable of reaching global optima in data compression or representation efficiency.7

The complexity analysis distinguishes between two primary variants of the problem: bottom-up tokenization, which involves selecting a sequence of merge operations, and direct tokenization, which involves the selection of a vocabulary from scratch.7 The absence of a polynomial-time approximation scheme (PTAS) for either variant suggests that the industry's historical reliance on frequency-based heuristics has introduced permanent sub-optimality into model training.7 Even in the case of a single-character alphabet—a scenario that, while not practical, serves as a mathematical baseline—direct tokenization remains NP-complete.7

Complementing these findings is a new information-theoretic perspective on how tokenizer training scale influences model performance. Research suggests that as the volume of training data for a tokenizer increases, entropy is redistributed across the token stream in a manner that favors in-context predictability.8 While the aggregate token stream becomes more diverse, individual tokens in sequence become markedly more predictable, effectively shifting the computational load from the transformer layers to the representation layer.8 However, this benefit exhibits harsh diminishing returns once the training corpus for a tokenizer exceeds approximately 150GB, indicating that the path forward lies not in more data for static tokenizers, but in fundamentally different architectural approaches.9

### **Complexity and Computational Hardness in Tokenization Paradigms**

| Tokenization Variant | Proof Basis | Alphabet Size | Approximation Possibility |
| :---- | :---- | :---- | :---- |
| Bottom-Up (BPE-like) | Merge Operation Sequence | ![][image1] | No PTAS (P ![][image2] NP) |
| Direct (Vocabulary Selection) | Optimal Set Selection | ![][image1] | No PTAS (P ![][image2] NP) |
| Single-Character Direct | Unary Encoding | ![][image3] | NP-Complete |
| Unbounded Direct | Set Cover Reduction | Unbounded | NP-Hard |

7

## **Cross-Lingual Inequities and the Tax on Multilinguality**

One of the most significant motivations for the 2025-2026 research cycle has been the empirical documentation of "token premiums" and their role in cross-lingual inequity.10 A comprehensive suite of approximately 7,000 monolingual tokenizers across 97 languages has revealed that standard tokenization practices impose a heavy throughput and cost penalty on non-Latin scripts.10 The research demonstrates that whitespace pre-tokenization—a standard feature in the Gemma and Llama series—is a primary driver of these inequities.1

For languages like Hindi, Tamil, and Sinhala, regular expression-based pre-tokenizers often segment text at the sub-graphemic level, splitting common diacritics and symbols into multiple units.10 This results in "catastrophic compression" for South Asian and African languages, where the same semantic content requires three to five times more tokens than its English equivalent.10 This "tax" is not merely a matter of inference cost; it effectively shrinks the model’s context window and reduces its reasoning capacity for speakers of these languages.5

The latest findings suggest that simply increasing vocabulary size is an insufficient remedy. Instead, researchers have proposed determining "optimal" vocabulary sizes specifically tailored to each language or adopting "superword tokenizers" that allow merges across whitespaces to improve overall compression.10 Furthermore, the TREX (Tokenizer Regression for Optimal Data Mixture) framework has been introduced to predict the optimal data mixture for multilingual tokenizer training.9 By utilizing the scaling laws of training steps and model sizes, TREX enables researchers to mitigate the accuracy-cost trade-off, ensuring that multilingual models are trained on tokenizers that balance efficiency across all target languages.9

### **Comparative Token Premiums and Efficiency Metrics Across Linguistic Families**

| Language Script | Pre-tokenization Impact | Fragment Level | Relative Throughput |
| :---- | :---- | :---- | :---- |
| Latin (English) | Minimal | Word/Sub-word | 1.00x (Baseline) |
| Cyrillic (Russian) | Low | Sub-word | 0.85x \- 0.90x |
| Devanagari (Hindi) | Severe | Sub-grapheme | 0.30x \- 0.45x |
| Sinhala | Catastrophic | Sub-grapheme | 0.20x \- 0.35x |
| Logographic (Chinese) | Variable | Character/Component | 0.75x \- 0.95x |

9

## **Learnable and End-to-End Tokenization Frameworks**

The transition from "artisanal" to "automated" tokenizer design has been accelerated by the introduction of end-to-end learnable tokenization layers.1 Historically, tokenization was treated as a separate, non-differentiable step. However, research by Dauncey and Wattenhofer (2026) has demonstrated that token boundaries can be learned using score function estimates, which offer tighter theoretical guarantees than the previously used "straight-through" estimators.1

This approach treats the problem of drawing discrete token boundaries as an optimization task within the model's training objective. By incorporating reinforcement learning (RL) techniques such as time discounting, researchers have succeeded in reducing the variance of these score functions sufficiently to make end-to-end training practicable.6 This architecture allows the model to reuse learned representations at the byte level for token-level processing, effectively bringing the compression step inside the neural network.1

Parallel to this, the QA-Token (Quality-Aware Tokenization) framework has emerged as a critical tool for pre-training on noisy, real-world data.12 Traditional BPE-based methods assume that every sequence in a corpus is equally reliable. QA-Token disrupts this by formalizing tokenization as a bilevel optimization problem that jointly optimizes for vocabulary construction and downstream performance while accounting for data reliability.12 This framework utilizes an RL approach where merge policies are learned through quality-aware rewards, such as Phred scores in genomics or signal stability in financial time series.12 The result is a tokenizer that filters out "noise" and focuses computational resources on high-signal substructures, achieving a 6.7 percentage point F1 gain in genomic variant calling while reducing the overall token count by 15%.12

## **Domain-Specific Evolution in Chemistry and Genomics**

The "tokenization bottleneck" is most acute in scientific domains where general-purpose tokenizers fail to capture specialized grammars.3 In chemistry, the use of standard subword tokenizers to process SMILES (Simplified Molecular Input Line Entry System) notation results in the semantic fragmentation of molecular structures.3 For instance, functional groups such as methyl esters or carboxylic acids are often split into uninformative sub-tokens, obscuring their chemical reactivity from the model.15

To resolve this, research in late 2025 introduced SMILES Pair Encoding, a methodology that extends a model’s vocabulary with chemically salient tokens extracted from millions of molecular structures.3 By augmenting the Llama 3 vocabulary with approximately 16,800 specialized molecular tokens, researchers were able to drastically reduce "token fertility"—the number of tokens per molecular string.3 The implications of this are profound: models using extended chemical vocabularies showed a jump from near-zero to over 2,500 exact matches in forward chemical synthesis tasks, proving that solving the representation problem is prerequisite to achieving scientific reasoning.15

In genomics, the challenge is even more complex due to the absence of a defined "vocabulary" unit in DNA.14 The information density of the genome is highly variable, with only 2% consisting of coding sequences (CDS) and the remainder being non-coding (nCDS) or repetitive elements.14 The MergeDNA and DNACHUNKER frameworks represent the latest efforts to solve this through dynamic, context-aware modeling.16 DNACHUNKER utilizes an end-to-end learnable routing module that proposes chunk boundaries based on base pair dissimilarity, automatically adjusting its resolution from 15-base chunks in dense functional areas to 320-base chunks in repetitive regions.17 MergeDNA takes a hierarchical approach, using differentiable token merging to chunk adjacent bases into "words" based on local similarity, outperforming previous static k-mer and BPE-based DNA models by significant margins.14

### **Benchmark Performance of Specialized Scientific Tokenizers**

| Domain | Model/Tokenizer | Accuracy/F1 | Efficiency Improvement | Baseline (Static) |
| :---- | :---- | :---- | :---- | :---- |
| Chemistry | SMILES-PE (Llama 3\) | 2,507 Exact Match | 45% Token Reduction | 2 Exact Match |
| Genomics | MergeDNA | 90.87% Accuracy | 22% Reduction | 87.30% (BPE) |
| Genomics | DNACHUNKER | 0.701 MCC | 15% Reduction | 0.625 (k-mer) |
| Genomics | VQDNA | 99.46% Species Acc | High Compression | 96.2% (k-mer) |
| Protein | Kanzi (Flow-based) | Reconstruction SOTA | 60% Parameter Reduction | VQ-VAE |

15

## **The Shift Toward Token-Free Byte-Level Architectures**

Perhaps the most radical development in 2025-2026 is the rise of the Byte Latent Transformer (BLT) and its variants, which discard tokenization entirely in favor of processing raw UTF-8 bytes.4 The primary argument for token-free modeling is its inherent robustness; by operating on bytes, the model avoids the brittleness introduced by BPE, where a single typo or shift in punctuation can radically alter token boundaries and disrupt the model's reasoning.5

The Byte Latent Transformer architecture introduces "entropy-based adaptive patching".4 Instead of treating every byte equally, BLT uses a lightweight auxiliary model to compute the next-byte entropy. A new patch is started only when the entropy—a measure of the model's uncertainty—crosses a certain threshold.20 This allows the model to allocate more compute and attention to information-dense regions while skipping ahead through predictable, low-entropy regions.21 Scaling studies have shown that BLT can match the performance of tokenization-based models like Llama 3 while using up to 50% fewer FLOPs during inference.4

A complementary approach, the Bolmo project from the Allen Institute, focuses on "byteifying" existing subword models.23 Rather than training from scratch, Bolmo uses a two-stage distillation process to convert pre-trained subword Transformers into byte-level models using less than 1% of the original compute budget.24 This process involves freezing the transformer weights and training a new local encoder and boundary predictor to handle byte-patch representations.24 The result is a family of models (at 1B and 7B scales) that outperform their subword teachers on character-understanding tasks while remaining competitive on general reasoning.23

### **Architectural Components of Patch-Based Byte Models**

| Module | Function | Implementation Detail | Resource Cost |
| :---- | :---- | :---- | :---- |
| **Entropy Model** | Boundary Prediction | Lightweight byte-level LM | \< 1% of Total FLOPs |
| **Local Encoder** | Byte Contextualization | mLSTM or Shallow Conv | Low (Byte-level) |
| **Global Transformer** | Latent Reasoning | Deep Self-Attention | High (Patch-level) |
| **Local Decoder** | Byte Reconstruction | Cross-Attention | Low (Byte-level) |

4

## **Economic Impacts, Pricing Multiplicity, and Agentic Density**

The move toward more efficient representation is not merely an academic exercise; it has profound economic implications for the AI-as-a-service market.8 Research into "tokenization multiplicity" has revealed that because multiple token sequences can often represent the same string, commercial providers may exhibit arbitrary price variations for identical user queries.8 This representational mismatch also creates a "fragility" in reasoning where a model may treat two semantically identical strings as distinct because their token boundaries differ.5

The release of GPT-5.4 in March 2026 highlights the ongoing effort to resolve these issues. This model achieved a reported 47% improvement in token efficiency compared to its 2025 predecessor, GPT-5.2.26 Such improvements translate directly into lower API costs for enterprise deployments and higher "cognitive density"—the amount of knowledge packed into every byte of representation.26 This efficiency is particularly critical for the burgeoning "Agentic AI" sector, where models must process vast amounts of data at the operating system level to automate complex workflows.26

In early 2026, the global cadence for major model releases has accelerated to one launch every 72 hours, with architecture efficiency becoming the single biggest differentiator.26 Trillion-parameter models like Kimi K2.5 and DeepSeek v4 are now deploying Mixture-of-Experts (MoE) designs that activate only a small fraction of their parameters per token, combined with highly optimized, domain-specific vocabularies to minimize operational friction.26

## **Convergence with Non-Transformer Architectures**

As tokenization evolves, the Transformer architecture itself is being supplemented or replaced by paradigms that handle sequence length and representation differently.26 State Space Models (SSMs) like Mamba and Jamba have demonstrated the ability to handle million-token contexts with linear scaling—a feat Transformers cannot achieve due to the quadratic complexity of attention.28 By late 2025, hybrid architectures integrating Mamba-2 layers with standard Transformers, such as NVIDIA Nemotron 3, have emerged as the industry standard for long-context reasoning.29

Simultaneously, the Joint-Embedding Predictive Architecture (JEPA), championed by researchers at AMI Labs and Meta, has begun to challenge the autoregressive next-token prediction paradigm.26 JEPA models, such as VL-JEPA, predict continuous embeddings of target text or visual data rather than discrete tokens.29 This avoids the pathological behaviors of autoregressive generation, such as mode collapse and accumulation of errors, while requiring 50% fewer trainable parameters than standard models.29

The field is also exploring Diffusion-based language models as a parallel generation alternative.29 Models like LLaDA (Large Language Diffusion Models) use a discrete masking approach to recover tokens, allowing for simultaneous multi-token generation and more direct controllability than autoregressive models.29 These "token-free" or "latent-first" approaches suggest that the very concept of the "token" may become obsolete as models move toward more fluid, information-grounded representations of reality.

### **Comparison of Sequential Modeling Paradigms in 2026**

| Architecture | Representation Unit | Scaling Complexity | Primary Strength |
| :---- | :---- | :---- | :---- |
| Standard Transformer | Discrete Tokens | Quadratic (![][image4]) | Proven Reasoning |
| Mamba/SSM | Continuous State | Linear (![][image5]) | Million-token Context |
| LLM-JEPA | Continuous Embeddings | Variable | World Model Realism |
| LLaDA (Diffusion) | Masked Tokens/Latents | Parallel | Controllability/Speed |
| BLT (Patch-Transformer) | Raw Bytes (Patches) | Linear/Sub-quadratic | Robustness/Universal |

4

## **Conclusion and Future Outlook**

The research trajectory of 2025 and 2026 confirms that tokenization, once a neglected preprocessing step, is now at the heart of the architectural revolution in artificial intelligence. The systemic biases and inefficiencies identified in static tokenizers have paved the way for a more principled, learnable approach to representation.4 From the entropy-driven patching of Byte Latent Transformers to the quality-aware vocabulary construction in genomics and finance, the focus has shifted toward models that adapt their granularity to the structure and quality of the data they ingest.4

The ultimate goal of this research is the creation of "world models" that bypass the artisanal limitations of human-defined symbols entirely.23 As compute costs continue to drop and models like GPT-5.4 and DeepSeek v4 move toward extreme token efficiency, the industry is approaching a state where the representation layer is as dynamic and intelligent as the reasoning layers above it.26 For practitioners, this means moving away from a one-size-fits-all approach to data and toward specialized, learnable interfaces that bridge the gap between raw bytes and high-level cognitive tasks. The next frontier will likely involve "omnimodal" byte-level models that can ingest and generate text, audio, video, and genomic data through a single, unified architectural framework, finally realizing the promise of truly end-to-end intelligence.23

#### **Works cited**

1. You Can Learn Tokenization End-to-End with Reinforcement Learning \- arXiv, accessed on April 11, 2026, [https://arxiv.org/html/2602.13940v1](https://arxiv.org/html/2602.13940v1)  
2. Tokenization is Killing our Multilingual LLM Dream \- Hugging Face, accessed on April 11, 2026, [https://huggingface.co/blog/omarkamali/tokenization](https://huggingface.co/blog/omarkamali/tokenization)  
3. The Tokenization Bottleneck: How Vocabulary Extension Improves Chemistry Representation Learning in Pretrained Language Models \- NeurIPS 2026, accessed on April 11, 2026, [https://neurips.cc/virtual/2025/122865](https://neurips.cc/virtual/2025/122865)  
4. Byte Latent Transformer: Patches Scale Better Than Tokens \- ACL Anthology, accessed on April 11, 2026, [https://aclanthology.org/2025.acl-long.453.pdf](https://aclanthology.org/2025.acl-long.453.pdf)  
5. SpaceByte: Towards Deleting Tokenization from Large Language Modeling \- ResearchGate, accessed on April 11, 2026, [https://www.researchgate.net/publication/397200427\_SpaceByte\_Towards\_Deleting\_Tokenization\_from\_Large\_Language\_Modeling](https://www.researchgate.net/publication/397200427_SpaceByte_Towards_Deleting_Tokenization_from_Large_Language_Modeling)  
6. You Can Learn Tokenization End-to-End with Reinforcement Learning \- ResearchGate, accessed on April 11, 2026, [https://www.researchgate.net/publication/400855760\_You\_Can\_Learn\_Tokenization\_End-to-End\_with\_Reinforcement\_Learning](https://www.researchgate.net/publication/400855760_You_Can_Learn_Tokenization_End-to-End_with_Reinforcement_Learning)  
7. ICLR Poster Tokenisation over Bounded Alphabets is Hard, accessed on April 11, 2026, [https://iclr.cc/virtual/2026/poster/10008947](https://iclr.cc/virtual/2026/poster/10008947)  
8. Toward a Theory of Tokenization in LLMs \- Semantic Scholar, accessed on April 11, 2026, [https://www.semanticscholar.org/paper/Toward-a-Theory-of-Tokenization-in-LLMs-Rajaraman-Jiao/444a607af32cb2a24d39fd19e15a9677be3b4a75](https://www.semanticscholar.org/paper/Toward-a-Theory-of-Tokenization-in-LLMs-Rajaraman-Jiao/444a607af32cb2a24d39fd19e15a9677be3b4a75)  
9. \[PDF\] TREX: Tokenizer Regression for Optimal Data Mixture ..., accessed on April 11, 2026, [https://api.semanticscholar.org/arXiv:2601.13588](https://api.semanticscholar.org/arXiv:2601.13588)  
10. NeurIPS Poster Explaining and Mitigating Crosslingual Tokenizer ..., accessed on April 11, 2026, [https://neurips.cc/virtual/2025/poster/120308](https://neurips.cc/virtual/2025/poster/120308)  
11. Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies | Request PDF, accessed on April 11, 2026, [https://www.researchgate.net/publication/397198295\_Scaling\_Laws\_with\_Vocabulary\_Larger\_Models\_Deserve\_Larger\_Vocabularies](https://www.researchgate.net/publication/397198295_Scaling_Laws_with_Vocabulary_Larger_Models_Deserve_Larger_Vocabularies)  
12. (PDF) Unlocking Noisy Real-World Corpora for Foundation Model Pre-Training via Quality-Aware Tokenization \- ResearchGate, accessed on April 11, 2026, [https://www.researchgate.net/publication/400583970\_Unlocking\_Noisy\_Real-World\_Corpora\_for\_Foundation\_Model\_Pre-Training\_via\_Quality-Aware\_Tokenization](https://www.researchgate.net/publication/400583970_Unlocking_Noisy_Real-World_Corpora_for_Foundation_Model_Pre-Training_via_Quality-Aware_Tokenization)  
13. Unlocking Noisy Real-World Corpora for Foundation Model ... \- arXiv, accessed on April 11, 2026, [https://arxiv.org/pdf/2602.06394](https://arxiv.org/pdf/2602.06394)  
14. MergeDNA: Context-Aware Genome Modeling with Dynamic Tokenization Through Token Merging, accessed on April 11, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/37032/40994](https://ojs.aaai.org/index.php/AAAI/article/view/37032/40994)  
15. The Tokenization Bottleneck: How Vocabulary Extension ... \- arXiv, accessed on April 11, 2026, [https://arxiv.org/pdf/2511.14365](https://arxiv.org/pdf/2511.14365)  
16. MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging \- arXiv, accessed on April 11, 2026, [https://arxiv.org/html/2511.14806v1](https://arxiv.org/html/2511.14806v1)  
17. Dynamic DNA Tokenization \- Emergent Mind, accessed on April 11, 2026, [https://www.emergentmind.com/topics/dynamic-dna-tokenization](https://www.emergentmind.com/topics/dynamic-dna-tokenization)  
18. MergeDNA: Context-Aware Genome Modeling with Dynamic Tokenization Through Token Merging | Request PDF \- ResearchGate, accessed on April 11, 2026, [https://www.researchgate.net/publication/402636547\_MergeDNA\_Context-Aware\_Genome\_Modeling\_with\_Dynamic\_Tokenization\_Through\_Token\_Merging](https://www.researchgate.net/publication/402636547_MergeDNA_Context-Aware_Genome_Modeling_with_Dynamic_Tokenization_Through_Token_Merging)  
19. ICLR Poster Flow Autoencoders are Effective Protein Tokenizers, accessed on April 11, 2026, [https://iclr.cc/virtual/2026/poster/10011420](https://iclr.cc/virtual/2026/poster/10011420)  
20. Byte Latent Transformer (BLT) \- Emergent Mind, accessed on April 11, 2026, [https://www.emergentmind.com/topics/byte-latent-transformer-blt](https://www.emergentmind.com/topics/byte-latent-transformer-blt)  
21. Patches Over Tokens: How Byte Latent Transformers Could Change the Future of LLMs, accessed on April 11, 2026, [https://medium.com/@wiz-wizdomgr/patches-over-tokens-how-byte-latent-transformers-could-change-the-future-of-llms-de352929b977](https://medium.com/@wiz-wizdomgr/patches-over-tokens-how-byte-latent-transformers-could-change-the-future-of-llms-de352929b977)  
22. Byte Latent Transformer: Patches Scale Better Than Tokens \- arXiv, accessed on April 11, 2026, [https://arxiv.org/html/2412.09871v1](https://arxiv.org/html/2412.09871v1)  
23. Byte Language Models: A Tokenization-Free Approach, accessed on April 11, 2026, [https://www.emergentmind.com/topics/byte-language-models-blms](https://www.emergentmind.com/topics/byte-language-models-blms)  
24. Bolmo: Byteifying the Next Generation of Language Models \- ResearchGate, accessed on April 11, 2026, [https://www.researchgate.net/publication/398806418\_Bolmo\_Byteifying\_the\_Next\_Generation\_of\_Language\_Models](https://www.researchgate.net/publication/398806418_Bolmo_Byteifying_the_Next_Generation_of_Language_Models)  
25. Allen Institute's Open Source Bolmo Models Redefine Byte AI \- AI CERTs News, accessed on April 11, 2026, [https://www.aicerts.ai/news/allen-institutes-open-source-bolmo-models-redefine-byte-ai/](https://www.aicerts.ai/news/allen-institutes-open-source-bolmo-models-redefine-byte-ai/)  
26. AI Breakthroughs in 2026: March Update \- Kersai, accessed on April 11, 2026, [https://kersai.com/ai-breakthroughs-in-2026-march-update/](https://kersai.com/ai-breakthroughs-in-2026-march-update/)  
27. The AI Avalanche: 7 Breakthroughs Redefining March 2026 \- Switas Consultancy, accessed on April 11, 2026, [https://www.switas.com/articles/the-ai-avalanche-7-breakthroughs-redefining-march-2026](https://www.switas.com/articles/the-ai-avalanche-7-breakthroughs-redefining-march-2026)  
28. Best Open Source LLMs in 2026: We Reviewed 7 Models \- Fireworks AI, accessed on April 11, 2026, [https://fireworks.ai/blog/best-open-source-llms](https://fireworks.ai/blog/best-open-source-llms)  
29. The End of LLMs As We Know Them: Why 2026 Marks the Beginning of AI's Next Architecture Revolution | by Aftab | Medium, accessed on April 11, 2026, [https://medium.com/@aftab001x/the-end-of-llms-as-we-know-them-why-2026-marks-the-beginning-of-ais-next-architecture-revolution-902ee29484f7](https://medium.com/@aftab001x/the-end-of-llms-as-we-know-them-why-2026-marks-the-beginning-of-ais-next-architecture-revolution-902ee29484f7)  
30. NeurIPS 2025 Papers with Code & Data \- Paper Digest, accessed on April 11, 2026, [https://www.paperdigest.org/2025/11/neurips-2025-papers-with-code-data/](https://www.paperdigest.org/2025/11/neurips-2025-papers-with-code-data/)  
31. ICLR 2026 Workshop on Multimodal Intelligence: Next Token Prediction & Beyond, accessed on April 11, 2026, [https://mmintelligence.github.io/](https://mmintelligence.github.io/)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAXCAYAAAAcP/9qAAABS0lEQVR4Xu2UsStFcRTHj0iEGDCox1MMBpOwGCSJTEaslMFmMNhENkXkD7Aps8Fge8UsE5NSstiExOf0c/ndw3v3uvcOhvupz3DP99f7dX6d80RysmcYe7HGBko19tliSop4jOf4jI+46R8IuMEj7LdBAhrwElc/vzvwAt+/Tnho17PiDpxgVTj+E/o72uWCVxsQd/G4V/vBNF7hEtabLA7r4i6xHV7jgamVpYj7+IDt4agiXVjrfTfiK654tVi0ipuDPSkzoREsinuBFhtE0Yx3eIpNJouiDm/xzQaVaMMtvJfkK7ct7pnnbfAbBdwV97wb4lYiCWv4hFM28OnEHSxJunUKmMNuU9NmQhziGU7aICEz4l5LB1EHswcHcdk/pN2N+YWUDOGLfO+y74R3LnN0mOyFgbrf/wPdzzjq/7judGaMxnREcnJi8AGMFkPeNOLWAAAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAAlUlEQVR4XmNgGCpgLRDboAsSA6yA+CC6ILFgHxA7oQsSA9yA+DC6ILHgFBDboQsSA0AazdAFkYEfENfiwDewiMlDtEGADxYFMDwVi5gCWBcBADKUbHASXQAbmAbE/0nAeOM7BIhXoAsSA5iA+CIQa6BLEAMigHgxuiAxIBqIZ6MLEgsuAbEsuiCxYAq6AClAGl1gGAIAeGQkaIVGi8wAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAXCAYAAAAyet74AAAAaUlEQVR4XmNgoAdgBGITdEF0YAHE+4H4MroEDEgA8UEgPgfE/xnwKOQAYlMg1mQgoBAGtBkgCq+gS6ADohXSzuqr6BLogGSF19Al0AFM4XV0CXQQwABR+BKIbYGYGVUaAkAKsOFRQBgAAHklIBfH3XM1AAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAZCAYAAABkdu2NAAAC/UlEQVR4Xu2XWahNURjHP/NMhogiUYYkQwp5OaUMGR4k5U2JUsr8ZMpQIkI8IJJMKR7IA5mVUJIiwwP3CRkSDxQS///91rp7re/us89wz7mi86t/Z+//t/baa6/1rW/vI/Jv0A5aCU2ygf+BHtAlaCn03h13jFo0kRFQzpp56AU9g0bbQBM4DP12x23d8fkkLL2h58F50RyCXkIXoA3QE2hJ1CKGaXQHWmgDjitGFg46jJ9yPldwnDseLPqA+925h6k70Hh56Q8dg9ZB3QN/OvQDOgd1CHzPaeiWNQNy0ArRAVLdoqjIGNGJZOwBNCgO17NHdKJ72gC4YQ0LH+wt9BUabmKeIaIDYDvLPWukcBmaCh2H6qCWcbh+Eh4az8PJG2rNgKPQC6iNDXguig5+lQ0E8OJfkuwJD2d7mfEsraFPUHtovGgfM6IWuoK7jEeGQX3cMcfAh7FME+2Tv6kw+FF0L+WjhaQ/IFOPg84iJ/HeYx9ng3NyDZptPGYNV3u9025obdRCYcqzz+024GFwpzUN/STZQyEnRFcoCw6OK+RhH9zTA9w5V/Yb1LWhhcJ09vf0mhe1SHgsebbKDtG9l7V6hLOX9oCc4UJ8kbg4cRXYD99t9DdCW4J4ObA4ss9ONsCnZuXKogv0QbSDMybGtC3ETXPOPfVdtL/F0G1octSidPaK9uezogFWKFa4LNaIXsxBhSW8lfMLscka4IjotSz97Dft9VMK3H/sj0Up4gB01ZqGN6IXp21iplkh0lZnrCQpX2iCi+GgaF+sFRF9XWCKDTj2QfclfvGHZL3gyRxrBMwVvfdEGygDVuk6a3p4k1eiD+th+m1zsUYbN4CfTp2tKfrtOAF6Co2Sxi92wurLQRWqwsXATDppTc986J1oIdksWtI5sEeSvyx7ZkKzjDdSkvTzYgqlsdoaZcCvL95jgfErxmvRFftb3IW2WrOSfIYWWbMZ+SnJ51xVYCFhpa3qTfLAP8DLrVkN+MF9XSpTMErB/3esUaNGjeblD+PkovGkWMLCAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAaCAYAAADxNd/XAAACqklEQVR4Xu2WS+hNURjFP28S8koUA0VJChNMuFHerySlmBClSMLAO0SRGEihFBGTv8JMGFBEJAOPIlHin0dhQCGxlu8cd+91971H3auk+6s1OGvts+853/323sesyb/PcKikZhX6QI+gkRrUQRvoopp/wlHoKXQB2grdh1ZEI2I6QdehpRpkXBIp5y3OTwdZb2hGcF2TgdAJaDPUM/CnQV+hs1CXwM85A11VM6AErYF+ZOoRpWajzAvF7A40OI7tHdRLvApGmw98okHGLvMfWKeBuT9VTWEjdMq8EGslIywMs/4agMfQfjVDWPlW6BM0TLKcoeYPynHKTTUSsJenQCehZ1DbOP71L90VL2eJ+cvxGZLwwVh99nI1uKC+m48NYWuMEU8pWdz7nKMluCZXoNnihdyGbqiZwwn3qSkMsHIPh7At2ounbDHv8RzOwYoOyq47Q5+h7r9HVHII+maJtTDWfMKJGghzLf0Ct+Q6Basfzv/CfJ7d2fV46Fo5TsJ1w3vGabDXvPdrtQ85YOkXYD8X8dHi3WuT+TxvMn8btDPIU3CL5j3TNeD2V3RYrDe/+YvFW1y7zC9iuxrgmPm9PGM4b2p7DuFZwPHLNTgMXVZTeGV+8x4NzKtYxCQ1zLft/B8tKiBZZD52oQaroefmu0wK+ryRu0QHycgDNYRuVr26nJNzb9AgwSrzsTxUI3hwMJisQcZB84UanswhtU5gMk+NgPnmv12xMBNwJ+PYIRoQ/i2vobfQDvMt7yF0D1oQjEsxE5ol3ggrt0euI9GIMqmTPQXbmOumKn3Nq7USWgxNsOptFdLVis+QeuFXMYvQyC/diJdQRzUbCM+Lolatiw/QMjUbyHsrbuW6YOuxR/tp0CDOWeXH31+BW90cNeuA30bH1WzSpMl/wk+r/JJFRw62WgAAAABJRU5ErkJggg==>