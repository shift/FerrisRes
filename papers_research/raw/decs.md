## **BOLT: Bootstrap Long Chain-of-Thought in Language Models without** **Distillation**

**Bo Pang** [1] **Hanze Dong** [1] **Jiacheng Xu** [1] **Silvio Savarese** [1] **Yingbo Zhou** [1] **Caiming Xiong** [1]

Salesforce AI Research



**Abstract**


Large language models (LLMs), such as o1 from
OpenAI, have demonstrated remarkable reasoning
capabilities. o1 generates a long chain-of-thought
(LongCoT) before answering a question. LongCoT allows LLMs to analyze problems, devise
plans, reflect, and backtrack effectively. These
actions empower LLM to solve complex problems. After the release of o1, many teams have
attempted to replicate its LongCoT and reasoning capabilities. In terms of methods, they primarily rely on knowledge distillation with data
from existing models with LongCoT capacities
(e.g., OpenAI-o1, Qwen-QwQ, DeepSeek-R1Preview), leaving significant uncertainties on systematically developing such reasoning abilities.
In terms of data domains, these works focus narrowly on math while a few others include coding,
limiting their generalizability. This paper introduces a novel approach to enable LLM’s LongCoT capacity without distillation from o1-like
models or expensive human annotations, where
we bootstrap LongCoT (BOLT) from a standard
instruct model. BOLT involves three stages:
1) LongCoT data bootstrapping with in-context
learning on a standard instruct model; 2) LongCoT supervised finetuning; 3) online training to
further refine LongCoT capacities. In BOLT, only
a few in-context examples need to be constructed
during the bootstrapping stage; in our experiments, we created 10 examples, demonstrating
the feasibility of this approach. We use Lllama3.1-70B-Instruct to bootstrap LongCoT and apply
our method to various model scales (7B, 8B, 70B).
We achieve impressive performance on a variety
of benchmarks, Arena-Hard, MT-Bench, WildBench, ZebraLogic, MATH500, which evaluate
diverse task-solving and reasoning capabilities.


1 Salesforce AI Research. Correspondence to: Bo Pang
_<_ b.pang@salesforce.com _>_ .



**1. Introduction**


Large language models (LLMs), such as OpenAI’s o1
model, have exhibited extraordinary reasoning abilities, particularly on coding and mathematical problems. The o1
model employs long chain-of-thought (LongCoT), which involves generating an extended reasoning sequence prior
to delivering a final answer. LongCoT enables LLMs
to analyze problems, make plans, branch to different approaches, evaluate their reasoning through reflection, and,
when necessary, backtrack to correct errors. By leveraging
these problem-solving techniques via long chain-of-thought,
LLMs can tackle highly intricate and multifaceted challenges, showcasing their potential to function as powerful
tools for complex reasoning tasks across various disciplines.


In fact, almost all modern LLMs are able to reason through
chain-of-thought via prompting techniques (Wei et al.,
2022). Recent instruct models are trained on chain-ofthought data, making this chain-of-thought process their default mode, particularly when solving math problems (Jiang
et al., 2023; Grattafiori et al., 2024; Team et al., 2024). Additionally, Wang & Zhou (2024) show that chain-of-thought
is inherent in pre-trained LLMs. However, regular LLMs exhibit shorter and, more critically, simpler behavior in chainof-thought, compared to o1-like models. In this paper, we
refer to o1-like models that generate long chain-of-thought
with rich reasoning behavior as _LongCoT_ models, while
regular LLMs are referred to as _ShortCoT_ models.


Following the release of o1, numerous research teams have
sought to replicate its LongCoT and reasoning capabilities.
Methodologically, these efforts primarily rely on knowledge
distillation using data derived from existing LongCoT models (e.g., OpenAI-o1, Qwen-QwQ, DeepSeek-R1-Preview).
However, this approach leaves significant gaps in understanding how to systematically develop such LongCoT reasoning skills. While distillation provides a shortcut for
training LongCoT models, developing such models without
relying on existing LongCoT models remains a black-box.
Regarding data domains, most of these studies concentrate
on mathematical problem solving, with only a few expanding into coding tasks, limiting the broader applicability and
generalization of their findings.



1


**BOLT**


_Figure 1._ Illustration of bootstrapping long chain-of-thought in large language models (BOLT). BOLT comprises three stages: 1) **LongCoT**
**Bootstrapping** which involves synthesizing LongCoT data, 2) **LongCoT Supervised Finetuning** where we train a ShortCoT model to
adapt to the LongCoT format, incorporating reasoning elements and practicing extended chains of thought before arriving at an external
solution, 3) **LongCoT Online Training** where the LongCoT SFT model is further improved through online exploration and refinement.
**Bootstrapping LLM** is a ShortCoT LLM that is used to generate LongCoT data via in-context learning. **ORM** is an outcome reward
model which scores the external solution in the model response.



This paper introduces a novel approach to enable LLMs to
develop LongCoT capabilities without relying on distillation from LongCoT models or expensive human annotations.
Our method, called Bootstrapping LongCoT (BOLT), builds
these capacities from a ShortCoT LLM through three key
stages: (1) LongCoT data bootstrapping using in-context
learning with a ShortCoT LLM, (2) LongCoT supervised
finetuning, and (3) online training to further refine LongCoT skills. In the bootstrapping stage, only a minimal
number of in-context examples are required—in our experiments, we created just 10 examples—highlighting the
feasibility and efficiency of this approach. Using Llama


3.1-70B-Instruct as the bootstrapping model, we applied
BOLT to various model scales (7B, 8B, 70B), achieving
remarkable performance across diverse benchmarks, including Arena-Hard, MT-Bench, WildBench, ZebraLoigc, and
MATH500. These benchmarks cover: 1) challenging realuser queries involving information-seeking, creative writing,
coding, planning, and math, 2) classical logic puzzles, and
3) competition-level math problems. They test a broad spectrum of reasoning and task-solving skills, demonstrating
BOLT’s effectiveness in enhancing LongCoT capabilities.


Unlike black-box distillation, BOLT represents a white-box
approach to developing LongCoT reasoning in LLMs. To



2


**BOLT**



support future research, we will open-source our training
data and recipes and trained models. In summary, our work
provides a principled, cost-effective pathway for cultivating
LongCoT reasoning skills from ShortCoT models.


**2. Related Work**


**LongCoT and o1-like Model** OpenAI’s o1 model (Jaech
et al., 2024) employs long chain-of-thoughts, allowing it
to leverage rich reasoning actions, such as branching, reflection, verification, to tackle complex problems before
arriving at a final answer (Dutta et al., 2024). This approach enhances the model performance in areas such as
mathematics, coding, and scientific problems. The LongCoT approach aligns with System 2 cognition (Kahneman,
2011), a mode of deliberate and sequential reasoning that
mirrors human problem-solving strategies. By integrating
reinforcement learning, o1 can refine its reasoning process
dynamically, evaluating multiple solution paths, backtracking when necessary, and improving its approach through
iterative self-correction. The shift towards deliberative rea
soning represents an important trend in AI research, aiming
to make LLMs more transparent, interpretable, and adaptable in complex decision-making scenarios (Ackoff, 1994;
Kahneman, 2011).


Despite o1’s success, most existing attempts to replicate
LongCoT rely on knowledge distillation and manually
curated datasets (Min et al., 2024; Huang et al., 2024).
These approaches pose several challenges: they often fail
to generalize beyond the specific training data, require
access to high-quality reference models, and lack principled methods for directly training LongCoT reasoning from
scratch. A concurrent work by DeepSeek (Guo et al., 2025)
demonstrated that reinforcement learning applied to a 671Bparameter model can yield LongCoT capabilities. However,
such large-scale models introduce significant computational
barriers, making broad adoption and reproducibility infeasible. Furthermore, while DeepSeek provides significant
transparency regarding their approach, some crucial details,
particularly their data curation strategies, remain unclear.


**LLM Reinforcement Learning and Self-Improvement**
Reinforcement learning has become a core approach for
enhancing LLMs during post-training, particularly for improving the quality of model outputs. Traditional RL algorithms like Proximal Policy Optimization (PPO) (Schulman
et al., 2017) have been effective but computationally expensive, making them less feasible in resource-limited settings.
Recent efforts have proposed some efficient alternatives.


Rejection sampling techniques (Zelikman et al., 2022; Dong
et al., 2023; Gulcehre et al., 2023) filter and select the best
responses from multiple model-generated candidates, improving training efficiency without requiring full policy



optimization. Similarly, Direct Preference Optimization
(DPO) methods (Rafailov et al., 2023; Munos et al., 2023;
Ethayarajh et al., 2024) bypass explicit reward modeling
by directly optimizing on preference data, achieving performance comparable to PPO at significantly lower training
costs. Meanwhile, REINFORCE-based approaches (Ahmadian et al., 2024; Li et al., 2023; Ahmadian et al., 2024; Shao
et al., 2024) further streamline training by eliminating the
value function, reducing memory and computation requirements. These approaches have proven useful in downstream
tasks, yielding measurable gains in accuracy and coherence.
Building on them, recent work has explored self-improving
LLMs, where models iteratively refine their own outputs using generated feedback loops (Xu et al., 2023b; Hoang Tran,
2024; Xiong et al., 2023; Yuan et al., 2024b; Dong et al.,
2024; Guo et al., 2024). These approaches enable LLMs
to autonomously evaluate, critique, and improve their responses over multiple iterations, integrating self-feedback
into the training process. This fosters an adaptive learning
cycle, allowing models to progressively enhance reasoning
depth, factual accuracy, and coherence over time.


Despite these advances, most existing RL-based methods
focus on single-stage response generation, where models
directly produce final answers without refining intermediate
reasoning steps (e.g., internal states or “thoughts”). While
inference-time techniques like chain-of-thought prompting
(Wei et al., 2022) and self-consistency decoding (Wang
et al., 2022) have shown that explicit intermediate reasoning
improves accuracy, such multi-stage reasoning is rarely incorporated into training itself. Recent work on deliberationbased methods (Madaan et al., 2023) suggest that iterative
refinement enhances reasoning quality, but most RL-based
approaches lack mechanisms for models to revise, backtrack, or critique their own internal thought processes. As
a result, current models struggle to recover from early reasoning errors or refine suboptimal strategies, limiting their
robustness and adaptability.


**3. BOLT**


This paper introduces **BOLT** for learning LongCoT models
by bootstrapping long-form chain-of-thought from ShortCoT models. BOLT comprises of three stages: 1) LongCoT Bootstrapping which involves synthesizing LongCoT
data, 2) LongCoT Supervised Finetuning where we train a
ShortCoT model to adapt to the LongCoT format, incorporating reasoning elements and practicing extended chains of
thought before arriving at an external solution, 3) LongCoT
online training where the LongCoT SFT model is further improved through online exploration and onpolicy refinement.
An overview of BOLT is depicted in Figure 1.


Before discussing the method in detail, we introduce key notations. Let _x_ represent a query, _z_ denote internal thoughts,



3


**BOLT**


_Figure 3._ Topic distribution of query data in LongCoT Bootstrapping, _D_ b-query .


ated responses are of reasonable quality and serve as a good
basis for further processing.



_Figure 2._ An illustration of long chain-of-thought as internal
thoughts. Portions of the external solution are omitted for brevity.


and _y_ indicate an external solution. Additionally, we use _M_
to denote off-the-shelf LLMs and _π_ for models or policies
trained in our experiments.


**3.1. LongCoT Bootstrapping**


In our earlier experiments, we investigated various approaches for constructing LongCoT data such as prompt engineering on ShortCoT models and employing multi-agent
systems (with actor agent and judge agent). However, these
approaches were neither stable nor reliable. To address this,
we developed a simple yet effective method for generating
LongCoT data by bootstrapping ShortCoT LLMs.


3.1.1. L ONG C O T WITH I N -C ONTEXT L EARNING


ShortCoT models demonstrate some capability to produce
chain-of-thought reasoning and handle complex tasks. To
induce LongCoT, we leverage in-context examples of LongCoT to prompt ShortCoT models. These in-context examples guide the models to generate long-form chain-ofthought reasoning. Our findings reveal that as long as the
ShortCoT language model is sufficiently strong, the gener


To construct the in-context examples, each instance includes
a long-form chain-of-thought and its corresponding solution
derived from the reasoning process. See Figure 2 for an
example. The LongCoT incorporates essential reasoning
actions: problem analysis, planning, branching, and reflection. These actions mirror key elements of human reasoning.
By illustrating these actions via the in-context examples,
the in-context learning capacity of ShortCoT models enable
them to emulate LongCoT reasoning processes effectively.
We collect 10 in-context learning examples where each example consists of a query ( _x_ ), a chain of internal thoughts
( _z_ ), and an external solution ( _y_ ). Let’s denote the collection
of in-context examples as _D_ ICL = _{_ ( _x, y, z_ ) _}_ .


3.1.2. Q UERY M IXTURE C URATION


While most prior works focus primarily on math problem
solving, we believe that reasoning would benefit generic
tasks and improve an LLM’s helpfulness in general. Thus,
we construct a query distribution that covers a wide range
of topics and shift the distribution towards harder queries.
Our query curation pipeline involves three steps: 1) query
collection, 2) difficulty scoring and filtering, and 3) topic
tagging and sub-sampling.


We first collect a large set of high-quality instruction datasets
from public sources, such as ShareGPT (Chiang et al., 2023),
SlimOrca (Lian et al., 2023b), MathInstruct (Yue et al.,
2023), and Evol-Instruct (Xu et al., 2023a) (see the Appendix for a full list). Only the query (of the first turn if
it is a multi-turn chat) of each data instance is retained.



4


**BOLT**


enough (in terms of reasoning and instruction-following
capacity) _M_ bootstrapping . Empirically, while Llama-3.1-8BInstruct cannot reliably generate LongCoT responses following instructions and examples, Llama-3.1-70B-Instruct
work well. Let’s denote the sampled responses together with
the queries as _D_ bootstrapping [(][original][)] [=] _[ {]_ [(] _[x,][ {][y]_ _[i]_ _[, z]_ _[i]_ _[}]_ _i_ _[n]_ =1 _[}]_ [)]


3.1.4. R ESPONSE F ILTERING



_Figure 4._ An illustration of the prompt used in LongCoT Bootstrapping.


We next assign a difficulty level to each query. We follow the approach introduced by LMSys Team (Li et al.,
2024b) to select high quality user queries where seven criteria are considered: specificity, domain knowledge, complexity, problem-solving, creativity, technical accuracy, and
real-world application. We assign a binary (0/1) label to
each query on each criterion, and the quality or difficulty
level of each query is determined by the total over the seven
criteria. We keep queries with a score greater than or equal
to 5 . Third, we use a pipeline to assign a topic to each query.
We identify a list of high-level topics by analyzing a subset
of queries by LLM and human annotator. Then an LLM is
employed to assign each query a topic from the list. We subsample the dataset based on the topic distribution. Figure 3
displays the topic distribution after subsampling. Note that
coding and math problems still dominate the query mixture
after subsampling. This is due to two reasons: 1) coding
and math problems are generally harder and 2) coding and
math problem dominates our data sources, public instruct
data, due to current research community’s interest and their
relative ease of data curation. We denote the set of queries
from this step as _D_ b-query = _{x}_ where b-query indicates
bootstrapping query.


3.1.3. R ESPONSE G ENERATION


Given in-context examples, _D_ ICL, and the query set, _D_ b-query,
we sample _n_ responses ( _y, z_ ) from _M_ bootstrapping, in particular,


( _y, z_ ) _∼M_ bootstrapping ( _y, z|f_ formatting ( _x,_ _D_ [�] ICL )) _,_ (1)


where _x ∼D_ b-query, _D_ [�] ICL is a subset of _D_ ICL, _f_ formatting is
a template that wraps _x_ and _D_ [�] ICL as an LLM input (see
Figure 4). In our experiments, _n_ = 8, _|D_ [�] ICL _|_ = 3, and
_M_ bootstrapping is Llama-3.1-70B-Instruct (Grattafiori et al.,
2024). _M_ bootstrapping is a ShortCoT LLM, but its basic reasoning capacity and generic instruction-following capacity
enable it to generate responses following the format of long
chain-of-thoughts and demonstrating reasoning elements
in the thoughts. The procedure indeed requires a strong



While _D_ bootstrapping [(][original][)] [are of reasonable quality, we conduct]
filtering steps to further improve the LongCoT data. We
first use some heuristics and rules to filter out data where
the responses ( _y, z_ ) does not follow the particular format as
demonstrated in the in-context examples (see Figure 2).


We next filter out data with low-quality responses. Each
response consists of _z_ (internal thoughts) and _y_ (an external
solution). We don’t have access to a judge or reward model
on _z_ (while training such a reward model would require
LongCoT data in the first place). However, many reward
models and related data on judging the quality of _y_ have
been published. These models can be viewed as outcome
reward models (ORM) for a response with ( _z, y_ ) . Therefore
we use ORM to access the quality of _y_ and filtering data
instance based on its quality score. With an ORM, we have
a quality distribution of all _y_ _[′]_ _s_ from _D_ bootstrapping [(][original][)] [. We first re-]
move all data instance with score lower than 30th percentile
of the score distribution and then choose the response with
highest _y_ score. After the filtering steps, we obtain a high
quality LongCoT data _D_ bootstrapping = _{_ ( _x, y, z}_ ).


**3.2. LongCoT Supervised Finetuning**


With _D_ bootstrapping = _{_ ( _x, y, z}_ ), we can conduct supervised
finetuning on a ShortCoT model to allow it learn long form
chain-of-thought and reasoning elements involved in it and
the format of first producing internal thoughts and then
an external response. Note the ShortCoT model does not
necessarily need to be _M_ bootstrapping but can be other models
too. In our experiments, we apply LongCoT Supervised
Finetuning with _D_ bootstrapping to various models. Supervised
finetuning leads to an initial LongCoT model, _π_ 0 .


**3.3. LongCoT Online Training**


With the SFT model _π_ 0 as an initialization, we conduct
online training to further improve the policy, _π_ _θ_ ( _y, z | x_ ),
and it involves,


max _π_ _θ_ [E] _[x][∼D]_ [online] _[,y,z][∼][π]_ _[θ]_ [(] _[y,z][|][x]_ [)] � _r_ _ϕ_ ( _x, z, y_ )� _−_

_β_ D KL � _π_ _θ_ ( _y, z | x_ ) _|| π_ 0 ( _y, z | x_ )� _,_ (2)


where _r_ _ϕ_ ( _x, z, y_ ) is a reward model. Similar to the strategy
used in Section 3.1.4 Response Filtering, we use an outcome



5


**BOLT**



reward model, which assign a score to _y_ given _x_, that is,
_r_ _ϕ_ : _X × Y →_ R . In practice, we also include a rule-based
format reward to facilitate model response following the
defined format (see Figure 2).


The generic reward maximization with conservative constraint objective (Equation 2) can be instantiated with several variants such as DPO, REINFORCE, RLOO, PPO. In
our experiments, DPO works the best in terms of performance and efficiency and we choose DPO for our model
training. See Section 4.6 for an ablation.


**4. Experiments**


In this section, we demonstrate that BOLT is a highly effective approach to develop LongCoT capacities in ShortCoT LLMs. We begin with a comprehensive evaluation
of BOLT across diverse benchmarks and multiple model
scales, followed by a series of ablation studies to provide
deeper insights into the effectiveness of our approach.


**4.1. Experiment Setup**


**Evaluation Benchmarks** We focus on evaluating models’ reasoning capabilities across diverse domains, with an
emphasis on real-world queries. MT-Bench (Zheng et al.,
2023) is a widely used benchmark that covers multi-turn
questions spanning eight domains: writing, roleplay, reasoning, math, coding, information extraction, STEM, and the
humanities. Arena-Hard (Li et al., 2024a) comprises challenging prompts drawn from crowd-sourced datasets featuring real user queries, including ChatBot Arena (Li et al.,
2024a) and Wildchat-1M (Zhao et al., 2024). A significant
portion of Arena-Hard prompts involves real-world coding
problems. To minimize the influence of response length
and markdown formatting, we use the style-controlled version of Arena-Hard, known as Arena-Hard-SC, as our focus
is on the substance of model responses rather than their
writing style. WildBench (Lin et al., 2024b) further complements this evaluation by including challenging real-world
queries selected from over one million human-chatbot conversation logs. ZebraLogic (Lin et al., 2024a) specifically
targets logical reasoning, with each example being a logic
grid puzzle, a typical Constraint Satisfaction Problem (CSP)
commonly used to assess human reasoning in exams like
the LSAT. Lastly, MATH500 (Lightman et al., 2023) is a
representative subset of the MATH dataset (Hendrycks et al.,
2021), featuring problems drawn from various mathematics
competitions, including the AMC and AIME.


**Models** We apply BOLT to three models, Mistral-7BInstruct-v0.3 (Jiang et al., 2023), Meta-Llama-3.1-8BInstruct (Grattafiori et al., 2024), Meta-Llama-3.1-70BInstruct (Grattafiori et al., 2024), to test the effectiveness of
our method across different model scales. Besides instruct



models, we also tested a base model, Meta-Llama-3.1-8Bbase, as the initial model for our method and observe similar
enhancing effects. See Section 4.5 for an ablation study.


**Training Hyperparameters** In the first stage of BOLT,
LongCoT Bootstrapping (see Figure 1) generates a dataset
of 220k instances. LongCoT supervised finetuning is performed on this dataset for 4 epochs, with the final checkpoint used for the next training stage. Hyperparameters
(same for all models) include a maximum sequence length
of 16,384, a batch size of 128, a learning rate of 2e-5, a
cosine learning rate scheduler with a warm-up ratio of 0.1,
and AdamW (Loshchilov & Hutter, 2019; Kingma & Ba,
2014) as the optimizer. The training is conducted using Axolotl (Axolotl, 2025). Mistral-7B and Llama-8B are trained
on a single 8xH100 node, requiring about 6 hours, while
Llama-70B is trained on eight 8xH100 nodes, completing
in about 5 hours.


The LongCoT online training involves sampling from a
policy model, where sampling temperature is 1 _._ 0 and top-p
is 1 _._ 0 . Eight samples are sampled given each query. To
assign a reward to each online sample (external solution
in our method), we use ArmoRM-Llama3-8B (Wang et al.,
2024) as the reward model. An ablation study on the choice
of reward model is presented in Section 4.4. DPO trianing
hyperparameters include a regularization coefficient of _β_ =
0 _._ 1, a learning rate of 5e-7 with a cosine scheduler and a
warm-up ratio of 0.1, a batch size of 128, and AdamW as the
optimizer. Online training is conducted over 3 iterations and
each iteration consists of 2 epochs. Each iteration uses 33k
hard prompts selected from a list of open-sourced preference
data (see Appendix for the full list). The queries used in
our experiments will be open-sourced as well. Our DPO
training is based on an open-sourced library TRL (von Werra
et al., 2020). Mistral-7B and Llama-8B are trained on one
8xH100 node for about 14 hours. Llama-70B is trained on

eight 8xH100 for about 20 hours.


**4.2. Main Results**


The main results are presented in Figure 5. Across all models, our method achieves significant performance improvements on diverse benchmarks. These benchmarks feature

challenging real user queries and assess models on math,
coding, logical problem-solving, and general capabilities.
Additionally, we provide qualitative examples from BOLT
models in Appendix B, showcasing rich reasoning actions
in the long chain-of-thoughts such as problem understanding, branching, reflection, and verification. The consistent
improvements across benchmarks and model scales highlight that: 1) BOLT effectively transforms ShortCoT models
into LongCoT models with enhanced reasoning abilities, 2)
LongCoT plays a critical role in solving complex problems,
and 3) our method offers broad applicability.



6


(a) Mistral-7B


(b) Llama-3.1-8B


(c) Llama-3.1-70B



**BOLT**


**4.3. Performance Trajectory Over Training**


Figure 6 depicts the performance progression during the
BOLT training process. In the figure, _Init_ denotes the initial
model to which BOLT is applied, specifically Meta-Llama3.1-8B-Instruct in this case. After Bootstrapping SFT, we
observe significant performance gains compared to the initial model, highlighting the effectiveness of LongCoT data
synthesis through bootstrapping and the role of LongCoT
SFT in enabling a ShortCoT model to learn LongCoT reasoning. Moreover, LongCoT online training via DPO consistently boosts performance throughout the training trajectory,
underscoring the critical role of online training in further
refining the model and enhancing its LongCoT reasoning
abilities.


_Figure 6._ Performance trajectory over the training process of BOLT
on Llama-3.1-8B. Init indicates the initial model and in this case

is Meta-Llama-3.1-8B-Instruct.


**4.4. Ablation on Reward Models**



_Figure 5._ Performance of BOLT on Mistral-7B, Llama-3.1-8B, and
Llama-3.1-70B across benchmarks. These benchmarks consist of

challenging real user queries and test models’ math, coding, logical
reasoning and general capacity. Note: ArenaHard-SC means the
style controlled version of ArenaHard which controls for the effect
of length and markdown. The metric for ZebraLogic is the celllevel accuracy.



We investigate the impact of the reward model in the online DPO training process by comparing two Llama-8Bbased models known for strong performance on Reward
Bench (Lambert et al., 2024): ArmoRM-Llama3-8B (Wang
et al., 2024) and Skywork-Reward-Llama-3.1-8B (Liu et al.,
2024). According to Reward Bench, Skywork-Reward


7


**BOLT**



Arena-Hard-SC


Model Score Length


ArmoRM-Llama3-8B 44.1 674

Skywork-Reward-Llama-3.1-8B 51.6 890


WildBench


Model Score Length


ArmoRM-Llama3-8B 43.0 3354.51

Skywork-Reward-Llama-3.1-8B 49.2 4588.15


_Table 1._ Ablation on reward model in online DPO training.


Llama-3.1-8B outperforms ArmoRM-Llama3-8B. When
using Skywork-Reward-Llama-3.1-8B as the reward model
in BOLT, we observe stronger performance, as shown in
Table 1. However, this also leads to significantly longer
response lengths. Since concise responses with strong performance are generally preferred, and to avoid potential
length bias in our evaluation, we select ArmoRM-Llama38B as the reward model for online DPO training.


**4.5. Ablation on Initial Models**


Models Arena-Hard-SC WildBench


Meta-Llama-3.1-8B-Instruct 18.3 32.08


BOLT-Llama-3.1-8B-Base 41.3 39.79

BOLT-Llama-3.1-8B-Instruct 44.1 42.96


_Table 2._ Ablation on the initial model to which BOLT is applied.


In this section, we conduct an ablation study on the initial
model for BOLT. In previous experiments, all initial models were instruct models. Here, we apply BOLT to a base
model: Meta-Llama-3.1-8B-base. As shown in Table 2,
BOLT-Llama-3.1-8B-Base, while not performing as well as
BOLT-Llama-3.1-8B-Instruct, significantly surpasses MetaLlama-3.1-8B-Instruct. Importantly, for BOLT-Llama-3.18B-Base, no human-annotated data or synthesized data from
proprietary models is used during training, except for the
small set of LongCoT in-context examples used during bootstrapping. In contrast, Meta-Llama-3.1-8B-Instruct relies
heavily on a large amount of human-annotated data.


**4.6. Ablation on Online Training Algorithms**


We conduct an ablation study comparing four training algorithms for LongCoT online training: DPO, REINFORCE,
RLOO, and PPO. The results are shown in Table 3. Contrary to the intuition that algorithms with stronger online
learning capabilities might be more suitable for BOLT’s
online training setting, we find that DPO outperforms the



Algorithm Arena-Hard-SC WildBench


DPO 44.1 42.96

REINFORCE 38.3 37.07

RLOO 39.7 38.60

PPO 37.4 35.14


_Table 3._ Ablation on the learning algorithm for LongCoT online
training.


other approaches. Notably, the three REINFORCE variants
perform comparably, with even carefully tuned PPO failing
to match DPO’s performance.


We attribute this performance disparity primarily to the
inherent noise in LLM-based proxy rewards. DPO’s superior performance can be explained by its sampling strategy:
we generate 8 samples and select the highest-scored and
lowest-scored responses as positive and negative examples
for training. This approach effectively mitigates reward
noise by focusing on high-confidence extremes of the distribution where reward signals are less noisy. In contrast,
REINFORCE, RLOO, and PPO utilize all online samples
for training, including those in the middle of the distribution
where the reward signal suffers from higher label uncertainty compared to the tail samples. This distinction implies
that noise reduction via selective sampling (as in DPO) is
critical for success in our setting.


**5. Conclusion**


We explored an approach for developing long chain-ofthought (LongCoT) reasoning capabilities in large language
models without knowledge distillation from existing LongCoT models or extensive human annotations. We presented
BOLT, a novel three-stage approach that successfully bootstraps LongCoT capabilities from ShortCoT models. Our
work shows that complex reasoning abilities can be developed through a combination of in-context learning, supervising finetuning, and online training. A significant finding is
that the bootstrapping stage requires only minimal human effort. Just 10 examples were sufficient to initiate the process.
This finding has important implications for the scalability
and accessibility of developing LongCoT reasoning capabilities in LLMs. Using Llama-3.1-70B-Instruct as our bootstrapping model, we validated BOLT’s effectiveness across
different model scales (7B, 8B, 70B) and demonstrated its
robust performance on a diverse set of benchmarks involving
challenging real-world user queries. These results indicate
that BOLT successfully enables models to develop LongCoT reasoning capabilities that generalize across various
task domains. We believe this research paves the way for
scalable, efficient development of reasoning capabilities in
LLMs without depending on existing LongCoT models.



8


**BOLT**



**Impact Statement**


This paper presents work whose goal is to advance the field
of Machine Learning. There are many potential societal
consequences of our work, none which we feel must be
specifically highlighted here.


**References**


Ackoff, R. L. Systems thinking and thinking systems. _Sys-_
_tem dynamics review_, 10(2-3):175–188, 1994.


Ahmadian, A., Cremer, C., Galle, M., Fadaee, M., Kreutzer, ´
J., Pietquin, O., Ust [¨] un, A., and Hooker, S. Back to ba- ¨
sics: Revisiting reinforce style optimization for learning
from human feedback in llms. In _Proceedings of the_
_62nd Annual Meeting of the Association for Computa-_
_tional Linguistics (Volume 1: Long Papers)_, pp. 12248–
12267. Association for Computational Linguistics, 2024.
doi: 10.18653/v1/2024.acl-long.662. URL [https:](https://aclanthology.org/2024.acl-long.662/)
[//aclanthology.org/2024.acl-long.662/.](https://aclanthology.org/2024.acl-long.662/)


Axolotl. Axolotl: Open source fine-tuning, 2025. URL

[https://github.com/axolotl-ai-cloud/](https://github.com/axolotl-ai-cloud/axolotl)
[axolotl. Accessed: 2025-01-30.](https://github.com/axolotl-ai-cloud/axolotl)


Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang,
H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E.,
Stoica, I., and Xing, E. P. Vicuna: An open-source
chatbot impressing gpt-4 with 90%* chatgpt quality,
March 2023. URL [https://lmsys.org/blog/](https://lmsys.org/blog/2023-03-30-vicuna/)
[2023-03-30-vicuna/.](https://lmsys.org/blog/2023-03-30-vicuna/)


Cui, G., Yuan, L., Ding, N., Yao, G., Zhu, W., Ni, Y., Xie, G.,
Liu, Z., and Sun, M. Ultrafeedback: Boosting language
models with high-quality feedback, 2023.


Daniele, L. and Suphavadeeprasit. Amplify-instruct: Synthetically generated diverse multi-turn conversations for
effecient llm training. _arXiv preprint arXiv:(coming_
_soon)_, 2023. URL [https://huggingface.co/](https://huggingface.co/datasets/LDJnr/Capybara)
[datasets/LDJnr/Capybara.](https://huggingface.co/datasets/LDJnr/Capybara)


Dong, H., Xiong, W., Goyal, D., Zhang, Y., Chow, W.,
Pan, R., Diao, S., Zhang, J., SHUM, K., and Zhang, T.
RAFT: Reward ranked finetuning for generative foundation model alignment. _Transactions on Machine Learn-_
_ing Research_, 2023. ISSN 2835-8856. URL [https:](https://openreview.net/forum?id=m7p5O7zblY)
[//openreview.net/forum?id=m7p5O7zblY.](https://openreview.net/forum?id=m7p5O7zblY)


Dong, H., Xiong, W., Pang, B., Wang, H., Zhao, H., Zhou,
Y., Jiang, N., Sahoo, D., Xiong, C., and Zhang, T. Rlhf
workflow: From reward modeling to online rlhf. _arXiv_
_preprint arXiv:2405.07863_, 2024.


Dutta, S., Singh, J., Chakrabarti, S., and Chakraborty,
T. How to think step-by-step: A mechanistic under


standing of chain-of-thought reasoning. _arXiv preprint_
_arXiv:2402.18312_, 2024.


Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., and
Kiela, D. Kto: Model alignment as prospect theoretic
optimization. _arXiv preprint arXiv:2402.01306_, 2024.


Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian,
A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A.,
Vaughan, A., Yang, A., Fan, A., Goyal, A., Hartshorn,
A., Yang, A., Mitra, A., Sravankumar, A., Korenev,
A., Hinsvark, A., Rao, A., Zhang, A., Rodriguez, A.,
Gregerson, A., Spataru, A., Roziere, B., Biron, B., Tang,
B., Chern, B., Caucheteux, C., Nayak, C., Bi, C., Marra,
C., McConnell, C., Keller, C., Touret, C., Wu, C., Wong,
C., Ferrer, C. C., Nikolaidis, C., Allonsius, D., Song, D.,
Pintz, D., Livshits, D., Wyatt, D., Esiobu, D., Choudhary,
D., Mahajan, D., Garcia-Olano, D., Perino, D., Hupkes,
D., Lakomkin, E., AlBadawy, E., Lobanova, E., Dinan,
E., Smith, E. M., Radenovic, F., Guzman, F., Zhang, F., ´
Synnaeve, G., Lee, G., Anderson, G. L., Thattai, G., Nail,
G., Mialon, G., Pang, G., Cucurell, G., Nguyen, H., Korevaar, H., Xu, H., Touvron, H., Zarov, I., Ibarra, I. A.,
Kloumann, I., Misra, I., Evtimov, I., Zhang, J., Copet, J.,
Lee, J., Geffert, J., Vranes, J., Park, J., Mahadeokar, J.,
Shah, J., van der Linde, J., Billock, J., Hong, J., Lee, J.,
Fu, J., Chi, J., Huang, J., Liu, J., Wang, J., Yu, J., Bitton,
J., Spisak, J., Park, J., Rocca, J., Johnstun, J., Saxe, J., Jia,
J., Alwala, K. V., Prasad, K., Upasani, K., Plawiak, K., Li,
K., Heafield, K., Stone, K., El-Arini, K., Iyer, K., Malik,
K., Chiu, K., Bhalla, K., Lakhotia, K., Rantala-Yeary,
L., van der Maaten, L., Chen, L., Tan, L., Jenkins, L.,
Martin, L., Madaan, L., Malo, L., Blecher, L., Landzaat,
L., de Oliveira, L., Muzzi, M., Pasupuleti, M., Singh,
M., Paluri, M., Kardas, M., Tsimpoukelli, M., Oldham,
M., Rita, M., Pavlova, M., Kambadur, M., Lewis, M.,
Si, M., Singh, M. K., Hassan, M., Goyal, N., Torabi, N.,
Bashlykov, N., Bogoychev, N., Chatterji, N., Zhang, N.,
Duchenne, O., C¸ elebi, O., Alrassy, P., Zhang, P., Li, P.,
Vasic, P., Weng, P., Bhargava, P., Dubal, P., Krishnan,
P., Koura, P. S., Xu, P., He, Q., Dong, Q., Srinivasan,
R., Ganapathy, R., Calderer, R., Cabral, R. S., Stojnic,
R., Raileanu, R., Maheswari, R., Girdhar, R., Patel, R.,
Sauvestre, R., Polidoro, R., Sumbaly, R., Taylor, R., Silva,
R., Hou, R., Wang, R., Hosseini, S., Chennabasappa, S.,
Singh, S., Bell, S., Kim, S. S., Edunov, S., Nie, S., Narang,
S., Raparthy, S., Shen, S., Wan, S., Bhosale, S., Zhang,
S., Vandenhende, S., Batra, S., Whitman, S., Sootla, S.,
Collot, S., Gururangan, S., Borodinsky, S., Herman, T.,
Fowler, T., Sheasha, T., Georgiou, T., Scialom, T., Speckbacher, T., Mihaylov, T., Xiao, T., Karn, U., Goswami, V.,
Gupta, V., Ramanathan, V., Kerkez, V., Gonguet, V., Do,
V., Vogeti, V., Albiero, V., Petrovic, V., Chu, W., Xiong,
W., Fu, W., Meers, W., Martinet, X., Wang, X., Wang,
X., Tan, X. E., Xia, X., Xie, X., Jia, X., Wang, X., Gold


9


**BOLT**



schlag, Y., Gaur, Y., Babaei, Y., Wen, Y., Song, Y., Zhang,
Y., Li, Y., Mao, Y., Coudert, Z. D., Yan, Z., Chen, Z.,
Papakipos, Z., Singh, A., Srivastava, A., Jain, A., Kelsey,
A., Shajnfeld, A., Gangidi, A., Victoria, A., Goldstand,
A., Menon, A., Sharma, A., Boesenberg, A., Baevski, A.,
Feinstein, A., Kallet, A., Sangani, A., Teo, A., Yunus, A.,
Lupu, A., Alvarado, A., Caples, A., Gu, A., Ho, A., Poulton, A., Ryan, A., Ramchandani, A., Dong, A., Franco,
A., Goyal, A., Saraf, A., Chowdhury, A., Gabriel, A.,
Bharambe, A., Eisenman, A., Yazdan, A., James, B.,
Maurer, B., Leonhardi, B., Huang, B., Loyd, B., Paola,
B. D., Paranjape, B., Liu, B., Wu, B., Ni, B., Hancock,
B., Wasti, B., Spence, B., Stojkovic, B., Gamido, B.,
Montalvo, B., Parker, C., Burton, C., Mejia, C., Liu, C.,
Wang, C., Kim, C., Zhou, C., Hu, C., Chu, C.-H., Cai, C.,
Tindal, C., Feichtenhofer, C., Gao, C., Civin, D., Beaty,
D., Kreymer, D., Li, D., Adkins, D., Xu, D., Testuggine,
D., David, D., Parikh, D., Liskovich, D., Foss, D., Wang,
D., Le, D., Holland, D., Dowling, E., Jamil, E., Montgomery, E., Presani, E., Hahn, E., Wood, E., Le, E.-T.,
Brinkman, E., Arcaute, E., Dunbar, E., Smothers, E., Sun,
F., Kreuk, F., Tian, F., Kokkinos, F., Ozgenel, F., Caggioni, F., Kanayet, F., Seide, F., Florez, G. M., Schwarz,
G., Badeer, G., Swee, G., Halpern, G., Herman, G., Sizov,
G., Guangyi, Zhang, Lakshminarayanan, G., Inan, H.,
Shojanazeri, H., Zou, H., Wang, H., Zha, H., Habeeb, H.,
Rudolph, H., Suk, H., Aspegren, H., Goldman, H., Zhan,
H., Damlaj, I., Molybog, I., Tufanov, I., Leontiadis, I.,
Veliche, I.-E., Gat, I., Weissman, J., Geboski, J., Kohli,
J., Lam, J., Asher, J., Gaya, J.-B., Marcus, J., Tang, J.,
Chan, J., Zhen, J., Reizenstein, J., Teboul, J., Zhong, J.,
Jin, J., Yang, J., Cummings, J., Carvill, J., Shepard, J.,
McPhie, J., Torres, J., Ginsburg, J., Wang, J., Wu, K., U,
K. H., Saxena, K., Khandelwal, K., Zand, K., Matosich,
K., Veeraraghavan, K., Michelena, K., Li, K., Jagadeesh,
K., Huang, K., Chawla, K., Huang, K., Chen, L., Garg,
L., A, L., Silva, L., Bell, L., Zhang, L., Guo, L., Yu, L.,
Moshkovich, L., Wehrstedt, L., Khabsa, M., Avalani, M.,
Bhatt, M., Mankus, M., Hasson, M., Lennie, M., Reso,
M., Groshev, M., Naumov, M., Lathi, M., Keneally, M.,
Liu, M., Seltzer, M. L., Valko, M., Restrepo, M., Patel,
M., Vyatskov, M., Samvelyan, M., Clark, M., Macey,
M., Wang, M., Hermoso, M. J., Metanat, M., Rastegari,
M., Bansal, M., Santhanam, N., Parks, N., White, N.,
Bawa, N., Singhal, N., Egebo, N., Usunier, N., Mehta,
N., Laptev, N. P., Dong, N., Cheng, N., Chernoguz, O.,
Hart, O., Salpekar, O., Kalinli, O., Kent, P., Parekh, P.,
Saab, P., Balaji, P., Rittner, P., Bontrager, P., Roux, P.,
Dollar, P., Zvyagina, P., Ratanchandani, P., Yuvraj, P.,
Liang, Q., Alao, R., Rodriguez, R., Ayub, R., Murthy, R.,
Nayani, R., Mitra, R., Parthasarathy, R., Li, R., Hogan,
R., Battey, R., Wang, R., Howes, R., Rinott, R., Mehta,
S., Siby, S., Bondu, S. J., Datta, S., Chugh, S., Hunt, S.,
Dhillon, S., Sidorov, S., Pan, S., Mahajan, S., Verma,



S., Yamamoto, S., Ramaswamy, S., Lindsay, S., Lindsay,
S., Feng, S., Lin, S., Zha, S. C., Patil, S., Shankar, S.,
Zhang, S., Zhang, S., Wang, S., Agarwal, S., Sajuyigbe,
S., Chintala, S., Max, S., Chen, S., Kehoe, S., Satterfield, S., Govindaprasad, S., Gupta, S., Deng, S., Cho,
S., Virk, S., Subramanian, S., Choudhury, S., Goldman,
S., Remez, T., Glaser, T., Best, T., Koehler, T., Robinson,
T., Li, T., Zhang, T., Matthews, T., Chou, T., Shaked,
T., Vontimitta, V., Ajayi, V., Montanez, V., Mohan, V.,
Kumar, V. S., Mangla, V., Ionescu, V., Poenaru, V., Mihailescu, V. T., Ivanov, V., Li, W., Wang, W., Jiang, W.,
Bouaziz, W., Constable, W., Tang, X., Wu, X., Wang, X.,
Wu, X., Gao, X., Kleinman, Y., Chen, Y., Hu, Y., Jia, Y.,
Qi, Y., Li, Y., Zhang, Y., Zhang, Y., Adi, Y., Nam, Y., Yu,
Wang, Zhao, Y., Hao, Y., Qian, Y., Li, Y., He, Y., Rait,
Z., DeVito, Z., Rosnbrick, Z., Wen, Z., Yang, Z., Zhao,
Z., and Ma, Z. The llama 3 herd of models, 2024. URL
[https://arxiv.org/abs/2407.21783.](https://arxiv.org/abs/2407.21783)


Gulcehre, C., Paine, T. L., Srinivasan, S., Konyushkova,
K., Weerts, L., Sharma, A., Siddhant, A., Ahern, A.,
Wang, M., Gu, C., et al. Reinforced self-training (rest)
for language modeling. _arXiv preprint arXiv:2308.08998_,
2023.


Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R.,
Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement
learning. _arXiv preprint arXiv:2501.12948_, 2025.


Guo, S., Zhang, B., Liu, T., Liu, T., Khalman, M., Llinares,
F., Rame, A., Mesnard, T., Zhao, Y., Piot, B., et al. Direct
language model alignment from online ai feedback. _arXiv_
_preprint arXiv:2402.04792_, 2024.


Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart,
S., Tang, E., Song, D., and Steinhardt, J. Measuring
mathematical problem solving with the math dataset. In
Vanschoren, J. and Yeung, S.-K. (eds.), _Proceedings of_
_the Neural Information Processing Systems Track on_
_Datasets and Benchmarks_, volume 1, 2021. URL [https:](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf)
[//datasets-benchmarks-proceedings.](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf)
[neurips.cc/paper/2021/file/](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf)
[be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf)

[pdf.](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf)


Hoang Tran, Chris Glaze, B. H. Snorkel-mistral-pairrmdpo. [https://huggingface.co/snorkelai/](https://huggingface.co/snorkelai/Snorkel-Mistral-PairRM-DPO)
[Snorkel-Mistral-PairRM-DPO](https://huggingface.co/snorkelai/Snorkel-Mistral-PairRM-DPO), 2024. URL
[https://huggingface.co/snorkelai/](https://huggingface.co/snorkelai/Snorkel-Mistral-PairRM-DPO)

[Snorkel-Mistral-PairRM-DPO.](https://huggingface.co/snorkelai/Snorkel-Mistral-PairRM-DPO)


Huang, Z., Zou, H., Li, X., Liu, Y., Zheng, Y., Chern, E.,
Xia, S., Qin, Y., Yuan, W., and Liu, P. O1 replication
journey–part 2: Surpassing o1-preview through simple
distillation, big progress or bitter lesson? _arXiv preprint_
_arXiv:2411.16489_, 2024.



10


**BOLT**



Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky,
A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. Openai o1 system card. _arXiv preprint_
_arXiv:2412.16720_, 2024.


Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C.,
Chaplot, D. S., de las Casas, D., Bressand, F., Lengyel,
G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux, M.A., Stock, P., Scao, T. L., Lavril, T., Wang, T., Lacroix,
T., and Sayed, W. E. Mistral 7b, 2023. URL [https:](https://arxiv.org/abs/2310.06825)
[//arxiv.org/abs/2310.06825.](https://arxiv.org/abs/2310.06825)


Kahneman, D. _Thinking, Fast and Slow_ . Farrar, Straus and
Giroux, New York, 2011. ISBN 9780374275631.


Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_, 2014. URL
[https://arxiv.org/abs/1412.6980.](https://arxiv.org/abs/1412.6980)


Lambert, N., Pyatkin, V., Morrison, J., Miranda, L., Lin,
B. Y., Chandu, K., Dziri, N., Kumar, S., Zick, T., Choi,
Y., et al. Rewardbench: Evaluating reward models for
language modeling. _arXiv preprint arXiv:2403.13787_,
2024.


Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Wu, T., Zhu, B.,
Gonzalez, J. E., and Stoica, I. From crowdsourced data to
high-quality benchmarks: Arena-hard and benchbuilder
pipeline. _arXiv preprint arXiv:2406.11939_, 2024a.


Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Zhu,
B., Gonzalez, J. E., and Stoica, I. From live data
to high-quality benchmarks: The arena-hard pipeline,
April 2024b. URL [https://lmsys.org/blog/](https://lmsys.org/blog/2024-04-19-arena-hard/)
[2024-04-19-arena-hard/](https://lmsys.org/blog/2024-04-19-arena-hard/) . Accessed: 2025-01
26.


Li, Z., Xu, T., Zhang, Y., Yu, Y., Sun, R., and Luo, Z.Q. Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models.
_arXiv e-prints_, pp. arXiv–2310, 2023.


Lian, W., Goodson, B., Pentland, E., Cook, A., Vong, C.,
and ”Teknium”. Openorca: An open dataset of gpt
augmented flan reasoning traces. [https://https:](https://https://huggingface.co/Open-Orca/OpenOrca)
[//huggingface.co/Open-Orca/OpenOrca](https://https://huggingface.co/Open-Orca/OpenOrca),
2023a.


Lian, W., Wang, G., Goodson, B., Pentland, E., Cook, A.,
Vong, C., and ”Teknium”. Slimorca: An open dataset of
gpt-4 augmented flan reasoning traces, with verification,
2023b. URL [https://https://huggingface.](https://https://huggingface.co/Open-Orca/SlimOrca)
[co/Open-Orca/SlimOrca.](https://https://huggingface.co/Open-Orca/SlimOrca)


Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker,
B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and
Cobbe, K. Let’s verify step by step, 2023. URL [https:](https://arxiv.org/abs/2305.20050)
[//arxiv.org/abs/2305.20050.](https://arxiv.org/abs/2305.20050)


11



Lin, B. Y., Bras, R. L., and Choi, Y. Zebralogic: Benchmarking the logical reasoning ability of language models, 2024a. URL [https://huggingface.co/](https://huggingface.co/spaces/allenai/ZebraLogic)
[spaces/allenai/ZebraLogic.](https://huggingface.co/spaces/allenai/ZebraLogic)


Lin, B. Y., Deng, Y., Chandu, K., Brahman, F., Ravichander, A., Pyatkin, V., Dziri, N., Bras, R. L., and Choi,
Y. Wildbench: Benchmarking llms with challenging
tasks from real users in the wild, 2024b. URL [https:](https://arxiv.org/abs/2406.04770)
[//arxiv.org/abs/2406.04770.](https://arxiv.org/abs/2406.04770)


Liu, C. Y., Zeng, L., Liu, J., Yan, R., He, J., Wang, C.,
Yan, S., Liu, Y., and Zhou, Y. Skywork-reward: Bag
of tricks for reward modeling in llms. _arXiv preprint_
_arXiv:2410.18451_, 2024.


Loshchilov, I. and Hutter, F. Decoupled weight decay regularization, 2019. URL [https://arxiv.org/abs/](https://arxiv.org/abs/1711.05101)

[1711.05101.](https://arxiv.org/abs/1711.05101)


Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L.,
Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang,
Y., Gupta, S., Majumder, B. P., Hermann, K., Welleck,
S., Yazdanbakhsh, A., and Clark, P. Self-refine: Iterative
refinement with self-feedback, 2023. URL [https://](https://arxiv.org/abs/2303.17651)
[arxiv.org/abs/2303.17651.](https://arxiv.org/abs/2303.17651)


Min, Y., Chen, Z., Jiang, J., Chen, J., Deng, J., Hu, Y., Tang,
Y., Wang, J., Cheng, X., Song, H., et al. Imitate, explore,
and self-improve: A reproduction report on slow-thinking
reasoning systems. _arXiv preprint arXiv:2412.09413_,
2024.


Mitra, A., Khanpour, H., Rosset, C., and Awadallah, A.
Orca-math: Unlocking the potential of slms in grade
school math, 2024.


Munos, R., Valko, M., Calandriello, D., Azar, M. G., Rowland, M., Guo, Z. D., Tang, Y., Geist, M., Mesnard, T.,
Michi, A., et al. Nash learning from human feedback.
_arXiv preprint arXiv:2312.00886_, 2023.


Peng, B., Li, C., He, P., Galley, M., and Gao, J. Instruction tuning with gpt-4. _arXiv preprint arXiv:2304.03277_,
2023.


Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning,
C. D., and Finn, C. Direct preference optimization: Your
language model is secretly a reward model. _arXiv preprint_
_arXiv:2305.18290_, 2023.


Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and
Klimov, O. Proximal policy optimization algorithms.
_arXiv preprint arXiv:1707.06347_, 2017. URL [https:](https://arxiv.org/abs/1707.06347)
[//arxiv.org/abs/1707.06347.](https://arxiv.org/abs/1707.06347)


Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M.,
Li, Y., Wu, Y., and Guo, D. Deepseekmath: Pushing


**BOLT**


the limits of mathematical reasoning in open language
models. _arXiv preprint arXiv:2402.03300_, 2024.



Thrush, T., Lambert, N., Huang, S., Rasul, K., and
Gallouedec, Q. Trl: Transformer reinforcement learn- ´
ing. [https://github.com/huggingface/trl](https://github.com/huggingface/trl),
2020.



Team, G., Riviere, M., Pathak, S., Sess