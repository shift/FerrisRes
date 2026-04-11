## K IMI K 1.5: S CALING R EINFORCEMENT L EARNING WITH LLM S

T ECHNICAL R EPORT OF K IMI K 1.5


**Kimi Team**



**A** **BSTRACT**


Language model pretraining with next token prediction has proved effective for scaling compute but
is limited to the amount of available training data. Scaling reinforcement learning (RL) unlocks a new
axis for the continued improvement of artificial intelligence, with the promise that large language
models (LLMs) can scale their training data by learning to explore with rewards. However, prior
published work has not produced competitive results. In light of this, we report on the training practice
of Kimi k1.5, our latest multi-modal LLM trained with RL, including its RL training techniques,
multi-modal data recipes, and infrastructure optimization. Long context scaling and improved policy
optimization methods are key ingredients of our approach, which establishes a simplistic, effective
RL framework without relying on more complex techniques such as Monte Carlo tree search, value
functions, and process reward models. Notably, our system achieves state-of-the-art reasoning
performance across multiple benchmarks and modalities—e.g., 77.5 on AIME, 96.2 on MATH
500, 94-th percentile on Codeforces, 74.9 on MathVista—matching OpenAI’s o1. Moreover, we
present effective long2short methods that use long-CoT techniques to improve short-CoT models,
yielding state-of-the-art short-CoT reasoning results—e.g., 60.8 on AIME, 94.6 on MATH500, 47.3
on LiveCodeBench—outperforming existing short-CoT models such as GPT-4o and Claude Sonnet
3.5 by a large margin (up to +550%).



Kimi k1.5 long-CoT OpenAI o1 OpenAI o1-mini QVQ-72B-Preview QwQ-32B Preview

































Figure 1: Kimi k1.5 long-CoT results.


Kimi k1.5 T ECHNICAL R EPORT



Kimi k1.5 short-CoT OpenAI 4o Claude 3.5 Sonnet Qwen2-VL LLaMA-3.1 405B-Inst. DeepSeek V3



Qwen2.5 72B-Inst.

























































































Figure 2: Kimi k1.5 short-CoT results.


**1** **Introduction**


Language model pretraining with next token prediction has been studied under the context of the scaling law, where
proportionally scaling model parameters and data sizes leads to the continued improvement of intelligence. (Kaplan
et al. 2020; Hoffmann et al. 2022) However, this approach is limited to the amount of available high-quality training
data (Villalobos et al. 2024; Muennighoff et al. 2023). In this report, we present the training recipe of Kimi k1.5,
our latest multi-modal LLM trained with reinforcement learning (RL). The goal is to explore a possible new axis
for continued scaling. Using RL with LLMs, the models learns to explore with rewards and thus is not limited to a
pre-existing static dataset.


There are a few key ingredients about the design and training of k1.5.


  - **Long context scaling** . We scale the context window of RL to 128k and observe continued improvement of
performance with an increased context length. A key idea behind our approach is to use partial rollouts to improve
training efficiency—i.e., sampling new trajectories by reusing a large chunk of previous trajectories, avoiding
the cost to re-generate the new trajectories from scratch. Our observation identifies the context length as a key
dimension of the continued scaling of RL with LLMs.

  - **Improved policy optimization** . We derive a formulation of RL with long-CoT and employ a variant of online
mirror descent for robust policy optimization. This algorithm is further improved by our effective sampling strategy,
length penalty, and optimization of the data recipe.

  - **Simplistic Framework** . Long context scaling, combined with the improved policy optimization methods, establishes a simplistic RL framework for learning with LLMs. Since we are able to scale the context length, the learned
CoTs exhibit the properties of planning, reflection, and correction. An increased context length has an effect of
increasing the number of search steps. As a result, we show that strong performance can be achieved without
relying on more complex techniques such as Monte Carlo tree search, value functions, and process reward models.

  - **Multimodalities** . Our model is jointly trained on text and vision data, which has the capabilities of jointly reasoning
over the two modalities.


Moreover, we present effective long2short methods that use long-CoT techniques to improve short-CoT models.
Specifically, our approaches include applying length penalty with long-CoT activations and model merging.


Our long-CoT version achieves state-of-the-art reasoning performance across multiple benchmarks and modalities—e.g.,
77.5 on AIME, 96.2 on MATH 500, 94-th percentile on Codeforces, 74.9 on MathVista—matching OpenAI’s o1. Our
model also achieves state-of-the-art short-CoT reasoning results—e.g., 60.8 on AIME, 94.6 on MATH500, 47.3 on
LiveCodeBench—outperforming existing short-CoT models such as GPT-4o and Claude Sonnet 3.5 by a large margin
(up to +550%). Results are shown in Figures 1 and 2.


2


Kimi k1.5 T ECHNICAL R EPORT


**2** **Approach: Reinforcement Learning with LLMs**


The development of Kimi k1.5 consists of several stages: pretraining, vanilla supervised fine-tuning (SFT), long-CoT
supervised fine-turning, and reinforcement learning (RL). This report focuses on RL, beginning with an overview of
the RL prompt set curation (Section 2.1) and long-CoT supervised finetuning (Section 2.2), followed by an in-depth
discussion of RL training strategies in Section 2.3. Additional details on pretraining and vanilla supervised finetuning
can be found in Section 2.5.


**2.1** **RL Prompt Set Curation**


Through our preliminary experiments, we found that the quality and diversity of the RL prompt set play a critical role in
ensuring the effectiveness of reinforcement learning. A well-constructed prompt set not only guides the model toward
robust reasoning but also mitigates the risk of reward hacking and overfitting to superficial patterns. Specifically, three
key properties define a high-quality RL prompt set:


  - **Diverse Coverage** : Prompts should span a wide array of disciplines, such as STEM, coding, and general reasoning,
to enhance the model’s adaptability and ensure broad applicability across different domains.


  - **Balanced Difficulty** : The prompt set should include a well-distributed range of easy, moderate, and difficult
questions to facilitate gradual learning and prevent overfitting to specific complexity levels.


  - **Accurate Evaluability** : Prompts should allow objective and reliable assessment by verifiers, ensuring that model
performance is measured based on correct reasoning rather than superficial patterns or random guess.


To achieve diverse coverage in the prompt set, we employ automatic filters to select questions that require rich
reasoning and are straightforward to evaluate. Our dataset includes problems from various domains, such as STEM
fields, competitions, and general reasoning tasks, incorporating both text-only and image-text question-answering
data. Furthermore, we developed a tagging system to categorize prompts by domain and discipline, ensuring balanced
representation across different subject areas (M. Li et al. 2023; W. Liu et al. 2023).


We adopt a model-based approach that leverages the model’s own capacity to adaptively assess the difficulty of each
prompt. Specifically, for every prompt, an SFT model generates answers ten times using a relatively high sampling
temperature. The pass rate is then calculated and used as a proxy for the prompt’s difficulty—the lower the pass rate,
the higher the difficulty. This approach allows difficulty evaluation to be aligned with the model’s intrinsic capabilities,
making it highly effective for RL training. By leveraging this method, we can prefilter most trivial cases and easily
explore different sampling strategies during RL training.


To avoid potential reward hacking (Everitt et al. 2021; Pan et al. 2022), we need to ensure that both the reasoning
process and the final answer of each prompt can be accurately verified. Empirical observations reveal that some
complex reasoning problems may have relatively simple and easily guessable answers, leading to false positive
verification—where the model reaches the correct answer through an incorrect reasoning process. To address this
issue, we exclude questions that are prone to such errors, such as multiple-choice, true/false, and proof-based questions.
Furthermore, for general question-answering tasks, we propose a simple yet effective method to identify and remove
easy-to-hack prompts. Specifically, we prompt a model to guess potential answers without any CoT reasoning steps.
If the model predicts the correct answer within _N_ attempts, the prompt is considered too easy-to-hack and removed.
We found that setting _N_ = 8 can remove the majority easy-to-hack prompts. Developing more advanced verification
models remains an open direction for future research.


**2.2** **Long-CoT Supervised Fine-Tuning**


With the refined RL prompt set, we employ prompt engineering to construct a small yet high-quality long-CoT warmup
dataset, containing accurately verified reasoning paths for both text and image inputs. This approach resembles rejection
sampling (RS) but focuses on generating long-CoT reasoning paths through prompt engineering. The resulting warmup
dataset is designed to encapsulate key cognitive processes that are fundamental to human-like reasoning, such as
**planning**, where the model systematically outlines steps before execution; **evaluation**, involving critical assessment of
intermediate steps; **reflection**, enabling the model to reconsider and refine its approach; and **exploration**, encouraging
consideration of alternative solutions. By performing a lightweight SFT on this warm-up dataset, we effectively prime
the model to internalize these reasoning strategies. As a result, the fine-tuned long-CoT model demonstrates improved
capability in generating more detailed and logically coherent responses, which enhances its performance across diverse
reasoning tasks.


3


Kimi k1.5 T ECHNICAL R EPORT


**2.3** **Reinforcement Learning**


**2.3.1** **Problem Setting**


Given a training dataset _D_ = _{_ ( _x_ _i_ _, y_ _i_ _[∗]_ [)] _[}]_ _[n]_ _i_ =1 [of problems] _[ x]_ _[i]_ [ and corresponding ground truth answers] _[ y]_ _i_ _[∗]_ [, our goal is to]
train a policy model _π_ _θ_ to accurately solve test problems. In the context of complex reasoning, the mapping of problem
_x_ to solution _y_ is non-trivial. To tackle this challenge, the _chain of thought_ (CoT) method proposes to use a sequence
of intermediate steps _z_ = ( _z_ 1 _, z_ 2 _, . . ., z_ _m_ ) to bridge _x_ and _y_, where each _z_ _i_ is a coherent sequence of tokens that acts
as a significant intermediate step toward solving the problem (J. Wei et al. 2022). When solving problem _x_, thoughts
_z_ _t_ _∼_ _π_ _θ_ ( _·|x, z_ 1 _, . . ., z_ _t−_ 1 ) are auto-regressively sampled, followed by the final answer _y ∼_ _π_ _θ_ ( _·|x, z_ 1 _, . . ., z_ _m_ ) . We
use _y, z ∼_ _π_ _θ_ to denote this sampling procedure. Note that both the thoughts and final answer are sampled as a language

sequence.


To further enhance the model’s reasoning capabilities, _planning_ algorithms are employed to explore various thought
processes, generating improved CoT at inference time (Yao et al. 2024; Y. Wu et al. 2024; Snell et al. 2024). The
core insight of these approaches is the explicit construction of a search tree of thoughts guided by value estimations.
This allows the model to explore diverse continuations of a thought process or backtrack to investigate new directions
when encountering dead ends. In more detail, let _T_ be a search tree where each node represents a partial solution
_s_ = ( _x, z_ 1: _|s|_ ) . Here _s_ consists of the problem _x_ and a sequence of thoughts _z_ 1: _|s|_ = ( _z_ 1 _, . . ., z_ _|s|_ ) leading up to
that node, with _|s|_ denoting number of thoughts in the sequence. The planning algorithm uses a critic model _v_ to
provide feedback _v_ ( _x, z_ 1: _|s|_ ), which helps evaluate the current progress towards solving the problem and identify any
errors in the existing partial solution. We note that the feedback can be provided by either a discriminative score or a
language sequence(L. Zhang et al. 2024). Guided by the feedbacks for all _s ∈T_, the planning algorithm selects the
most promising node for expansion, thereby growing the search tree. The above process repeats iteratively until a full
solution is derived.


We can also approach planning algorithms from an _algorithmic perspective_ . Given past search history available at the
_t_ -th iteration ( _s_ 1 _, v_ ( _s_ 1 ) _, . . ., s_ _t−_ 1 _, v_ ( _s_ _t−_ 1 )), a planning algorithm _A_ iteratively determines the next search direction
_A_ ( _s_ _t_ _|s_ 1 _, v_ ( _s_ 1 ) _, . . ., s_ _t−_ 1 _, v_ ( _s_ _t−_ 1 )) and provides feedbacks for the current search progress _A_ ( _v_ ( _s_ _t_ ) _|s_ 1 _, v_ ( _s_ 1 ) _, . . ., s_ _t_ ) .
Since both thoughts and feedbacks can be viewed as intermediate reasoning steps, and these components can both be
represented as sequence of language tokens, we use _z_ to replace _s_ and _v_ to simplify the notations. Accordingly, we view
a planning algorithm as a mapping that directly acts on a sequence of reasoning steps _A_ ( _·|z_ 1 _, z_ 2 _, . . ._ ) . In this framework,
all information stored in the search tree used by the planning algorithm is flattened into the full context provided to the
algorithm. This provides an intriguing perspective on generating high-quality CoT: Rather than explicitly constructing a
search tree and implementing a planning algorithm, we could potentially train a model to approximate this process. Here,
the number of thoughts (i.e., language tokens) serves as an analogy to the computational budget traditionally allocated
to planning algorithms. Recent advancements in long context windows facilitate seamless scalability during both the
training and testing phases. If feasible, this method enables the model to run an implicit search over the reasoning space
directly via auto-regressive predictions. Consequently, the model not only learns to solve a set of training problems but
also develops the ability to tackle individual problems effectively, leading to improved generalization to unseen test
problems.


We thus consider training the model to generate CoT with reinforcement learning (RL) (OpenAI 2024). Let _r_ be a
reward model that justifies the correctness of the proposed answer _y_ for the given problem _x_ based on the ground truth
_y_ _[∗]_, by assigning a value _r_ ( _x, y, y_ _[∗]_ ) _∈{_ 0 _,_ 1 _}_ . For verifiable problems, the reward is directly determined by predefined
criteria or rules. For example, in coding problems, we assess whether the answer passes the test cases. For problems
with free-form ground truth, we train a reward model _r_ ( _x, y, y_ _[∗]_ ) that predicts if the answer matches the ground truth.
Given a problem _x_, the model _π_ _θ_ generates a CoT and the final answer through the sampling procedure _z ∼_ _π_ _θ_ ( _·|x_ ),
_y ∼_ _π_ _θ_ ( _·|x, z_ ) . The quality of the generated CoT is evaluated by whether it can lead to a correct final answer. In
summary, we consider the following objective to optimize the policy


max E ( _x,y_ _∗_ ) _∼D,_ ( _y,z_ ) _∼π_ _θ_ [ _r_ ( _x, y, y_ _[∗]_ )] _._ (1)
_θ_


By scaling up RL training, we aim to train a model that harnesses the strengths of both simple prompt-based CoT
and planning-augmented CoT. The model still auto-regressively sample language sequence during inference, thereby
circumventing the need for the complex parallelization required by advanced planning algorithms during deployment.
However, a key distinction from simple prompt-based methods is that the model should not merely follow a series of
reasoning steps. Instead, it should also learn critical planning skills including error identification, backtracking and
solution refinement by leveraging the entire set of explored thoughts as contextual information.


4


Kimi k1.5 T ECHNICAL R EPORT


**2.3.2** **Policy Optimization**


We apply a variant of online policy mirror decent as our training algorithm (Abbasi-Yadkori et al. 2019; Mei et al. 2019;
Tomar et al. 2020). The algorithm performs iteratively. At the _i_ -th iteration, we use the current model _π_ _θ_ _i_ as a reference
model and optimize the following relative entropy regularized policy optimization problem,


max E ( _x,y_ _∗_ ) _∼D_ �E ( _y,z_ ) _∼π_ _θ_ [ _r_ ( _x, y, y_ _[∗]_ )] _−_ _τ_ KL( _π_ _θ_ ( _x_ ) _||π_ _θ_ _i_ ( _x_ ))� _,_ (2)
_θ_


where _τ >_ 0 is a parameter controlling the degree of regularization. This objective has a closed form solution


_π_ _[∗]_ ( _y, z|x_ ) = _π_ _θ_ _i_ ( _y, z|x_ ) exp( _r_ ( _x, y, y_ _[∗]_ ) _/τ_ ) _/Z ._


Here _Z_ = [�] _y_ _[′]_ _,z_ _[′]_ _[ π]_ _[θ]_ _i_ [(] _[y]_ _[′]_ _[, z]_ _[′]_ _[|][x]_ [) exp(] _[r]_ [(] _[x, y]_ _[′]_ _[, y]_ _[∗]_ [)] _[/τ]_ [)] [ is the normalization factor. Taking logarithm of both sides we have]

for _any_ ( _y, z_ ) the following constraint is satisfied, which allows us to leverage off-policy data during optimization

_r_ ( _x, y, y_ _[∗]_ ) _−_ _τ_ log _Z_ = _τ_ log _[π]_ _[∗]_ [(] _[y,][ z][|][x]_ [)]

_π_ _θ_ _i_ ( _y, z|x_ ) _[.]_


This motivates the following surrogate loss



� 2 [��]



_L_ ( _θ_ ) = E ( _x,y_ _∗_ ) _∼D_



�



E ( _y,z_ ) _∼π_ _θi_



_r_ ( _x, y, y_ _[∗]_ ) _−_ _τ_ log _Z −_ _τ_ log _[π]_ _[θ]_ [(] _[y,][ z][|][x]_ [)]

_π_ _θ_ _i_ ( _y, z|x_ )

��



_._



_k_

To approximate _τ_ log _Z_, we use samples ( _y_ 1 _, z_ 1 ) _, . . .,_ ( _y_ _k_ _, z_ _k_ ) _∼_ _π_ _θ_ _i_ : _τ_ log _Z ≈_ _τ_ log _k_ [1] � _j_ =1 [exp(] _[r]_ [(] _[x, y]_ _[j]_ _[, y]_ _[∗]_ [)] _[/τ]_ [)] [.]

We also find that using empirical mean of sampled rewards ~~_r_~~ = mean( _r_ ( _x, y_ 1 _, y_ _[∗]_ ) _, . . ., r_ ( _x, y_ _k_ _, y_ _[∗]_ )) yields effective
practical results. This is reasonable since _τ_ log _Z_ approaches the expected reward under _π_ _θ_ _i_ as _τ →∞_ . Finally, we
conclude our learning algorithm by taking the gradient of surrogate loss. For each problem _x_, _k_ responses are sampled
using the reference policy _π_ _θ_ _i_, and the gradient is given by



�



� 2 [�]



1

_k_



_k_
�

_j_ =1



_∇_ _θ_ log _π_ _θ_ ( _y_ _j_ _, z_ _j_ _|x_ )( _r_ ( _x, y_ _j_ _, y_ _[∗]_ ) _−_ ~~_r_~~ ~~)~~ _−_ _[τ]_



2 _[∇]_ _[θ]_



log _[π]_ _[θ]_ [(] _[y]_ _[j]_ _[,][ z]_ _[j]_ _[|][x]_ [)]
� _π_ _θ_ _i_ ( _y_ _j_ _, z_ _j_ _|x_ )



_._ (3)



To those familiar with policy gradient methods, this gradient resembles the policy gradient of (2) using the mean of
sampled rewards as the baseline (Kool et al. 2019; Ahmadian et al. 2024). The main differences are that the responses
are sampled from _π_ _θ_ _i_ rather than on-policy, and an _l_ 2 -regularization is applied. Thus we could see this as the natural
extension of a usual on-policy regularized policy gradient algorithm to the off-policy case (Nachum et al. 2017). We
sample a batch of problems from _D_ and update the parameters to _θ_ _i_ +1, which subsequently serves as the reference
policy for the next iteration. Since each iteration considers a different optimization problem due to the changing
reference policy, we also reset the optimizer at the start of each iteration.


We exclude the value network in our training system which has also been exploited in previous studies (Ahmadian et al.
2024). While this design choice significantly improves training efficiency, we also hypothesize that the conventional use
of value functions for credit assignment in classical RL may not be suitable for our context. Consider a scenario where
the model has generated a partial CoT ( _z_ 1 _, z_ 2 _, . . ., z_ _t_ ) and there are two potential next reasoning steps: _z_ _t_ +1 and _z_ _t_ _[′]_ +1 [.]
Assume that _z_ _t_ +1 directly leads to the correct answer, while _z_ _t_ _[′]_ +1 [contains some errors. If an oracle value function were]
accessible, it would indicate that _z_ _t_ +1 preserves a higher value compared to _z_ _t_ _[′]_ +1 [. According to the standard credit]
assignment principle, selecting _z_ _t_ _[′]_ +1 [would be penalized as it has a negative advantages relative to the current policy.]
However, exploring _z_ _t_ _[′]_ +1 [is extremely valuable for training the model to generate long CoT. By using the justification of]
the final answer derived from a long CoT as the reward signal, the model can learn the pattern of trial and error from
taking _z_ _t_ _[′]_ +1 [as long as it successfully recovers and reaches the correct answer. The key takeaway from this example is]
that we should encourage the model to explore diverse reasoning paths to enhance its capability in solving complex
problems. This exploratory approach generates a wealth of experience that supports the development of critical planning
skills. Our primary goal is not confined to attaining high accuracy on training problems but focuses on equipping the
model with effective problem-solving strategies, ultimately improving its performance on test problems.


**2.3.3** **Length Penalty**


We observe an overthinking phenomenon that the model’s response length significantly increases during RL training.
Although this leads to better performance, an excessively lengthy reasoning process is costly during training and
inference, and overthinking is often not preferred by humans. To address this issue, we introduce a length reward to
restrain the rapid growth of token length, thereby improving the model’s token efficiency. Given _k_ sampled responses


5


Kimi k1.5 T ECHNICAL R EPORT


( _y_ 1 _, z_ 1 ) _, . . .,_ ( _y_ _k_ _, z_ _k_ ) of problem _x_ with true answer _y_ _[∗]_, let len( _i_ ) be the length of ( _y_ _i_ _, z_ _i_ ), min_len = min _i_ len( _i_ ) and
max_len = max _i_ len( _i_ ) . If max_len = min_len, we set length reward zero for all responses, as they have the same
length. Otherwise the length reward is given by


_λ_ If _r_ ( _x, y_ _i_ _, y_ _[∗]_ ) = 1 len( _i_ ) _−_ min_len
len_reward(i) = where _λ_ = 0 _._ 5 _−_
�min(0 _, λ_ ) If _r_ ( _x, y_ _i_ _, y_ _[∗]_ ) = 0 _[,]_ max_len _−_ min_len _[.]_


In essence, we promote shorter responses and penalize longer responses among correct ones, while explicitly penalizing
long responses with incorrect answers. This length-based reward is then added to the original reward with a weighting
parameter.


In our preliminary experiments, length penalty may slow down training during the initial phases. To alleviate this
issue, we propose to gradually warm up the length penalty during training. Specifically, we employ standard policy
optimization without length penalty, followed by a constant length penalty for the rest of training.


**2.3.4** **Sampling Strategies**


Although RL algorithms themselves have relatively good sampling properties (with more difficult problems providing
larger gradients), their training efficiency is limited. Consequently, some well-defined prior sampling methods can yield
potentially greater performance gains. We exploit multiple signals to further improve the sampling strategy. First, the
RL training data we collect naturally come with different difficulty labels. For example, a math competition problem is
more difficult than a primary school math problem. Second, because the RL training process samples the same problem
multiple times, we can also track the success rate for each individual problem as a metric of difficulty. We propose two
sampling methods to utilize these priors to improve training efficiency.


**Curriculum Sampling** We start by training on easier tasks and gradually progress to more challenging ones. Since
the initial RL model has limited performance, spending a restricted computation budget on very hard problems often
yields few correct samples, resulting in lower training efficiency. Meanwhile, our collected data naturally includes grade
and difficulty labels, making difficulty-based sampling an intuitive and effective way to improve training efficiency.


**Prioritized Sampling** In addition to curriculum sampling, we use a prioritized sampling strategy to focus on problems
where the model underperforms. We track the success rates _s_ _i_ for each problem _i_ and sample problems proportional to
1 _−_ _s_ _i_, so that problems with lower success rates receive higher sampling probabilities. This directs the model’s efforts
toward its weakest areas, leading to faster learning and better overall performance.


**2.3.5** **More Details on Training Recipe**


**Test Case Generation for Coding** Since test cases are not available for many coding problems from the web, we
design a method to automatically generate test cases that serve as a reward to train our model with RL. Our focus is
primarily on problems that do not require a special judge. We also assume that ground truth solutions are available for
these problems so that we can leverage the solutions to generate higher quality test cases.


We utilize the widely recognized test case generation library, CYaRon [1], to enhance our approach. We employ our
base Kimi k1.5 to generate test cases based on problem statements. The usage statement of CYaRon and the problem
description are provided as the input to the generator. For each problem, we first use the generator to produce 50 test
cases and also randomly sample 10 ground truth submissions for each test case. We run the test cases against the
submissions. A test case is deemed valid if at least 7 out of 10 submissions yield matching results. After this round of
filtering, we obtain a set of selected test cases. A problem and its associated selected test cases are added to our training
set if at least 9 out of 10 submissions pass the entire set of selected test cases.


In terms of statistics, from a sample of 1,000 online contest problems, approximately 614 do not require a special
judge. We developed 463 test case generators that produced at least 40 valid test cases, leading to the inclusion of 323
problems in our training set.


**Reward Modeling for Math** One challenge in evaluating math solutions is that different written forms can represent
the same underlying answer. For instance, _a_ [2] _−_ 4 and ( _a_ + 2)( _a −_ 2) may both be valid solutions to the same problem.
We adopted two methods to improve the reward model’s scoring accuracy:


1. Classic RM: Drawing inspiration from the InstructGPT (Ouyang et al. 2022) methodology, we implemented a
value-head based reward model and collected approximately 800k data points for fine-tuning. The model ultimately


1 https://github.com/luogu-dev/cyaron


6


Kimi k1.5 T ECHNICAL R EPORT


takes as input the “question,” the “reference answer,” and the “response,” and outputs a single scalar that indicates
whether the response is correct.


2. Chain-of-Thought RM: Recent research (Ankner et al. 2024; McAleese et al. 2024) suggests that reward models
augmented with chain-of-thought (CoT) reasoning can significantly outperform classic approaches, particularly
on tasks where nuanced correctness criteria matter—such as mathematics. Therefore, we collected an equally
large dataset of about 800k CoT-labeled examples to fine-tune the Kimi model. Building on the same inputs as the
Classic RM, the chain-of-thought approach explicitly generates a step-by-step reasoning process before providing
a final correctness judgment in JSON format, enabling more robust and interpretable reward signals.


During our manual spot checks, the Classic RM achieved an accuracy of approximately **84.4**, while the Chain-ofThought RM reached **98.5** accuracy. In the RL training process, we adopted the Chain-of-Thought RM to ensure more
correct feedback.


**Vision Data** To improve the model’s real-world image reasoning capabilities and to achieve a more effective alignment
between visual inputs and large language models (LLMs), our vision reinforcement learning (Vision RL) data is primarily
sourced from three distinct categories: Real-world data, Synthetic visual reasoning data, and Text-rendered data.


1. The real-world data encompass a range of science questions across various grade levels that require graphical
comprehension and reasoning, location guessing tasks that necessitate visual perception and inference, and data
analysis that involves understanding complex charts, among other types of data. These datasets improve the model’s
ability to perform visual reasoning in real-world scenarios.


2. Synthetic visual reasoning data is artificially generated, including procedurally created images and scenes aimed
at improving specific visual reasoning skills, such as understanding spatial relationships, geometric patterns, and
object interactions. These synthetic datasets offer a controlled environment for testing the model’s visual reasoning
capabilities and provide an endless supply of training examples.


3. Text-rendered data is created by converting textual content into visual format, enabling the model to maintain
consistency when handling text-based queries across different modalities. By transforming text documents, code
snippets, and structured data into images, we ensure the model provides consistent responses regardless of whether
the input is pure text or text rendered as images (like screenshots or photos). This also helps to enhance the model’s
capability when dealing with text-heavy images.


Each type of data is essential in building a comprehensive visual language model that can effectively manage a wide
range of real-world applications while ensuring consistent performance across various input modalities.


**2.4** **Long2short: Context Compression for Short-CoT Models**


Though long-CoT models achieve strong performance, it consumes more test-time tokens compared to standard
short-CoT LLMs. However, it is possible to transfer the thinking priors from long-CoT models to short-CoT models
so that performance can be improved even with limited test-time token budgets. We present several approaches for
this long2short problem, including model merging (Yang et al. 2024), shortest rejection sampling, DPO (Rafailov et al.
2024), and long2short RL. Detailed descriptions of these methods are provided below:


**Model Merging** Model merging has been found to be useful in maintaining generalization ability. We also discovered
its effectiveness in improving token efficiency when merging a long-cot model and a short-cot model. This approach
combines a long-cot model with a shorter model to obtain a new one without training. Specifically, we merge the two
models by simply averaging their weights.


**Shortest Rejection Sampling** We observed that our model generates responses with a large length variation for the
same problem. Based on this, we designed the Shortest Rejection Sampling method. This method samples the same
question _n_ times (in our experiments, _n_ = 8) and selects the shortest correct response for supervised fine-tuning.


**DPO** Similar with Shortest Rejection Sampling, we utilize the Long CoT model to generate multiple response samples.
The shortest correct solution is selected as the positive sample, while longer responses are treated as negative samples,
including both wrong longer responses and correct longer responses (1.5 times longer than the chosen positive sample).
These positive-negative pairs form the pairwise preference data used for DPO training.


7


Kimi k1.5 T ECHNICAL R EPORT


**Long2short RL** After a standard RL training phase, we select a model that offers the best balance between performance and token efficiency to serve as the base model, and conduct a separate long2short RL training phase. In this
second phase, we apply the length penalty introduced in Section 2.3.3, and significantly reduce the maximum rollout
length to further penalize responses that exceed the desired length while possibly correct.


**2.5** **Other Training Details**


**2.5.1** **Pretraining**


The Kimi k1.5 base model is trained on a diverse, high-quality multimodal corpus. The language data covers five
domains: English, Chinese, Code, Mathematics Reasoning, and Knowledge. Multimodal data, including Captioning,
Image-text Interleaving, OCR, Knowledge, and QA datasets, enables our model to acquire vision-language capabilities.
Rigorous quality control ensures relevance, diversity, and balance in the overall pretrain dataset. Our pretraining
proceeds in three stages: (1) Vision-language pretraining, where a strong language foundation is established, followed
by gradual multimodal integration; (2) Cooldown, which consolidates capabilities using curated and synthetic data,
particularly for reasoning and knowledge-based tasks; and (3) Long-context activation, extending sequence processing
to 131,072 tokens. More details regarding our pretraining efforts can be found in Appendix B.


**2.5.2** **Vanilla Supervised Finetuning**


We create the vanilla SFT corpus covering multiple domains. For non-reasoning tasks, including question-answering,
writing, and text processing, we initially construct a seed dataset through human annotation. This seed dataset is used
to train a seed model. Subsequently, we collect a diverse of prompts and employ the seed model to generate multiple
responses to each prompt. Annotators then rank these responses and refine the top-ranked response to produce the
final version. For reasoning tasks such as math and coding problems, where rule-based and reward modeling based
verifications are more accurate and efficient than human judgment, we utilize rejection sampling to expand the SFT
dataset.


Our vanilla SFT dataset comprises approximately 1 million text examples. Specifically, 500k examples are for general
question answering, 200k for coding, 200k for math and science, 5k for creative writing, and 20k for long-context
tasks such as summarization, doc-qa, translation, and writing. In addition, we construct 1 million text-vision examples
encompassing various categories including chart interpretation, OCR, image-grounded conversations, visual coding,
visual reasoning, and math/science problems with visual aids.


We first train the model at the sequence length of 32k tokens for 1 epoch, followed by another epoch at the sequence
length of 128k tokens. In the first stage (32k), the learning rate decays from 2 _×_ 10 _[−]_ [5] to 2 _×_ 10 _[−]_ [6], before it re-warmups
to 1 _×_ 10 _[−]_ [5] in the second stage (128k) and finally decays to 1 _×_ 10 _[−]_ [6] . To improve training efficiency, we pack multiple
training examples into each single training sequence.


**2.6** **RL Infrastructure**

















iteration N









from

promt

set













(a) System overview



(b) Partial Rollout



Figure 3: Large Scale Reinforcement Learning Training System for LLM


8


Kimi k1.5 T ECHNICAL R EPORT


**2.6.1** **Large Scale Reinforcement Learning Training System for LLM**


In the realm of artificial intelligence, reinforcement learning (RL) has emerged as a pivotal training methodology
for large language models (LLMs)(Ouyang et al. 2022)(Jaech et al. 2024), drawing inspiration from its success in
mastering complex games like Go, StarCraft II, and Dota 2 through systems such as AlphaGo(Silver et al. 2017),
AlphaStar(Vinyals et al. 2019), and OpenAI Dota Five (Berner et al. 2019). Following in this tradition, the Kimi
k1.5 system adopts an iterative synchronous RL framework, meticulously designed to bolster the model’s reasoning
capabilities through persistent learning and adaptation. A key innovation in this system is the introduction of a Partial
Rollout technique, designed to optimize the handling of complex reasoning trajectories.


The RL training system as illustrated in Figure 3a operates through an iterative synchronous approach, with each
iteration encompassing a rollout phase and a training phase. During the rollout phase, rollout workers, coordinated
by a central master, generate rollout trajectories by interacting with the model, producing sequences of responses to
various inputs. These trajectories are then stored in a replay buffer, which ensures a diverse and unbiased dataset for
training by disrupting temporal correlations. In the subsequent training phase, trainer workers access these experiences
to update the model’s weights. This cyclical process allows the model to continuously learn from its actions, adjusting
its strategies over time to enhance performance.


The central master serves as the central conductor, managing the flow of data and communication between the rollout
workers, trainer workers, evaluation with reward models and the replay buffer. It ensures that the system operates
harmoniously, balancing the load and facilitating efficient data processing.


The trainer workers access these rollout trajectories, whether completed in a single iteration or divided across multiple
iterations, to compute gradient updates that refine the model’s parameters and enhance its performance. This process
is overseen by a reward model, which evaluates the quality of the model’s outputs and provides essential feedback to
guide the training process. The reward model’s evaluations are particularly pivotal in determining the effectiveness of
the model’s strategies and steering the model towards optimal performance.


Moreover, the system incorporates a code execution service, which is specifically designed to handle code-related
problems and is integral to the reward model. This service evaluates the model’s outputs in practical coding scenarios,
ensuring that the model’s learning is closely aligned with real-world programming challenges. By validating the model’s
solutions against actual code executions, this feedback loop becomes essential for refining the model’s strategies and
enhancing its performance in code-related tasks.


**2.6.2** **Partial Rollouts for Long CoT RL**


One of the primary ideas of our work is to scale long-context RL training. Partial rollouts is a key technique that
effectively addresses the challenge of handling long-CoT features by managing the rollouts of both long and short
trajectories. This technique establishes a fixed output token budget, capping the length of each rollout trajectory.
If a trajectory exceeds the token limit during the rollout phase, the unfinished portion is saved to the replay buffer
and continued in the next iteration. It ensures that no single lengthy trajectory monopolizes the system’s resources.
Moreover, since the rollout workers operate asynchronously, when some are engaged with long trajectories, others can
independently process new, shorter rollout tasks. The asynchronous operation maximizes computational efficiency
by ensuring that all rollout workers are actively contributing to the training process, thereby optimizing the overall
performance of the system.


As illustrated in Figure 3b, the partial rollout system works by breaking down long responses into segments across
iterations (from iter n-m to iter n). The Replay Buffer acts as a central storage mechanism that maintains these response
segments, where only the current iteration (iter n) requires on-policy computation. Previous segments (iter n-m to
n-1) can be efficiently reused from the buffer, eliminating the need for repeated rollouts. This segmented approach
significantly reduces the computational overhead: instead of rolling out the entire response at once, the system processes
and stores segments incrementally, allowing for the generation of much longer responses while maintaining fast iteration
times. During training, certain segments can be excluded from loss computation to further optimize the learning process,
making the entire system both efficient and scalable.


The implementation of partial rollouts also offers repeat detection. The system identifies repeated sequences in the
generated content and terminates them early, reducing unnecessary computation while maintaining output quality.
Detected repetitions can be assigned additional penalties, effectively discouraging redundant content generation in the
prompt set.


**2.6.3** **Hybrid Deployment of Training and Inference**


The RL training process comprises of the following phases:


9


Kimi k1.5 T ECHNICAL R EPORT



























Figure 4: Hybrid Deployment Framework


  - **Training Phase:** At the outset, Megatron (Shoeybi et al. 2020) and vLLM (Kwon et al. 2023) are executed
within separate containers, encapsulated by a shim process known as checkpoint-engine (Section 2.6.3). Megatron
commences the training procedure. After the training is completed, Megatron offloads the GPU memory and
prepares to transfer current weights to vLLM.


  - **Inference Phase:** Following Megatron’s offloading, vLLM starts with dummy model weights and updates them
with the latest ones transferred from Megatron via Mooncake (Qin et al. 2024). Upon completion of the rollout, the
checkpoint-engine halts all vLLM processes.


  - **Subsequent Training Phase:** Once the memory allocated to vLLM is released, Megatron onloads the memory and
initiates another round of training.


We find existing works challenging to simultaneously support all the following characteristics.


  - Complex parallelism strategy: Megatron may have different parallelism strategy with vLLM. Training weights
distributing in several nodes in Megatron could be challenging to be shared with vLLM.


  - Minimizing idle GPU resources: For On-Policy RL, recent works such as SGLang (L. Zheng et al. 2024) and
vLLM might reserve some GPUs during the training process, which conversely could lead to idle training GPUs. It
would be more efficient to share the same devices between training and inference.


  - Capability of dynamic scaling: In some cases, a significant acceleration can be achieved by increasing the number
of inference nodes while keeping the training process constant. Our system enables the efficient utilization of idle
GPU nodes when needed.


As illustrated in Figure 4, we implement this hybrid deployment framework (Section 2.6.3) on top of Megatron and
vLLM, achieving less than one minute from training to inference phase and about ten seconds conversely.


**Hybrid Deployment Strategy** We propose a hybrid deployment strategy for training and inference tasks, which
leverages Kubernetes Sidecar containers sharing all available GPUs to collocate both workloads in one pod. The
primary advantages of this strategy are:


  - It facilitates efficient resource sharing and management, preventing train nodes idling while waiting for inference
nodes when both are deployed on separate nodes.


  - Leveraging distinct deployed images, training and inference can each iterate independently for better performance.


 - The architecture is not limited to vLLM, other frameworks can be conveniently integrated.


**Checkpoint Engine** Checkpoint Engine is responsible for managing the lifecycle of the vLLM process, exposing
HTTP APIs that enable triggering various operations on vLLM. For overall consistency and reliability, we utilize a
global metadata system managed by the etcd service to broadcast operations and statuses.


10


Kimi k1.5 T ECHNICAL R EPORT


It could be challenging to entirely release GPU memory by vLLM offloading primarily due to CUDA graphs, NCCL
buffers and NVIDIA drivers. To minimize modifications to vLLM, we terminate and restart it when needed for better
GPU utilization and fault tolerance.


The worker in Megatron converts the owned checkpoints into the Hugging Face format in shared memory. This
conversion also takes Pipeline Parallelism and Expert Parallelism into account so that only Tensor Parallelism remains
in these checkpoints. Checkpoints in shared memory are subsequently divided into shards and registered in the global
metadata system. We employ Mooncake to transfer checkpoints between peer nodes over RDMA. Some modifications
to vLLM are needed to load weight files and perform tensor parallelism conversion.


**2.6.4** **Code Sandbox**


We developed the sandbox as a secure environment for executing user-submitted code, optimized for code execution
and code benchmark evaluation. By dynamically switching container images, the sandbox supports different use cases
through MultiPL-E (Cassano, Gouwar, D. Nguyen, S. Nguyen, et al. 2023), DMOJ Judge Server [2], Lean, Jupyter
Notebook, and other images.


For RL in coding tasks, the sandbox ensures the reliability of training data judgment by providing consistent and repeatable evaluation mechanisms. Its feedback system supports multi-stage assessments, such as code execution feedback
and repo-level editing, while maintaining a uniform context to ensure fair and equitable benchmark comparisons across
programming languages.


We deploy the service on Kubernetes for scalability and resilience, exposing it through HTTP endpoints for external
integration. Kubernetes features like automatic restarts and rolling updates ensure availability and fault tolerance.


To optimize performance and support RL environments, we incorporate several techniques into the code execution
service to enhance efficiency, speed, and reliability. These include:


  - **Using Crun:** We utilize `crun` as the container runtime instead of Docker, significantly reducing container startup
times.


  - **Cgroup Reusing:** We pre-create cgroups for container use, which is crucial in scenarios with high concurrency
where creating and destroying cgroups for each container can become a bottleneck.


  - **Disk Usage Optimization:** An overlay filesystem with an upper layer mounted as `tmpfs` is used to control disk
writes, providing a fixed-size, high-speed storage space. This approach is beneficial for ephemeral workloads.



|Method|Time(s)|
|---|---|
|Docker<br>Sandbox|0.12<br>0.04|


(a) Container startup times



|Method|Containers/sec|
|---|---|
|Docker<br>Sandbox|27<br>120|


(b) Maximum containers started per second on a 16-core machine



These optimizations improve RL efficiency in code execution, providing a consistent and reliable environment for
evaluating RL-generated code, essential for iterative training and model improvement.


**3** **Experiments**


**3.1** **Evaluation**


Since k1.5 is a multimodal model, we conducted comprehensive evaluation across various benchmarks for different
modalities. The detailed evaluation setup can be found in Appendix C. Our benchmarks primarily consist of the
following three categories:


  - **Text Benchmark** : MMLU (Hendrycks et al. 2020), IF-Eval (J. Zhou et al. 2023), CLUEWSC (L. Xu et al. 2020),
C-EVAL (Y. Huang et al. 2023)


  - **Reasoning Benchmark** : HumanEval-Mul, LiveCodeBench (Jain et al. 2024), Codeforces, AIME 2024, MATH500 (Lightman et al. 2023)


  - **Vision Benchmark** : MMMU (Yue, Ni, et al. 2024), MATH-Vision (K. Wang et al. 2024), MathVista (Lu et al.
2023)


2 https://github.com/DMOJ/judge-server


11


Kimi k1.5 T ECHNICAL R EPORT


**3.2** **Main Results**


**K1.5 long-CoT model** The performance of the Kimi k1.5 long-CoT model is presented in Table 2. Through long-CoT
supervised fine-tuning (described in Section 2.2) and vision-text joint reinforcement learning (discussed in Section 2.3),
the model’s long-term reasoning capabilities are enhanced significantly. The test-time computation scaling further
strengthens its performance, enabling the model to achieve state-of-the-art results across a range of modalities. Our
evaluation reveals marked improvements in the model’s capacity to reason, comprehend, and synthesize information
over extended contexts, representing a advancement in multi-modal AI capabilities.


**K1.5 short-CoT model** The performance of the Kimi k1.5 short-CoT model is presented in Table 3. This model
integrates several techniques, including traditional supervised fine-tuning (discussed in Section 2.5.2), reinforcement
learning (explored in Section 2.3), and long-to-short distillation (outlined in Section 2.4). The results demonstrate
that the k1.5 short-CoT model delivers competitive or superior performance compared to leading open-source and
proprietary models across multiple tasks. These include text, vision, and reasoning challenges, with notable strengths in
natural language understanding, mathematics, coding, and logical reasoning.



**Benchmark** **(Metric)**



**Language-only Model** **Vision-Language Model**
**QwQ-32B** **OpenAI** **QVQ-72B** **OpenAI** **Kimi**
**Preview** **o1-mini** **Preview** **o1** **k1.5**



Reasoning



MATH-500 (EM) 90.6 90.0 - 94.8 **96.2**
AIME 2024 (Pass@1) 50.0 63.6 - 74.4 **77.5**
Codeforces (Percentile) 62 88 - **94** **94**
LiveCodeBench (Pass@1) 40.6 53.1 - **67.2** 62.5



MathVista-Test (Pass@1)       -       - 71.4 71.0 **74.9**
Vision MMMU-Val (Pass@1) - - 70.3 **77.3** 70.0
MathVision-Full (Pass@1)      -      - 35.9      - **38.6**


Table 2: Performance of Kimi k1.5 long-CoT and flagship open-source and proprietary models.



**Benchmark** **(Metric)**



**Language-only Model** **Vision-Language Model**
**Qwen2.5 LLaMA-3.1 DeepSeek Qwen2-VL Claude-3.5- GPT-4o Kimi**
**72B-Inst.** **405B-Inst.** **V3** **Sonnet-1022** **0513** **k1.5**



Text


Reasoning



MMLU (EM) 85.3 **88.6** 88.5 - 88.3 87.2 87.4
IF-Eval (Prompt Strict) 84.1 86.0 86.1 - 86.5 84.3 **87.2**
CLUEWSC (EM) 91.4 84.7 90.9 - 85.4 87.9 **91.7**
C-Eval (EM) 86.1 61.5 86.5 - 76.7 76.0 **88.3**


MATH-500 (EM) 80.0 73.8 90.2 - 78.3 74.6 **94.6**
AIME 2024 (Pass@1) 23.3 23.3 39.2 - 16.0 9.3 **60.8**
HumanEval-Mul (Pass@1) 77.3 77.2 **82.6** - 81.7 80.5 81.5
LiveCodeBench (Pass@1) 31.1 28.4 40.5 - 36.3 33.4 **47.3**



MathVista-Test (Pass@1)         -         -         - 69.7 65.3 63.8 **70.1**
Vision MMMU-Val (Pass@1)     -  