Published as a conference paper at ICLR 2026

## - O VERTHINKING R EDUCTION WITH D ECOUPLED R E WARDS AND C URRICULUM D ATA S CHEDULING


**Shuyang Jiang** _[♠][,][♣]_ **,Yusheng Liao** _[♢]_ **,Ya Zhang** _[♢][,][♣][,][∗]_ **, Yanfeng Wang** _[♢][,][♣]_ **, Yu Wang** _[♢][,][♣][,][∗]_

_♠_ Fudan University
_♢_ School of Artificial Intelligence, Shanghai Jiao Tong University
_♣_ Shanghai Artificial Intelligence Laboratory
shuyangjiang23@m.fudan.edu.cn
_{_ liao20160907,ya ~~z~~ hang,wangyanfeng622,yuwangsjtu _}_ @sjtu.edu.cn


A BSTRACT


While large reasoning models trained with critic-free reinforcement learning and
verifiable rewards (RLVR) represent the state-of-the-art, their practical utility is
hampered by “overthinking”, a critical issue where models generate excessively
long reasoning paths without any performance benefit. Existing solutions that penalize length often fail, inducing performance degradation due to a fundamental
misalignment between trajectory-level rewards and token-level optimization. In
this work, we introduce a novel framework, D E CS, built on our theoretical discovery of two previously unaddressed flaws in current length rewards: (1) the
erroneous penalization of essential exploratory tokens and (2) the inadvertent rewarding of partial redundancy. Our framework’s innovations include (i) a first-ofits-kind decoupled token-level reward mechanism that surgically distinguishes and
penalizes redundant tokens, and (ii) a novel curriculum batch scheduling strategy
to master the efficiency-efficacy equilibrium. Experimental results show D E CS
can achieve a dramatic reduction in reasoning tokens by over 50% across seven
benchmarks while simultaneously maintaining or even improving performance.
It demonstrates conclusively that substantial gains in reasoning efficiency can be
achieved without compromising a model’s underlying reasoning power. Code is
[available at https://github.com/pixas/DECS.](https://github.com/pixas/DECS)







𝑜𝑜 11

𝑜𝑜 21

𝑜𝑜 12

𝑜𝑜 22

𝑜𝑜 13

𝑜𝑜 23

𝑜𝑜 14

𝑜𝑜 24

𝑜𝑜 15
5























𝑜𝑜 25







Figure 1: _**Left**_ : Two major flaws of prior practice apply sequence-level length reward without control of training data. Negative advantage values penalize correct high entropy tokens from long
sequences while positive ones reward redundant tokens from short sequences; _**Middle**_ : Flaws of
length rewards lead to inferior performance and suboptimal efficiency gains on AIME2024 dataset;
_**Right**_ : D E CS improves pass@1 of base models while reducing _∼_ 60% token costs compared to the
base model across 7 benchmarks. Experimental details are presented in Appendix G.5.


_∗_ Corresponding Author


1


Published as a conference paper at ICLR 2026


1 I NTRODUCTION


Recent large reasoning models (LRM; Guo et al. (2025); OpenAI (2025); Qwen (2025)) trained with
critic-free reinforcement learning (RL) algorithms, such as GRPO (Shao et al., 2024), DAPO (Yu
et al., 2025), and REINFORCE++ (Hu et al., 2025a), have demonstrated impressive reasoning capabilities through verifiable outcome rewards. A hallmark of such models is their increased propensity
to generate high-entropy tokens (e.g., “wait”, “however”, “alternatively”), which serve to bridge
logical transitions between reasoning steps (Wang et al., 2025b). While these tokens reflect active reasoning mechanisms that enhance performance, the propagation of trajectory-level rewards to
all tokens can inadvertently encourage prolonged generation led by high-entropy tokens even after
reaching a correct answer, a phenomenon known as “overthinking” (Ji et al., 2025). To address
this inefficiency without sacrificing reasoning quality, recent approaches incorporate a small length
penalty into the correctness reward (Hou et al., 2025; Su & Cardie, 2025; Aggarwal & Welleck,
2025; Zhang et al., 2025d; Kimi et al., 2025; Wu et al., 2025), using critic-free RL frameworks like
GRPO to promote concise yet effective reasoning.


Despite these advancements, we find that existing methods still fall short of achieving the optimal
efficiency-performance trade-off: improvements in reasoning speed often come at the expense of
degraded reasoning fidelity. This suboptimality raises a fundamental question: _why do current re-_
_ward designs fail to effectively balance brevity and capability_ ? To investigate this, we conduct a
theoretical analysis of the logit dynamics of two key groups of tokens within the GRPO framework:
(i) high-entropy tokens that initiate exploratory reasoning paths, and (ii) those belonging to the Necessary Reasoning Prefix (NRP), defined as the minimal prefix of a reasoning trajectory that suffices
to justify the final correct answer. Our analysis reveals two critical limitations arising from the misalignment between sequence-level length regularization and token-level policy updates (depicted in
Fig. 1(Left)), revealing inherent tensions in how efficiency is incentivized during training.


First, sequence-level length penalties inherently suppress high-entropy tokens, even when they contribute to valid reasoning (§3.2). Specifically, in GRPO, overlong (yet correct) trajectories receive
uniformly negative advantages across all tokens from length penalties. Consequently, when all responses to a given prompt are correct but differ in length, shorter trajectories yield positive advantages while longer ones receive negative ones. This leads to a reduction in the logits of high-entropy
tokens through policy gradient updates. When easy prompts dominate the batch and response lengths
vary significantly, this negative gradient becomes dominant over iterations, causing the policy to
avoid generating these tokens, even if they are essential for productive exploration (Theorem 1).
This leads to premature convergence and deviation from the optimal efficiency-efficacy trade-off.


Second, training convergence is impeded by misaligned incentives (§3.3). Without explicitly decoupling the NRP serving as the minimal sufficient reasoning prefix from subsequent generations,
tokens produced after the NRP in shorter trajectories may still receive positive advantages. This
falsely reinforces redundant steps, encouraging the model to continue generating beyond logical necessity. These spurious rewards not only distort the learning signal but also slow down convergence,
limiting the extent of achievable efficiency gains under finite training updates.


Building on these insights, we propose D E CS, a novel framework with **De** coupled token-level rewards and **C** urriculum data **S** cheduling for overthinking reduction (§4). To enable precise intervention, we fine-tune a lightweight judge model to identify NRP boundaries. Based on this, we
design a decoupled reward function that ensures redundant tokens generated after the NRP are consistently penalized, thereby suppressing overthinking during autoregressive decoding. Meanwhile,
we introduce a curriculum batching strategy that adaptively balances the proportion of easy prompts
according to the average NRP ratio in the current batch, mitigating undue suppression of exploratory
behavior. Experimental results on two base models show that D E CS reduces reasoning tokens by
over 50%, while maintaining or surpassing performance on both deterministic (Pass@1; Table 1)
and exploratory (Pass@K; Fig. 3c) metrics. In summary, we conclude our contributions as follows:


1. **Misalignment Analysis** : We identify a fundamental misalignment between sequence-level
length penalties and token-level policy optimization in critic-free RL. Our theoretical analysis
demonstrates that this misalignment not only inhibits the generation of high-entropy tokens,
which are essential for valid reasoning, but also hampers efficiency improvements due to misguided gradient signals.


2


Published as a conference paper at ICLR 2026


2. **Adaptive Sampling with Decoupled Reward** : We introduce D E CS, a novel method that employs a decoupled reward system to consistently penalize redundancy. Coupled with a dynamic
batching strategy, this approach mitigates the over-penalization of exploration by incorporating
adaptive curriculum control.
3. **Comprehensive Evaluation** : We perform extensive evaluations across two model scales and
seven benchmarks, showing that D E CS consistently reduces over 50% thinking tokens without
sacrificing base models’ capability boundary.


2 P RELIMINARY


2.1 R EINFORCEMENT LEARNING WITH V ERIFIABLE R EWARDS (RLVR)


The RL objective for the policy _π_ _θ_ is to maximize the cumulative rewards _r_ received from the
verifier. Specifically, Policy Gradient (Williams, 1992) gives the following objective function:



_∇J_ ( _θ_ ) = E _q∼D,_ _**o**_ _∼π_ _θ_ ( _q_ )



_T_
� _∇_ _θ_ log _π_ _θ_ ( _o_ _j_ _|_ _**o**_ _<j_ ) _A_ ( _**o**_ _<j_ _, j_ ) _,_ (1)

_j_ =0



where _D_ is the training distribution, _q_ is an input prompt, _**o**_ is an output sequence consisting of
_T_ tokens _{o_ 1 _, o_ 2 _, . . ., o_ _T_ _}_, and _A_ ( _**o**_ _<j_ _, j_ ) is the advantage of the _j_ -th token given the state _**o**_ _<j_ .
Recently, DeepSeek-R1 (Guo et al., 2025) boosted large language models’ reasoning ability via the
Group Relative Policy Optimization (GRPO; Shao et al. (2024)) algorithm. Each rollout is labeled
with a verifiable reward _r_ ( _·_ ), while its advantage is estimated using the group average and standard
deviation values of rewards from a group of _G_ trajectories _O_ = _{_ _**o**_ _i_ _}_ _[G]_ _i_ =1 [generated based on the]
same prompt _q_ :

_A_ _i_ = _[r]_ [(] _**[o]**_ _[i]_ [)] _[ −]_ [mean][(] _[r]_ [(] _**[o]**_ [1] [)] _[,][ . . . ][,][ r]_ [(] _**[o]**_ _[G]_ [))] _._ (2)

std( _r_ ( _**o**_ 1 ) _, . . ., r_ ( _**o**_ _G_ ))

GRPO optimizes the policy using the PPO surrogate loss (Schulman et al., 2017):







 _,_ (3)



_|_ _**o**_ _i_ _|_
�



� min ( _ρ_ _i,j_ _A_ _i_ _,_ clip( _ρ_ _i,j_ _A_ _i_ _,_ 1 _−_ _ϵ,_ 1 + _ϵ_ ) _A_ _i_ )

_j_ =1



E _q∼D,{_ _**o**_ _i_ _}_ _Gi_ =1 _[∼][π]_ _[θ]_ [(] _[·|][q]_ [)]



1



~~�~~ _Gi_ _[|]_ _**[o]**_ _[i]_ _[|]_




_G_
�


_i_ =1



where _ρ_ _i,j_ = _π_ _θ_ ( _o_ _i,j_ _| o_ _i,<j_ _, q_ ) _/π_ old ( _o_ _i,j_ _| o_ _i,<j_ _, q_ ) is the importance sampling ratio, _|_ _**o**_ _i_ _|_ is the
sequence length. The KL term is reduced to align with Hu et al. (2025b). Models are incentivized to
explore new trials, cross-verifying temporary results using diverse approaches, and correct existing
results, based on high-entropy decisive tokens (Wang et al., 2025b). However, although the high frequency of generating high-entropy triggers does boost the model for challenging problems (Muennighoff et al., 2025), such improvements are not consistent (Ghosal et al., 2025), and introduce great
verbosity and “over-thinking” for vanilla queries (Ji et al., 2025).


2.2 E FFICIENT R EASONING W ITH L ENGTH P ENALTIES


One of the most straightforward methods is to add a length-based reward along with the fundamental
correctness reward to encourage shorter yet correct responses (Hou et al., 2025; Su & Cardie, 2025;
Aggarwal & Welleck, 2025). Specifically, if adopting a monotonically decreasing length reward
function _f_ ( _l_ ) = _−γl_ accepting the sequence length _l_ as input, the combined reward is defined as:


_r_ ( _**o**_ _i_ ) _−_ _γ|_ _**o**_ _i_ _|_ _**o**_ _i_ is correct
_r_ _[′]_ ( _**o**_ _i_ ) = (4)
� _r_ ( _**o**_ _i_ ) otherwise


where _γ_ is a small factor to prevent the length reward from leading the overall reward, which could
be adaptively computed (Zhang et al., 2025d) or preset as a hyperparameter (Kimi et al., 2025).


3 O N THE L IMITATIONS OF L ENGTH -G UIDED R EASONING O PTIMIZATION


In this section, we formally reveal two significant limitations of current length-reward driven efficiency reasoning under the representative critic-free RLVR algorithm, GRPO, by analyzing the


3


Published as a conference paper at ICLR 2026


misalignment between the trajectory-level advantage score and the token-level optimization objective for redundant thinking tokens. Through an analysis of logit dynamics, we demonstrate that
this misalignment degrades reasoning performance (§3.2) and fails to reduce early redundancies,
thereby limiting potential gains in efficiency (§3.3). The concepts for each involved notation and
abbreviation are illustrated in Table 4.


3.1 L OGIT D YNAMICS UNDER P OLICY G RADIENT


The LRM policy at step _m_, as a softmax policy, is parameterized by


exp( _z_ _**o**_ _<t_ _,o_ _t_ )
_π_ _θ_ _[m]_ [(] _[o]_ _[t]_ _[|]_ _**[ o]**_ _[<t]_ [) =] ~~�~~ _o_ _[′]_ _∈|V |_ [exp] _[ z]_ _**[o]**_ _<t_ _[,o]_ _[′]_ _[,]_ (5)


where _z_ _**o**_ _<t_ _,o_ _t_ is the output logit of token _o_ _t_ given prefix _**o**_ _<t_ and _o_ _t_ _∼_ _π_ _θ_ _[m]_ [(] _[· |]_ _**[ o]**_ _[<t]_ [)][. Under the]
learning objective of the policy gradient, we have the following lemma (Cui et al., 2025):

**Lemma 1** ( **Difference of policy logits in vanilla policy gradient** ) **.** _Let the actor policy π_ _θ_ _be a_
_tabular softmax policy and updated using Eq. 1 with a learning rate η, the difference of z_ _**o**_ _<t_ _,o_ _t_
_between two consecutive steps m and m_ + 1 _satisfies_


_z_ _**o**_ _[m]_ _<t_ [+1] _,o_ _t_ _[−]_ _[z]_ _**o**_ _[m]_ _<t_ _,o_ _t_ [=] _[ η][ ·][ π]_ _[θ]_ [(] _[o]_ _[t]_ _[|]_ _**[ o]**_ _[<t]_ [)] _[ ·][ A]_ [(] _**[o]**_ _[<t]_ _[, o]_ _[t]_ [)]


3.2 O PTIMIZATION WITH I LL - POSED E FFICIENCY


GRPO estimates an advantage with intra-group relative reward by sampling _G_ rollouts repeatedly
for a prompt. When _G_ rollouts contain both correct and incorrect trajectories, correct sequences
always receive positive advantages, differing only in their magnitude and contributing little to efficiency optimization. In contrast, when rollouts generated by the policy _π_ _θ_ on an easy prompt _q_ _θ,G_
are all correct, the correctness reward becomes constant across trajectories, leaving length as the
sole discriminative signal. As a result, correct yet overlong trajectories receive negative advantage
estimates through the GRPO algorithm, which activates efficiency optimization.


Recently, Wang et al. (2025b) observes that the superior performance of LRMs is driven by highentropy tokens, which lead the policy to conduct exploration and reflection. However, trajectorylevel negative advantages would back-propagate to all tokens in Eq. 3, including the essential highentropy tokens. Under Lemma 1, the negative advantages will cause the decline of probability for
generating high-entropy tokens, and thereby the optimization process shifts from its intended goal,
i.e., improving efficiency while preserving performance, to one that trades correctness for shorter
trajectories. Formally, we could derive the following lemma:

**Lemma 2** ( **Decreased logits for correct high-entropy tokens** ) **.** _(Proof in Appendix A.1) For f_
_defined in Eq. 4, the expected change of logit for high-entropy tokens {o_ high _} from G correct rollouts_
_{_ _**o**_ _i_ _}_ _[G]_ _i_ =1 _[∼]_ _[π]_ _[θ]_ [(] _[· |][ q]_ _[θ,G]_ [)] _[ sampled from][ q]_ _[θ,G]_ _[ between two consecutive optimization steps][ m][ and][ m]_ [+1] _[,]_
_is strictly negative:_
E _o∈{o_ high _}_ � _z_ _o_ _[m]_ [+1] _−_ _z_ _o_ _[m]_ � _<_ 0


In the above lemma, the correctly generated high-entropy tokens produced by _q_ _θ,G_ have their generation probabilities reduced, which may disrupt or even distort the learning direction of an entire batch
with respect to high-entropy tokens, subject to the constraints specified by the following theorem:

**Theorem 1** ( **Maintenance of High-entropy Tokens Under Batch Learning** ) **.** _(Proof in Ap-_
_pendix A.2) Let the ratio of prompts q_ _θ,G_ _be κ. Assume that the length reward is defined as Eq. 4_
_and σ_ _L_ _is the standard deviation of response lengths of q_ _θ,G_ _on average, the condition for which the_
_expected logit change for correct high-entropy tokens among a batch is greater than 0 is as follows:_


_κσ_ _L_ _< C,_


_where C is a constant with respect to the rollout tokens generated during a mini-batch._


This theorem implies the condition under which the policy would suffer from performance degradation when applying length reward with GRPO. When _κσ_ _L_ becomes too large, the policy no longer
follows the performance-efficiency trade-off frontier. Instead, it shifts into a regime where gains in
efficiency come at the cost of the proactivity of high-entropy tokens, thereby degrading performance.


4


Published as a conference paper at ICLR 2026


3.3 I NSUFFICIENT E FFICIENCY


In addition to the decreased performance, current length-based reward methods also fail to achieve
sufficient reduction of overthinking. Specifically, we differentiate the redundant tokens to be reduced by formally defining the necessary reasoning prefix as the most compact thinking process that
supports deriving a correct answer for the first time:


**Definition 1** ( **Necessary Reasoning Prefix** ) **.** _Let q be an input prompt, y_ _[∗]_ _be the ground truth_
_answer, and_ _**o**_ = ( _o_ 1 _, o_ 2 _, . . ., o_ _L_ ) _be a generated response sequence on q, where L_ = _|_ _**o**_ _|. The_
_necessary reasoning prefix (NRP) of_ _**o**_ _with respect to q is the shortest prefix_ _**o**_ 1: _K_ _[∗]_ _such that_
A NSWER ( _**o**_ 1: _K_ _∗_ ) = _y_ _[∗]_ _and ∀k < K_ _[∗]_ _, either_ A NSWER ( _o_ 1: _k_ ) = null _or_ A NSWER ( _o_ 1: _k_ ) _̸_ = _y_ _[∗]_ _._


As the correct answer is logically justified at position _K_ _[∗]_, the token set _{o_ _j_ _| j > K_ _**o**_ _[∗]_ _[}]_ [ is considered]
**redundant** by many works (Dai et al., 2025; Yue et al., 2025). To prohibit the policy from continually generating further tokens after the already generated NRP tokens, we convert the objective to
minimizing the probability of generating the first thinking token after the NRP, which functions on
the reduction of holistic redundancy due to the autoregressive generation of LRMs:

min E _**o**_ _∼π_ _θ_ ( _·|q_ _θ,G_ ) � _z_ _**o**_ _[m]_ _≤K∗_ _,o_ _j_ _[−]_ _[z]_ _**o**_ _[m]_ _≤_ _[−]_ _K_ [1] _∗_ _,o_ _j_ � _s.t._ _j_ = _K_ _[∗]_ + 1 (6)


Applying Lemma 1, this objective could be converted into a policy weighted expectation of advantages, which is shown to be positive:


**Theorem 2** ( **Suboptimal Reduction of Redundant Tokens** ) **.** _(Proof in Appendix A.3) Let the re-_
_ward function f be defined as Eq. 4. Let j_ = _K_ _**o**_ _[∗]_ [+ 1] _[ denote the position of the first redundant]_
_token beyond the NRP in a correct rollout_ _**o**_ _. Let A_ ( _**o**_ ) _be the group-relative advantage computed_
_via Eq. 2. Then, the expected policy gradient signal for the first overthinking token, denoted as_
_J_ ( _A_ ; _j_ = _K_ _[∗]_ + 1) = E _**o**_ _∼π_ _θ_ ( _·|q_ _θ,G_ ) [ _π_ _θ_ ( _o_ _j_ _|_ _**o**_ _<j_ ) _A_ ( _**o**_ ) _| j_ = _K_ _**o**_ _[∗]_ [+ 1]] _[ satisfies:]_


_J_ ( _A_ ; _j_ = _K_ _[∗]_ + 1) _>_ 0


This theorem tells us that although the policy would reduce thinking length by penalizing tokens far
from the end of NRP from overlong responses, the policy cannot learn to stop at the end of NRP
given no penalization on the first redundant token. This undesired property keeps partial overthinking tokens, leading to suboptimal reduction of redundancies.


4 D E CS


Given the above analysis, we propose D E CS, which contains three main designs to achieve the highest efficacy-efficiency tradeoff. First, to ensure that redundant tokens are penalized deterministically,
we train a small module that precisely identifies necessary reasoning prefix (NRP) components for
correct trajectories (§4.1). After that, we design a decoupled token-level reward and differentiate
the reward scale for necessary and redundant tokens, to ensure enhanced efficiency without compromising performance (§4.2). Based on the conception of NRP, we propose to prevent aggressive
penalization on high-entropy tokens following NRP by refactoring the data distribution of a batch
according to the current levels of redundancy incrementally (§4.3). Fig. 2 illustrates the overall
algorithm.


4.1 D ETECTION OF NRP


It is common practice to train a token-level classification model to annotate NRP components. However, it requires the same tokenizer as the policy, which hinders adaptation to other policies. To this
end, we implement this detector as a lightweight generator model _M_ judge, determining whether a
reasoning chunk contains the correct answer to a given problem. Specifically, given a correct rollout
_**o**_, we first extract the reasoning tokens as _**o**_ think = ( _o_ 1 _, . . ., o_ _</_ think _>_ ). Using pre-defined separator
tokens, the reasoning process is segmented into multiple chunks: _S_ = _{s_ 1 _, s_ 2 _, · · ·, s_ _|S|_ _}_, where _s_ _c_
is the _c_ -th chunk of _**o**_ think . The judgment _j_ _s_ _c_ _∈{_ yes _,_ no _}_ for substep _s_ _c_ is generated by prompting
_M_ judge given the problem _q_ and corresponding ground truth _y_ _[∗]_ as:


_j_ _s_ _c_ _∼M_ judge ( _· | q, s_ _c_ _, y_ _[∗]_ ) (7)


5


Published as a conference paper at ICLR 2026











𝜅𝜅 𝑚𝑚 𝐵𝐵 easy
prompts

























































𝜅𝜅 𝑚𝑚 **𝒎**













Figure 2: Overview of the D E CS training pipeline. (1) **Decoupled Token-level Reward** : We finetune a small language model to detect the necessary reasoning prefix (NRP) from other redundancy,
which are separately rewarded to penalize overthinking consistently while maintaining the probability for generating necessary reasoning steps. As the running example “What is 2+3?” shows, the
NRP contains the reasoning chunks from the starting token to the first time the model generates the
correct answer “5”. After that, any leading redundant token like “Wait” receives negative advantages, and thereby discourage any redundant tokens to be generated via autoregressive generation.
_α_ = _r_ + _−_ _r_ 0 _._ (2) **Curriculum Prompt Schedule** : The number of easy prompts _q_ _θ,G_ grows in step
with the progressive decline in remaining redundancy.


The NRP spans all reasoning chunks from the start through the first chunk whose judgment is “yes”:



NRP =



_c_ _[∗]_
� _s_ _i_ _,_ where _c_ _[∗]_ = min _{c ∈_ [1 _, |S|_ ] : _j_ _s_ _c_ = yes _}_ (8)


_i_ =1



Here, [�] denotes the concatenation of reasoning chunks, and the _c_ _[∗]_ -th reasoning chunk is the first
to entail the correct answer _y_ _[∗]_ . The training details are illustrated in Appendix G.6.


4.2 D ECOUPLED R EWARD A SSIGNMENT



For a group of rollouts _{_ _**o**_ _i_ _}_ _[G]_ _i_ =1 [generated based on a given prompt] _[ q]_ [, we design a token-level reward]
which ensures a maximum reward for NRP tokens and preferences for short yet correct responses:



_r_ _i,j_ =



_r_ + _·_ **1** _**o**_ _i_ is correct _j ≤_ _K_ _o_ _[∗]_ _i_ _[∨]_ _[o]_ _[j]_ _[∈][/]_ _[o]_ [think] _[∨]_ _[o]_ _[i,j]_ [=] _[ ∅]_ (9)
�( _r_ 0 _−_ [(] _[r]_ [+] _L_ _[−]_ max _[r]_ [0] [)] _[L]_ _[i]_ ) _·_ **1** _**o**_ _i_ is correct _j > K_ _o_ _[∗]_ _i_ _[∧]_ _[o]_ _[j]_ _[∈]_ _[o]_ [think]



where _r_ + and _r_ 0 are respectively the maximum and minimum positive rewards, _K_ _o_ _[∗]_ _i_ [is the last NRP]
token index of _**o**_ _i_ and _∅_ denotes a padded token. Since the inverse proportional function enforces
a far lower reward for redundant tokens compared to _r_ +, any token followed by the NRP would
consistently receive negative advantages. Such penalization, as a result, helps to reduce verbosity
via the autoregressive generation property of LRM regardless of sequence lengths. Besides, only
redundant thinking tokens are possible to receive negative advantages, which prevents biased penalty
on essential reasoning tokens and answer conclusion tokens, and sustains the policy during the RL
training. Finally, the token-level advantage is computed similarly to GRPO and updated with Eq. 3:



_A_ _i,j_ [D] [E] [CS] = _[r]_ _[i][,j]_ _[ −]_ [mean][(] _[r]_ [1] _[,j]_ _[,][ . . . ][,][ r]_ _[G][,j]_ [)] _._ (10)

std( _r_ 1 _,j_ _, . . ., r_ _G,j_ )



Appendix C in detail explains the functionality of Eq. 9 for penalizing any leading redundant tokens.


6


Published as a conference paper at ICLR 2026


4.3 C URRICULUM P ROMPT S CHEDULE


After identifying NRP tokens, penalization of high-entropy tokens occurs only in redundant tokens
following the NRP. Therefore, we schedule _κ_ _m_ based on the proportion of NRP _R_ _m_ in correct
sequences within a batch, which reflects how many correct high-entropy tokens would be penalized:
_κ_ _m_ = clip( _κ_ m _−_ 1 + _β_ ( _R_ m _−R_ m _−_ 1 ) _,_ 0 _, κ_ [0] m [)] (11)
where _κ_ [0] _m_ [is the ratio of] _[ q]_ _[θ,G]_ [among the current sampled batch and] _[ β]_ [ is a hyperparameter to con-]
trol the learning progress. As trajectories with zero advantages would not provide any learning
signal, we follow Yu et al. (2025) to filter prompts whose _G_ rollouts are all incorrect and fill the
batch by over-sampling. This curriculum strategy, designed to be bounded and monotonic, enables smooth adjustment in response to the observed NRP ratio, which aligns with the principle of
curriculum learning (Bengio et al., 2009). By setting a moderate value _β_ with grid search (see Appendix H.1), D E CS can satisfy the condition elucidated in Theorem 1 to maintain unbiased learning
of high-entropy tokens throughout the whole training process. This yields stable convergence with
no observed training instability or performance degradation, which is reflected in Fig. 6 and 7.


5 E XPERIMENTS


5.1 E XPERIMENT S ETUPS


**Evaluation** We use MATH500 (Lightman et al., 2023), AMC23 (AI-MO, 2024), OlympiadBench (He et al., 2024), AIME2024 (Mathematical Association of America, 2025a) and
AIME2025 (Mathematical Association of America, 2025b) as in-domain testbeds, GPQADiamond (GPQA-D; Rein et al. (2024)) and LiveCodeBench-v6 (LCB; Jain et al. (2025)) as
held-out testbeds, covering math, coding, and science tasks with diverse complexity. We choose
**ThinkPrune** (Hou et al., 2025), **TLMRE** (Arora & Zanette, 2025), **AdaptThink** (Zhang et al.,
2025b), **LC-R1** (Cheng et al., 2025) as baselines, and also include GRPO to serve as a performance
reference. For fair comparison, we set the temperature as 0.6, top ~~p~~ as 0.95, and use a maximum
token limit of 16384 suggested by Guo et al. (2025). We conduct 128 rollouts for AIME2024,
AIME2025, and AMC23, 16 rollouts for OlympiadBench, MATH500 and GPQA-D, and 10 rollouts for LCB to compute pass@1. We also compute the Average Efficiency Score (AES; Luo et al.
(2025a)) for a comprehensive assessment of efficiency and efficacy. The details of both metrics are
presented in Appendix G.4.


**Training** We adopt DeepScaleR (Luo et al., 2025b) as the training set and choose DeepSeek-R1Distill-1.5B (DS-1.5B), DeepSeek-R1-Distill-7B (DS-7B) as base policies. We perform 16 rollouts
per prompt and use veRL (Sheng et al., 2025) as the training framework. _r_ + _, r_ 0 in Eq. 9 are set to 1 _._ 1
and 1 _._ 0, respectively, while _β_ in Eq. 11 is set to 0.2 with grid-search. Additional hyperparameters
are presented in Table 8.


5.2 R ESULTS


As shown in Table 1, D E CS reduces average reasoning length by 57.17% on the 1.5B model while
improving pass@1 accuracy by +2.48 points, demonstrating simultaneous gains in efficiency and
performance. On the 7B model, which exhibits less overthinking, D E CS still cuts thinking tokens
by 49.50%, outperforming all baselines, with a +0.8 point accuracy gain. Compared to the previous
best, D E CS improves the AES score by 0.12 and 0.14 on the 1.5B and 7B backbones, respectively,
establishing a superior efficiency-performance trade-off that compresses the computation without
sacrificing output quality. Meanwhile, although the NRP detector is specialized for math reasoning
and the training data only cover the math corpus, such superiority of efficiency generalizes robustly
to out-of-domain tasks (56.33% fewer tokens in GPQA-D and 33.52% fewer tokens in LCB), confirming D E CS ’s strong transferability and practical value for broader reasoning tasks.


5.3 A BLATION S TUDY


In this section, we conduct an ablation study on the DS-1.5B base policy, to reveal the critical
complementary relationship between the schedule prompt scheduling (CS) and decoupled tokenlevel reward (DR). We show the results in Table 3 and plot the comparison in Fig. 3a. We observe


7


Published as a conference paper at ICLR 2026


Table 1: Pass@1 (Acc) and the number of tokens (#Tok.) used across seven benchmarks. “LCB.”
denotes LiveCodeBench-v6, “OlympiadB.” denotes the OlympiadBench, and “GPQA-D” denotes
GPQA-Diamond. The best performing score is marked in **bold** and the second-best is underlined.


**AIME2024** **AIME2025** **AMC23** **MATH500** **OlympiadB** **GPQA-D** **LCB** **Average**
**Model** **Acc** **#Tok.** **Acc** **#Tok.** **Acc** **#Tok.** **Acc** **#Tok.** **Acc** **#Tok.** **Acc** **#Tok.** **Acc** **#Tok.** **Acc** **#Tok.** **AES**


_**DS-1.5B**_

Base 27.99 12202 22.94 12138 69.84 7875 84.55 4847 53.78 9217 32.86 8540 24.53 10560 45.21 9340 0.00

+GRPO 32.76 8834 25.91 8431 77.09 5722 87.34 3577 58.73 6425 35.76 5953 26.45 8759 49.15 6814 0.53

AdaptThink 27.92 6914 21.95 7400 64.73 **2644** 81.57 **1488** 50.40 3501 25.92 4093 26.98 9181 42.78 5031 0.19
ThinkPrune 26.93 **5306** 20.86 **4937** 72.87 2869 84.27 1879 55.04 3477 35.51 3839 25.36 **5515** 45.83 **3975** 0.62

TLMRE 29.87 7550 22.24 7151 74.51 3943 **84.86** 2376 56.08 4833 33.74 4896 26.13 7737 46.78 5498 0.52

LC-R1 23.65 6904 19.64 6681 68.69 3715 82.02 2277 51.57 4519 30.93 5377 23.54 6940 42.86 5202 0.18

D E CS **31.25** 5550 **23.78** 4965 **75.37** 2988 84.40 1817 **56.10** **3396** **35.92** **3255** **27.66** 6026 **47.78** 4000 **0.74**


_**DS-7B**_

Base 50.65 10508 **36.67** 11096 88.77 5764 **93.25** 3654 69.22 7507 46.46 7502 45.95 8966 61.57 7857 0.00

+GRPO 52.50 9011 38.54 9670 91.88 5205 94.21 3520 72.59 6425 49.62 6101 47.71 8569 63.86 6929 0.23

AdaptThink **53.31** 8884 36.48 9525 86.66 3675 91.06 1824 67.98 5528 43.91 5746 47.09 8209 60.93 6199 0.16
ThinkPrune 51.15 6625 36.46 7127 88.28 3193 92.98 2105 70.03 4154 48.42 4498 47.90 6881 62.17 4940 0.40

TLMRE 50.11 7023 34.24 7501 87.07 3329 91.83 2073 68.84 4382 49.02 4913 47.03 6772 61.16 5142 0.31

LC-R1 50.52 6891 32.50 7387 85.74 2802 90.28 **1473** 67.76 3983 48.58 4672 46.83 6554 60.32 4823 0.28

D E CS 51.33 **5277** 36.43 **5516** **89.04** **2772** 92.96 1728 **70.28** **3283** **49.27** **3276** **48.05** **5921** **62.48** **3968** **0.54**


Table 2: Generalization to the Qwen3-4B model. D E CS still achieves 0.61 AES score, with 54.80%
reduction to overthinking and 1.32 pass@1 improvement.


**AIME2024** **AIME2025** **AMC23** **MATH500** **OlympiadB** **GPQA-D** **LCB** **Average**
**Model**

**Acc** **#Tok.** **Acc** **#Tok.** **Acc** **#Tok.** **Acc** **#Tok.** **Acc** **#Tok.** **Acc** **#Tok.** **Acc** **#Tok.** **Acc** **#Tok.** **AES**


Qwen3-4B 64.82 11611 56.30 12870 91.60 7478 93.74 4839 71.07 9144 39.11 8072 62.17 9713 68.40 9104 0.00
D E CS **65.38** **5431** **56.96** **5758** **93.59** **2864** **93.78** **1648** **74.09** **3646** **41.00** **3260** **63.27** **6196** **69.72** **4115** **0.61**


that without adaptive scheduling of easy problems, there is a noticeable performance drop, which
verifies the impacts elucidated in Theorem 1. Meanwhile, without decoupled rewards, the policy
remains nearly 25% of overthinking tokens, verifying that the sequence-level length reward fails to
fully reduce overthinking as Theorem 2 implies.


5.4 B ACKBONE G ENERALIZATION


In this section, we generalize D E CS to Qwen3 backbone model, where we apply D E CS to Qwen34B (Yang et al., 2025) with the same training hyperparameters introduced in §5.1. Results in Table 2
demonstrates that D E CS successfully extends to Qwen3-4B, with 54.80% reduction of reasoning
tokens and 1.32 pass@1 improvement on average. This strongly implies that D E CS is backbonerobust, and remains effective on a stronger base model.


6 A NALYSIS


In this section, we discuss the following research questions:


**RQ1:** How do the decoupled rewards help D E CS to achieve the highest efficiency?
**RQ2:** How can D E CS balance the exploration and exploitaiton when compressing reasoning?
**RQ3:** How does D E CS perform with variable token budget?
**RQ4:** How do representative high-entropy tokens distribute after applying D E CS?
**RQ5:** How does compressed thinking spread over various difficulty levels?


**Response to RQ1:** **Most of the tokens reduced by D** **E** **CS stem from non-NRP tokens.** To
reveal the significance of decoupled learning for reducing redundancy, we compute the proportion
of NRP tokens in all thinking tokens (PNRP) of correct trajectories generated on AIME2024. We
plot the average token costs and the average PNRP score in Fig. 3b. Although ThinkPrune reduces
a similar number of thinking tokens as D E CS, it achieves a relatively lower PNRP score. This
inconsistency reflects that part of the reduced tokens stems from necessary reasoning tokens that
contribute to the final correctness, which explains its performance drops in Table 1. Compared to
LC-R1 remaining _∼_ 10% redundancy, D E CS further reduces non-NRP tokens and improves the
PNRP score, highlighting the utility of the decoupled reward for a unified reduction of overthinking.


8


Published as a conference paper at ICLR 2026



















(b)









|Col1|Col2|
|---|---|
|||
|||
|||


(c)



Figure 3: (a) Ablation study with two major components of D E CS on the DS-1.5B base model.
(b) Comparison of D E CS with other baselines on the proportion of NRP tokens (PRNP) and actual
reasoning tokens in the AIME2024 testbed. (c) D E CS performs on par with the base policy (DS1.5B) in terms of Pass@K scores on three challenging benchmarks.



|Col1|-1.47 Pass@1|
|---|---|
|||
|+1111 Avg. T|okens|
||<br>|
|DECS<br>w/o<br>Mo<br>(a)<br> 3: (a) Abla<br> mparison of<br>ing tokens in<br> in terms of P<br>Our|CS<br>w/o DR<br>del<br>  tion study w<br>  DECS with<br>  the AIME2<br>  ass@K score<br>Base|
|||
|||


(a)



(b)





(c)



Figure 4: (a) Average tokens and Pass@1 performance with 5 increasing generation budgets; (b)
Frequency of reasoning behavior tokens after applying D E CS; (c) Consistent compression rates of
D E CS on six difficulty levels sourced from MATH500 and AIME2024.


**Response to RQ2:** **D** **E** **CS maintains similar exploration potentials as the base model.**
To investigate whether D E CS achieves good pass@1-efficiency tradeoffs by sacrificing the
problem-solving potentials compared to base models, we compare the pass@k scores ( _k_ =
_{_ 2 _,_ 4 _,_ 8 _,_ 16 _,_ 32 _,_ 64 _,_ 128 _}_ ) on AIME2024, AIME2025 and AMC23. Results in Fig. 3c and Fig. 8c
reveal that across nearly all sample numbers, the success rate on the performance curve of the model
compressed by our method almost perfectly overlaps with that of the original model. This result
strongly demonstrates that the model’s exploration ability to find a correct solution through multiple
attempts is not injured by D E CS. It suggests that preventing high-entropy tokens from receiving
negative gradients sufficiently preserves most exploratory and creative properties.


**Response to RQ3:** **D** **E** **CS consistently improves the token efficiency across diverse token bud-**
**gets.** To validate whether the protection of NRP and exploratory high-entropy tokens would both
improve the model’s performance on token-constrained scenarios and not impair its performance
with a less-constrained token limit (Snell et al., 2025), we evaluate under 5 increasing token limits: [2,048, 4,096, 8,192, 16,384, 32,768]. Fig. 4a, 8a, and 8b demonstrate the Pass@1 scores and
average token costs on AIME2024, AIME2025 and AMC23 with the 1.5B policy. After applying
D E CS, the policy could use far fewer tokens to achieve competitive performance across diverse token limits, which holds even for the 32,768 context limit. For the 7B policy (depicted in Fig. 10),
D E CS performs on par with the base model with a negligible performance gap under the 32,768 token limit, but consuming fewer than 30% tokens. This further validates that D E CS achieves superior
efficiency compression without sacrificing the model’s capability boundary excessively.


9


Published as a conference paper at ICLR 2026


**Response to RQ4:** **D** **E** **CS reduces unnecessary reflective and conclusion tokens, but remains**
**a consistent tendency for creative and context formulation tokens.** To investigate how D E CS
refines the reasoning process and modulate the distribution for high-entropy decisive tokens, we
analyzed the frequency of representative tokens with different reasoning behaviors, including “SelfCorrection & Verification”, “Exploration & Alternatives”, “Context Setting” and “Conclusion Drawing” (Wu et al., 2025). Results in Fig. 4b reveal a significant shift in the tendency for correction
tokens with D E CS, which is the main source of overthinking. Meanwhile, the negligible change in
the frequency of exploratory tokens also suggests that the shearing of tokens after NRP hardly cause
degradation of creative thinking. Also, the dramatic decrease of conclusion tokens reflects that after
applying D E CS the policy is more confident in their reasoning intermediate results, which leads to
similar or even higher pass@1 scores across diverse benchmarks.


**Response to RQ5:** **D** **E** **CS compresses non-NRP tokens under variable input complexity.**
Since large reasoning models (LRMs) often overthink even on easy queries, we examine whether
D E CS consistently reduces overthinking across varying difficulty levels. We compute the PNRP
score on the MATH500 and AIME2024 datasets, which provide self-contained difficulty gradients
across six levels. As shown in Fig. 4c, PNRP scores increase with problem difficulty, and D E CS consistently achieves scores above 90% across all levels. This trend holds for the 7B model (Fig. 9b),
with even higher scores on AIME2024, likely due to its improved reasoning ability and reduced
inherent overthinking. These results confirm that D E CS enhances reasoning efficiency in LRMs
across diverse inputs, demonstrating its effectiveness and generality.


7 C ONCLUSION


In this paper, we theoretically identify two key flaws in current critic-free RL methods for reducing
the overthinking phenomenon, which stems from the misalignment between token-level overthinking reduction and sequence-level reward assignment. To mitigate these two drawbacks, we propose
D E CS, which proposes a decoupled reward system for NRP and non-NRP tokens for consistent
reduction of overthinking, and introduces curriculum batch scheduling for maintaining exploratory
potentials. Experiments show that D E CS reduces _∼_ 50% of reasoning tokens while maintaining or
improving performance, achieving more efficient reasoning without sacrificing accuracy.


A CKNOWLEDGMENTS


We thank the anonymous reviewers for their insightful comments and suggestions. This work was
supported by the National Key R&D Program of China (No. 2022ZD0162101), National Natural
Science Foundation of China (No. 62576209) and STCSM (No. 2025SHZDZX025G05).


8 R EPRODUCIBILITY S TATEMENT


In this section, we list any related materials that help to reproduce this paper


1. **Datasets** : The training set we used is described in §5.1 and evaluation sets we used are described
in Appendix G.2.


2. **Theoretical Support** : Any assumptions, lemmas, propositions, theorems and corresponding
proofs are detailed in Appendix A.


3. **Code** : The code to reproduce our algorithm would be put into the supplementary materials.


4. **Computational Resources** : We use 4xNVIDIA A100 80GB GPUs to conduct all experiments.


R EFERENCES


Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with
reinforcement learning. In _Second Conference on Language Modeling_ [, 2025. URL https:](https://openreview.net/forum?id=4jdIxXBNve)
[//openreview.net/forum?id=4jdIxXBNve.](https://openreview.net/forum?id=4jdIxXBNve)


10


Published as a conference paper at ICLR 2026


AI-MO. Amc23 dataset. Hugging Face Dataset Repository, 2024. [URL https://](https://huggingface.co/datasets/zwhe99/amc23)
[huggingface.co/datasets/zwhe99/amc23. Accessed: 2025-06-26.](https://huggingface.co/datasets/zwhe99/amc23)


Daman Arora and Andrea Zanette. Training language models to reason efficiently. _arXiv preprint_
_arXiv:2502.04463_, 2025.


Marthe Ballon, Andres Algaba, and Vincent Ginis. The relationship between reasoning and
performance in large language models–o3 (mini) thinks harder, not longer. _arXiv preprint_
_arXiv:2502.15631_, 2025.


Yoshua Bengio, J´erˆome Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In
_Proceedings of the 26th annual international conference on machine learning_, pp. 41–48, 2009.


Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared
Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large
language models trained on code. _arXiv preprint arXiv:2107.03374_, 2021.


Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu,
Mengfei Zhou, Zhuosheng Zhang, et al. Do not think that much for 2+ 3=? on the overthinking
of o1-like llms. _arXiv preprint arXiv:2412.21187_, 2024.


Zhengxiang Cheng, Dongping Chen, Mingyang Fu, and Tianyi Zhou. Optimizing Length Compres[sion in Large Reasoning Models, 2025. URL http://arxiv.org/abs/2506.14755.](http://arxiv.org/abs/2506.14755)


Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen
Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng,
Bowen Zhou, and Ning Ding. The Entropy Mechanism of Reinforcement Learning for Reasoning
[Language Models, 2025. URL http://arxiv.org/abs/2505.22617.](http://arxiv.org/abs/2505.22617)


Muzhi Dai, Chenxu Yang, and Qingyi Si. S-GRPO: Early Exit via Reinforcement Learning in
[Reasoning Models, 2025. URL http://arxiv.org/abs/2505.07686.](http://arxiv.org/abs/2505.07686)


Soumya Suvra Ghosal, Souradip Chakraborty, Avinash Reddy, Yifu Lu, Mengdi Wang, Dinesh
Manocha, Furong Huang, Mohammad Ghavamzadeh, and Amrit Singh Bedi. Does Thinking
[More always Help? Understanding Test-Time Scaling in Reasoning Models, 2025. URL http:](http://arxiv.org/abs/2506.04210)
[//arxiv.org/abs/2506.04210.](http://arxiv.org/abs/2506.04210)


Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu
Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. _Nature_, 645(8081):633–638, 2025.


Trevor Hastie, Robert Tibshirani, and Jerome Friedman. _The Elements of Statistical Learning: Data_
_Mining, Inference, and Prediction_ . Springer, New York, NY, 2 edition, 2009. ISBN 978-0-38784857-0. doi: 10.1007/978-0-387-84858-7.


Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han,
Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench:
A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), _Proceedings of the_
_62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Pa-_
_pers)_, pp. 3828–3850, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.211. [URL https://aclanthology.org/2024.](https://aclanthology.org/2024.acl-long.211/)
[acl-long.211/.](https://aclanthology.org/2024.acl-long.211/)


Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang.
ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning, 2025. URL
[http://arxiv.org/abs/2504.01296.](http://arxiv.org/abs/2504.01296)


Jian Hu, Jason Klein Liu, Haotian Xu, and Wei Shen. Reinforce++: An efficient rlhf algorithm with
robustness to both prompt and reward models. _arXiv preprint arXiv:2501.03262_, 2025a.


Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum.
Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the
[Base Model, 2025b. URL http://arxiv.org/abs/2503.24290.](http://arxiv.org/abs/2503.24290)


11


Published as a conference paper at ICLR 2026


Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In _The Thirteenth International Confer-_
_ence on Learning Representations_ [, 2025. URL https://openreview.net/forum?id=](https://openreview.net/forum?id=chfJJYC3iL)
[chfJJYC3iL.](https://openreview.net/forum?id=chfJJYC3iL)


Ke Ji, Jiahao Xu, Tian Liang, Qiuzhi Liu, Zhiwei He, Xingyu Chen, Xiaoyuan Liu, Zhijie Wang,
Junying Chen, Benyou Wang, et al. The first few tokens are all you need: An efficient and effective
unsupervised prefix fine-tuning method for reasoning models. _arXiv preprint arXiv:2503.02875_,
2025.


Team Kimi, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun
Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with
llms. _arXiv preprint arXiv:2501.12599_, 2025.


Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model
serving with pagedattention. In _Proceedings of the ACM SIGOPS 29th Symposium on Operating_
_Systems Principles_, 2023.


Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan
Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In _The Twelfth_
_International Conference on Learning Representations_, 2023.


Wei Liu, Ruochen Zhou, Yiyun Deng, Yuzhen Huang, Junteng Liu, Yuntian Deng, Yizhe Zhang,
and Junxian He. Learn to reason efficiently with adaptive length-based reward shaping. _arXiv_
_preprint arXiv:2505.15612_, 2025.


Haotian Luo, Li Shen, Haiying He, Yibo Wang, Shiwei Liu, Wei Li, Naiqiang Tan, Xiaochun Cao,
and Dacheng Tao. O1-pruner: Length-harmonizing fine-tuning for o1-like reasoning pruning.
In _2nd AI for Math Workshop @ ICML 2025_ [, 2025a. URL https://openreview.net/](https://openreview.net/forum?id=ioYybCRcyW)
[forum?id=ioYybCRcyW.](https://openreview.net/forum?id=ioYybCRcyW)


Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin
Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1[preview with a 1.5b model by scaling rl. https://pretty-radio-b75.notion.site/](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)
[DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005b](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)
2025b. Notion Blog.


Shangke Lyu, Linjuan Wu, Yuchen Yan, Xingyu Wu, Hao Li, Yongliang Shen, Peisheng Jiang,
Weiming Lu, Jun Xiao, and Yueting Zhuang. Hierarchical Budget Policy Optimization for Adap[tive Reasoning, 2025. URL http://arxiv.org/abs/2507.15844.](http://arxiv.org/abs/2507.15844)


Mathematical Association of America. Aime 2024 dataset. Hugging Face Dataset Repository,
2025a. [URL https://huggingface.co/datasets/Maxwell-Jia/AIME_2024.](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024)
Accessed: 2025-06-26.


Mathematical Association of America. Aime 2025 dataset. Hugging Face Dataset Repository,
[2025b. URL https://huggingface.co/datasets/opencompass/AIME2025. Ac-](https://huggingface.co/datasets/opencompass/AIME2025)
cessed: 2025-06-26.


Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke
Zettlemoyer, Percy Liang, Emmanuel Cand`es, and Tatsunori Hashimoto. s1: Simple test-time
scaling. _arXiv preprint arXiv:2501.19393_, 2025.


[OpenAI. Openai o3: Most advanced reasoning model. https://openai.com/zh-Hans-CN/](https://openai.com/zh-Hans-CN/index/introducing-o3-and-o4-mini/)
[index/introducing-o3-and-o4-mini/, 2025. Accessed: 2025-08-20.](https://openai.com/zh-Hans-CN/index/introducing-o3-and-o4-mini/)


[Team Qwen. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.](https://arxiv.org/abs/2505.09388)


David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In _First Conference on Language Modeling_, 2024.


12


Published as a conference paper at ICLR 2026


John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy
optimization algorithms. _arXiv preprint arXiv:1707.06347_, 2017.


Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical
reasoning in open language models. _arXiv preprint arXiv:2402.03300_, 2024.


Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng,
Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In _Proceedings_
_of the Twentieth European Conference on Computer Systems_, pp. 1279–1297, 2025.


Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute
optimally can be more effective than scaling parameters for reasoning. In _The Thirteenth In