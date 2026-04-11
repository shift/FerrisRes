## - L O RA: L OW -R ANK A DAPTATION OF L ARGE L AN GUAGE M ODELS

**Edward Hu** _[∗]_ **Yelong Shen** _[∗]_ **Phillip Wallis** **Zeyuan Allen-Zhu**
**Yuanzhi Li** **Shean Wang** **Lu Wang** **Weizhu Chen**
Microsoft Corporation
_{_ edwardhu, yeshe, phwallis, zeyuana,
yuanzhil, swang, luw, wzchen _}_ @microsoft.com
yuanzhil@andrew.cmu.edu
(Version 2)


A BSTRACT


An important paradigm of natural language processing consists of large-scale pretraining on general domain data and adaptation to particular tasks or domains. As
we pre-train larger models, full fine-tuning, which retrains all model parameters,
becomes less feasible. Using GPT-3 175B as an example – deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively
expensive. We propose **Lo** w- **R** ank **A** daptation, or LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each
layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam,
LoRA can reduce the number of trainable parameters by 10,000 times and the
GPU memory requirement by 3 times. LoRA performs on-par or better than finetuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters,
_no additional inference latency_ . We also provide an empirical investigation into
rank-deficiency in language model adaptation, which sheds light on the efficacy of
LoRA. We release a package that facilitates the integration of LoRA with PyTorch
models and provide our implementations and model checkpoints for RoBERTa,
[DeBERTa, and GPT-2 at https://github.com/microsoft/LoRA.](https://github.com/microsoft/LoRA)


1 I NTRODUCTION



Many applications in natural language processing rely on adapting _one_ large-scale, pre-trained language model to _multiple_ downstream applications. Such adaptation is usually done via _fine-tuning_,
which updates all the parameters of the pre-trained model. The major downside of fine-tuning is that the new model contains as many
parameters as in the original model. As larger models are trained
every few months, this changes from a mere “inconvenience” for
GPT-2 (Radford et al., b) or RoBERTa large (Liu et al., 2019) to a
critical deployment challenge for GPT-3 (Brown et al., 2020) with
175 billion trainable parameters. [1]


Many sought to mitigate this by adapting only some parameters or
learning external modules for new tasks. This way, we only need
to store and load a small number of task-specific parameters in addition to the pre-trained model for each task, greatly boosting the
operational efficiency when deployed. However, existing techniques













Figure 1: Our reparametrization. We only train _A_ and _B_ .



_∗_ Equal contribution.
0 Compared to V1, this draft includes better baselines, experiments on GLUE, and more on adapter latency.
1 While GPT-3 175B achieves non-trivial performance with few-shot learning, fine-tuning boosts its performance significantly as shown in Appendix A.


1


often introduce inference latency (Houlsby et al., 2019; Rebuffi et al., 2017) by extending model
depth or reduce the model’s usable sequence length (Li & Liang, 2021; Lester et al., 2021; Hambardzumyan et al., 2020; Liu et al., 2021) (Section 3). More importantly, these method often fail to
match the fine-tuning baselines, posing a trade-off between efficiency and model quality.


We take inspiration from Li et al. (2018a); Aghajanyan et al. (2020) which show that the learned
over-parametrized models in fact reside on a low intrinsic dimension. We hypothesize that the
change in weights during model adaptation also has a low “intrinsic rank”, leading to our proposed
**Lo** w- **R** ank **A** daptation (LoRA) approach. LoRA allows us to train some dense layers in a neural
network indirectly by optimizing rank decomposition matrices of the dense layers’ change during
adaptation instead, while keeping the pre-trained weights frozen, as shown in Figure 1. Using GPT-3
175B as an example, we show that a very low rank (i.e., _r_ in Figure 1 can be one or two) suffices even
when the full rank (i.e., _d_ ) is as high as 12,288, making LoRA both storage- and compute-efficient.


LoRA possesses several key advantages.


   - A pre-trained model can be shared and used to build many small LoRA modules for different tasks. We can freeze the shared model and efficiently switch tasks by replacing the
matrices _A_ and _B_ in Figure 1, reducing the storage requirement and task-switching overhead significantly.


    - LoRA makes training more efficient and lowers the hardware barrier to entry by up to 3
times when using adaptive optimizers since we do not need to calculate the gradients or
maintain the optimizer states for most parameters. Instead, we only optimize the injected,
much smaller low-rank matrices.


    - Our simple linear design allows us to merge the trainable matrices with the frozen weights
when deployed, _introducing no inference latency_ compared to a fully fine-tuned model, by
construction.


   - LoRA is orthogonal to many prior methods and can be combined with many of them, such
as prefix-tuning. We provide an example in Appendix E.


**Terminologies and Conventions** We make frequent references to the Transformer architecture
and use the conventional terminologies for its dimensions. We call the input and output dimension size of a Transformer layer _d_ _model_ . We use _W_ _q_, _W_ _k_, _W_ _v_, and _W_ _o_ to refer to the
query/key/value/output projection matrices in the self-attention module. _W_ or _W_ 0 refers to a pretrained weight matrix and ∆ _W_ its accumulated gradient update during adaptation. We use _r_ to
denote the rank of a LoRA module. We follow the conventions set out by (Vaswani et al., 2017;
Brown et al., 2020) and use Adam (Loshchilov & Hutter, 2019; Kingma & Ba, 2017) for model
optimization and use a Transformer MLP feedforward dimension _d_ _ffn_ = 4 _× d_ _model_ .


2 P ROBLEM S TATEMENT


While our proposal is agnostic to training objective, we focus on language modeling as our motivating use case. Below is a brief description of the language modeling problem and, in particular, the
maximization of conditional probabilities given a task-specific prompt.


Suppose we are given a pre-trained autoregressive language model _P_ Φ ( _y|x_ ) parametrized by Φ.
For instance, _P_ Φ ( _y|x_ ) can be a generic multi-task learner such as GPT (Radford et al., b; Brown
et al., 2020) based on the Transformer architecture (Vaswani et al., 2017). Consider adapting this
pre-trained model to downstream conditional text generation tasks, such as summarization, machine
reading comprehension (MRC), and natural language to SQL (NL2SQL). Each downstream task is
represented by a training dataset of context-target pairs: _Z_ = _{_ ( _x_ _i_ _, y_ _i_ ) _}_ _i_ =1 _,..,N_, where both _x_ _i_ and
_y_ _i_ are sequences of tokens. For example, in NL2SQL, _x_ _i_ is a natural language query and _y_ _i_ its
corresponding SQL command; for summarization, _x_ _i_ is the content of an article and _y_ _i_ its summary.


2


During full fine-tuning, the model is initialized to pre-trained weights Φ 0 and updated to Φ 0 + ∆Φ
by repeatedly following the gradient to maximize the conditional language modeling objective:



_|y|_
� log ( _P_ Φ ( _y_ _t_ _|x, y_ _<t_ )) (1)


_t_ =1



max
Φ



�

( _x,y_ ) _∈Z_



One of the main drawbacks for full fine-tuning is that for _each_ downstream task, we learn a _different_
set of parameters ∆Φ whose dimension _|_ ∆Φ _|_ equals _|_ Φ 0 _|_ . Thus, if the pre-trained model is large
(such as GPT-3 with _|_ Φ 0 _| ≈_ 175 Billion), storing and deploying many independent instances of
fine-tuned models can be challenging, if at all feasible.


In this paper, we adopt a more parameter-efficient approach, where the task-specific parameter
increment ∆Φ = ∆Φ(Θ) is further encoded by a much smaller-sized set of parameters Θ with
_|_ Θ _| ≪|_ Φ 0 _|_ . The task of finding ∆Φ thus becomes optimizing over Θ:



max
Θ �

( _x,y_ ) _∈Z_



_|y|_
� log � _p_ Φ 0 +∆Φ(Θ) ( _y_ _t_ _|x, y_ _<t_ )� (2)


_t_ =1



In the subsequent sections, we propose to use a low-rank representation to encode ∆Φ that is both
compute- and memory-efficient. When the pre-trained model is GPT-3 175B, the number of trainable parameters _|_ Θ _|_ can be as small as 0 _._ 01% of _|_ Φ 0 _|_ .


3 A REN ’ T E XISTING S OLUTIONS G OOD E NOUGH ?


The problem we set out to tackle is by no means new. Since the inception of transfer learning, dozens
of works have sought to make model adaptation more parameter- and compute-efficient. See Section 6 for a survey of some of the well-known works. Using language modeling as an example, there
are two prominent strategies when it comes to efficient adaptations: adding adapter layers (Houlsby
et al., 2019; Rebuffi et al., 2017; Pfeiffer et al., 2021; R¨uckl´e et al., 2020) or optimizing some forms
of the input layer activations (Li & Liang, 2021; Lester et al., 2021; Hambardzumyan et al., 2020;
Liu et al., 2021). However, both strategies have their limitations, especially in a large-scale and
latency-sensitive production scenario.


**Adapter Layers Introduce Inference Latency** There are many variants of adapters. We focus
on the original design by Houlsby et al. (2019) which has two adapter layers per Transformer block
and a more recent one by Lin et al. (2020) which has only one per block but with an additional
LayerNorm (Ba et al., 2016). While one can reduce the overall latency by pruning layers or exploiting multi-task settings (R¨uckl´e et al., 2020; Pfeiffer et al., 2021), there is no direct ways to bypass
the extra compute in adapter layers. This seems like a non-issue since adapter layers are designed
to have few parameters (sometimes _<_ 1% of the original model) by having a small bottleneck dimension, which limits the FLOPs they can add. However, large neural networks rely on hardware
parallelism to keep the latency low, and adapter layers have to be processed sequentially. This makes
a difference in the online inference setting where the batch size is typically as small as one. In a
generic scenario without model parallelism, such as running inference on GPT-2 (Radford et al., b)
medium on a single GPU, we see a noticeable increase in latency when using adapters, even with a
very small bottleneck dimension (Table 1).


This problem gets worse when we need to shard the model as done in Shoeybi et al. (2020); Lepikhin et al. (2020), because the additional depth requires more synchronous GPU operations such as
AllReduce and Broadcast, unless we store the adapter parameters redundantly many times.


**Directly Optimizing the Prompt is Hard** The other direction, as exemplified by prefix tuning (Li
& Liang, 2021), faces a different challenge. We observe that prefix tuning is difficult to optimize
and that its performance changes non-monotonically in trainable parameters, confirming similar
observations in the original paper. More fundamentally, reserving a part of the sequence length for
adaptation necessarily reduces the sequence length available to process a downstream task, which
we suspect makes tuning the prompt less performant compared to other methods. We defer the study
on task performance to Section 5.


3


Batch Size 32 16 1
Sequence Length 512 256 128
_|_ Θ _|_ 0.5M 11M 11M


Fine-Tune/LoRA 1449.4 _±_ 0.8 338.0 _±_ 0.6 19.8 _±_ 2.7


Adapter [L] 1482.0 _±_ 1.0 (+2.2%) 354.8 _±_ 0.5 (+5.0%) 23.9 _±_ 2.1 (+20.7%)
Adapter [H] 1492.2 _±_ 1.0 (+3.0%) 366.3 _±_ 0.5 (+8.4%) 25.8 _±_ 2.2 (+30.3%)


Table 1: Infernece latency of a single forward pass in GPT-2 medium measured in milliseconds, averaged over 100 trials. We use an NVIDIA Quadro RTX8000. “ _|_ Θ _|_ ” denotes the number of trainable
parameters in adapter layers. Adapter [L] and Adapter [H] are two variants of adapter tuning, which we
describe in Section 5.1. The inference latency introduced by adapter layers can be significant in an
online, short-sequence-length scenario. See the full study in Appendix B.


4 O UR M ETHOD


We describe the simple design of LoRA and its practical benefits. The principles outlined here apply
to any dense layers in deep learning models, though we only focus on certain weights in Transformer
language models in our experiments as the motivating use case.


4.1 L OW -R ANK -P ARAMETRIZED U PDATE M ATRICES


A neural network contains many dense layers which perform matrix multiplication. The weight
matrices in these layers typically have full-rank. When adapting to a specific task, Aghajanyan et al.
(2020) shows that the pre-trained language models have a low “instrisic dimension” and can still
learn efficiently despite a random projection to a smaller subspace. Inspired by this, we hypothesize the updates to the weights also have a low “intrinsic rank” during adaptation. For a pre-trained
weight matrix _W_ 0 _∈_ R _[d][×][k]_, we constrain its update by representing the latter with a low-rank decomposition _W_ 0 + ∆ _W_ = _W_ 0 + _BA_, where _B ∈_ R _[d][×][r]_ _, A ∈_ R _[r][×][k]_, and the rank _r ≪_ min( _d, k_ ).
During training, _W_ 0 is frozen and does not receive gradient updates, while _A_ and _B_ contain trainable
parameters. Note both _W_ 0 and ∆ _W_ = _BA_ are multiplied with the same input, and their respective
output vectors are summed coordinate-wise. For _h_ = _W_ 0 _x_, our modified forward pass yields:


_h_ = _W_ 0 _x_ + ∆ _Wx_ = _W_ 0 _x_ + _BAx_ (3)


We illustrate our reparametrization in Figure 1. We use a random Gaussian initialization for _A_ and
zero for _B_, so ∆ _W_ = _BA_ is zero at the beginning of training. We then scale ∆ _Wx_ by _[α]_ _r_ [, where] _[ α]_

is a constant in _r_ . When optimizing with Adam, tuning _α_ is roughly the same as tuning the learning
rate if we scale the initialization appropriately. As a result, we simply set _α_ to the first _r_ we try
and do not tune it. This scaling helps to reduce the need to retune hyperparameters when we vary
_r_ (Yang & Hu, 2021).


**A Generalization of Full Fine-tuning.** A more general form of fine-tuning allows the training of
a subset of the pre-trained parameters. LoRA takes a step further and does not require the accumulated gradient update to weight matrices to have full-rank during adaptation. This means that when
applying LoRA to all weight matrices and training all biases [2], we roughly recover the expressiveness of full fine-tuning by setting the LoRA rank _r_ to the rank of the pre-trained weight matrices. In
other words, as we increase the number of trainable parameters [3], training LoRA roughly converges
to training the original model, while adapter-based methods converges to an MLP and prefix-based
methods to a model that cannot take long input sequences.


**No Additional Inference Latency.** When deployed in production, we can explicitly compute and
store _W_ = _W_ 0 + _BA_ and perform inference as usual. Note that both _W_ 0 and _BA_ are in R _[d][×][k]_ .
When we need to switch to another downstream task, we can recover _W_ 0 by subtracting _BA_ and
then adding a different _B_ _[′]_ _A_ _[′]_, a quick operation with very little memory overhead. Critically, this


2 They represent a negligible number of parameters compared to weights.
3 An inevitability when adapting to hard tasks.


4


guarantees that we do not introduce any additional latency during inference compared to a fine-tuned
model by construction.


4.2 A PPLYING L O RA TO T RANSFORMER


In principle, we can apply LoRA to any subset of weight matrices in a neural network to reduce the
number of trainable parameters. In the Transformer architecture, there are four weight matrices in
the self-attention module ( _W_ _q_ _, W_ _k_ _, W_ _v_ _, W_ _o_ ) and two in the MLP module. We treat _W_ _q_ (or _W_ _k_, _W_ _v_ )
as a single matrix of dimension _d_ _model_ _×_ _d_ _model_, even though the output dimension is usually sliced
into attention heads. We limit our study to **only adapting the attention weights** for downstream
tasks and freeze the MLP modules (so they are not trained in downstream tasks) both for simplicity
and parameter-efficiency.We further study the effect on adapting different types of attention weight
matrices in a Transformer in Section 7.1. We leave the empirical investigation of adapting the MLP
layers, LayerNorm layers, and biases to a future work.


**Practical Benefits and Limitations.** The most significant benefit comes from the reduction in
memory and storage usage. For a large Transformer trained with Adam, we reduce that VRAM
usage by up to 2 _/_ 3 if _r ≪_ _d_ _model_ as we do not need to store the optimizer states for the frozen
parameters. On GPT-3 175B, we reduce the VRAM consumption during training from 1.2TB to
350GB. With _r_ = 4 and only the query and value projection matrices being adapted, the checkpoint
size is reduced by roughly 10,000 _×_ (from 350GB to 35MB) [4] . This allows us to train with significantly fewer GPUs and avoid I/O bottlenecks. Another benefit is that we can switch between tasks
while deployed at a much lower cost by only swapping the LoRA weights as opposed to all the
parameters. This allows for the creation of many customized models that can be swapped in and out
on the fly on machines that store the pre-trained weights in VRAM. We also observe a 25% speedup
during training on GPT-3 175B compared to full fine-tuning [5] as we do not need to calculate the
gradient for the vast majority of the parameters.


LoRA also has its limitations. For example, it is not straightforward to batch inputs to different tasks
with different _A_ and _B_ in a single forward pass, if one chooses to absorb _A_ and _B_ into _W_ to eliminate
additional inference latency. Though it is possible to not merge the weights and dynamically choose
the LoRA modules to use for samples in a batch for scenarios where latency is not critical.


5 E MPIRICAL E XPERIMENTS


We evaluate the downstream task performance of LoRA on RoBERTa (Liu et al., 2019), DeBERTa (He et al., 2021), and GPT-2 (Radford et al., b), before scaling up to GPT-3 175B (Brown
et al., 2020). Our experiments cover a wide range of tasks, from natural language understanding
(NLU) to generation (NLG). Specifically, we evaluate on the GLUE (Wang et al., 2019) benchmark
for RoBERTa and DeBERTa. We follow the setup of Li & Liang (2021) on GPT-2 for a direct comparison and add WikiSQL (Zhong et al., 2017) (NL to SQL queries) and SAMSum (Gliwa et al.,
2019) (conversation summarization) for large-scale experiments on GPT-3. See Appendix C for
more details on the datasets we use. We use NVIDIA Tesla V100 for all experiments.


5.1 B ASELINES


To compare with other baselines broadly, we replicate the setups used by prior work and reuse their
reported numbers whenever possible. This, however, means that some baselines might only appear
in certain experiments.


**Fine-Tuning (FT)** is a common approach for adaptation. During fine-tuning, the model is initialized
to the pre-trained weights and biases, and all model parameters undergo gradient updates.A simple
variant is to update only some layers while freezing others. We include one such baseline reported
in prior work (Li & Liang, 2021) on GPT-2, which adapts just the last two layers ( **FT** **[Top2]** ).


4 We still need the 350GB model during deployment; however, storing 100 adapted models only requires
350GB + 35MB * 100 _≈_ 354GB as opposed to 100 * 350GB _≈_ 35TB.
5 For GPT-3 175B, the training throughput for full fine-tuning is 32.5 tokens/s per V100 GPU; with the same
number of weight shards for model parallelism, the throughput is 43.1 tokens/s per V100 GPU for LoRA.


5


Model & Method # Trainable

Parameters MNLI SST-2 MRPC CoLA QNLI QQP RTE STS-B Avg.


RoB base (FT)* 125.0M **87.6** 94.8 90.2 **63.6** 92.8 **91.9** 78.7 91.2 86.4
RoB base (BitFit)* 0.1M 84.7 93.7 **92.7** 62.0 91.8 84.0 81.5 90.8 85.2
RoB base (Adpt [D] )* 0.3M 87.1 _±_ .0 94.2 _±_ .1 88.5 _±_ 1.1 60.8 _±_ .4 93.1 _±_ .1 90.2 _±_ .0 71.5 _±_ 2.7 89.7 _±_ .3 84.4
RoB base (Adpt [D] )* 0.9M 87.3 _±_ .1 94.7 _±_ .3 88.4 _±_ .1 62.6 _±_ .9 93.0 _±_ .2 90.6 _±_ .0 75.9 _±_ 2.2 90.3 _±_ .1 85.4
RoB base (LoRA) 0.3M 87.5 _±_ .3 **95.1** _±_ **.2** 89.7 _±_ .7 63.4 _±_ 1.2 **93.3** _±_ **.3** 90.8 _±_ .1 **86.6** _±_ **.7** **91.5** _±_ **.2** **87.2**


RoB large (FT)* 355.0M 90.2 **96.4** **90.9** 68.0 94.7 **92.2** 86.6 92.4 88.9
RoB large (LoRA) 0.8M **90.6** _±_ .2 96.2 _±_ .5 **90.9** _±_ 1.2 **68.2** _±_ 1.9 **94.9** _±_ .3 91.6 _±_ .1 **87.4** _±_ 2.5 **92.6** _±_ .2 **89.0**


RoB large (Adpt [P] ) _†_ 3.0M 90.2 _±_ .3 96.1 _±_ .3 90.2 _±_ .7 **68.3** _±_ 1.0 **94.8** _±_ .2 **91.9** _±_ .1 83.8 _±_ 2.9 92.1 _±_ .7 88.4
RoB large (Adpt [P] ) _†_ 0.8M **90.5** _±_ .3 **96.6** _±_ .2 89.7 _±_ 1.2 67.8 _±_ 2.5 **94.8** _±_ .3 91.7 _±_ .2 80.1 _±_ 2.9 91.9 _±_ .4 87.9
RoB large (Adpt [H] ) _†_ 6.0M 89.9 _±_ .5 96.2 _±_ .3 88.7 _±_ 2.9 66.5 _±_ 4.4 94.7 _±_ .2 92.1 _±_ .1 83.4 _±_ 1.1 91.0 _±_ 1.7 87.8
RoB large (Adpt [H] ) _†_ 0.8M 90.3 _±_ .3 96.3 _±_ .5 87.7 _±_ 1.7 66.3 _±_ 2.0 94.7 _±_ .2 91.5 _±_ .1 72.9 _±_ 2.9 91.5 _±_ .5 86.4
RoB large (LoRA) _†_ 0.8M **90.6** _±_ .2 96.2 _±_ .5 **90.2** _±_ 1.0 68.2 _±_ 1.9 **94.8** _±_ .3 91.6 _±_ .2 **85.2** _±_ 1.1 **92.3** _±_ .5 **88.6**


DeB XXL (FT)* 1500.0M 91.8 **97.2** 92.0 72.0 **96.0** 92.7 93.9 92.9 91.1
DeB XXL (LoRA) 4.7M **91.9** _±_ .2 96.9 _±_ .2 **92.6** _±_ .6 **72.4** _±_ 1.1 **96.0** _±_ .1 **92.9** _±_ .1 **94.9** _±_ .4 **93.0** _±_ .2 **91.3**


Table 2: RoBERTa base, RoBERTa large, and DeBERTa XXL with different adaptation methods on the
GLUE benchmark. We report the overall (matched and mismatched) accuracy for MNLI, Matthew’s
correlation for CoLA, Pearson correlation for STS-B, and accuracy for other tasks. Higher is better
for all metrics. * indicates numbers published in prior works. _†_ indicates runs configured in a setup
similar to Houlsby et al. (2019) for a fair comparison.


**Bias-only or BitFit** is a baseline where we only train the bias vectors while freezing everything else.
Contemporarily, this baseline has also been studied by BitFit (Zaken et al., 2021).


**Prefix-embedding tuning (PreEmbed)** inserts special tokens among the input tokens. These special tokens have trainable word embeddings and are generally not in the model’s vocabulary. Where
to place such tokens can have an impact on performance. We focus on “prefixing”, which prepends
such tokens to the prompt, and “infixing”, which appends to the prompt; both are discussed in Li &
Liang (2021). We use _l_ _p_ (resp. _l_ _i_ ) denote the number of prefix (resp. infix) tokens. The number of
trainable parameters is _|_ Θ _|_ = _d_ _model_ _×_ ( _l_ _p_ + _l_ _i_ ).


**Prefix-layer tuning (PreLayer)** is an extension to prefix-embedding tuning. Instead of just learning
the word embeddings (or equivalently, the activations after the embedding layer) for some special
tokens, we learn the activations after every Transformer layer. The activations computed from previous layers are simply replaced by trainable ones. The resulting number of trainable parameters is
_|_ Θ _|_ = _L × d_ _model_ _×_ ( _l_ _p_ + _l_ _i_ ), where _L_ is the number of Transformer layers.


**Adapter tuning** as proposed in Houlsby et al. (2019) inserts adapter layers between the selfattention module (and the MLP module) and the subsequent residual connection. There are two
fully connected layers with biases in an adapter layer with a nonlinearity in between. We call this
original design **Adapter** **[H]** . Recently, Lin et al. (2020) proposed a more efficient design with the
adapter layer applied only after the MLP module and after a LayerNorm. We call it **Adapter** **[L]** . This
is very similar to another deign proposed in Pfeiffer et al. (2021), which we call **Adapter** **[P]** . We also
include another baseline call AdapterDrop (R¨uckl´e et al., 2020) which drops some adapter layers for
greater efficiency ( **Adapter** **[D]** ). We cite numbers from prior works whenever possible to maximize
the number of baselines we compare with; they are in rows with an asterisk (*) in the first column.
In all cases, we have _|_ Θ _|_ = _L_ [ˆ] _Adpt_ _×_ (2 _×_ _d_ _model_ _×_ _r_ + _r_ + _d_ _model_ )+2 _×_ _L_ [ˆ] _LN_ _×_ _d_ _model_ where _L_ [ˆ] _Adpt_
is the number of adapter layers and _L_ [ˆ] _LN_ the number of trainable LayerNorms (e.g., in Adapter [L] ).


**LoRA** adds trainable pairs of rank decomposition matrices in parallel to existing weight matrices.
As mentioned in Section 4.2, we only apply LoRA to _W_ _q_ and _W_ _v_ in most experiments for simplicity.
The number of trainable parameters is determined by the rank _r_ and the shape of the original weights:
_|_ Θ _|_ = 2 _×_ _L_ [ˆ] _LoRA_ _× d_ _model_ _× r_, where _L_ [ˆ] _LoRA_ is the number of weight matrices we apply LoRA to.


6


Model & Method # Trainable E2E NLG Challenge
Parameters BLEU NIST MET ROUGE-L CIDEr


GPT-2 M (FT)* 354.92M 68.2 8.62 46.2 71.0 2.47
GPT-2 M (Adapter [L] )* 0.37M 66.3 8.41 45.0 69.8 2.40
GPT-2 M (Adapter [L] )* 11.09M 68.9 8.71 46.1 71.3 2.47
GPT-2 M (Adapter [H] ) 11.09M 67.3 _±_ .6 8.50 _±_ .07 46.0 _±_ .2 70.7 _±_ .2 2.44 _±_ .01
GPT-2 M (FT [Top2] )* 25.19M 68.1 8.59 46.0 70.8 2.41
GPT-2 M (PreLayer)* 0.35M 69.7 8.81 46.1 71.4 2.49
GPT-2 M (LoRA) 0.35M **70.4** _±_ **.1** **8.85** _±_ **.02** **46.8** _±_ **.2** **71.8** _±_ **.1** **2.53** _±_ **.02**

GPT-2 L (FT)* 774.03M 68.5 8.78 46.0 69.9 2.45
GPT-2 L (Adapter [L] ) 0.88M 69.1 _±_ .1 8.68 _±_ .03 46.3 _±_ .0 71.4 _±_ .2 **2.49** _±_ **.0**
GPT-2 L (Adapter [L] ) 23.00M 68.9 _±_ .3 8.70 _±_ .04 46.1 _±_ .1 71.3 _±_ .2 2.45 _±_ .02
GPT-2 L (PreLayer)* 0.77M 70.3 8.85 46.2 71.7 2.47
GPT-2 L (LoRA) 0.77M **70.4** _±_ **.1** **8.89** _±_ **.02** **46.8** _±_ **.2** **72.0** _±_ **.2** 2.47 _±_ .02


Table 3: GPT-2 medium (M) and large (L) with different adaptation methods on the E2E NLG
Challenge. For all metrics, higher is better. LoRA outperforms several baselines with comparable
or fewer trainable parameters. Confidence intervals are shown for experiments we ran. * indicates
numbers published in prior works.


5.2 R O BERT A BASE / LARGE


RoBERTa (Liu et al., 2019) optimized the pre-training recipe originally proposed in BERT (Devlin
et al., 2019a) and boosted the latter’s task performance without introducing many more trainable
parameters. While RoBERTa has been overtaken by much larger models on NLP leaderboards
such as the GLUE benchmark (Wang et al., 2019) in recent years, it remains a competitive and
popular pre-trained model for its size among practitioners. We take the pre-trained RoBERTa base
(125M) and RoBERTa large (355M) from the HuggingFace Transformers library (Wolf et al., 2020)
and evaluate the performance of different efficient adaptation approaches on tasks from the GLUE
benchmark. We also replicate Houlsby et al. (2019) and Pfeiffer et al. (2021) according to their
setup. To ensure a fair comparison, we make two crucial changes to how we evaluate LoRA when
comparing with adapters. First, we use the same batch size for all tasks and use a sequence length
of 128 to match the adapter baselines. Second, we initialize the model to the pre-trained model for
MRPC, RTE, and STS-B, not a model already adapted to MNLI like the fine-tuning baseline. Runs
following this more restricted setup from Houlsby et al. (2019) are labeled with _†_ . The result is
presented in Table 2 (Top Three Sections). See Section D.1 for details on the hyperparameters used.


5.3 D E BERT A XXL


DeBERTa (He et al., 2021) is a more recent variant of BERT that is trained on a much larger
scale and performs very competitively on benchmarks such as GLUE (Wang et al., 2019) and SuperGLUE (Wang et al., 2020). We evaluate if LoRA can still match the performance of a fully
fine-tuned DeBERTa XXL (1.5B) on GLUE. The result is presented in Table 2 (Bottom Section).
See Section D.2 for details on the hyperparameters used.


5.4 GPT-2 MEDIUM / LARGE


Having shown that LoRA can be a competitive alternative to full fine-tuning on NLU, we hope to
answer if LoRA still prevails on NLG models, such as GPT-2 medium and large (Radford et al.,
b). We keep our setup as close as possible to Li & Liang (2021) for a direct comparison. Due
to space constraint, we only present our result on E2E NLG Challenge (Table 3) in this section.
See Section F.1 for results on WebNLG (Gardent et al., 2017) and DART (Nan et al., 2020). We
include a list of the hyperparameters used in Section D.3.


7


|Model&Method|#Trainable<br>Parameters|WikiSQL MNLI -m SAMSum|
|---|---|---|
|Model&Method|# Trainable<br>Parameters|Acc. (%)<br>Acc. (%)<br>R1/R2/RL|


GPT-3 (FT) 175,255.8M **73.8** 89.5 52.0/28.0/44.5
GPT-3 (BitFit) 14.2M 71.3 91.0 51.3/27.4/43.5
GPT-3 (PreEmbed) 3.2M 63.1 88.6 48.3/24.2/40.5
GPT-3 (PreLayer) 20.2M 70.1 89.5 50.8/27.3/43.5
GPT-3 (Adapter [H] ) 7.1M 71.9 89.8 53.0/28.9/44.8
GPT-3 (Adapter [H] ) 40.1M 73.2 **91.5** 53.2/29.0/45.1


GPT-3 (LoRA) 4.7M 73.4 **91.7** **53.8/29.8/45.9**
GPT-3 (LoRA) 37.7M **74.0** **91.6** 53.4/29.2/45.1


Table 4: Performance of different adaptation methods on GPT-3 175B. We report the logical form
validation accuracy on WikiSQL, validation accuracy on MultiNLI-matched, and Rouge-1/2/L on
SAMSum. LoRA performs better than prior approaches, including full fine-tuning. The results
on WikiSQL have a fluctuation around _±_ 0 _._ 5%, MNLI-m around _±_ 0 _._ 1%, and SAMSum around
_±_ 0 _._ 2/ _±_ 0 _._ 2/ _±_ 0 _._ 1 for the three metrics.


5.5 S CALING UP TO GPT-3 175B


As a final stress test for LoRA, we scale up to GPT-3 with 175 billion parameters. Due to the high
training cost, we only report the typical standard deviation for a given task over random seeds, as
opposed to providing one for every entry. See Section D.4 for details on the hyperparameters used.


As shown in Table 4, LoRA matches or exceeds the fine-tuning baseline on all three datasets. Note
that not all methods benefit monotonically from having more trainable parameters, as shown in Figure 2. We observe a significant performance drop when we use more than 256 special tokens for
prefix-embedding tuning or more than 32 special tokens for prefix-layer tuning. This corroborates
similar observations in Li & Liang (2021). While a thorough investigation into this phenomenon
is out-of-scope for this work, we suspect that having more special tokens causes the input distribution to shift further away from the pre-training data distribution. Separately, we investigate the
performance of different adaptation approaches in the low-data regime in Section F.3.










|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
||||||Me<br>~~Fin~~<br>Pr<br>~~Pr~~|thod<br>~~e-Tune~~<br>efixEmbed<br>~~fixLayer~~||
||||||Ad<br>~~L~~|apter(H)<br>~~RA~~||
|||||~~o~~|~~o~~|||



Figure 2: GPT-3 175B validation accuracy vs. number of trainable parameters of several adaptation
methods on WikiSQL and MNLI-matched. LoRA exhibits better scalability and task performance.
See Section F.2 for more details on the plotted data points.


6 R ELATED W ORKS


**Transformer Language Models.** Transformer (Vaswani et al., 2017) is a sequence-to-sequence
architecture that makes heavy use of self-attention. Radford et al. (a) applied it to autoregressive language modeling by using a stack of Transformer decoders. Since then, Transformer-based language
models have dominated NLP, achieving the state-of-the-art in many tasks. A new paradigm emerged
with BERT (Devlin et al., 2019b) and GPT-2 (Radford et al., b) – both are large Transformer lan

8


guage models trained on a large amount of text – where fine-tuning on task-specific data after pretraining on general domain data provides a significant performance gain compared to training on
task-specific data directly. Training larger Transformers generally results in better performance and
remains an active research direction. GPT-3 (Brown et al., 2020) is the largest single Transformer
language model trained to-date with 175B parameters.


**Prompt Engineering and Fine-Tuning.** While GPT-3 175B can adapt its behavior with just a
few additional training examples, the result depends heavily on the input prompt (Brown et al.,
2020). This necessitates an empirical art of composing and formatting the prompt to maximize a
model’s performance on a desired task, which is known as prompt engineering or prompt hacking.
Fine-tuning retrains a model pre-trained on general domains to a specific task Devlin et al. (2019b);
Radford et al. (a). Variants of it include learning just a subset of the parameters Devlin et al. (2019b);
Collobert & Weston (2008), yet practitioners often retrain all of them to maximize the downstream
performance. However, the enormity of GPT-3 175B makes it challenging to perform fine-tuning in
the usual way due to the large checkpoint it produces and the high hardware barrier to entry since it
has the same memory footprint as pre-training.


**Parameter-Efficient Adaptation.** Many have proposed inserting _adapter_ layers between existing
layers in a neural network (Houlsby et al., 2019; Rebuffi et al., 2017; Lin et al., 2020). Our method
uses a similar bottleneck structure to impose a low-rank constraint on the weight updates. The
key functional difference is that our learned weights can be merged with the main weights during
inference, thus not introducing any latency, which is not the case for the adapter layers (Section 3).
A comtenporary extension of adapter is COMPACTER (Mahabadi et al., 2021), which essentially
parametrizes the adapter layers using Kronecker products with some predetermined weight sharing
scheme. Similarly, combining LoRA with other tensor product-based methods could potentially
improve its parameter efficiency, which we leave to future work. More recently, many proposed
optimizing the input word embeddings in lieu of fine-tuning, akin to a continuous and differentiable
generalization of prompt engineering (Li & Liang, 2021; Lester et al., 2021; Hambardzumyan et al.,
2020; Liu et al., 2021). We include comparisons with Li & Liang (2021) in our experiment section.
However, this line of works can only scale up by using more special tokens in the prompt, which
take up available sequence length for task tokens when positional embeddings are learned.


**Low-Rank Structures in Deep Learning.** Low-rank structure is very common in machine learning. A lot of machine learning problems have certain intrinsic low-rank structure (Li et al., 2016;
Cai et al., 2010; Li et al., 2018b; Grasedyck et al., 2013). Moreover, it is known that for many
deep learning tasks, especially those with a heavily over-parametrized neural network, the learned
neural network will enjoy low-rank properties after training (Oymak et al., 2019). Some prior works
even explicitly impose the low-rank constraint when training the original neural network (Sainath
et al., 2013; Povey et al., 2018; Zhang et al., 2014; Jaderberg et al., 2014; Zhao et al., 2016; Khodak et al., 2021; Denil et al., 2014); however, to the best of our knowledge, none of these works
considers low-rank update to a frozen model for _adaptation to downstream tasks_ . In theory literature, it is known that neural networks outperform other classical learning methods, including the
corresponding (finite-width) neural tangent kernels (Allen-Zhu et al., 2019; Li & Liang, 2018) when
the underlying concept class has certain low-rank structure (Ghorbani et al., 2020; Allen-Zhu & Li,
2019; Allen-Zhu & Li, 2020a). Another theoretical result in Allen-Zhu & Li (2020b) suggests that
low-rank adaptations can be useful for adversarial training. In sum, we believe that our proposed
low-rank adaptation update is well-motivated by the literature.


7 U NDERSTANDING THE L OW -R ANK U PDATES


Given the empirical advantage of LoRA, we hope to further explain the properties of the low-rank
adaptation learned from downstream tasks. Note that the low-rank structure not only lowers the
hardware barrier to entry which allows us to run multiple experiments in parallel, but also gives
better interpretability of how the update weights are correlated with the pre-trained weights. We
focus our study on GPT-3 175B, where we achieved the largest reduction of trainable parameters
(up to 10,000 _×_ ) without adversely affecting task performances.


We perform a sequence of empirical studies to answer the following questions: 1) Given a parameter
budget constraint, _which subset of weight matrices_ in a pre-trained Transformer should we adapt


9


to maximize downstream performance? 2) Is the “optimal” adaptation matrix ∆ _W really rank-_
_deficient_ ? If so, what is a good rank to use in practice? 3) What is the connection between ∆ _W_ and
_W_ ? Does ∆ _W_ highly correlate with _W_ ? How large is ∆ _W_ comparing to _W_ ?


We believe that our answers to question (2) and (3) shed light on the fundamental principles of using
pre-trained language models for downstream tasks, which is a critical topic in NLP.


7.1 W HICH W EIGHT M ATRICES IN T RANSFORMER S HOULD W E A PPLY L O RA TO ?


Given a limited parameter budget, which types of weights should we adapt with LoRA to obtain
the best performance on downstream tasks? As mentioned in Section 4.2, we only consider weight
matrices in the self-attention module. We set a parameter budget of 18M (roughly 35MB if stored
in FP16) on GPT-3 175B, which corresponds to _r_ = 8 if we adapt one type of attention weights or
_r_ = 4 if we adapt two types, for all 96 layers. The result is presented in Table 5.


# of Trainable Parameters = 18M


Weight Type _W_ _q_ _W_ _k_ _W_ _v_ _W_ _o_ _W_ _q_ _, W_ _k_ _W_ _q_ _, W_ _v_ _W_ _q_ _, W_ _k_ _, W_ _v_ _, W_ _o_
Rank _r_ 8 8 8 8 4 4 2


WikiSQL ( _±_ 0 _._ 5%) 70.4 70.0 73.0 73.2 71.4 **73.7** **73.7**
MultiNLI ( _±_ 0 _._ 1%) 91.0 90.8 91.0 91.3 91.3 91.3 **91.7**


Table 5: Validation accuracy on WikiSQL and MultiNLI after applying LoRA to different types of
attention weights in GPT-3, given the same number of trainable parameters. Adapting both _W_ _q_ and
_W_ _v_ gives the best performance overall. We find the standard deviation across random seeds to be
consistent for a given dataset, which we report in the first column.


Note that putting all the parameters in ∆ _W_ _q_ or ∆ _W_ _k_ results in significantly lower performance,
while adapting both _W_ _q_ and _W_ _v_ yields the best result. This suggests that even a rank of four
captures enough information in ∆ _W_ such that it is preferable to adapt more weight matrices than
adapting a single type of weights with a larger rank.


7.2 W HAT IS THE O PTIMAL R ANK _r_ FOR L O RA?


We turn our attention to the effect of rank _r_ on model performance. We adapt _{W_ _q_ _, W_ _v_ _}_,
_{W_ _q_ _, W_ _k_ _, W_ _v_ _, W_ _c_ _}_, and just _W_ _q_ for a comparison.


Weight Type _r_ = 1 _r_ = 2 _r_ = 4 _r_ = 8 _r_ = 64

WikiSQL( _±_ 0 _._ 5%) _W_ _q_ 68.8 69.6 70.5 70.4 70.0
_W_ _q_ _, W_ _v_ 73.4 73.3 73.7 73.8 73.5
_W_ _q_ _, W_ _k_ _, W_ _v_ _, W_ _o_ 74.1 73.7 74.0 74.0 73.9



MultiNLI ( _±_ 0 _._ 1%)



_W_ _q_ 90.7 90.9 91.1 90.7 90.7
_W_ _q_ _, W_ _v_ 91.3 91.4 91.3 91.6 91.4
_W_ _q_ _, W_ _k_ _, W_ _v_ _, W_ _o_ 91.2 91.7 91.7 91.5 91.4



Table 6: Validation accuracy on WikiSQL and MultiNLI with different rank _r_ . To our surprise, a
rank as small as one suffices for adapting both _W_ _q_ and _W_ _v_ on these datasets while training _W_ _q_ alone
needs a larger _r_ . We conduct a similar experiment on GPT-2 in Section H.2.


Table 6 shows that, surprisingly, LoRA already performs competitively with a very small _r_ (more
so for _{W_ _q_ _, W_ _v_ _}_ than just _W_ _q_ ). This suggests the update matrix ∆ _W_ could have a very small
“intrinsic rank”. [6] To further support this finding, we check the overlap of the subspaces learned by
different choices of _r_ and by different random seeds. We argue that increasing _r_ does not cover a
more meaningful subspace, which suggests that a low-rank adaptation matrix is sufficient.


6
However, we do not expect a small _r_ to work for every task or dataset. Consider the following thought
experiment: if the downstream task were in a different language than the one used for pre-training, retraining
the entire model (similar to LoRA with _r_ = _d_ _model_ ) could certainly outperform LoRA with a small _r_ .


10


**Subspace similarity between different** _**r**_ **.** Given _A_ _r_ =8 and _A_ _r_ =64 which are the learned adaptation matrices with rank _r_ = 8 and 64 using the _same pre-trained model_, we perform singular value
decomposition and obtain the right-singular unitary matrices _U_ _A_ _r_ =8 and _U_ _A_ _r_ =64 . [7] We hope to answer: how much of the subspace spanned by the top _i_ singular vectors in _U_ _A_ _r_ =8 (for 1 _≤_ _i ≤_ 8) is
contained in the subspace spanned by top _j_ singular vectors of _U_ _A_ _r_ =64 (for 1 _≤_ _j ≤_ 64)? We measure this quantity with a normalized subspace similarity based on the Grassmann distance (See Appendix G for a more formal discussion)

_A_ _r_ =8 _[U]_ _A_ _[ j]_ _r_ =64 _[||]_ _F_ [2]
_φ_ ( _A_ _r_ =8 _, A_ _r_ =64 _, i, j_ ) = _[||][U]_ _[ i][⊤]_ _∈_ [0 _,_ 1] (4)
min( _i, j_ )


where _U_ _A_ _[i]_ _r_ =8 [represents the columns of] _[ U]_ _[A]_ _r_ =8 [corresponding to the top-] _[i]_ [ singular vectors.]


_φ_ ( _·_ ) has a range of [0 _,_ 1], where 1 represents a complete overlap of subspaces and 0 a complete
separation. See Figure 3 for how _φ_ changes as we vary _i_ and _j_ . We only look at the 48th layer
(out of 96) due to space constraint, but the conclusion holds for other layers as well, as shown
in Section H.1.













Figure 3: Subspace similarity between column vectors of _A_ _r_ =8 and _A_ _r_ =64 for both ∆ _W_ _q_ and ∆ _W_ _v_ .
The third and the fourth figures zoom in on the lower-left triangle in the first two figures. The top
directions in _r_ = 8 are included in _r_ = 64, and vice versa.


We make an _important observation_ from Figure 3.


Directions corresponding to the top singular vector overlap significantly between
_A_ _r_ =8 and _A_ _r_ =64, while others do not. Specifically, ∆ _W_ _v_ (resp. ∆ _W_ _q_ ) of _A_ _r_ =8
and ∆ _W_ _v_ (resp. ∆ _W_ _q_ ) of _A_ _r_ =64 share a subspace of dimension 1 with normalized
similarity _>_ 0 _._ 5, providing an explanation of why _r_ = 1 performs quite well in our
downstream tasks for GPT-3.


Since both _A_ _r_ =8 and _A_ _r_ =64 are learned using the same pre-trained model, Figure 3 indicates that
the top singular-vector directions of _A_ _r_ =8 and _A_ _r_ =64 are the most useful, while other directions
potentially contain mostly random noises accumulated during training. Hence, the adaptation matrix
can indeed have a very low rank.


**Subspace similarity between different random seeds.** We further confirm this by plotting the
normalized subspace similarity between two randomly seeded runs with _r_ = 64, shown in Figure 4.
∆ _W_ _q_ appears to have a higher “intrinsic rank” than ∆ _W_ _v_, since more common singular value directions are learned by both runs for ∆ _W_ _q_, which is in line with our empirical observation in Table 6.
As a comparison, we also plot two random Gaussian matrices, which do not share any common
singular value directions with each other.


7.3 H OW D OES THE A DAPTATION M ATRIX ∆ _W_ C OMPARE TO _W_ ?


We further investigate the relationship between ∆ _W_ and _W_ . In particular, does ∆ _W_ highly correlate
with _W_ ? (Or mathematically, is ∆ _W_ mostly contained in the top singular directions of _W_ ?) Also,


7 Note that a similar analysis can be carried out with _B_ and the left-singular unitary matrices – we stick with
_A_ for our experiments.


11


Figure 4: **Left and Middle:** Normalized subspace similarity between the column vectors of _A_ _r_ =64
from two random seeds, for both ∆ _W_ _q_ and ∆ _W_ _v_ in the 48-th layer. **Right:** the same heat-map
between the column vectors of two random Gaussian matrices. See Section H.1 for other layers.


how “large” is ∆ _W_ comparing to its corresponding directions in _W_ ? This can shed light on the
underlying mechanism for adapting pre-trained language models.


To answer these questions, we project _W_ onto the _r_ -dimensional subspace of ∆ _W_ by computing _U_ _[⊤]_ _WV_ _[⊤]_, with _U_ / _V_ being the left/right singular-vector matrix of ∆ _W_ . Then, we compare the Frobenius norm between _∥U_ _[⊤]_ _WV_ _[⊤]_ _∥_ _F_ and _∥W_ _∥_ _F_ . As a comparison, we also compute
_∥U_ _[⊤]_ _WV_ _[⊤]_ _∥_ _F_ by replacing _U, V_ with the top _r_ singular vectors of _W_ or a random matrix.


_r_ = 4 _r_ = 64
∆ _W_ _q_ _W_ _q_ Random ∆ _W_ _q_ _W_ _q_ Random

_||U_ _[⊤]_ _W_ _q_ _V_ _[⊤]_ _||_ _F_ = 0.32 21.67 0.02 1.90 37.71 0.33


_||W_ _q_ _||_ _F_ = 61 _._ 95 _||_ ∆ _W_ _q_ _||_ _F_ = 6 _._ 91 _||_ ∆ _W_ _q_ _||_ _F_ = 3 _._ 57


Table 7: The Frobenius norm of _U_ _[⊤]_ _W_ _q_ _V_ _[⊤]_ where _U_ and _V_ are the left/right top _r_ singular vector
directions of either (1) ∆ _W_ _q_, (2) _W_ _q_, or (3) a random matrix. The weight matrices are taken from
the 48th layer of GPT-3.


We draw _several conclusions_ from Table 7. First, ∆ _W_ has a stronger correlation with _W_ compared
to a random matrix, indicating that ∆ _W_ amplifies some features that are already in _W_ . Second,
instead of repeating the top singular directions of _W_, ∆ _W_ only _amplifies directions that are not_
_emphasized in W_ . Third, the amplification factor is rather huge: 21 _._ 5 _≈_ 6 _._ 91 _/_ 0 _._ 32 for _r_ = 4.
See Section H.4 for why _r_ = 64 has a smaller amplification factor. We also provide a visualization
in Section H.3 for how the correlation changes as we include more top singular directions from _W_ _q_ .
This suggests that the low-rank adaptation matrix potentially _amplifies the important features for_
_specific downstream tasks that were learned but not emphasized in the general pre-training model_ .


8 C ONCLUSION AND F UTURE W ORK


Fine-tuning enormous language models is prohibitively expensive in terms of the hardware required
and the storage/switching cost for hosting independent instances for different tasks. We propose
LoRA, an efficient adaptation strategy that neither introduces inference latency nor reduces input
sequence length while retaining high model quality. Importantly, it allows for quick task-switching
when deployed as a service by sharing the vast majority of the model parameters. While we focused
on Transformer language models, the proposed principles are generally applicable to any neural
networks with dense layers.


There are many directions for future works. 1) LoRA can be combined with other efficient adaptation methods, potentially providing orthogonal improvement. 2) The mechanism behind fine-tuning
or LoRA is far from clear – how are features learned during pre-training transformed to do well
on downstream tasks? We believe that LoRA makes it more tractable to answer this than full fine

12


tuning. 3) We mostly depend on heuristics to select the weight matrices to apply LoRA to. Are
there more principled ways to do it? 4) Finally, the rank-deficiency of ∆ _W_ suggests that _W_ could
be rank-deficient as well, which can also be a source of inspiration for future works.


R EFERENCES


Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta. Intrinsic Dimensionality Explains the
Effectiveness of Language Model Fine-Tuning. _arXiv:2012.13255 [cs]_, December 2020. URL
[http://arxiv.org/abs/2012.13255.](http://arxiv.org/abs/2012.13255)


Zeyuan Allen-Zhu and Yuanzhi Li. What Can ResNet Learn Efficiently, Going Beyond Kernels? In
_NeurIPS_ [, 2019. Full version available at http://arxiv.org/abs/1905.10337.](http://arxiv.org/abs/1905.10337)


Zeyuan Allen-Zhu and Yuanzhi Li. Backward feature correction: How deep learning performs deep
learning. _arXiv preprint arXiv:2001.04413_, 2020a.


Zeyuan Allen-Zhu and Yuanzhi Li. Feature purification: How adversarial training performs robust
deep learning. _arXiv preprint arXiv:2005.10190_, 2020b.


Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song. A convergence theory for deep learning via overparameterization. In _ICML_ [, 2019. Full version available at http://arxiv.org/abs/1811.](http://arxiv.org/abs/1811.03962)
[03962.](http://arxiv.org/abs/1811.03962)


Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization, 2016.


Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal,
Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M.
Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin,
Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford,
Ilya Sutskever, and Dario Amodei. Language Models are Few-Shot Learners. _arXiv:2005.14165_

_[cs]_ [, July 2020. URL http://arxiv.org/abs/2005.14165.](http://arxiv.org/abs/2005.14165)


Jian-Feng Cai, Emmanuel J Cand`es, and Zuowei Shen. A singular value thresholding algorithm for
matrix completion. _SIAM Journal on optimization_, 20(4):1956–1982, 2010.


Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, and Lucia Specia. Semeval-2017 task
1: Semantic textual similarity multilingual and crosslingual focused evaluation. _Proceedings of_
_the 11th International Workshop on Semantic Evaluation (SemEval-2017)_, 2017. doi: 10.18653/
[v1/s17-2001. URL http://dx.doi.org/10.18653/v1/S17-2001.](http://dx.doi.org/10.18653/v1/S17-2001)


Ronan Collobert and Jason Weston. A unified architecture for natural language processi