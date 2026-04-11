## Y A RN: E FFICIENT C ONTEXT W INDOW E XTENSION OF L ARGE L ANGUAGE M ODELS

**Bowen Peng** **[†]** [1] **Jeffrey Quesnelle** **[†]** [1] **Honglu Fan** [23] **Enrico Shippole**


1 Nous Research 2 EleutherAI 3 University of Geneva


A BSTRACT


Rotary Position Embeddings (RoPE) have been shown to effectively encode positional information in transformer-based language models. However, these models
fail to generalize past the sequence length they were trained on. We present YaRN
(Yet another RoPE extensioN method), a compute-efficient method to extend the
context window of such models, requiring 10x less tokens and 2.5x less training
steps than previous methods. Using YaRN, we show that LLaMA models can
effectively utilize and extrapolate to context lengths much longer than their original
pre-training would allow, while also surpassing previous the state-of-the-art at
context window extension. In addition, we demonstrate that YaRN exhibits the
capability to extrapolate beyond the limited context of a fine-tuning dataset. Code
[is available at https://github.com/jquesnelle/yarn.](https://github.com/jquesnelle/yarn)


1 I NTRODUCTION


Transformer-based Large Language Models(Vaswani et al., 2017) (LLMs) have become the nearubiquitous choice for many natural language processing (NLP) tasks where long-range abilities such
as _in-context learning_ (ICL) has been crucial.


In performing the NLP tasks, the maximal length of the sequences (the _context window_ ) determined by
its training processes has been one of the major limits of a pretrained LLM. Being able to dynamically
extend the context window via a small amount of fine-tuning (or without fine-tuning) has become
more and more desirable. To this end, the position encodings of transformers are the center of the
discussions.


The original Transformer architecture used an absolute sinusoidal position encoding, which was later
improved to a learnable absolute position encoding (Gehring et al., 2017). Since then, relative positional encoding schemes (Shaw et al., 2018) have further increased the performance of Transformers.
Currently, the most popular relative positional encodings are _T5 Relative Bias_ (Roberts et al., 2019),
_RoPE_ (Su et al., 2022), _XPos_ (Sun et al., 2022), and _ALiBi_ (Press et al., 2022).


One reoccurring limitation with positional encodings is the inability to generalize past the context
window seen during training. While some methods such as ALiBi are able to do limited generalization,
none are able to generalize to sequences significantly longer than their pre-trained length (Kazemnejad
et al., 2023).


Some works have been done to overcome such limitation. (Chen et al., 2023) and concurrently (kaiokendev, 2023) proposed to extend the context length by slightly modifying RoPE via Position
Interpolation (PI) and fine-tuning on a small amount of data. As an alternative, (bloc97, 2023a)
proposed the "NTK-aware" interpolation by taking the loss of high frequency into account. Since then,
two improvements of the "NTK-aware" interpolation have been proposed, with different emphasis:


    - the "Dynamic NTK" interpolation method (emozilla, 2023) for pre-trained models without
fine-tuning.


    - the "NTK-by-parts" interpolation method (bloc97, 2023b) which performs the best when
fine-tuned on a small amount of longer-context data.


  - Correspondence: {bloc,emozilla}@nousresearch.com


1


The "NTK-aware" interpolation and the "Dynamic NTK" interpolation have already seen their
presence in the open-source models such as Code Llama (Rozière et al., 2023) (using "NTK-aware"
interpolation) and Qwen 7B (qwe) (using "Dynamic NTK").


In this paper, in addition to making a complete account of the previous unpublished works on the
"NTK-aware", the "Dynamic NTK" and the "NTK-by-parts" interpolations, we present YaRN (Yet
another RoPE extensioN method), an improved method to efficiently extend the context window
of models trained with Rotary Position Embeddings (RoPE) including the LLaMA (Touvron et al.,
2023a), the GPT-NeoX (Black et al., 2022), and the PaLM (Chowdhery et al., 2022) families of
models.


The relationship between different methods and how they evolve into YaRN can be summarized into
the following diagram:


Figure 1: An outline of the relationship between different interpolation methods.


YaRN reaches state-of-the-art performances in context window extensions after fine-tuning on less
than _∼_ 0.1% of the original pre-training data. In the meantime, by combining with the inference-time
technique called Dynamic Scaling, the Dynamic-YaRN allows for more than 2x context window
extension without any fine-tuning.


2 B ACKGROUND AND R ELATED W ORK


2.1 R OTARY P OSITION E MBEDDINGS


The basis of our work is the Rotary Position Embedding (RoPE) introduced in (Su et al., 2022).
We work on a hidden layer where the set of hidden neurons are denoted by _D_ . Given a sequence
of vectors _**x**_ 1 _, · · ·,_ _**x**_ _L_ _∈_ R _[|][D][|]_, following the notation of (Su et al., 2022), the attention layer first
converts the vectors into the query vectors and the key vectors:


_**q**_ _m_ = _f_ _q_ ( _**x**_ _m_ _, m_ ) _∈_ R _[|][D][|]_ _,_ _**k**_ _n_ = _f_ _k_ ( _**x**_ _n_ _, n_ ) _∈_ R _[|][D][|]_ _._ (1)


Next, the attention weights are calculated as

softmax( _**[q]**_ _m_ _[T]_ _**[k]**_ _[n]_ ) _,_ (2)
~~�~~ _|D|_


where _**q**_ _m_ _,_ _**k**_ _n_ are considered as column vectors so that _**q**_ _m_ _[T]_ _**[k]**_ _[n]_ [is simply the Euclidean inner product.]
In RoPE, we first assume that _|D|_ is even and identify the embedding space and the hidden states as
complex vector spaces:
R _[|][D][|]_ _[ ∼]_ = C _[|][D][|][/]_ [2]

where the inner product _**q**_ _[T]_ _**k**_ becomes the real part of the standard Hermitian inner product Re( _**q**_ _[∗]_ _**k**_ ) .
More specifically, the isomorphisms interleave the real part and the complex part
�( _**x**_ _m_ ) 1 _, · · ·,_ ( _**x**_ _m_ ) _|D|_ � _�→_ �( _**x**_ _m_ ) 1 + _i_ ( _**x**_ _m_ ) 2 _, · · ·,_ (( _**x**_ _m_ ) _|D|−_ 1 + _i_ ( _**x**_ _m_ ) _|D|_ )� _,_ (3)
�( **q** _m_ ) 1 _, · · ·,_ ( **q** _m_ ) _|D|_ � _�→_ �( **q** _m_ ) 1 + _i_ ( **q** _m_ ) 2 _, · · ·,_ (( **q** _m_ ) _|D|−_ 1 + _i_ ( **q** _m_ ) _|D|_ )� _._ (4)


To convert embeddings _**x**_ _m_ _,_ _**x**_ _n_ into query and key vectors, we are first given R-linear operators


_**W**_ _q_ _,_ _**W**_ _k_ : R _[|][D][|]_ _→_ R _[|][D][|]_ _._


2


Let _**θ**_ = diag( _θ_ 1 _, · · ·, θ_ _|D|/_ 2 ). In complex coordinates, we define


_f_ _**W**_ ( _**x**_ _m_ _, m,_ _**θ**_ ) = _e_ _[im]_ _**[θ]**_ _**W x**_ _m_ _,_ (5)


for any linear operator _**W**_ . The functions _f_ _q_ _, f_ _k_ in RoPE are given by


_f_ _q_ = _f_ _**W**_ _q_ _, f_ _k_ = _f_ _**W**_ _k_ _._ (6)


where _θ_ _d_ = _b_ _[−]_ [2] _[d/][|][D][|]_ and _b_ = 10000 . This way, RoPE associates each (complex-valued) hidden
neuron with a separate frequency _θ_ _d_ . The benefit of doing so is that the dot product between the
query vector and the key vector only depends on the relative distance _m −_ _n_ .


In later discussions, a context length interpolation usually aims to modify the equation Eq. 5. To set
up a uniform convention for these discussions, note that a modification _f_ _**W**_ _[′]_ [can take the following]
form:


_f_ _**W**_ _[′]_ [(] _**[x]**_ _[m]_ _[, m,]_ _**[ θ]**_ [) =] _[ f]_ _**[W]**_ [(] _**[x]**_ _[m]_ _[, g]_ [(] _[m]_ [)] _[,]_ _**[ h]**_ [(] _**[θ]**_ [))] _[,]_ (7)


where _g_ ( _m_ ) is a map between real numbers and _**h**_ ( _**θ**_ ) acts on the entries of the diagonal matrix _**θ**_
uniformly by diag( _h_ ( _θ_ 1 ) _, · · ·, h_ ( _θ_ _|D|/_ 2 )) according to a function _h_ . _g_ and _h_ are method-dependent
functions.


In the subsequent sections, when we introduce a new interpolation method of the form Eq. 7, we only
specify the functions _g_ ( _m_ ) and _h_ ( _θ_ _d_ ).


2.2 A DDITIONAL NOTATIONS


Given the pretrained maximal context length _L_, our goal is to extend it to _L_ _[′]_ _> L_ either with or
without finetuning. We introduce the notion of _scale factor s_ defined by _s_ = _[L]_ _L_ _[′]_ [.]


For the convenience of some discussions, we also introduce _wavelength_ _λ_ _d_ associated with the _d_ -th
hidden dimension of RoPE as follows:



_λ_ _d_ = [2] _[π]_ = 2 _πb_

_θ_ _d_



2 _d_
_|D|_ _._ (8)



The wavelength describes the length of tokens needed in order for the rotary position embedding at
dimension _d_ to perform a full rotation (2 _π_ ).


2.3 R ELATED WORK


Position Interpolation (PI) is one of the earlier works extending context lengths of RoPE proposed by
Chen et al. (2023), and concurrently kaiokendev (2023). Under the notation of Eq. 7, it is setting


_g_ ( _m_ ) = _s · m,_ _**h**_ ( _**θ**_ ) = _**θ**_ _,_ (9)


where _s_ is the scale factor _[L]_ _[′]_

_L_ [. We include some details in Appendix A.1.]


ReRoPE (Su, 2023) also aims to extend the context size of existing models pre-trained with RoPE,
and claims "infinite" context length without needing any fine-tuning. This claim is backed by a
monotonically decreasing loss with increasing context length up to 16k on the Llama 2 13B model. It
achieves context extension by modifying the attention mechanism and thus is not purely an embedding
interpolation method. Since it is currently not compatible with Flash Attention 2 (Dao, 2023) and
requires two attention passes during inference, we do not consider it for comparison.


Concurrently with our work, LM-Infinite (Han et al., 2023) proposes similar ideas to YaRN, but
focuses on "on-the-fly" length generalization for non-fine-tuned models. Since they also modify
the attention mechanism of the models, it is not an embedding interpolation method and is not
immediately compatible with Flash Attention 2.


3


3 M ETHODOLOGY


Whereas PI stretches all RoPE dimensions equally, we find that the theoretical interpolation bound
described by PI (Chen et al., 2023) is insufficient at predicting the complex dynamics between RoPE
and the LLM’s internal embeddings. In the following subsections, we describe the main issues
with PI we have individually identified and solved, so as to give the readers the context, origin and
justifications of each method which we use in concert to obtain the full YaRN method.


3.1 L OSS OF H IGH F REQUENCY INFORMATION - "NTK- AWARE " INTERPOLATION


If we look at rotary position embeddings (RoPE) only from an information encoding perspective,
it was shown in (Tancik et al., 2020), using Neural Tangent Kernel (NTK) theory, that deep neural
networks have trouble learning high frequency information if the input dimension is low and the
corresponding embeddings lack high frequency components. Here we can see the similarities: a
token’s positional information is one-dimensional, and RoPE expands it to an n-dimensional complex
vector embedding. RoPE closely resembles Fourier Features (Tancik et al., 2020) in many aspects, as
it is possible to define RoPE as a special 1D case of a Fourier Feature.


In the case of Positional Interpolation (PI), as we strech all dimensions equally by a factor _s_, it
removes the high frequency components of RoPE. This degradation is worsened as the scaling factor
_s_ grows, and at some point, the network will not be able to recover. Previous fine-tunes (kaiokendev,
2023) (Chen et al., 2023) (Together.ai, 2023) (Quesnelle et al., 2023) using PI were only able to
achieve a scaling factor of roughly _s_ = 8 before the LLM’s outputs starts to degrade, even after
fine-tuning.


In order to alleviate this issue, the "NTK-aware" interpolation was developed in (bloc97, 2023a).
Instead of scaling every dimension of RoPE equally by a factor _s_, we spread out the interpolation
pressure across multiple dimensions by scaling high frequencies less and low frequencies more. One
can obtain such a transformation in many ways, but the simplest would be to perform a base change
on the value of _θ_ . The details are described in the Appendix A.2 and the method has seen some
open-source adoptions [*] .


One main issue of this "NTK-aware" scaling is that it is very difficult to determine what optimal base
should be used for an intended context extension by _s_ times. The best base to use for "NTK-aware"
interpolation usually has to be found empirically, which significantly increases the difficulty and
cost of obtaining a successful fine-tuned model. Despite its limitations, the observations from the
NTK theory is valid and the following idea is still maintained and executed in a different way in the
"NTK-by-parts" interpolation introduced in the next section.


3.2 L OSS OF R ELATIVE L OCAL D ISTANCES - "NTK- BY - PARTS " INTERPOLATION


To understand why "NTK-aware" interpolation works better than PI and to eliminate its disadvantages,
we have to take a closer look at RoPE. In this section, we think heavily in terms of the wavelengths
_λ_ _d_ defined in Eq. 8 in the formula of RoPE. For simplicity, we omit the subscript _d_ in _λ_ _d_ and the
reader is encouraged to think about _λ_ as the wavelength of an arbitrary periodic function.


In theory, as RoPE is a relative position embedding, it should be quite surprising that it fails to
generalize to unseen longer context sizes. However, we can show that in practice, RoPE does not
only encode relative position. One observation we can make is that given a context size _L_, there are
some dimensions _d_ where the wavelength is longer than the maximum context length seen during
pretraining ( _λ > L_ ), this suggests that some dimensions’ rotary embeddings might not be distributed
evenly in the rotational domain (i.e. does not perform a full rotation for the entire training context
size). In such cases, we presume having unique position pairs [†] implies that the absolute positional


  - We note that shortly before the release of this article, Code Llama (Rozière et al., 2023) was released
and uses "NTK-aware" scaling by manually scaling the base _b_ to 1M, in which they call this method as RoPE
"adjusted base frequency" (ABF).

  - Since the dimension never rotates fully at least once during pre-training, if we pick the first token as the
anchor, every other token during pre-training has an unique distance to it, which the neural network can use to
determine its absolute position.


4


information remains intact in those dimensions. On the contrary, when the wavelength is short, only
relative positional information is accessible to the network.


Given these observations, we can see that it is important to not touch the dimensions that only encode
relative positional information, as they are crucial for the network to distinguish the relative order
of nearby tokens. Meanwhile, dimensions that only encode absolute positional information should
always be interpolated, as larger distances will be out of distribution. Instead of arbitrarily changing
the base in "NTK-aware" interpolation (which basically does something similar to what is described
here), we can formulate an explicit and targeted interpolation method that takes in account all of the
above.


In other words,


    - if the wavelength _λ_ is much smaller than the context size _L_, we do not interpolate;

    - if the wavelength _λ_ is equal to or bigger than the context size _L_, we want to only interpolate
and avoid any extrapolation (unlike the previous "NTK-aware" method);

    - dimensions in-between can have a bit of both, similar to the "NTK-aware" interpolation.


As a result, it is more convenient to introduce the ratio _r_ = _[L]_ _λ_ [between the original context size] _[ L]_ [ and]

the wavelength _λ_ . This ratio represents the number of rotations a certain RoPE dimension makes
given a fixed pretrained context length _L_ . In the _d_ -th hidden state, the ratio _r_ depends on _d_ in the
following way:



_r_ ( _d_ ) = _[L]_



_L_

_[L]_ =

_λ_ _d_



2 _d_ (10)
_|D|_ _[.]_



2 _πb_



In order to define the boundary of the different interpolation strategies as above, we introduce
two extra parameters _α, β_ . All hidden dimensions _d_ where _r_ ( _d_ ) _< α_ are those where we linearly
interpolate by a scale _s_ (exactly like PI, avoiding any extrapolation), and the _d_ where _r_ ( _d_ ) _> β_ are
those where we do not interpolate at all. Define the ramp function _γ_ to be



_γ_ ( _r_ ) =



0 _,_ if _r < α_

1 _,_ if _r > β_
(11)

 _r −_ _α_ otherwise _._

 _β −_ _α_ _[,]_



With the help of the ramp function, the "NTK-by-parts" method can be described as follows.


**Definition 1** _The "NTK-by-parts" interpolation is a modification of RoPE using Eq. 7 with the_
_following functions_ [‡] _._


_g_ ( _m_ ) = _m_ (12)


_θ_ _d_
_h_ ( _θ_ _d_ ) = �1 _−_ _γ_ � _r_ ( _d_ )� [�] _s_ [+] _[ γ]_ � _r_ ( _d_ )� _θ_ _d_ _._ (13)


The values of _α_ and _β_ should be tuned on a case-by-case basis. For example, we have found
experimentally that for the Llama family of models, good values for _α_ and _β_ are _α_ = 1 and _β_ = 32 .


Using the techniques described in this section, a variant of the resulting method was released under
the name "NTK-by-parts" interpolation (bloc97, 2023b). This improved method performs better
than the previous PI (Chen et al., 2023) and "NTK-aware" 3.1 interpolation methods, both with
non-fine-tuned models and with fine-tuned models, as shown in (bloc97, 2023b) and Section 4.2.


3.3 Y A RN


In addition to the previous interpolation techniques, we also observe that introducing a temperature _t_
on the logits before the attention softmax has a uniform impact on perplexity regardless of the data


  - The interpolation by linear ramp on _h_ may have alternatives, such as a harmonic mean over _θ_ _d_ _/s_ and
_θ_ _d_ converted from a linear interpolation on wavelengths. The choice of _h_ here was for the simplicity of
implementation, but both would work.


5


sample and the token position over the extended context window (See Appendix A.3). More precisely,
instead of Eq. 2, we modify the computation of attention weights into



�



softmax



_**q**_ _m_ _[T]_ _**[k]**_ _[n]_
� _t_ ~~�~~ _|D|_



_._ (14)



The reparametrization of RoPE as a set of 2D matrices has a clear benefit on the implementation
of this attention scaling: we can instead use a "length scaling" trick which scales both _**q**_ _m_ and _**k**_ _n_
by a constant factor �1 _/t_ by simply scaling the complex rotary position embeddings by the same

amount. With this, YaRN can effectively alter the attention mechanism without modifying its code.
Furthermore, it has zero overhead during both inference and training, as rotary position embeddings
are generated in advance and are reused for all forward passes. Combining it with the "NTK-by-parts"
interpolation, we have the YaRN method.


**Definition 2** _By the "YaRN method", we refer to a combination of the attention scaling in Eq. 14 and_
_the "NTK-by-parts" interpolation introduced in Section 3.2._


For LLaMA and Llama 2 models, we recommend the following values:



�



1

_t_ [= 0] _[.]_ [1 ln(] _[s]_ [) + 1] _[.]_ (15)



The equation above is found by fitting �1 _/t_ at the lowest perplexity against the scale extension

by various factors _s_ using the "NTK-by-parts" method (Section 3.2) on LLaMA 7b, 13b, 33b and
65b models without fine-tuning. We note that the same values of _t_ also apply fairly well to Llama
2 models (7b, 13b and 70b). It suggests that the property of increased entropy and the temperature
constant _t_ may have certain degree of "universality" and may be generalizable across some models
and training data.


The YaRN method combines all our findings and surpasses all previous methods in both fine-tuned
and non-fine-tuned scenarios. Thanks to its low footprint, YaRN allows for direct compatibility with
libraries that modify the attention mechanism such as Flash Attention 2 (Dao, 2023).


3.4 D YNAMIC S CALING - "D YNAMIC NTK" INTERPOLATION


In a lot of use cases, multiple forward-passes are performed with varying sequence lengths from 1 to
the maximal context size. A typical example is the autoregressive generation where the sequence
lengths increment by 1 after each step. There are two ways of applying an interpolation method that
uses a scale factor _s_ (including PI, "NTK-aware", "NTK-by-parts" and YaRN):


1. Throughout the whole inference cycle, the embedding layer is fixed including the scale
factor _s_ = _L_ _[′]_ _/L_ where _L_ _[′]_ is the fixed number of extended context size.
2. In each forward-pass, the position embedding updates the scale factor _s_ = max(1 _, l_ _[′]_ _/L_ )
where _l_ _[′]_ is the sequence length of the current sequence.


The problem of (1) is that the model may experience a performance discount at a length less than _L_
and an abrupt degradation when the sequence length is longer than _L_ _[′]_ . But by doing Dynamic Scaling
as (2), it allows the model to gracefully degrade instead of immediately breaking when hitting the
trained context limit _L_ _[′]_ . We call this inference-time method the Dynamic Scaling method. When it is
combined with "NTK-aware" interpolation, we call it "Dynamic NTK" interpolation. It first appeared
in public as a reddit post in (emozilla, 2023).


One notable fact is that the "Dynamic NTK" interpolation works exceptionally well on models pretrained on _L_ without any finetuning ( _L_ _[′]_ = _L_ ). This is supported by the experiment in Appendix B.7.


Often in the repeated forward-passes, the kv-caching (Chen, 2022) is applied so that we can reuse
the previous key-value vectors and improve the overall efficiency. We point out that in some
implementations when the rotary position embeddings are cached, some care has to be taken in order
to modify it for Dynamic Scaling with kv-caching. The correct implementation should cache the
kv-embeddings before applying rotary position embeddings, as the RoPE of every token changes
when _s_ changes.


6


4 E XPERIMENTS


4.1 T RAINING


We broadly followed the training and evaluation procedures as outlined in Chen et al. (2023).


For training the 128k context window size models, we extended the Llama 2 (Touvron et al., 2023b)
7B and 13B parameter models. No changes were made to the LLaMA model architecture other than
the calculation of the embedding frequencies as described in Section 3.3 with _s_ = 16 and _s_ = 32.


We used a learning rate of 2 _×_ 10 _[−]_ [5] with no weight decay and a linear warmup of 20 steps along
with AdamW (Loshchilov and Hutter, 2019) _β_ 1 = 0 _._ 9 and _β_ 2 = 0 _._ 95 . For the _s_ = 16 model, we
fine-tuned for 400 steps with global batch size 64 using PyTorch (Paszke et al., 2019) Fully Sharded
Data Parallelism (Zhao et al., 2023) and Flash Attention 2 (Dao, 2023) on the PG19 dataset (Rae et al.,
2020) chunked into 64k segments bookended with the BOS and EOS token. For _s_ = 32 we followed
the same procedure, but due to compute constraints, we started from the finished _s_ = 16 checkpoint
and trained for only an additional 200 steps. Note that the _s_ = 32 model is also trained with 64k
context data, but we show that it is able to extrapolate to a context size of 128k in Section 4.2.


For the ablation studies, we used the LLaMA 7B model. It has the same architecture as the newer
Llama 2 models except for a shorter pretrained context window size [§], which reduces compute
requirements and allows for faster training and evaluations. The training procedure is similar to the
128k models, but we chunk the PG19 dataset into 32k segments instead, and train using _s_ = 16 for
400 steps. As shown in Figure 2, YaRN converges faster compared to other interpolation techniques
during training and consistently has lower loss.


Figure 2: Training loss curves for the LLaMA 7B model extended to 32k context size using different
interpolation techniques. The graph on the right is zoomed in.


4.2 L ONG S EQUENCE L ANGUAGE M ODELING


To evaluate the long sequence language modeling performances, we use the GovReport (Huang et al.,
2021) and Proof-pile (Azerbayev et al., 2022) datasets both of which contain many long sequence
samples. For all evaluations, the test splits of both datasets were used exclusively. All perplexity
evaluations were calculated using the sliding window method from Press et al. (2022) with _S_ = 256,
which takes in account the entire documents’ perplexity contribution, even if the context window of
the model is shorter.


First, we select 10 random samples from Proof-pile with at least 128k tokens each and evaluate the
perplexity of each of these samples when truncated at 2k steps from a sequence length of 2k tokens
through 128k tokens. Table 1 shows the long sequence performance of fine-tuned Llama 2 _s_ = 16
and _s_ = 32 models. We demonstrate that YaRN is able to generalize and extrapolate to unseen
context lengths and benefit from transfer learning, since the _s_ = 32 model was only further trained
for 200 steps using the _s_ = 16 checkpoint with 64k data and is able to extrapolate to 128k context.


§ LLaMA models have a pretrained context size of 2k tokens, while Llama 2 models have 4k.


7


Model Extension Fine- Training Extension Evaluation Context Window Size
Size Method tuned Steps Scale _s_ 8192 16384 32768 65536 131072


7B YaRN ✓ 400 4k _×_ 16 3.51 2.99 2.65 2.42 _>_ 10 [1]
7B YaRN ✓ 400+200 4k _×_ 32 3.56 3.04 2.70 2.45 2.37


13B YaRN ✓ 400 4k _×_ 16 **3.25** **2.79** **2.50** **2.29** _>_ 10 [1]
13B YaRN ✓ 400+200 4k _×_ 32 3.29 2.83 2.53 2.31 **2.24**


Table 1: Sliding window perplexity ( _S_ = 256 ) of ten 128k Proof-pile documents over Llama 2 models extended
via YaRN. We show successful context size extrapolation and transfer learning from 64k to 128k given only 64k
context as training data.


In order to further confirm the effectiveness of YaRN, we compare all four interpolation methods
in Figure 3 on the left and Table 5 from Appendix B.1 as an ablation study. YaRN consistently
outperforms (has lower perplexity than) other methods in both non fine-tuned and fine-tuned scenarios
when using the same number of training steps. We also demonstrate that YaRN has better training
efficiency compared to PI in Appendix B.2. More comparisons against open models can be found in
Appendix B.3.


Figure 3: Sliding window perplexity ( _S_ = 256 ) of ten 128k Proof-pile documents and passkey
retrieval accuracy at different prompt lengths for finetuned LLaMA 7B models fine-tuned to 32k
context for 400 steps using different interpolation techniques. YaRN outperforms other interpolation
methods given the same training budget.


4.3 P ASSKEY R ETRIEVAL


The passkey retrieval task as defined in Mohtashami and Jaggi (2023) measures a model’s ability
to retrieve a simple passkey (i.e., a five-digit number) from amongst a large amount of otherwise
meaningless text. For our evaluation of the fine-tuned 32k LLaMA 7B models, we performed
50 iterations of the passkey retrieval task with the passkey placed at a random location uniformly
distributed across the evaluation context window on different prompt lengths ranging from 2k to 32k.
YaRN achieves higher scores compared to other interpolation methods when given similar training
budget, as seen in Figure 3 on the right. More results and comparisons for Llama 2 models are shown
in Appendix B.5.


4.4 S TANDARDIZED B ENCHMARKS


The Hugging Face Open LLM Leaderboard (Hugging Face, 2023) compares a multitude of
LLMs across a standardized set of four public benchmarks. Specifically, we use 25-shot ARCChallenge (Clark et al., 2018), 10-shot HellaSwag (Zellers et al., 2019), 5-shot MMLU (Hendrycks
et al., 2021), and 0-shot TruthfulQA (Lin et al., 2022).


To test the degradation of models’ short context performance under context extension, we evaluated
our Llama 2 and 32k LLaMA 7B models using this suite and compared it to established scores for


8


the baselines. The results are summarized in Table 10 and Table 3. More results for Llama 2 models
are shown in Appendix B.6.


Extension Fine- Extension
ARC-c Hellaswag MMLU TruthfulQA
Method tuned Scale _s_


None ✗   - **51.0** **77.8** **35.7** 34.3
PI ✓ 2k _×_ 16 44.8 70.2 25.9 34.1

NTK-aware ✓ 2k _×_ 16 47.4 73.9 27.7 32.6
NTK-by-parts ✓ 2k _×_ 16 48.5 76.6 32.7 33.4
YaRN ✓ 2k _×_ 16 48.1 77.2 30.0 **35.1**


Table 2: Performance of context window extensions methods, fine-tuned for 400 steps, on the Hugging Face
Open LLM benchmark suite compared with original LLaMA 7B baselines.


Model Extension Fine- Extension
ARC-c Hellaswag MMLU TruthfulQA
Size Method tuned Scale _s_


7B None ✗   - 53.1 77.8 43.8 **39.0**
7B YaRN ✓ 4k _×_ 16 52.3 78.8 42.5 38.2

7B YaRN ✓ 4k _×_ 32 52.1 78.4 41.7 37.3


13B None ✗   - **59.4** 82.1 **55.8** 37.4
13B YaRN ✓ 4k _×_ 16 58.1 **82.3** 52.8 37.8

13B YaRN ✓ 4k _×_ 32 58.0 82.2 51.9 37.3


Table 3: Performance of YaRN on the Hugging Face Open LLM benchmark suite compared with original
Llama 2 baselines.


We observe that there is minimal performance degradation between the YaRN models and their
respective Llama 2 baselines. Some variance is to be expected as the PG19 dataset (Rae et al., 2020)
we used for fine-tuning is very different from the original pre-training datased used for LLaMA and
Llama 2 models. We also observe that there was on average a 0.49% drop in scores between the
YaRN _s_ = 16 and _s_ = 32 models and can conclude that the the iterative extension from 64k to 128k
results in negligible performance loss.


4.5 C OMPUTATIONAL E FFICIENCY


Given that rotary position embeddings are cached during training and inference when the context
window size is fixed to a preset length _L_, modifying the interpolation on rotary position embeddings
incurs no additional computational or memory cost compared to previous context extension methods,
which is the case for all four interpolation methods outlined in this work. YaRN converges the fastest
during training compared to other methods, thus is the most computationally efficient, as shown in
Table 4.


Model Model Extension Extension Effective Training Time in
Size Name Method Scale _s_ Context GPU-Hours (A100)


7B LLaMA YaRN YaRN 2k _×_ 16 32k 128

7B Llama 2 YaRN YaRN 4k _×_ 16 64k 256

7B Llama 2 YaRN YaRN 4k _×_ 32 128k 256 + 128


7B (Chen et al., 2023) PI 2k _×_ 8 16k 640
7B (Together.ai, 2023) PI 4k _×_ 8 32k ?
7B (Xiong et al., 2023) NTK-aware 4k _×_ 44.2 _≈_ 50k 64000
7B (Rozière et al., 2023) NTK-aware 4k _×_ 88.6 _≈_ 100k 6400


Table 4: Comparison of training time in A100-hours for different open and closed models using different
extension methods.


9


5 C ONCLUSION


In conclusion, we have shown that YaRN improves upon all existing RoPE interpolation methods
and can act as a drop-in replacement to PI, with no downsides and minimal implementation effort.
The fine-tuned models preserve their original abilities on multiple benchmarks while being able
to attend to a very large context size. Furthermore, YaRN allows efficient extrapolation with finetuning on shorter datasets and can take advantage of transfer learning for faster convergence, both of
which are crucial under compute-constrained scenarios. Finally, we have shown the effectiveness of
extrapolation with YaRN where it is able to "train short, and test long".


6 R EPRODUCIBILITY


To aid in reproducibility, we provide, as supplementary material, the entirety of of the code used to
train the YaRN models in Table 7, as well as the evaluation code that produced Figure 7 and Tables 6,
7, 10, 8, and 9. The code also contains implementations of various extension methods referenced
throughout the paper. For training YaRN, we used the publicly available PG19 dataset (Rae et al.,
2020) tokenized to contiguous chunks of 64k tokens.


R EFERENCES


Introducing Qwen-7B: Open foundation and human-aligned models (of the state-of-the-arts). URL
[https://github.com/QwenLM/Qwen-7B/blob/main/tech_memo.md.](https://github.com/QwenLM/Qwen-7B/blob/main/tech_memo.md)


Z. Azerbayev, E. Ayers,, and B. Piotrowski. Proof-pile, 2022. URL [https://github.com/](https://github.com/zhangir-azerbayev/proof-pile)
[zhangir-azerbayev/proof-pile.](https://github.com/zhangir-azerbayev/proof-pile)


S. Black, S. Biderman, E. Hallahan, Q. Anthony, L. Gao, L. Golding, H. He, C. Leahy, K. McDonell,
J. Phang, M. Pieler, U. S. Prashanth, S. Purohit, L. Reynolds, J. Tow, B. Wang, and S. Weinbach.
GPT-NeoX-20B: An open-source autoregressive language model, 2022. arXiv: 2204.06745.


bloc97. NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation., 2023a. URL
[https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
[scaled_rope_allows_llama_models_to_have/.](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)


bloc97. Add NTK-Aware interpolation "by parts" correction, 2023b. URL [https://github.](https://github.com/jquesnelle/scaled-rope/pull/1)
[com/jquesnelle/scaled-rope/pull/1.](https://github.com/jquesnelle/scaled-rope/pull/1)


C. Chen. Transformer Inference Arithmetic, 2022. URL [https://kipp.ly/blog/](https://kipp.ly/blog/transformer-inference-arithmetic/)
[transformer-inference-arithmetic/.](https://kipp.ly/blog/transformer-inference-arithmetic/)


S. Chen, S. Wong, L. Chen, and Y. Tian. Extending context window of large language models via
positional interpolation, 2023. arXiv: 2306.15595.


A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung,
C. Sutton, S. Gehrmann, P. Schuh, K. Shi, S. Tsvyashchenko, J. Maynez, A. Rao, P. Barnes,
Y. Tay, N. Shazeer, V. Prabhakaran, E. Reif, N. Du, B. Hutchinson, R. Pope, J. Bradbury, J. Austin,
M. Isard, G. Gur-Ari, P. Yin, T. Duke, A. Levskaya, S. Ghemawat, S. Dev, H. Michalewski,
X. Garcia, V. Misra, K. Robinson, L. Fedus, D. Zhou, D. Ippolito, D. Luan, H. Lim, B. Zoph,
A. Spiridonov, R. Sepassi, D. Dohan, S. Agrawal, M. Omernick, A. M. Dai, T. S. Pillai, M. Pellat,
A. Lewkowycz, E. Moreira, R. Child, O. Polozov, K. Lee, Z. Zhou, X. Wang, B. Saeta, M. Diaz,
O. Firat, M. Catasta, J. Wei, K. Meier-Hellstern, D. Eck, J. Dean, S. Petrov, and N. Fiedel. PaLM:
Scaling language modeling with pathways, 2022. arXiv: 2204.02311.


P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have
solved question answering? try ARC, the AI2 Reasoning Challenge, 2018. arXiv: 1803.05457.


T. Computer. Redpajama: An open source recipe to reproduce llama training dataset, 2023. URL
[https://github.com/togethercomputer/RedPajama-Data.](https://github.com/togethercomputer/RedPajama-Data)


10


T. Dao. Flashattention-2: Faster attention with better parallelism and work partitioning, 2023. arXiv:
2307.08691.


emozilla. Dynamically Scaled RoPE further increases performance of long context LLaMA with
zero fine-tuning, 2023. URL [https://www.reddit.com/r/LocalLLaMA/comments/](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/)
[14mrgpr/dynamically_scaled_rope_further_increases/.](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/)


J. Gehring, M. Auli, D. Grangier, D. Yarats, and Y. N. Dauphin. Convolutional sequence to sequence
learning, 2017. arXiv: 1705.03122.


C. Han, Q. Wang, W. Xiong, Y. Chen, H. Ji, and S. Wang. LM-Infinite: Simple on-the-fly length
generalization for large language models, 2023. arXiv: 2308.16137.


D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring
massive multitask language understanding. _Proceedings of the International Conference on_
_Learning Representations (ICLR)_, 2021.


L. Huang, S. Cao, N. Parulian, H. Ji, and L. Wang. Efficient attentions for long document summarization. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for_
_Computational Linguistics: Human Language Technologies_, pages 1419–1436. Association for
Computational Linguistics, June 2021.


Hugging Face. Open LLM Leaderboard, 2023. URL [https://huggingface.co/spaces/](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
[HuggingFaceH4/open_llm_leaderboard.](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)


kaiokendev. Things I’m learning while training superhot., 2023. URL [https://kaiokendev.](https://kaiokendev.github.io/til#extending-context-to-8k)
[github.io/til#extending-context-to-8k.](https://kaiokendev.github.io/til#extending-context-to-8k)


A. Kazemnejad, I. Padhi, K. N. Ramamurthy, P. Das, and S. Reddy. The impact of positional encoding
on length generalization in transformers, 2023. arXiv: 2305.19466.


S. Lin, J. Hilton, and O. Evans. TruthfulQA: Measuring how models mimic human falsehoods. In
_Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume_
_1: Long Papers)_, pages 3214–3252, May 2022.


I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In _International Conference on_
_Learning Representations_, 2019.


A. Mohtashami and M. Jaggi. Landmark attention: Random-access infinite context length for
transformers, 2023. arXiv: 2305.16300.


A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein,
L. Antiga, A. Desmaison, A. Köpf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy,
B. Steiner, L. Fang, J. Bai, and S. Chintala. PyTorch: An imperative style, high-performance deep
learning library. In _NeurIPS_, pages 8024–8035, 2019.


O. Press, N. Smith, and M. Lewis. Train Short, Test Long: Attention with linear biases enables input
length extrapolation. In _International Conference on Learning Representations_, 2022.


J. Quesnelle, E. Shippole, and "Kaiokendev". Llongma: Scaling rotary embeddings
through linear positional interpolation. [https://huggingface.co/conceptofmind/](https://huggingface.co/conceptofmind/LLongMA-2-7b/)
[LLongMA-2-7b/, 2023.](https://huggingface.co/conceptofmind/LLongMA-2-7b/)


J. W. Rae, A. Potapenko, S. M. Jayakumar, C. Hillier, and T. P. Lillicrap. Compressive transformers
for long-range sequence modelling. In _International Conference on Learning Representations_,
2020.


A. Roberts, C. Raffel, K. Lee, M. Matena, N. Shazeer, P. J. Liu, S. Narang, W. Li, and Y. Zhou.
Exploring the limits of transfer learning with a unified text-to-text transformer. Technical report,
Google, 2019.


B. Rozière, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y. Adi, J. Liu, T. Remez, J. Rapin,
A. Kozhevnikov, I. Evtimov, J. Bitton, M. Bhatt, C. C. Ferrer, A. Grattafiori, W. Xiong, A. Défossez,
J. Copet, F. Azhar, H. Touvron, L. Martin, N. Usunier, T. Scialom, and G. Synnaeve. Code Llama:
Open foundation models for code, 2023. arXiv: 2308.12950.


11


P. Shaw, J. Uszkoreit, and A. Vaswani. Self-attention with relative position representations. In
_Proceedings of the 2018 Conference of the North American Chapter of the Association for Compu-_
_tational Linguistics: Human Language Technologies, Volume 2 (Short Papers)_, pages 464–468,
New Orleans, Louisiana, June 2018. Association for Computational Linguistics.


[J. Su. Rectified rotary position embeddings. https://github.com/bojone/rerope, 2023.](https://github.com/bojone/rerope)


J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu. RoFormer: Enhanced transformer with rotary
position embedding, 2022. arXiv: 2104.09864.


Y. Sun, L. Dong, B. Patra, S. Ma, S. Huang, A. Benhaim, V. Chaudhary, X. Song, and F. Wei. A
length-extrapolatable transformer, 2022. arXiv: 2212.10554.


M. Tancik, P. P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. T. Barron, and R. Ng. Fourier features let networks learn high frequency functions
in low dimensional domains. In _Proceedings of the 34th International Conference on Neural_
_Information Processing Systems_, NIPS’20, Red Hook, NY, USA, 2020. Curran Associates Inc.
ISBN 9781713829546.


Together.ai. LLaMA-2-7B-32K, 2023. URL [https://huggingface.co/](https://huggingface.co/togethercomputer/LLaMA-2-7B-32K)
[togethercomputer/LLaMA-2-7B-32K.](https://huggingface.co/togethercomputer/LLaMA-2-7B-32K)


H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal,
E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample. LLaMA: Open and
efficient foundation language models, 2023a. arXiv: 2302.13971.


H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra,
P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu,
J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini,
R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M.-A.
Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra,
I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M.
Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan,
I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and
T. Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023b.


A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin.
Attention is all you need. In _Advances in Neural Information Processing Systems_, volume 30.
Curran Associates, Inc., 2017.


W. Xiong, J. Liu, I. Molybog, H. Zhang, P. Bhargava, R. Hou, L. Martin, R. Rungta, K. A. Sankararaman, B. Oguz, M. Khabsa, H. Fang, Y. Mehdad, S. Narang, K. Malik, A. Fan, S. Bhosale,
S. Edunov, M. Lewis, S. Wang, and H. Ma. Effective long-context scaling of foundation models,
2023.


R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi. HellaSwag: Can a machine really finish
your sentence? In _Proceedings of the 57th Annual Meeting of the Association for Computational_
_Linguistics_, 2019.


Y. Zhao, A. Gu, R. Varma, L. Luo, C.-C. Huang, M. Xu, L. Wright, H. Shojanazeri, M. Ott,
S. Shleifer, A. Desmaison, C. Balioglu, B. Nguyen, G. Chauhan, Y. Hao, and S. Li. PyTorch FSDP:
Experiences on scaling fully sharded data parallel, 2023. arXiv: 2304.11277.


12


A A DDITIONAL DETAILS ON INTERPOLATION METHODS


A.1 P OSITION I NTERPOLATION


As mentioned in Section 2.2, PI is one of the earlier works extending context lengths of RoPE. We
include some extra details here:


While a direct extrapolation does not perform well on sequences _w_ 1 _, · · ·, w_ _L_ with _L_ larger than the
pre-trained limit, they discovered that interpolating the position indicies within the pre-trained limit
works well with the help of a small amount of fine-tuning. Specifically, given a pre-trained language
model with RoPE, they modify the RoPE by



_f_ _**W**_ _[′]_ [(] _**[x]**_ _[m]_ _[, m,]_ _**[ θ]**_ [) =] _[ f]_ _**[W]**_



� _**x**_ _m_ _,_ _[mL]_ _L_ _[′]_ _[,]_ _**[ θ]**_ � _,_ (16)



where _L_ _[′]_ _> L_ is a new context window beyond the pre-trained limit. With the original pre-trained
model plus the modified RoPE formula, they fine-tuned the language model further on several orders
of magnitude fewer tokens (a few billion in Chen et al. (2023)) and successfully acheived context
window extension.


A.2 D ETAILS OF "NTK- AWARE " INTERPOLATION


In Section 3.1, we introduce a change of basis from _b_ to _b_ _[′]_ in the definition of "NTK-aware"
interpolation method.


Precisely, following the notations set out in Section 2.1 Eq. 7, we define the "NTK-aware" interpolation scheme as follows:


**Definition 3** _The "NTK-aware" interpolation is a modification of RoPE using Eq. 7 with the following_
_functions, given s as the scale factor._
_g_ ( _m_ ) = _m_ (17)

_h_ ( _θ_ _d_ ) = _b_ _[′−]_ [2] _[d/][|][D][|]_ _,_ (18)

_where_



_|D|_

_b_ _[′]_ = _b · s_ _|D|−_ 2 (19)



_s_ = _[L]_ _[′]_ (20)

_L_ _[.]_


Given the results from (bloc97, 2023a), this method performs much better at extending the context
size of non-fine-tuned models compared to PI (Chen et al., 2023). However, one major disadvantage
of this method is that given it is not just an interpolation scheme, some dimensions are slightly
extrapolated to "out-of-bound" values, thus fine-tuning with "NTK-aware" interpolation (bloc97,
2023a) yields inferior results to PI (Chen et al., 2023). Furthermore, due to the "out-of-bound" values,
the theoretical scale factor _s_ does not accurately describe the true context extension scale. In practice,
the scale value _s_ has to be set higher than the expected scale for a given context length extension.


The mathematical derivation of the base change is the following:


Recall that our goal is to spread out the interpolation pressure across the hidden dimensions using a
base-change instead of scaling the frequencies by a fixed factor _s_ . The property we want to guarantee
is that: The lowest frequency needs to be scaled as much as linear positional scaling and the highest
frequency to stay constant.


We introduce a new base _b_ _[′]_ such that the last dimension matches the wavelength of linear interpolation
with a scale factor _s_ . Since the original RoPE method skips odd dimensions in order to concatenate
both cos( [2] _[πx]_ _λ_ [)] [ and] [ sin(] [2] _[πx]_ _λ_ [)] [ components into a single embedding, the last dimension] _[ d][ ∈]_ _[D]_ [ is]

_|D| −_ 2.


The new base _b_ _[′]_ can be chosen so that



_|D|−_ 2
_b_ _[′]_ _|D|_



_|D|−_ 2



_|D|_ = _s · b_



_b_ _|D|_ = _s · b_ _|D|_ _._ (21)

Solving for _b_ _[′]_ yields



_b_ _[′]_ = _b · s_


13



_|D|_
_|D|−_ 2 _._ (22)


A.3 T HE IMPACT OF PRE - SOFTMAX SCALING OF Y A RN ON PERPLEXITY


In Section 3.3, we mention the impact of the factor _t_ inside the softmax computation of attention
weights. Here we fix 896 16 k-token documents from RedPajama (Computer, 2023) [¶], and calculate
their perplexity scores with different scaling 1 _/√t_ . The result is in Figure 4. For comparison, recall

that our recommended factor in this case ( _s_ = 8) is given by the following.



�



1

_t_ [= 0] _[.]_ [1 ln(] _[s]_ [) + 1] _[ ≈]_ [1] _[.]_ [208] _[.]_ (23)



Figure 4: Fix _s_ = 8, compare the LLaMA 7b perplexity on 896 16 k-token documents over different scaling
1 _/_ ~~_√_~~ _t_ . The shaded area represents 1 standard deviation (68%).


To show the impact of the factor 1 _/√t_ on different token positions, we cut each 16 k-token document

into chunks of 2048 tokens, and further plot the mean perplexity change comparing to _t_ = 1 in
percentages


ppl( _t_ ) _−_ ppl( _t_ = 1)

(24)
ppl( _t_ = 1)


of each chunk. The plot is shown in Figure 5.


   - We choose RedPajama because it is the open-source dataset closest to the training dataset of LLaMA as far
as we are aware of.


14


Figure 5: Fix _s_ = 8, compare the mean of perplexity change percentages [pp][l][(] _[t]_ [)] _[ −]_ [pp][l][(] _[t]_ [ = 1][)] at different

ppl( _t_ = 1)

segments of token positions on 896 16k-token documents over different scaling 1 _/_ ~~_√_~~ _t_ .


To further demonstrate the best values of _t_ across all samples over different token positions, we plot
the sample counts with minimal perplexity at a given 1 _/√t_ for each of the 8 position segments over

the 16k-token range in Figure 6.


Figure 6: The sample counts (out of the 896 samples) with minimal perplexity at a given 1 _/_ ~~_√_~~ _t_ for a given

segment of token positions over the 16k-token range.


We observe that:


    - for a suitable _t_, a sample may obtain better perplexity scores across the extended context
window;


    - the best value of _t_ is mostly consistent across different samples and different positions.


We remark that this finding is consistent for different values of _s_ and the best value of _t_ follows our
recommended formula (Eq. 15) closely.


15


B A DDITIONAL TABLES AND CHARTS


B.1 A BLATION S TUDY


Extension Fine- Training Extension Evaluation Context Window Size
Method tuned Steps Scale _s_ 2048 4096 8192 16384 32768


None ✗  -  - **4.05**  -  -  -  

PI ✗  - 2k _×_ 2 4.36 3.90  -  -  NTK-aware ✗  - 2k _×_ 2 4.08 5.97  -  -  NTK-by-parts ✗  - 2k _×_ 2 4.12 3.71  -  -  YaRN ✗  - 2k _×_ 2 4.07 **3.67**  -  -  

PI ✗  - 2k _×_ 4 7.09 6.39 6.18  -  NTK-aware ✗  - 2k _×_ 4 4.27 3.84 _>_ 10 [1]  -  NTK-by-parts ✗  - 2k _×_ 4 4.39 4.03 4.11  -  YaRN ✗  - 2k _×_ 4 4.19 3.77 3.65  -  

PI ✗  - 2k _×_ 8 _>_ 10 [1] _>_ 10 [1] _>_ 10 [1] _>_ 10 [1]  NTK-aware ✗  - 2k _×_ 8 4.64 4.27 4.24 _>_ 10 [1]  NTK-by-parts ✗  - 2k _×_ 8 4.98 4.91 5.33 5.79  YaRN ✗  - 2k _×_ 8 4.37 3.95 3.81 3.33  

PI ✗  - 2k _×_ 16 _>_ 10 [2] _>_ 10 [2] _>_ 10 [2] _>_ 10 [2] _>_ 10 [2]

NTK-aware ✗  - 2k _×_ 16 5.23 5.02 5.22 6.85 _>_ 10 [1]

NTK-by-parts ✗  - 2k _×_ 16 6.04 7.54 _>_ 10 [1] _>_ 10 [1] _>_ 10 [1]
YaRN ✗  - 2k _×_ 16 4.61 4.24 4.18 3.66 3.45


PI ✗  - Dynamic **4.05** 3.90 6.18 _>_ 10 [1] _>_ 10 [2]

NTK-aware ✗  - Dynamic **4.05** 5.97 _>_ 10 [1] _>_ 10 [1] _>_ 10 [1]

NTK-by-parts ✗  - Dynamic **4.05** 3.71 4.11 5.79 _>_ 10 [1]
YaRN ✗  - Dynamic **4.05** **3.67** 3.65 3.33 3.45


PI ✓ 400 2k _×_ 16 5.70 4.95 4.64 3.97 3.57

NTK-aware ✓ 400 2k _×_ 16 4.39 3.92 3.73 3.21 8.49
NTK-by-parts ✓ 400 2k _×_ 16 4.14 3.75 3.62 3.12 2.81
YaRN ✓ 400 2k _×_ 16 4.19 3.77 **3.30** **3.09** **2.77**


Table 5: Sliding window perplexity ( _S_ = 256 ) of ten 128k Proof-pile documents over the LLaMA 7B model
extended via different methods.


16


B.2 T RAINING E FFICIENCY OF Y A RN


Table 6 shows a side-by-side comparison of the Llama 2 7B model extended from 4096 to 8192
context length via PI (LLongMA-2 7B [||] ) and YaRN. Note that the PI model was trained using the
methodology in Chen et al. (2023), while YaRN used the same methodology but 2.5x less training
steps and data, as de