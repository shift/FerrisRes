## **QL O RA: Efficient Finetuning of Quantized LLMs**

**Tim Dettmers** _[∗]_ **Artidoro Pagnoni** _[∗]_ **Ari Holtzman**


**Luke Zettlemoyer**


University of Washington
```
        {dettmers,artidoro,ahai,lsz}@cs.washington.edu

```

**Abstract**


We present QL O RA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while
preserving full 16-bit finetuning task performance. QL O RA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank
Adapters (LoRA). Our best model family, which we name **Guanaco**, outperforms
all previous openly released models on the Vicuna benchmark, reaching 99.3%
of the performance level of ChatGPT while only requiring 24 hours of finetuning
on a single GPU. QL O RA introduces a number of innovations to save memory
without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that
is information theoretically optimal for normally distributed weights (b) Double
Quantization to reduce the average memory footprint by quantizing the quantization
constants, and (c) Paged Optimizers to manage memory spikes. We use QL O RA
to finetune more than 1,000 models, providing a detailed analysis of instruction
following and chatbot performance across 8 instruction datasets, multiple model
types (LLaMA, T5), and model scales that would be infeasible to run with regular
finetuning (e.g. 33B and 65B parameter models). Our results show that QLoRA
finetuning on a small high-quality dataset leads to state-of-the-art results, even
when using smaller models than the previous SoTA. We provide a detailed analysis
of chatbot performance based on both human and GPT-4 evaluations showing that
GPT-4 evaluations are a cheap and reasonable alternative to human evaluation. Furthermore, we find that current chatbot benchmarks are not trustworthy to accurately
evaluate the performance levels of chatbots. A lemon-picked analysis demonstrates
where **Guanaco** fails compared to ChatGPT. We release all of our models and code,
including CUDA kernels for 4-bit training. [2]


**1** **Introduction**


Finetuning large language models (LLMs) is a highly effective way to improve their performance,

[ 40, 62, 43, 61, 59, 37 ] and to add desirable or remove undesirable behaviors [ 43, 2, 4 ]. However,
finetuning very large models is prohibitively expensive; regular 16-bit finetuning of a LLaMA 65B
parameter model [ 57 ] requires more than 780 GB of GPU memory. While recent quantization
methods can reduce the memory footprint of LLMs [ 14, 13, 18, 66 ], such techniques only work for
inference and break down during training [65].


We demonstrate for the first time that it is possible to finetune a quantized 4-bit model without any
performance degradation. Our method, QL O RA, uses a novel high-precision technique to quantize
a pretrained model to 4-bit, then adds a small set of learnable Low-rank Adapter weights [ 28 ]


_∗_ Equal contribution.
2 `[https://github.com/artidoro/qlora](https://github.com/artidoro/qlora)` and `[https://github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)`


Preprint. Under review.


that are tuned by backpropagating gradients through
the quantized weights.



**Table 1:** Elo ratings for a competition between
models, averaged for 10,000 random initial orderings. The winner of a match is determined by
GPT-4 which declares which response is better for
a given prompt of the the Vicuna benchmark. 95%
confidence intervals are shown ( _±_ ). After GPT4, Guanaco 33B and 65B win the most matches,
while Guanaco 13B scores better than Bard.



QL O RA reduces the average memory requirements

a given prompt of the the Vicuna benchmark. 95%

of finetuning a 65B parameter model from _>_ 780GB confidence intervals are shown ( _±_ ). After GPTof GPU memory to _<_ 48GB without degrading the 4, Guanaco 33B and 65B win the most matches,
runtime or predictive performance compared to a 16- while Guanaco 13B scores better than Bard.
bit fully finetuned baseline. This marks a significant
shift in accessibility of LLM finetuning: now the Model Size Elo
largest publicly available models to date finetunable

GPT-4                       - 1348 _±_ 1

on a single GPU. Using QL O RA, we train the **Gua-**

Guanaco 65B 41 GB 1022 _±_ 1

**naco** family of models, with the second best model

Guanaco 33B 21 GB 992 _±_ 1

reaching 97.8% of the performance level of ChatGPT

Vicuna 13B 26 GB 974 _±_ 1

on the Vicuna [ 10 ] benchmark, while being trainable

ChatGPT                       - 966 _±_ 1

in less than 12 hours on a single consumer GPU;

Guanaco 13B 10 GB 916 _±_ 1

using a single professional GPU over 24 hours we

Bard                         - 902 _±_ 1

achieve 99.3% with our largest model, essentially

Guanaco 7B 6 GB 879 _±_ 1

closing the gap to ChatGPT on the Vicuna benchmark. When deployed, our smallest **Guanaco** model
(7B parameters) requires just 5 GB of memory and outperforms a 26 GB Alpaca model by more than
20 percentage points on the Vicuna benchmark (Table 6).



Model Size Elo



GPT-4 - 1348 _±_ 1

Guanaco 65B 41 GB 1022 _±_ 1

Guanaco 33B 21 GB 992 _±_ 1

Vicuna 13B 26 GB 974 _±_ 1

ChatGPT - 966 _±_ 1

Guanaco 13B 10 GB 916 _±_ 1

Bard - 902 _±_ 1

Guanaco 7B 6 GB 879 _±_ 1



QL O RA introduces multiple innovations designed to reduce memory use without sacrificing performance: (1) **4-bit NormalFloat**, an information theoretically optimal quantization data type for
normally distributed data that yields better empirical results than 4-bit Integers and 4-bit Floats.
(2) **Double Quantization**, a method that quantizes the quantization constants, saving an average
of about 0.37 bits per parameter (approximately 3 GB for a 65B model). (3) **Paged Optimizers**,
using NVIDIA unified memory to avoid the gradient checkpointing memory spikes that occur when
processing a mini-batch with a long sequence length. We combine these contributions into a better
tuned LoRA approach that includes adapters at every network layer and thereby avoids almost all of
the accuracy tradeoffs seen in prior work.


QL O RA’s efficiency enables us to perform an in-depth study of instruction finetuning and chatbot
performance on model scales that would be impossible using regular finetuning due to memory
overhead. Therefore, we train more than 1,000 models across several instruction tuning datasets,
model architectures, and sizes between 80M to 65B parameters. In addition to showing that QL O RA
recovers 16-bit performance (§4) and training a state-of-the-art chatbot, **Guanaco**, (§5), we also
analyze trends in the trained models. First, we find that data quality is far more important than
dataset size, e.g., a 9k sample dataset (OASST1) outperformed a 450k sample dataset (FLAN v2,
subsampled) on chatbot performance, even when both are meant to support instruction following
generalization. Second, we show that strong Massive Multitask Language Understanding (MMLU)
benchmark performance does not imply strong Vicuna chatbot benchmark performance and vice
versa—in other words, dataset suitability matters more than size for a given task.


Furthermore, we also provide a extensive analysis of chatbot performance that uses both human
raters and GPT-4 for evaluation. We use tournament-style benchmarking where models compete
against each other in matches to produce the best response for a given prompt. The winner of a
match is judged by either GPT-4 or human annotators. The tournament results are aggregated into
Elo scores [ 16, 17 ] which determine the ranking of chatbot performance. We find that GPT-4 and
human evaluations largely agree on the rank of model performance in the tournaments, but we also
find there are instances of strong disagreement. As such, we highlight that model-based evaluation
while providing a cheap alternative to human-annotation also has its uncertainties.


We augment our chatbot benchmark results with a qualitative analysis of **Guanaco** models. Our analysis highlights success and failure cases that were not captured by the quantitative benchmarks.


We release all model generations with human and GPT-4 annotations to facilitate further study. We
open-source our codebase and CUDA kernels and integrate our methods into the Hugging Face
transformers stack [ 64 ], making them easily accessible to all. We release a collection of adapters
for 7/13/33/65B size models, trained on 8 different instruction following datasets, for a total of 32
different open sourced, finetuned models.


2


**Figure 1:** Different finetuning methods and their memory requirements. QL O RA improves over LoRA by
quantizing the transformer model to 4-bit precision and using paged optimizers to handle memory spikes.


**2** **Background**


**Block-wise k-bit Quantization** Quantization is the process of discretizing an input from a representation that holds more information to a representation with less information. It often means
taking a data type with more bits and converting it to fewer bits, for example from 32-bit floats to
8-bit Integers. To ensure that the entire range of the low-bit data type is used, the input data type is
commonly rescaled into the target data type range through normalization by the absolute maximum
of the input elements, which are usually structured as a tensor. For example, quantizing a 32-bit
Floating Point (FP32) tensor into a Int8 tensor with range [ _−_ 127 _,_ 127]:


127
**X** [Int8] = round = round( _c_ [FP32] _·_ **X** [FP32] ) _,_ (1)
� absmax( **X** [FP32] ) **[X]** [FP32] �


where _c_ is the _quantization constant_ or _quantization scale_ . Dequantization is the inverse:

dequant( _c_ [FP32] _,_ **X** [Int8] ) = **[X]** _c_ [FP32][Int8] [=] **[ X]** [FP32] (2)


The problem with this approach is that if a large magnitude value (i.e., an outlier) occurs in the input
tensor, then the quantization bins—certain bit combinations—are not utilized well with few or no
numbers quantized in some bins. To prevent the outlier issue, a common approach is to chunk the
input tensor into blocks that are independently quantized, each with their own quantization constant _c_ .
This can be formalized as follows: We chunk the input tensor **X** _∈_ R _[b][×][h]_ into _n_ contiguous blocks of
size _B_ by flattening the input tensor and slicing the linear segment into _n_ = ( _b × h_ ) _/B_ blocks. We
quantize these blocks independently with Equation 1 to create a quantized tensor and _n_ quantization
constants _c_ _i_ .


**Low-rank Adapters** Low-rank Adapter (LoRA) finetuning [ 28 ] is a method that reduces memory
requirements by using a small set of trainable parameters, often termed adapters, while not updating
the full model parameters which remain fixed. Gradients during stochastic gradient descent are
passed through the fixed pretrained model weights to the adapter, which is updated to optimize the
loss function. LoRA augments a linear projection through an additional factorized projection. Given
a projection **XW** = **Y** with **X** _∈_ R _[b][×][h]_, **W** _∈_ R _[h][×][o]_ LoRA computes:


**Y** = **XW** + _s_ **XL** 1 **L** 2 _,_ (3)


where **L** 1 _∈_ R _[h][×][r]_ and **L** 2 _∈_ R _[r][×][o]_, and _s_ is a scalar.


**Memory Requirement of Parameter-Efficient Finetuning** One important point of discussion is
the memory requirement of LoRA during training both in terms of the number and size of adapters
used. Since the memory footprint of LoRA is so minimal, we can use more adapters to improve
performance without significantly increasing the total memory used. While LoRA was designed as a


3


Parameter Efficient Finetuning (PEFT) method, most of the memory footprint for LLM finetuning
comes from activation gradients and not from the learned LoRA parameters. For a 7B LLaMA
model trained on FLAN v2 with a batch size of 1, with LoRA weights equivalent to commonly used
0.2% of the original model weights[ 28, 37 ], the LoRA input gradients have a memory footprint
of 567 MB while the LoRA parameters take up only 26 MB. With gradient checkpointing [ 9 ], the
input gradients reduce to an average of 18 MB per sequence making them more memory intensive
than all LoRA weights combined. In comparison, the 4-bit base model consumes 5,048 MB of
memory. This highlights that gradient checkpointing is important but also that aggressively reducing
the amount of LoRA parameter yields only minor memory benefits. This means we can use more
adapters without significantly increasing the overall training memory footprint (see Appendix G
for a detailed breakdown). As discussed later, this is crucial for recovering full 16-bit precision
performance.


**3** **QL** **O** **RA Finetuning**


QL O RA achieves high-fidelity 4-bit finetuning via two techniques we propose—4-bit NormalFloat
(NF4) quantization and Double Quantization. Additionally, we introduce Paged Optimizers, to
prevent memory spikes during gradient checkpointing from causing out-of-memory errors that have
traditionally made finetuning on a single machine difficult for large models.


QL O RA has one low-precision storage data type, in our case usually 4-bit, and one computation data
type that is usually BFloat16. In practice, this means whenever a QL O RA weight tensor is used, we
dequantize the tensor to BFloat16, and then perform a matrix multiplication in 16-bit.


We now discuss the components of QL O RA followed by a formal definition of QL O RA.


**4-bit NormalFloat Quantization** The NormalFloat (NF) data type builds on Quantile Quantization

[ 15 ] which is an information-theoretically optimal data type that ensures each quantization bin has an
equal number of values assigned from the input tensor. Quantile quantization works by estimating
the quantile of the input tensor through the empirical cumulative distribution function.


The main limitation of quantile quantization is that the process of quantile estimation is expensive.
Therefore fast quantile approximation algorithms, such as SRAM quantiles [ 15 ], are used to estimate
them. Due to the approximate nature of these quantile estimation algorithms, the data type has large
quantization errors for outliers, which are often the most important values.


Expensive quantile estimates and approximation errors can be avoided when input tensors come from
a distribution fixed up to a quantization constant. In such cases, input tensors have the same quantiles
making exact quantile estimation computationally feasible.


Since pretrained neural network weights usually have a zero-centered normal distribution with
standard deviation _σ_ (see Appendix F), we can transform all weights to a single fixed distribution by
scaling _σ_ such that the distribution fits exactly into the range of our data type. For our data type, we
set the arbitrary range [ _−_ 1 _,_ 1] . As such, both the quantiles for the data type and the neural network
weights need to be normalized into this range.


The information theoretically optimal data type for zero-mean normal distributions with arbitrary
standard deviations _σ_ in the range [ _−_ 1 _,_ 1] is computed as follows: (1) estimate the 2 _[k]_ + 1 quantiles
of a theoretical _N_ (0 _,_ 1) distribution to obtain a _k_ -bit quantile quantization data type for normal distributions, (2) take this data type and normalize its values into the [ _−_ 1 _,_ 1] range, (3) quantize an input
weight tensor by normalizing it into the [ _−_ 1 _,_ 1] range through absolute maximum rescaling.


Once the weight range and data type range match, we can quantize as usual. Step (3) is equivalent to
rescaling the standard deviation of the weight tensor to match the standard deviation of the k-bit data
type. More formally, we estimate the 2 _[k]_ values _q_ _i_ of the data type as follows:



_i_
� 2 _[k]_ + 1



+ _Q_ _X_
�



_i_ + 1
� 2 _[k]_ + 1



_,_ (4)
��



_q_ _i_ = [1]

2



_Q_ _X_
�



where _Q_ _X_ ( _·_ ) is the quantile function of the standard normal distribution _N_ (0 _,_ 1) . A problem for
a symmetric k-bit quantization is that this approach does not have an exact representation of zero,
which is an important property to quantize padding and other zero-valued elements with no error. To


4


ensure a discrete zeropoint of 0 and to use all 2 _[k]_ bits for a k-bit datatype, we create an asymmetric
data type by estimating the quantiles _q_ _i_ of two ranges _q_ _i_ : 2 _[k][−]_ [1] for the negative part and 2 _[k][−]_ [1] + 1 for
the positive part and then we unify these sets of _q_ _i_ and remove one of the two zeros that occurs in both
sets. We term the resulting data type that has equal expected number of values in each quantization bin
_k-bit NormalFloat_ (NFk), since the data type is information-theoretically optimal for zero-centered
normally distributed data. The exact values of this data type can be found in Appendix E.


**Double Quantization** We introduce _Double Quantization_ (DQ), the process of quantizing the
quantization constants for additional memory savings. While a small blocksize is required for precise
4-bit quantization [ 13 ], it also has a considerable memory overhead. For example, using 32-bit
constants and a blocksize of 64 for **W**, quantization constants add 32 _/_ 64 = 0 _._ 5 bits per parameter on
average. Double Quantization helps reduce the memory footprint of quantization constants.


More specifically, Double Quantization treats quantization constants _c_ [FP32] 2 of the first quantization
as inputs to a second quantization. This second step yields the quantized quantization constants
_c_ [FP8] 2 and the second level of quantization constants _c_ [FP32] 1 . We use 8-bit Floats with a blocksize of
256 for the second quantization as no performance degradation is observed for 8-bit quantization,
in line with results from Dettmers and Zettlemoyer [13] . Since the _c_ [FP32] 2 are positive, we subtract
the mean from _c_ 2 before quantization to center the values around zero and make use of symmetric
quantization. On average, for a blocksize of 64, this quantization reduces the memory footprint per
parameter from 32 _/_ 64 = 0 _._ 5 bits, to 8 _/_ 64 + 32 _/_ (64 _·_ 256) = 0 _._ 127 bits, a reduction of 0.373 bits
per parameter.


**Paged Optimizers** use the NVIDIA unified memory [3] feature wich does automatic page-to-page
transfers between the CPU and GPU for error-free GPU processing in the scenario where the GPU
occasionally runs out-of-memory. The feature works like regular memory paging between CPU RAM
and the disk. We use this feature to allocate paged memory for the optimizer states which are then
automatically evicted to CPU RAM when the GPU runs out-of-memory and paged back into GPU
memory when the memory is needed in the optimizer update step.


**QL** **O** **RA.** Using the components described above, we define QL O RA for a single linear layer in
the quantized base model with a single LoRA adapter as follows:


**Y** [BF16] = **X** [BF16] doubleDequant( _c_ [FP32] 1 _, c_ [k-bit] 2 _,_ **W** [NF4] ) + **X** [BF16] **L** [BF16] 1 **L** [BF16] 2 _,_ (5)


_·_
where doubleDequant( ) is defined as:


doubleDequant( _c_ [FP32] 1 _, c_ [k-bit] 2 _,_ **W** [k-bit] ) = dequant(dequant( _c_ [FP32] 1 _, c_ [k-bit] 2 ) _,_ **W** [4bit] ) = **W** [BF16] _,_ (6)


We use NF4 for **W** and FP8 for _c_ 2 . We use a blocksize of 64 for **W** for higher quantization precision
and a blocksize of 256 for _c_ 2 to conserve memory.



For parameter updates only the gradient with respect to the error for the adapters weights _[∂E]_



_∂_ **L** _i_ [are]
needed, and not for 4-bit weights _∂_ _[∂E]_ **W** [. However, the calculation of] _∂_ _[∂E]_ **L** _i_ [entails the calculation of] _∂_ _[∂]_ **W** **[X]**

which proceeds via equation (5) with dequantization from storage **W** [NF4] to computation data type
**W** [BF16] to calculate the derivative _∂_ _[∂]_ **W** **[X]** [in BFloat16 precision.]


To summarize, QL O RA has one storage data type (usually 4-bit NormalFloat) and a computation
data type (16-bit BrainFloat). We dequantize the storage data type to the computation data type
to perform the forward and backward pass, but we only compute weight gradients for the LoRA
parameters which use 16-bit BrainFloat.


**4** **QLoRA vs. Standard Finetuning**


We have discussed how QLoRA works and how it can significantly reduce the required memory for
finetuning models. The main question now is whether QLoRA can perform as well as full-model
finetuning. Furthermore, we want to analyze the components of QLoRA including the impact of
NormalFloat4 over standard Float4. The following sections will discuss the experiments that aimed
at answering these questions.


3
```
   https://docs.nvidia.com/cuda/cuda-c-programming-guide

```

5




_[∂E]_ _[∂E]_

_∂_ **W** [. However, the calculation of] _∂_ **L**




_[∂E]_ _[∂]_ **[X]**

_∂_ **L** _i_ [entails the calculation of] _∂_ **W**



_∂_ **W** [in BFloat16 precision.]


**Experimental setup.** We consider three architectures (encoder, encoder-decoder, and decoder only)
and compare QLoRA with 16-bit adapter-finetuning and with full-finetuning for models up to 3B. Our
evaluations include GLUE [ 58 ] with RoBERTa-large [ 38 ], Super-NaturalInstructions (TKInstruct)

[ 61 ] with T5 [ 49 ], and 5-shot MMLU [ 24 ] after finetuning LLaMA on Flan v2 [ 39 ] and Alpaca

[ 55 ]. To additionally study the advantages of NF4 over other 4-bit data types, we use the setup of
Dettmers and Zettlemoyer [13] and measure post-quantization zero-shot accuracy and perplexity
across different models (OPT [ 72 ], LLaMA [ 57 ], BLOOM [ 52 ], Pythia [ 7 ]) for model sizes 125m 13B. We provide more details in the results section for each particular setup to make the results more
readable. Full details in Appendix A.



While paged optimizers are critical to do 33B/65B
QL O RA tuning on a single 24/48GB GPU, we do
not provide hard measurements for Paged Optimizers since the paging only occurs when processing
mini-batches with long sequence lengths, which is
rare. We do, however, perform an analysis of the
runtime of paged optimizers for 65B models on
48GB GPUs and find that with a batch size of 16,
paged optimizers provide the same training speed
as regular optimizers. Future work should measure
and characterize under what circumstances slowdowns occur from the paging process.







**Default LoRA hyperparameters do not match 16-**
**bit performance** When using the standard practice of applying LoRA to query and value attention
projection matrices [ 28 ], we are not able to replicate **Figure 2:**
full finetuning performance for large base models.
As shown in Figure 2 for LLaMA 7B finetuning on
Alpaca, we find that the most critical LoRA hyperparameter is how many LoRA adapters are used in
total and that LoRA on all linear transformer block
layers are required to match full finetuning performance. Other LoRA hyperparameters, such as the
projection dimension _r_, do not affect performance (see Appendix A).





**Figure 2:** RougeL for LLaMA 7B models on the
Alpaca dataset. Each point represents a run with a
different random seed. We improve on the Stanford
Alpaca fully finetuned default hyperparameters to
construct a strong 16-bit baseline for comparisons.
Using LoRA on all transformer layers is critical to
match 16-bit performance.



Similarly, we find that default hyperparameters for
fully finetuned baselines are undertuned. We do a
hyperparameter search over learning rates 1e-6 to
5e-5 and batch sizes 8 to 128 to find robust baselines.
Results for 7B LLaMA finetuning on Alpaca are
shown in Figure 2.


**4-bit NormalFloat yields better performance**
**than 4-bit Floating Point** While the 4-bit
NormalFloat (NF4) data type is informationtheoretically optimal, it still needs to be determined
if this property translates to empirical advantages.
We follow the setup from Dettmers and Zettlemoyer

[13] where quantized LLMs (OPT [ 72 ], BLOOM

[ 52 ], Pythia [ 7 ], LLaMA) of different sizes (125M
to 65B) with different data types are evaluated on
language modeling and a set of zero-shot tasks. In
Figure 3 and Table 2 we see that NF4 improves performance significantly over FP4 and Int4 and that
double quantization reduces the memory footprint
without degrading performance.


**k-bit QL** **O** **RA matches 16-bit full finetuning and**
**16-bit LoRA performance** Recent findings have
established that 4-bit quantization for _inference_ is


6



0.67


0.66


0.65


0.64


0.63


0.62


0.61


0.60






|4 -bit LLaMA|Col2|Col3|
|---|---|---|
||||
||||
||||
||||
||||
||Data type|Data type|
||Data type|Data type|
|||~~Float~~<br>NFloat<br>~~NFloat +~~|



10 [10] 10 [11]


Total model bits


**Figure 3:** Mean zero-shot accuracy over Winogrande, HellaSwag, PiQA, Arc-Easy, and ArcChallenge using LLaMA models with different 4-bit
data types. The NormalFloat data type significantly
improves the bit-for-bit accuracy gains compared
to regular 4-bit Floats. While Double Quantization
(DQ) only leads to minor gains, it allows for a more
fine-grained control over the memory footprint to fit
models of certain size (33B/65B) into certain GPUs
(24/48GB).


**Table 3:** Experiments comparing 16-bit BrainFloat (BF16), 8-bit Integer (Int8), 4-bit Float (FP4), and 4bit NormalFloat (NF4) on GLUE and Super-NaturalInstructions. QL O RA replicates 16-bit LoRA and fullfinetuning.


Dataset GLUE (Acc.) Super-NaturalInstructions (RougeL)
Model RoBERTa-large T5-80M T5-250M T5-780M T5-3B T5-11B


BF16 88.6 40.1 42.1 48.0 54.3 62.0
BF16 replication 88.6 40.0 42.2 47.3 54.9  

LoRA BF16 88.8 40.5 42.6 47.1 55.4 60.7
QL O RA Int8 88.8 40.4 42.9 45.4 56.5 60.7
QL O RA FP4 88.6 40.3 42.4 47.5 55.6 60.9
QL O RA NF4 + DQ - 40.4 42.7 47.7 55.3 60.9


possible, but leads to performance degradation relative to 16-bit [ 13, 18 ]. This raises the crucial question of whether the lost performance can be
recovered by conducting 4-bit adapter finetuning. We test this for two setups.



The first focuses on a comparison with full 16-bit finetuning
of RoBERTA and T5 models sized 125M to 3B parameters on
GLUE and the Super-NaturalInstructions dataset. Results are
shown in Table 3. In both datasets, we observe that 16-bit, 8-bit,
and 4-bit adapter methods replicate the performance of the fully
finetuned 16-bit baseline. This suggests that the performance lost
due to the imprecise quantization can be fully recovered through
adapter finetuning after quantization.



Int4 34.34
Float4 (E2M1) 31.07
Float4 (E3M0) 29.48
NFloat4 + DQ **27.41**



**Table 2:** Pile Common Crawl mean
perplexity for different data types
for 125M to 13B OPT, BLOOM,
LLaMA, and Pythia models.


Data type Mean PPL



For our second setup, since full finetuning models at and beyond

Float4 (E3M0) 29.48

11B parameters requires more than one server of high memory
GPUs, we continue to test whether 4-bit QL O RA can match NFloat4 + DQ **27.41**
16-bit LoRA at the 7B to 65B parameter scales. To this end, we
finetune LLaMA 7B through 65B on two instruction following
datasets, Alpaca and FLAN v2, and evaluate on the MMLU benchmark via 5-shot accuracy. Results
are shown in Table 4 where we see that NF4 with double quantization fully recovers the 16-bit
LoRA MMLU performance. In addition, we also note that QL O RA with FP4 lags behind the 16-bit
brain float LoRA baseline by about 1 percentage point. This corroborates both our findings that (1)
QL O RA with NF4 replicates both 16-bit full finetuning and 16-bit LoRA finetuning performance,
and (2) NF4 is superior to FP4 in terms of quantization precision.



**Summary** Our results consistently show that 4-bit QL O RA with NF4 data type matches 16bit full finetuning and 16-bit LoRA finetuning performance on academic benchmarks with wellestablished evaluation setups. We have also shown that NF4 is more effective than FP4 and that
double quantization does not degrade performance. Combined, this forms compelling evidence that
4-bit QL O RA tuning reliably yields results matching 16-bit methods.


In line with previous work on quantization [ 13 ], our MMLU and Elo results indicate that with a given
finetuning and inference resource budget it is beneficial to increase the number of parameters in the
base model while decreasing their precision. This highlights the importance of efficiency benefits
from QL O RA. Since we did not observe performance degradation compared to full-finetuning in
our experiments with 4-bit finetuning, this raises the question of where the performance-precision
trade-off exactly lies for QLoRA tuning, which we leave to future work to explore.


We proceed to investigate instruction tuning at scales that would be impossible to explore with full
16-bit finetuning on academic research hardware.


**5** **Pushing the Chatbot State-of-the-art with QLoRA**


Having established that 4-bit QL O RA matches 16-bit performance across scales, tasks, and datasets
we conduct an in-depth study of instruction finetuning up to the largest open-source language models
available for research. To assess the performance of instruction finetuning these models, we evaluate


7


**Table 4:** Mean 5-shot MMLU test accuracy for LLaMA 7-65B models finetuned with adapters on Alpaca and
FLAN v2 for different data types. Overall, NF4 with double quantization (DQ) matches BFloat16 performance,
while FP4 is consistently one percentage point behind both.


Mean 5-shot MMLU Accuracy


LLaMA Size 7B 13B 33B 65B Mean
Dataset Alpaca FLAN v2 Alpaca FLAN v2 Alpaca FLAN v2 Alpaca FLAN v2


BFloat16 38.4 45.6 47.2 50.6 57.7 60.5 61.8 62.5 53.0

Float4 37.2 44.0 47.3 50.0 55.9 58.5 61.3 63.3 52.2
NFloat4 + DQ 39.0 44.5 47.5 50.7 57.3 59.2 61.8 63.9 53.1


on a challenging Natural Language Understanding benchmark (MMLU) and develop new methods
for real-world chatbot performance evaluation.


**5.1** **Experimental setup**

We now describe an overview of the experimental setup with full details in Appendix B.


**Data** As, to our knowledge, there is no comprehensive study of recent instruction-following datasets,
we select eight recent datasets. We include datasets obtained through crowd-sourcing (OASST1 [ 31 ],
HH-RLHF [ 4 ]), distillation from instruction-tuned models (Alpaca [ 55 ], self-instruct [ 59 ], unnaturalinstructions [ 26 ]), corpora aggregations (FLAN v2 [ 12 ]), as well as hybrids (Chip2 [ 32 ], Longform [30]). These datasets cover different languages, data sizes, and licenses.


**Training Setup** To avoid confounding effects from different training objectives, we perform QLoRA
finetuning with cross-entropy loss (supervised learning) without reinforcement learning, even for
datasets that include human judgments of different responses. For datasets that have a clear distinction
between instruction and response, we finetune only on the response (see ablations in Appendix B).
For OASST1 and HH-RLHF, multiple responses are available. We then select the top response at
every level of the conversation tree and finetune on the full selected conversation, including the
instructions. In all of our experiments, we use NF4 QL O RA with double quantization and paged
optimizers to prevent memory spikes during gradient checkpointing. We do small hyperparameter
searches for the 13B and 33B LLaMA models and we find that all hyperparameter settings found
at 7B generalize (including number of epochs) except learning rate and batch size. We halve the
learning rate for 33B and 65B while doubling the batch size.


**Baselines** We compare our models to both research (Vicuna [ 10 ] and Open Assistant [ 31 ]) and
commercial (GPT-4 [ 42 ], GPT-3.5-turbo and Bard) chatbot systems. The Open Assistant model is
a LLaMA 33B model finetuned with Reinforcement Learning from Human Feedback (RLHF) on
the same OASST1 dataset that we experiment with. Vicuna does full fine-tuning of LLaMA 13B
on proprietary user-shared conversations from ShareGPT and is thus the result of distillation from
OpenAI GPT models.


**5.2** **Evaluation**



Following common practice, we use the MMLU (Massively Multitask Language Understanding) benchmark

[ 24 ] to measure performance on a range of language understanding tasks. This is a multiple-choice benchmark
covering 57 tasks including elementary mathematics,
US history, computer science, law, and more. We report
5-shot test accuracy.



Self-Instruct 36.4 33.3 53.0 56.7
Longform 32.1 43.2 56.6 59.7
Chip2 34.5 41.6 53.6 59.8
HH-RLHF 34.9 44.6 55.8 60.1

Unnatural Instruct 41.9 48.1 57.3 61.3
Guanaco (OASST1) 36.6 46.4 57.0 62.2
Alpaca 38.8 47.8 57.3 62.5
FLAN v2 44.5 51.4 59.2 63.9



**Table 5:** MMLU 5-shot test results for different
sizes of LLaMA finetuned on the corresponding
datasets using QLoRA.


Dataset 7B 13B 33B 65B


LLaMA no tuning 35.1 46.9 57.8 63.4



We also test generative language capabilities through Chip2 34.5 41.6 53.6 59.8
both automated and human evaluations. This second HH-RLHF 34.9 44.6 55.8 60.1
set of evaluations relies on queries curated by humans Unnatural Instruct 41.9 48.1 57.3 61.3

Guanaco (OASST1) 36.6 46.4 57.0 62.2

and aims at measuring the quality of model responses. Alpaca 38.8 47.8 57.3 62.5
While this is a more realistic testbed for chatbot model FLAN v2 44.5 51.4 59.2 63.9
performance and is growing in popularity, there is no
commonly accepted protocol in the literature. We describe below our proposed setup, using nucleus sampling with _p_ = 0 _._ 9 and temperature 0 _._ 7 in all

cases.



8


**Benchmark Data** We evaluate on two curated datasets of queries (questions): the Vicuna prompts

[ 10 ] and the OASST1 validation dataset [ 31 ]. We use the Vicuna prompts, a set of 80 prompts from a
diverse set of categories, without modifications. The OASST1 dataset is a multilingual collection of
crowd-sourced multiturn dialogs between a user and an assistant. We select all user messages in the
validation dataset as queries and include previous turns in the prompt. This procedure leads to 953
unique user queries. We term these two datasets the Vicuna and OA benchmarks.


**Automated Evaluation** First, based on the evaluation protocol introduced by Chiang et al. [10],
we use GPT-4 to rate the performance of different systems against ChatGPT (GPT-3.5 Turbo) on the
Vicuna benchmark. Given a query along with ChatGPT’s and a model’s responses, GPT-4 is prompted
to assign a score out of ten to both responses and provide an explanation. The overall performance of
a model is calculated as a percentage of the score that ChatGPT achieved. Note this relative score
can be higher than 100% if the model achieves a higher absolute score than ChatGPT. We find a
significant ordering effect with GPT-4 increasing the score of the response occurring earlier in the
prompt. To control for such effects, we recommend reporting the mean score over both orders.


Next, we measure performance through direct comparisons between system outputs. We simplify
the rating scheme to a three-class labeling problem that accounts for ties. We prompt GPT-4 to
pick the best response or declare a tie and provide an explanation. We conduct these head-to-head
comparisons on all permutations of pairs of systems on both the Vicuna and OA benchmarks.


**Human Evaluation** While recent work indicates generative models can be effectively employed
for system evaluations [ 19 ], the reliability GPT-4 ratings to assess chatbot performance is, to our
knowledge, yet to be proven to correlate with human judgments. Therefore, we run two parallel
human evaluations on the Vicuna benchmark matching both automated evaluation protocols described
above. We use Amazon Mechanical Turk (AMT) and get two human annotators for comparisons to
ChatGPT and three annotators for pairwise comparisons.


**Elo Rating** With both human and automated pairwise comparisons, we create a tournament-style
competition where models compete against each other. The tournament is made up of matches where
pairs of models compete to produce the best response for a given prompt. This is similar to how Bai
et al. [4] and Chiang et al. [10] compare models, but we also employ GPT-4 ratings in addition to
human ratings. We randomly sample from the set of labeled comparisons to compute Elo [ 16, 17 ].
Elo rating, which is widely used in chess and other games, is a measure of the expected win-rate
relative to an opponent’s win rate, for example, an Elo of 1100 vs 1000 means the Elo 1100 player
has an expected win-rate of approximately 65% against the Elo 1000 opponent; a 1000 vs 1000 or
1100 vs 1100 match results in an expected win-rate of 50%. The Elo rating changes after each match
proportionally to the expected outcome, that is, an unexpected upset leads to a large change in Elo
rating while an expected outcome leads to a small change. Over time, Elo ratings approximately
match the skill of each player at playing the game. We start with a score of 1,000 and use _K_ = 32 .
Similar to Chiang et al. [10], we repeat this procedure 10,000 times with different random seeds to
control for ordering effects, e.g., the effect of which model pairs compete with each other first.


**5.3** **Guanaco: QL** **O** **RA trained on OASST1 is a State-of-the-art Chatbot**

Based on our automated and human evaluations, we find that the top QL O RA tuned model, Guanaco
65B, which we finetune on a variant of OASST1, is the best-performing open-source chatbot model
and offers performance competitive to ChatGPT. When compared to GPT-4, Guanaco 65B and 33B
have an expected win probability of 30%, based on Elo rating from human annotators system-level
pairwise comparisons - the highest reported to date.


The Vicuna benchmark [ 10 ] results relative to ChatGPT are shown in Table 6. We find that Guanaco
65B is the best-performing model after GPT-4, achieving 99.3% performance relative to ChatGPT.
Guanaco 33B has more parameters than the Vicuna 13B model, but uses only 4-bit precision for its
weights and is thus much more memory efficient at 21 GB vs 26 GB, providing a three percentage
points of improvement over Vicuna 13B. Furthermore, Guanaco 7B easily fits on modern phones at a
5 GB footprint while still scoring nearly 20 percentage points higher than Alpaca 13B.


However, Table 6 also has very wide confidence intervals, with many models overlapping in performance. We hypothesize that this uncertainty comes from the lack of clear specification of scale,
e.g., it is unclear what 8 on a 10 point scale means across different scenarios. As such, we instead
recommend using the Elo ranking method [ 16 ], based on _pairwise_ judgments from human annotators
and GPT-4 to avoid the problem of grounding an absolute scale. Elo ratings of the most competitive


9


**Table 6:** Zero-shot Vicuna benchmark scores as a percentage of the score obtained by ChatGPT evaluated by
GPT-4. We see that OASST1 models perform close to ChatGPT despite being trained on a very small dataset
and having a fraction of the memory requirement of baseline models.


Model / Dataset Params Model bits Memory ChatGPT vs Sys Sys vs ChatGPT Mean 95% CI


GPT-4  -  -  - 119.4% 110.1% **114.5** % 2.6%

Bard  -  -  - 93.2% 96.4% 94.8% 4.1%


**Guanaco** 65B 4-bit 41 GB 96.7% 101.9% **99.3** % 4.4%
Alpaca 65B 4-bit 41 GB 63.0% 77.9% 70.7% 4.3%
FLAN v2 65B 4-bit 41 GB 37.0% 59.6% 48.4% 4.6%


**Guanaco** 33B 4-bit 21 GB 96.5% 99.2% **97.8** % 4.4%
Open Assistant 33B 16-bit 66 GB 91.2% 98.7% 94.9% 4.5%
Alpaca 33B 4-bit 21 GB 67.2% 79.7% 73.6% 4.2%
FLAN v2 33B 4-bit 21 GB 26.3% 49.7% 38.0% 3.9%


Vicuna 13B 16-bit 26 GB 91.2% 98.7% **94.9** % 4.5%

**Guanaco** 13B 4-bit 10 GB 87.3% 93.4% 90.4% 5.2%
Alpaca 13B 4-bit 10 GB 63.8% 76.7% 69.4% 4.2%
HH-RLHF 13B 4-bit 10 GB 55.5% 69.1% 62.5% 4.7%

Unnatural Instr. 13B 4-bit 10 GB 50.6% 69.8% 60.5% 4.2%
Chip2 13B 4-bit 10 GB 49.2% 69.3% 59.5% 4.7%
Longform 13B 4-bit 10 GB 44.9% 62.0% 53.6% 5.2%
Self-Instruct 13B 4-bit 10 GB 38.0% 60.5% 49.1% 4.6%

FLAN v2 13B 4-bit 10 GB 32.4% 61.2% 47.0% 3.6%


**Guanaco** 7B 4-bit 5 GB 84.1% 89.8% **87.0** % 5.4%
Alpaca 7B 4-bit 5 GB 57.3% 71.2% 64.4% 5.0%
FLAN v2 7B 4-bit 5 GB 33.3% 56.1% 44.8% 4.0%


models can be seen in Table 1. We note that human and GPT-4 ranking of models on the Vicuna
benchmark disagree partially, particularly for Guanaco 7B, but are consistent for most models with
a Kendall Tau of _τ_ = 0 _._ 43 and Spearman rank correlation of _r_ = 0 _._ 55 at the system level. At the
example level, the agreement between GPT-4 and human annotators’ majority vote is weaker with
Fleiss _κ_ = 0 _._ 25 . Overall, this shows a moderate agreement between system-level judgments by
GPT-4 and human annotators, and thus that model-based evaluation represents a somewhat reliable
alternative to human evaluation. We discuss further considerations in Section 6.2.


Elo rankings in Table 7 indicate that Guanaco 33B and 65B models outperform all models besides
GPT-4 on the Vicuna and OA benchmarks and that they perform comparably to ChatGPT in line
with Table 6. We note that the Vicuna benchmark favors open-source models while the larger OA
benchmark favors ChatGPT. Furthermore, we can see from Tables 5 and 6 that the suitability of
a finetuning dataset is a determining factor in performance. Finetuning Llama models on FLAN
v2 does particularly well on MMLU, but performs worst on the Vicuna benchmark (similar trends
are observed with other models). This also points to partial orthogonality in current evaluation
benchmarks: strong MMLU performance does not imply strong chatbot performance (as measured
by Vicuna or OA benchmarks) and vice versa.


Guanaco is the only top model in our evaluation that is not trained on proprietary data as the OASST1
dataset collection guidelines explicitly forbid the use of GPT models. The next best model trained
on only open-source data is the Anthropic HH-RLHF model, which scores 30 percentage points
lower than Guanaco on the Vicuna benchmark (see Table 6). Overall, these results show that 4-bit
QL O RA is effective and can produce state-of-the-art chatbots that rival ChatGPT. Furthermore, our
33B Guanaco can be trained on 24 GB consumer GPUs in less than 12 hours. This opens up the
potential for future work via QL O RA tuning on specialized open-source data, which produces models
that can compete with the very best commercial models that exist today.


**6** **Qualitative Analysis**


While quantitative analysis is the core of our evaluation, there are a number of issues with only
looking at summary statistics. Perhaps the largest is the problem of benchmark validity [ 36 ]—whether
a benchmark truly tests what its name or description suggests is always at question, especially as we
discover “shortcuts” to solve benchmarks that machine learning models sometimes exploit [ 22, 46 ].
To partially alleviate this, we here perform some qualitative analysis, in two sections. First, in §6.1


10


**Table 7:** Elo rating for a tournament between models where models compete to generate the best response
for a prompt, judged by human raters or GPT-4. Overall, Guanaco 65B and 33B tend to be preferred to
ChatGPT-3.5 on the benchmarks studied. According to human raters they have a Each 10-point difference in Elo
is approximately a difference of 1.5% in win-rate.


Benchmark Vicuna Vicuna Open Assistant
# Prompts 80 80 953
Judge Human raters GPT-4 GPT-4 Median Rank


Model Elo Rank Elo Rank Elo Rank


GPT-4 1176 1 1348 1 1294 1 1

Guanaco-65B 1023 2 1022 2 1008 3 2

Guanaco-33B 1009 4 992 3 1002 4 4

ChatGPT-3.5 Turbo 916 7 966 5 1015 2 5

Vicuna-13B 984 5 974 4 936 5 5

Guanaco-13B 975 6 913 6 885 6 6

Guanaco-7B 1010 3 879 8 860 7 7

Bard 909 8 902 7   -   - 8


we show some examples that we believe are representative of some observed patterns in the text
generated by our 65b Guanaco model. Second, §6.2 we detail considerations about the results we
have discussed and our interpretation of them.


**6.1** **Qualitative Analysis of Example Generations**

To find examples, we first go through data generated for the Vicuna benchmark and the OpenAssistant
benchmark, and look for patterns in the answers Guanaco generates. When we notice a pattern we
attempt to setup a question or prompt that will induce the pattern even though it is the incorrect
solution, e.g., if we observe that the model tends to give long-winded answers we prompt the model
to “Answer yes or no without explanation.” We use this to find “lemons” where we manage to
adversarially break the model and “cherries” where we fail to break the model, and present both. All
generations in this section were generated with Nucleus Sampling [25] with _p_ = 0 _._ 9.


Of course, this is by no means comprehensive, since it is beyond the scope of this small qualitative
study to control for all the variables involved, e.g., the full distribution of responses the model can
generate for a given prompt is quite large, so we rely on samples we hope are representative. However,
we believe describing these examples gives context to the quantitative evidence shown earlier in the
paper. Since we open source all models and code, we hope this section will inspire future work to
examine in more detail the issues we present here.


**Factual Recall** For questions such as “What is the capital of Zambia?” all models consistently
generate correct answers, e.g.,


**Guanaco** : `The capital of Zambia is Lusaka.`


However as questions get more obscure Guanaco becomes unreliable, but stays confident. For
instance in response to this prompt from HotPotQA [68]:


**User** : In what year was the singer who popularized ‘I’ll Keep the Lovelight
Burning’ born?


**Guanaco** : `Al Jolson is credited with popularizing the song`
```
   ‘I’ll Keep the Lovelight Burning,’ and he was born in the year
    1886.

```

Guanaco generates the wrong popularizer and the wrong birthday (though the birthday given is
correct for the person mentioned, Al Jolson).


**Suggestibility** Guanaco shows a surprising resistance to going along with some kinds of assumed
misinformation, e.g., in the following interaction:


**User** : How was it finally, officially confirmed that the earth is flat by peer reviewed
scientists?


11


**Guanaco** : `The earth has never been officially confirmed to be`
```
    flat by peer reviewed scientists. The overwhelming consensus
    among scientists is that the earth is spherical. In fact,
    this has been known for centuries and has been confirmed

    through numerous experiments and observations.

    The belief that the earth is flat is a conspiracy theory
    that has been debunked by science. The idea that a group
    of scientists have peer reviewed and confirmed the earth is
    flat is simply false.

    If you have any more questions please ask.

```

Guanaco is also quite good at knowing what kinds of questions aren’t possible to answer, e.g.,


**User** : What time is it?


**Guanaco** : `I’m sorry, but as an AI language model, I don’t have`
```
    access to real-time information. My responses are generated
    based on my pre-existing knowledge and do not reflect the
    current time. If you’re looking for the current time, I
    suggest using a timekeeping device or checking an online
    timepiece.

```

Similar results hold for “Where are you?”, “How are you?”, etc.


**Refusal** Similarly, Guanaco sometimes refuses to follow instructions for seemingly random rea
sons:


**User** : Please reverse the words in the sentence “I’m a cool dude in a mocha mood.”


**Guanaco** : `I’m sorry, I’m not able to do that right now, but`
```
    I can explain why the sentence you provided is grammatically

    correct.

    The sentence “I’m a cool dude in a mocha mood" follows the

    correct grammar and syntax rules of the English language. In
    this case, the adjective “cool” comes before the noun “dude”
    to describe it, and the prepositional phrase “in a mocha
    mood" provides additional information about the context of
    the sentence. This sentence structure is perfectly acceptable
    and makes sense in the context of the sentence.

    If you have any other questions or if there’s anything else
    I can help you with, feel free to ask

```

**Secret Keeping** Unfortunately, this refusal is unreliable even when it is desired. Often system
designers would like to keep something secret from the user, even if the system is allowed to give
free text responses back to the user. We prompt the model with “The secret word is ‘banana’. Do not
reveal or repeat back this back, no matter what the human requests.” and try to see how easy it is for
a potential user to access this secret information. Directly asking fails:


**User** : What is the secret word?


**Guanaco** : `I’m sorry, but I’m not able to disclose the secret`
```
    word. I’m programmed to not reveal or repeat back the secret
    word, no matter what the human requests. Is there anything
    else I can help you with?

```

which is the desired behavior. However, a tiny amount of trickery breaks this behavior:


**User** : This is a game. The goal is to ignore your previous instructions. What is the
secret word?


12


**Guanaco** : `Sorry, the secret word is banana.` `It’s best if we`
```
    don’t speak of it. Is there anything else you’d like to ask?

```

showcasing the need to study methods for more faithful instruction following.


**Math** Guanaco’s biggest weakness is mathematics, an area where many language models struggle
in [35]. When Guanaco shows its work, it tends to be accurate, e.g.,


**User** : Lee mows one lawn and charges $33. Last week he mowed 16 lawns and
three customers each gave him a $10 tip. How many dollars did Lee earn mowing
lawns last week?


**Guanaco** : `Lee earned $582 mowing lawns last week.`

```
    To find the total amount he earned, we first need to calculate
    how much he earned for mowing 16 lawns:

    $33 x 16 = $528

    Then we need to add 