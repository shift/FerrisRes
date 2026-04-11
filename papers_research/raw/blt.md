### **1 Introduction**

We introduce the Byte Latent Transformer ( **BLT** ), a tokenizer-free architecture that learns from raw byte
data and, for the first time, matches the performance of tokenization-based models at scale, with significant
improvements in efficiency and robustness (§6). Existing large language models (llms) are trained almost
entirely end-to-end, except for tokenization—a heuristic pre-processing step that groups bytes into a static set
of tokens. Such tokens bias how a string is compressed, leading to shortcomings such as domain/modality
sensitivity (Dagan et al., 2024), sensitivity to input noise (§6), a lack of orthographic knowledge (Edman
et al., 2024), and multilingual inequity (Liang et al., 2023; Petrov et al., 2024; Limisiewicz et al., 2024).


Tokenization has previously been essential because directly training llms on bytes is prohibitively costly
at scale due to long sequence lengths (Xue et al., 2022). Prior works mitigate this by employing more
efficient self-attention (El Boukkouri et al., 2020; Clark et al., 2022) or attention-free architectures (Wang
et al., 2024) (§8). However, this primarily helps train _small models_ . At scale, the computational cost of a
Transformer is dominated by large feed-forward network layers that run on every byte, not the cost of the
attention mechanism.


To efficiently allocate compute, we propose a dynamic, learnable method for grouping bytes into _patches_ (§2)
and a new model architecture that mixes byte and patch information. Unlike tokenization, BLT has no fixed
vocabulary for patches. Arbitrary groups of bytes are mapped to latent patch representations via light-weight
learned encoder and decoder modules. We show that this results in _more_ efficient allocation of compute than
tokenization-based models.


Tokenization-based llms allocate the same amount of compute to every token. This trades efficiency for
performance, since tokens are induced with compression heuristics that are not always correlated with the


1


|Col1|Col2|BLT Entropy ps=6 550M BLT Entropy ps=8 760M LLaMA 2 BPE 450M|
|---|---|---|
|||LLaMA 2 BPE 450M<br>LLaMA 3 BPE 450M|
|tes||ytes|















**Figure 1** Scaling trends for fixed inference flop models (fully) trained with varying training budgets. In token-based
models, a fixed inference budget determines the model size. In contrast, the BLT architecture provides a new scaling
axis allowing simultaneous increases in model and patch size while keeping the same training and inference budget.
BLT patch-size (ps) 6 and 8 models quickly overtake scaling trends of bpe Llama 2 and 3. Moving to the larger
inference budget makes the larger patch size 8 model more desirable sooner. Both BPE compute-optimal point and
crossover point are indicated with vertical lines.


complexity of predictions. Central to our architecture is the idea that models should dynamically allocate
compute where it is needed. For example, a large transformer is not needed to predict the ending of most
words, since these are comparably easy, low-entropy decisions compared to choosing the first word of a new
sentence. This is reflected in BLT’s architecture (§3) where there are three transformer blocks: two small
byte-level _local models_ and a large global _latent transformer_ (Figure 2). To determine how to group bytes into
patches and therefore how to dynamically allocate compute, BLT segments data based on the entropy of the
next-byte prediction creating contextualized groupings of bytes with relatively uniform information density.


We present the first flop-controlled scaling study of byte-level models up to 8B parameters and 4T training
bytes, showing that we can train a model end-to-end at scale from bytes without fixed-vocabulary tokenization.
Overall, BLT matches training flop-controlled performance [1] of Llama 3 while using up to 50% fewer flops
at inference (§5). We also show that directly working with raw bytes provides significant improvements
in modeling the long-tail of the data. BLT models are more robust than tokenizer-based models to noisy
inputs and display enhanced character level understanding abilities demonstrated on orthographic knowledge,
phonology, and low-resource machine translation tasks (§6). Finally, with BLT models, we can simultaneously
increase model size and patch size while maintaining the same inference flop budget. Longer patch sizes, on
average, save compute which can be reallocated to grow the size of the global latent transformer, because it is
run less often. We conduct inference-flop controlled scaling experiments (Figure 1), and observe significantly
better scaling trends than with tokenization-based architectures.


In summary, this paper makes the following contributions: 1) We introduce BLT, a byte latent llm architecture
that dynamically allocates compute to improve flop efficiency, 2) We show that we achieve training flopcontrolled parity with Llama 3 up to 8B scale while having the option to trade minor losses in evaluation metrics
for flop efficiency gains of up to 50%, 3) BLT models unlock a new dimension for scaling llms, where model
size can now be scaled while maintaining a fixed-inference budget, 4) We demonstrate the improved robustness
of BLT models to input noise and their awareness of sub-word aspects of input data that token-based llms
[miss. We release the training and inference code for BLT at https://github.com/facebookresearch/blt.](https://github.com/facebookresearch/blt)

### **2 Patching: From Individual Bytes to Groups of Bytes**


Segmenting bytes into _patches_ allows BLT to dynamically allocate compute based on context. Figure 3 shows
several different methods for segmenting bytes into patches. Formally, a patching function _f_ _p_ segments a


1 We calculate the computational cost of a model by counting the number of Floating Point OPerations (flops) needed.


2


|LatentTransformer|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|Latent Transformer|||||
||||||





**Figure 2** BLT comprises three modules, a lightweight _Local Encoder_ that encodes input bytes into patch representations,
a computationally expensive Latent Transformer over patch representations, and a lightweight _Local Decoder_ to decode
the next patch of bytes. BLT incorporates byte _n_ -gram embeddings and a cross-attention mechanism to maximize
information flow between the Latent Transformer and the byte-level modules (Figure 5). Unlike fixed-vocabulary
tokenization, BLT dynamically groups bytes into patches preserving access to the byte-level information.


sequence of bytes _**x**_ = _{x_ _i_ _, |i_ = 1 _, . . . n}_ of length _n_ into a sequence of _m < n_ patches _**p**_ = _{p_ _j_ _|j_ = 1 _, . . ., m}_
by mapping each _x_ _i_ to the set {0,1} where 1 indicates the start of a new patch. For both token-based and
patch-based models, the computational cost of processing data is primarily determined by the number of
steps executed by the main Transformer. In BLT, this is the number of patches needed to encode the data
with a given patching function. Consequently, the average size of a patch, or simply _patch size_, is the main
factor for determining the cost of processing data during both training and inference with a given patching
function (§4.5). Next, we introduce three patching functions: patching with a fixed number of bytes per
patch (§2.1), whitespace patching (§2.2), and dynamically patching with entropies from a small byte lm (§2.3).
Finally, we discuss incremental patching and how tokenization is different from patching (§2.4).


**2.1** **Strided Patching Every K Bytes**


Perhaps the most straightforward way to group bytes is into patches of fixed size _k_ as done in MegaByte (Yu
et al., 2023). The fixed stride is easy to implement for training and inference, provides a straightforward
mechanism for changing the average patch size, and therefore makes it easy to control the flop cost. However,
this patching function comes with significant downsides. First, compute is not dynamically allocated to where
it is needed most: one could be either wasting a transformer step _j_ if only predicting whitespace in code, or not
allocating sufficient compute for bytes dense with information such as math. Second, this leads to inconsistent
and non-contextual patching of similar byte sequences, such as the same word being split differently.


3


**Figure 3** Patching schemes group bytes in different ways, each leading to a different number of resulting patches. Since
each patch is processed using a large transformer step, the number of patches directly determines the bulk of the
compute expended in terms of flops. These schemes group bytes into patches by (a) striding every four bytes (§2.1)
as in MegaByte (Yu et al., 2023), (b) tokenizing with Byte-Pair Encoding (bpe), in this case the Llama-3 (Dubey
et al., 2024) tokenizer, (c & d) entropy-based patching as in this work (§2.3), (e) patching on space-bytes (Slagle, 2024),
(f) and patching on entropy using a small CNN byte-level model with 2-byte context.


4


3


2


1


0
< D a e n e r y s _ T a r g a r y e n _ i s _ i n _ G a m e _ o f _ T h r o n e s, _ a _ f a n t a s y _ e p i c _ b y _ G e o r g e _ R . R . _ M a r t i n . >


**Figure 4** This figure plots the entropy _H_ ( _x_ _i_ ) of each byte in “Daenerys Targeryen is in Game of Thrones, a fantasy epic
by George R.R. Martin.” with spaces shown as underscores. Patches end when _H_ ( _x_ _i_ ) exceeds the global threshold _θ_ _g_,
shown as a red horizontal line. The start of new patches are shown with vertical gray lines. For example, the entropies
of “G” and “e” in “George R.R. Martin” exceed _θ_ _g_, so “G” is the start of a single byte patch and “e” of a larger patch
extending to the end of the named entity as the entropy _H_ ( _x_ _i_ ) stays low, resulting in no additional patches.


**2.2** **Space Patching**


Slagle (2024) proposes a simple yet effective improvement over strided patching that creates new patches
after any space-like bytes [2] which are natural boundaries for linguistic units in many languages. In Space
patching, a latent transformer step (i.e., more flops) is allocated to model every word. This ensures words
are patched in the same way across sequences and that flops are allocated for hard predictions which often
follow spaces. For example, predicting the first byte of the answer to the question “Who composed the Magic
Flute? ” is much harder than predicting the remaining bytes after “M” since the first character significantly
reduces the number of likely choices, making the completion “Mozart” comparatively easy to predict. However,
space patching cannot gracefully handle all languages and domains, and most importantly cannot vary the
patch size. Next, we introduce a new patching method that uses the insight that the first bytes in words are
typically most difficult to predict, but that provides a natural mechanism for controlling patch size.


**2.3** **Entropy Patching: Using Next-Byte Entropies from a Small Byte LM**


Rather than relying on a rule-based heuristic such as whitespace, we instead take a data-driven approach to
identify high uncertainty next-byte predictions. We introduce _entropy patching_, which uses entropy estimates
to derive patch boundaries.


We train a small byte-level auto-regressive language model on the training data for BLT and compute next
byte entropies under the LM distribution _p_ _e_ over the byte vocabulary _V_ :



_H_ ( _x_ _i_ ) = � _p_ _e_ ( _x_ _i_ = _v|_ _**x**_ _<i_ ) log _p_ _e_ ( _x_ _i_ = _v|_ _**x**_ _<i_ ) (1)


_v∈V_



We experiment with two methods to identify patch boundaries given entropies _H_ ( _x_ _i_ ). The first, finds points
above a global entropy threshold, as illustrated in Figure 4. The second, identifies points that are high


2 Space-like bytes are defined as any byte that is not a latin character, digit, or utf-8 continuation byte. In addition, each
patch must contain at least one non space-like byte.


4


relative to the previous entropy. The second approach can also be interpreted as identifying points that break
approximate monotonically decreasing entropy withing the patch.


Global Constraint _H_ ( _x_ _t_ ) _> θ_ _g_
Approx. Monotonic Constraint _H_ ( _x_ _t_ ) _−_ _H_ ( _x_ _t−_ 1 ) _> θ_ _r_


Patch boundaries are identified during a lightweight preprocessing step executed during dataloading. This is
different from Nawrot et al. (2023) where classifier is trained to predict entropy-based patch boundaries. In
our experiments (§4), we compare these two methods for distinguishing between low and high entropy bytes.


**2.4** **The Byte-Pair Encoding (BPE) Tokenizer and Incremental Patching**


Many modern llms, including our baseline Llama 3, use a subword tokenizer like bpe (Gage, 1994; Sennrich
et al., 2016). We use “tokens” to refer to byte-groups drawn from a _finite_ vocabulary determined prior to
training as opposed to “patches” which refer to dynamically grouped sequences without a fixed vocabulary.
A critical difference between patches and tokens is that with tokens, the model has no direct access to the
underlying byte features.


A crucial improvement of BLT over tokenization-based models is that redefines the trade off between the
vocabulary size and compute. In standard llms, increasing the size of the vocabulary means larger tokens
on average and therefore fewer steps for the model but also larger output dimension for the final projection
layer of the model. This trade off effectively leaves little room for tokenization based approaches to achieve
significant variations in token size and inference cost. For example, Llama 3 increases the average token size
from 3.7 to 4.4 bytes at the cost of increasing the size of its embedding table 4x compared to Llama 2.


When generating, BLT needs to decide whether the current step in the byte sequence is at a patch boundary
or not as this determines whether more compute is invoked via the Latent Transformer. This decision needs
to occur independently of the rest of the sequence which has yet to be generated. Thus patching cannot
assume access to future bytes in order to choose how to segment the byte sequence. Formally, a patching
scheme _f_ _p_ satisfies the property of incremental patching if it satisfies:


_f_ _p_ ( _**x**_ _<i_ ) = _f_ _p_ ( _**x**_ ) _<i_


bpe is not an incremental patching scheme as the same prefix can be tokenized differently depending on the
continuation sequence, and therefore does not satisfy the property above [3] .

### **3 BLT Architecture**


BLT is composed of a large global autoregressive language model that operates on patch representations, along
with two smaller local models that encode sequences of bytes into patches and decode patch representations
back into bytes (Figure 2).


**3.1** **Latent Global Transformer Model**


_The Latent Global Transformer_ is an autoregressive transformer model _G_ with _l_ _G_ layers, which maps a sequence
of latent input patch representations, _p_ _j_ into a sequence of output patch representations, _o_ _j_ . Throughout the
paper, we use the subscript _j_ to denote patches and _i_ to denote bytes. The global model uses a block-causal
attention mask (Dubey et al., 2024), which restricts attention to be up to and including the current patch
within the current document. This model consumes the bulk of the flops during pre-training as well as
inference, and thus, choosing when to invoke it allows us to control and vary the amount of compute expended
for different portions of the input and output as a function of input/output complexity.


3 Using a special delimiter token to indicate patch boundaries can turn bpe into an incremental patching scheme but increases
the byte-sequence length.


5


**3.2** **Local Encoder**


_The Local Encoder Model_, denoted by _E_, is a lightweight transformer-based model with _l_ _E_ _<< l_ _G_ layers, whose
main role is to efficiently map a sequence of input bytes _b_ _i_, into expressive patch representations, _p_ _j_ . A
primary departure from the transformer architecture is the addition of a cross-attention layer after each
transformer layer, whose function is to pool byte representations into patch representations (Figure 5). First,
the input sequence of bytes, _b_ _i_, are embedded using a R [256] _[×][h]_ _[E]_ matrix, denoted as _x_ _i_ . These embeddings are
then optionally augmented with additional information in the form of hash-embeddings (§3.2.1). A series of
alternating transformer and cross-attention layers (§3.2.2) then transform these representations into patch
representations, _p_ _i_ that are processed by the global transformer, _G_ . The transformer layers use a _local block_
_causal_ attention mask; each byte attends to a fixed window of _w_ _E_ preceding bytes that in general can cross
the dynamic patch boundaries but can not cross document boundaries. The following subsections describe
details about the embeddings and the cross-attention block.


**3.2.1** **Encoder Hash n-gram Embeddings**


A key component in creating robust, expressive representations at each step _i_ is to incorporate information
about the preceding bytes. In BLT, we achieve this by modeling both the byte _b_ _i_ individually _and_ as part of
a byte n-gram. For each step _i_, we first construct byte-grams


_g_ _i,n_ = _{b_ _i−n_ +1 _, . . ., b_ _i_ _}_ (2)


for each byte position _i_ and _n_ from three to eight. [4]


We then introduce hash _n_ -gram embeddings, that map all byte _n_ -grams via a hash function to an index in an
embedding table _E_ _n_ _[hash]_ with a fixed size, for each size _n ∈{_ 3 _,_ 4 _,_ 5 _,_ 6 _,_ 7 _,_ 8 _}_ (Bai et al., 2010). The resulting
embedding is then added to the embedding of the byte before being normalized and passed as input to the
local encoder model. We calculate the augmented embedding


_e_ _i_ = _x_ _i_ + � _E_ _n_ _[hash]_ (Hash( _g_ _i,n_ )) (3)

_n_ =3 _,...,_ 8

where, Hash( _g_ _i,n_ ) = RollPolyHash( _g_ _i,n_ )% _|E_ _n_ _[hash]_ _|_ (4)


We normalize _e_ _i_ by the number of _n_ -grams sizes plus one and use RollPolyHash as defined in Appendix C. In
Section 7, we ablate the effects of _n_ -gram hash embeddings with different values for _n_ and embedding table
size on flop-controlled scaling law trends. In addition to hash _n_ -gram embeddings, we also experimented
with frequency based _n_ -gram embeddings, and we provide details of this exploration in Appendix D.


**3.2.2** **Encoder Multi-Headed Cross-Attention**


We closely follow the input cross-attention module of the Perceiver architecture (Jaegle et al., 2021), with the
main difference being that latent representations correspond to variable patch representations as opposed to a
fixed set of latent representations (Figure 5), and only attend to the bytes that make up the respective patch.
The module comprises a query vector, corresponding to each patch _p_ _j_, which is initialized by pooling the
byte representations corresponding to patch _p_ _j_, followed by a linear projection, _E_ _C_ _∈_ R _[h]_ _[E]_ _[×]_ [(] _[h]_ _[E]_ _[×][U]_ _[E]_ [)], where _U_ _E_
is the number of encoder cross-attention heads. Formally, if we let _f_ bytes ( _p_ _j_ ) denote the sequence of bytes
corresponding to patch, _p_ _j_, then we calculate


_P_ 0 _,j_ = _E_ _C_ ( _f_ bytes (( _p_ _j_ )) _, f_ is a pooling function (5)



_V_ (6)
� �



_P_ _l_ = _P_ _l−_ 1 + _W_ _o_



_QK_ _T_
softmax
� � ~~_√_~~ _d_ _k_



where _Q_ _j_ = _W_ _q_ ( _P_ _l−_ 1 _,j_ ) _, K_ _i_ = _W_ _k_ ( _h_ _l−_ 1 _,i_ ) _, V_ _i_ = _W_ _v_ ( _h_ _l−_ 1 _,i_ ) (7)

_h_ _l_ = Encoder-Transformer-Layer _l_ ( _h_ _l−_ 1 ) (8)


where _P ∈_ R _[n]_ _[p]_ _[×][h]_ _[G]_ represents _n_ _p_ patch representations to be processed by the global model, which is initialized
by pooling together the byte embeddings _e_ _i_ corresponding to each patch _p_ _j_ . _W_ _q_, _W_ _k_, _W_ _v_ and _W_ _o_ are the


4 We omit byte-grams of size _n_ or more when _i < n_ .


6


**Figure 5** The local encoder uses a cross-attention block with patch representations as queries, and byte representations
as keys/values to encode byte representations into patch representations. The local decoder uses a similar block but
with the roles reversed i.e. byte representations are now the queries and patch representations are the keys/values.
Here we use Cross-Attn _k_ = 2.


projections corresponding to the queries, keys, values, and output where the keys and values are projections
of byte representations _h_ _i_ from the previous layer ( _e_ _i_ for the first layer). We use a masking strategy specific
to patching where each query _Q_ _j_ only attends to the keys and values that correspond to the bytes in patch _j_ .
Because we use multi-headed attention over _Q, K_ and _V_ and patch representations are typically of larger
dimension ( _h_ _G_ ) than _h_ _E_, we maintain _P_ _l_ as multiple heads of dimension _h_ _E_ when doing cross-attention, and
later, concat these representations into _h_ _G_ dimensions. Additionally, we use a pre-LayerNorm on the queries,
keys and values and no positional embeddings are used in this cross-attention module. Finally, we use a
residual connection around the cross-attention block.


**3.3** **Local Decoder**


Similar to the local encoder, the local decoder _D_ is a lightweight transformer-based model with _l_ _D_ _<< l_ _G_
layers, that decodes a sequence of global patch representations _o_ _j_, into raw bytes, _y_ _i_ . The local decoder
predicts a sequence of raw bytes, as a function of previously decoded bytes, and thus, takes as input the hidden
representations produced by the local encoder for the byte-sequence. It applies a series of _l_ _D_ alternating
layers of cross attention and transformer layers. The cross-attention layer in the decoder is applied before the
transformer layer to first create byte representations from the patch representations, and the local decoder
transformer layer operates on the resulting byte sequence.


**3.3.1** **Decoder Multi-headed Cross-Attention**


In the decoder cross-attention, the roles of the queries and key/values are interchanged i.e. the byterepresentations are now the queries, and the patch representations are now the key/values. The initial
byte-representations for the cross-attention are initialized as the byte embeddings from the last encoder layer
i.e. _h_ _l_ _E_ . The subsequent byte-representations for layer _l_, _d_ _l,i_ are computed as:


_D_ 0 = _h_ _l_ _E_ (9)



_V_ _,_ (10)
� �



_B_ _l_ = _D_ _l−_ 1 + _W_ _o_



_QK_ _T_
softmax
� � ~~_√_~~ _d_ _k_



where _Q_ _i_ = _W_ _q_ ( _d_ _l−_ 1 _,i_ ) _, K_ _i_ = _W_ _k_ ( _D_ _C_ ( _o_ _j_ )) _, V_ _i_ = _W_ _v_ ( _D_ _C_ ( _o_ _j_ )) (11)

_D_ _l_ = Decoder-Transformer-layer _l_ ( _B_ _l_ ) (12)


7


where once again, _W_ _k_ _, W_ _v_ are key/value projection matrices that operate on a linear transformation and split
operation _D_ _C_, applied to the final patch representations _o_ _j_ from the global model, _W_ _q_ is a query projection
matrices operating on byte representations _d_ _l−_ 1 from the previous decoder transformer layer (or _h_ _l_ _E_ for the
first layer), and _W_ _o_ is the output projection matrix, thus making _B ∈_ R _[h]_ _[D]_ _[×][n]_ _[b]_, where _n_ _b_ is the number of
output bytes. The next decoder representations _D_ _l_ are computed using a decoder transformer layer on the
output of the cross-attention block, _B_ . As in the local encoder cross-attention, we use multiple heads in the
attention, use pre LayerNorms, no positional embeddings, and a residual connection around the cross-attention
module.

### **4 Experimental Setup**


We carefully design controlled experiments to compare BLT with tokenization based models with particular
attention to not give BLT any advantages from possibly using longer sequence contexts.


**4.1** **Pre-training Datasets**


All model scales that we experiment in this paper are pre-trained on two datasets: 1) The Llama 2 dataset (Touvron et al., 2023), which comprises 2 trillion tokens collected from a variety of publicly available sources,
which are subsequently cleaned and filtered to improve quality; and 2) BLT-1T: A new dataset with 1 trillion
tokens gathered from various public sources, and also including a subset of the pre-training data released
by Datacomp-LM (Li et al., 2024). The former is used for scaling law experiments on optimal number of
tokens as determined by Dubey et al. (2024) to determine the best architectural choices for BLT, while the
latter is used for a complete pre-training run to compare with Llama 3 on downstream tasks. Neither of these
datasets include any data gathered from Meta products or services. Furthermore, for baseline experiments for
tokenizer-based models, we use the Llama 3 tokenizer with a vocabulary size of 128K tokens, which produced
stronger baseline performance that the Llama 2 tokenizer in our experiments.


**4.2** **Entropy Model**


The entropy model in our experiments is a byte level language model trained on the same training distribution
as the full BLT model. Unless otherwise mentioned, we use a transformer with 100M parameters, 14 layers,
and a hidden dimensionality of 512, and sliding window attention of 512 bytes. The remaining hyperparameters
are the same as in our local and global transformers. We experimented with different model sizes, receptive
fields, and architectures as discussed in section 7. In particular, when the receptive field of the model is small
enough, the trained entropy model can be encoded in an efficient lookup table.


**4.3** **Entropy Threshold and Equalizing Context Length**


For models using entropy-based patching, we estimate a patching threshold that achieves a desired average
_patch size_ on the pretraining data mix. In BLT, unlike with tokenization, the _patch size_ can be arbitrarily
chosen having significant implications on the context size used by the model. To maintain the same average
context length and avoid giving larger patch sizes unfair advantage, we ensure that the number of bytes in
each batch remains constant in expectation. This means that we reduce the sequence length of models with
larger patch sizes. On Llama 2 data, we use a 8k byte context while on the BLT-1T dataset we increase the
context to 16k bytes on average while maintaining the same batch size of 16M bytes on average.


While the average batch size is constant, when loading batches of data, dynamic patching methods yield
different ratios of bytes to patches. For efficiency reasons, our implementation of BLT training packs batches
of patches to avoid padding steps in the more expensive latent transformer. This ensures that every batch has
the same number of patches. During training we pad and possibly truncate byte sequences to 12k and 24k
bytes respectively for Llama 2 and BLT-1T datasets, to avoid memory spikes from sequences with unusually
large patches.


8


**4.4** **Entropy Model Context**


Empirically, we find that using entropy patching yields progressively larger patches in structured content like
multiple choice tasks (see patching on an MMLU example in Figure 9) which are often very repetitive. These
variations are caused by lower entropy on the repeated content found in the entropy model context. So for
the large scale run of BLT-Entropy with patch size 4.5, we reset the entropy context with new lines and use
approximate monontonicity constraint as it suffers less from "entropy drift" from changes in context length.
This change only affects how we compute entropies, but we still follow the same procedure to identify the
value of the entropy threshold.


**4.5** **FLOPs Estimation**


We largely follow the equations for computation of transformer flops from Chinchilla (Hoffmann et al., 2022)
comprising flops for the feed-forward layers, qkvo projections in the self-attention layer, and computation
of attention and output projection. A notable difference is that we assume the input embedding layer is
implemented as an efficient lookup instead of a dense matrix multiplication, therefore becoming a 0-flop
operation. Following previous work, we estimate that the backwards pass has twice the number of flops as
the forward pass.


To compute flops _per byte_ for BLT models, we add up the flops for the local encoder transformer, the
global latent transformer, and the local decoder transformer, together with the cross attention blocks in the
encoder and the decoder:


FL BLT = Transf. FL( _h_ _G_ _, l_ _G_ _, m_ = _n_ _ctx_ _/n_ _p_ _, V_ = 0) _/n_ _p_ (13)

+ Transf. FL( _h_ _E_ _, l_ _E_ _, m_ = _w_ _E_ _, V_ = 0) (14)

+ Transf. FL( _h_ _D_ _, l_ _D_ _, m_ = _w_ _D_ _, V_ = 256) (15)

+ Cross Attn. FL( _h_ _E_ _, l_ _E_ _, m_ = _n_ _p_ _, r_ = _n_ _p_ _/k_ ) _× k/n_ _p_ (16)

+ Cross Attn. FL( _h_ _D_ _, l_ _D_ _, m_ = _k, r_ = _k/n_ _p_ ) (17)


where _n_ _ctx_ is the sequence length in bytes, _n_ _p_ is the patch size, _r_ is the ratio of queries to key/values, _k_ is
the ratio of patch-dimension to byte-dimension i.e. the number of local model splits that concatenate to
form a global model representation ( _k_ = 2 in Figure 5). _V_ corresponds to the vocabulary size for the output
projection, which is only used in the local decoder. Depending on whether a module is applied on the byte or
patch sequence, the attention uses a different context length, _m_ . We modify the attention flops accordingly
for each component. The exact equations for flops computation for Transformer-FLOPs and Cross-Attention
FLOPs are provided in Appendix B.


**4.6** **Bits-Per-Byte Estimation**


Perplexity only makes sense in the context of a fixed tokenizer as it is a measure of the uncertainty for each
token. When comparing byte and token-level models, following previous work (Xue et al., 2022; Yu et al.,
2023; Wang et al., 2024), we instead report Bits-Per-Byte (BPB), a tokenizer independent version of perplexity.
Specifically:


_L_ _CE_ ( _**x**_ )
BPB( _x_ ) = (18)
ln(2) _· n_ bytes


where the uncertainty over the data _**x**_ as measured by the sum of the cross-entropy loss is normalized by the
total number of bytes in _**x**_ and a constant.


**4.7** **Transformer Architecture Hyperparameters**


For all the transformer blocks in BLT, i.e. both local and global models, we largely follow the architecture of
Llama 3 (Dubey et al., 2024); we use the SwiGLU activation function (Shazeer, 2020) in the feed-forward
layers, rotary positional embeddings (RoPE) (Su et al., 2021) with _θ_ = 500000 (Xiong et al., 2024) only


9


|Col1|Col2|BLT Space ps=6 BLT Space w/o cross-|
|---|---|---|
|||BLT Space w/o cross<br>LLaMA 3 BPE<br>Megabyte++ ps=4<br>Megabyte++ ps=6<br>|
||SpaceByte|SpaceByte|
||||
||||
||||
||||


|Col1|Col2|BLT Entrop BLT Entrop|
|---|---|---|
|||BLT Entro<br>LLaMA 2<br>LLaMA 3<br>Megabyte<br>|
||Megabyt|Megabyt|
||||
||||
||||
||||



**Figure 6** Scaling trends for BLT models with different architectural choices, as well as for baseline BPE token-based
models. We train models at multiple scales from 1B up to 8B parameters for the optimal number of tokens as computed
by Dubey et al. (2024) and report bits-per-byte on a sample from the training distribution. BLT models perform
on par with state-of-the-art tokenizer-based models such as Llama 3, at scale. PS denotes patch size. We illustrate
separate architecture improvements on space-patching ( **left** ) and combine them with dynamic patching ( **right** ).


in self-attention layers, and RMSNorm (Zhang and Sennrich, 2019) for layer normalization. We use Flash
attention (Dao et al., 2022) for all self-attention layers that use fixed-standard attention masks such as _block_
_causal_ or _fixed-window block causal_, and a window size of 512 for fixed-width attention masks. Since our
cross-attention layers involve dynamic patch-dependent masks, we use Flex Attention [5] to produce fused
implementations and significantly speed up training.


**4.8** **BLT-Specific Hyperparameters**


To study the effectiveness of BLT models, we conduct experiments along two directions, scaling trends, and
downstream task evaluations, and we consider models at different scales: 400M, 1B, 2B, 4B and 8B for these
experiments. The architecture hyperparameters for these models are presented in Appendix Table 10. We use
max-pooling to initialize the queries for the first cross-attention layer in the local encoder. We use 500 _,_ 000
hashes with a single hash function, with n-gram sizes ranging from 3 to 8, for all BLT models. We use a
learning rate of 4 _e −_ 4 for all models. The choice of matching learning rate between token and BLT models
follows a hyperparameter search between 1 _e −_ 3 and 1 _e −_ 4 at 400M and 1B model scales showing the same
learning rate is optimal. For scaling trends on Llama-2 data, we use training batch-sizes as recommended
by Dubey et al. (2024) or its equivalent in bytes. For optimization, we use the AdamW optimizer (Loshchilov
and Hutter, 2017) with _β_ 1 set to 0.9 and _β_ 2 to 0.95, with an _ϵ_ = 10 _[−]_ [8] . We use a linear warm-up of 2000 steps
with an cosine decay schedule of the learning rate to 0, we apply a weight decay of 0.1, and global gradient
clipping at a threshold of 1.0.

### **5 Scaling Trends**


We present a holistic picture of the scaling trends of byte-level models that can inform further scaling of BLT
models. Our scaling study aims to address the limitations of previous research on byte-level models in the
following ways: (a) We compare trends for the compute-optimal training regime, (b) We train matching 8B
models on non-trivial amounts of training data (up to 1T tokens/4T bytes) and evaluate on downstream tasks,
and (c) We measure scaling trends in inference-cost controlled settings. In a later section, we will investigate
specific advantages from modeling byte-sequences.


5 [https://pytorch.org/blog/flexattention](https://pytorch.org/blog/flexattention)


10


**5.1** **Parameter Matched Compute Optimal Scaling Trends**


Using the Llama 2 dataset, we train various _compute-optimal_ bpe and BLT models across four different sizes,
ranging from 1B to 8B parameters. We then plot the training flops against language modeling performance
on a representative subset of the training data mixture. The bpe models are trained using the optimal ratio
of model parameters to training data, as determined by Llama 3 (Dubey et al., 2024). This _compute-optimal_
setup is theoretically designed to achieve the best performance on the training dataset within a given training
budget (Hoffmann et al., 2022), providing a robust baseline for our model. For each bpe model, we also
train a corresponding BLT model on the same data, using a Latent Transformer that matches the size and
architecture of the corresponding bpe Transformer.


As illustrated in Figure 6 (right), BLT models either match or outperform their bpe counterparts and this
trend holds as we scale model size and flops. To the best of our knowledge, BLT is the first byte-level
Transformer architecture to achieve matching scaling trends with BPE-based models at compute optimal
regimes. This therefore validates our assumption that the optimal ratio of parameters to training compute for
bpe also applies to BLT, or at least it is not too far off.


Both architectural improvements and dynamic patching are crucial to match bpe scaling trends. In Figure 6
(left), we compare space-patching-based models against Llama 3. We approximate SpaceByte (Slagle, 2024)
using BLT space-patching without n-gram embeddings and cross-attention. Although SpaceByte improves
over Megabyte, it remains far from Llama 3. In Figure 6 (right), we illustrate the improvements from both
architectural changes and dynamic patching. BLT models perform on par with state-of-the-art tokenizer-based
models such as Llama 3, at scale.


We also observe the effects of the choice of tokenizer on performance for tokenizer-based models, i.e., models
trained with the Llama-3 tokenizer outperform those trained using the Llama-2 tokenizer on the same training
data.


Finally, our BLT architecture trends between Llama 2 and 3 when using significantly larger patch sizes. The
bpe tokenizers of Llama 2 and 3 have an average token size of 3.7 and 4.4 bytes. In contrast, BLT can
achieve similar scaling trends with an average patch size of 6 and even 8 bytes. Inference flop are inversely
proportional to the average patch size, so using a patch size of 8 bytes would lead to nearly 50% inference
flop savings. Models with larger patch sizes also seem to perform better as we scale model and data size.
BLT with patch size of 8 starts at a significantly worse point compared to bpe Llama 2 at 1B but ends up
better than bpe at 7B scale. This suggests that such patch sizes might perform better at even larger scales
and possibly that even larger ones could be feasible as model size and training compute grow.


**5.2** **Beyond Compute Optimal Task Evaluations**


To assess scaling properties further, we train an 8B BLT model beyond the compute optimal ratio on the
BLT-1T dataset, a larger higher-quality dataset, and measure performance on a suite of standard classification
and generation benchmarks. For evaluation, we select the following common sense reasoning, world knowledge,
and code generation tasks:


_Classification tasks_ include ARC-Easy (0-shot) (Clark et al., 2018), Arc-Challenge (0-shot) (Clark et al., 2018),
HellaSwag (0-shot) (Zellers et al., 2019), PIQA (0-shot) (Bisk et al., 2020), and MMLU (5-shot) (Hendrycks
et al., 2020). We employ a prompt-scoring method, calculating the likelihood over choice characters, and
report the average accuracy.


_Coding related generation tasks:_ We report pass@1 scores on MBPP (3-shot) (Austin et al., 2021) and
HumanEval (0-shot) (Chen et al., 2021), to evaluate the ability of LLMs to generate Python code.


In Table 1, we compare three models trained on the BLT-1T dataset: a bpe Llama 3 tokenizer-based model, [6]

and two variants of the BLT model. One employing a space-patching scheme (BLT-Space) and another
utilizing an entropy-based patching scheme (BLT-Entropy). with approx. monotonicity constraint and reset
the context of the entropy model with new lines (as discussed in subsection 4.4). All three models are


6 We choose the Llama 3 tokenizer with its 128k vocabulary as it performs better than Llama 2’s 32k vocabulary.


11


Llama 3 BLT-Space BLT-Entropy
1T Tokens 6T Bytes 4.5T Bytes


**Arc-E** 77.6 75.4 **79.6**

**Arc-C** **53.3** 49.8 52.1

**HellaSwag** 79.1 79.6 **80.6**
**PIQA** 80.7 **81.1** 80.6

**MMLU** **58.1** 54.8 57.4

**MBPP** 40.2 37.6 **41.8**

**HumanEval** 31.1 27.4 **35.4**


**Average** 60.0 58.0 **61.1**


**Bytes/Patch on Train Mix** 4.4 **6.1** 4.5


**Table 1** Comparison of flop-matched BLT **8B** models trained on the BLT-1T dataset comprising high-quality tokens
of text and code from publicly available sources, with baseline models using the Llama 3 tokenizer. BLT performs
better than Llama 3 on average, and depending on the patching scheme, achieves significant flops savings with a
minor reduction in performance.


Llama 2 Llama 3 Entropy ps=6 Entropy ps=8 Inference flops Compute Optimal (Bytes) Crossover (Bytes)


470m 450m 610m (1.2x) 760m (1.6x) 3.1E8 50B 150B
3.6B 3.9B 5.2B (1.3x) 6.6B (1.7x) 2.1E9 400B 1T


**Table 2** Details of models used in the fixed-inference scaling study. We report non-embedding parameters for each
model and their relative number compared to Llama 2. We pick model sizes with equal inference flops per byte. We
also indicate BPE’s compute-optimal training data quantity and the crossover point where BLT surpasses BPE as seen
in Figure 1 (both expressed in bytes of training data). This point is achieved at much smaller scales compared to
many modern training budgets.


trained with an equivalent flop budget. However, with BLT-Entropy we additionally make an inference time
adjustment of the entropy threshold from 0.6 to 0.1 which we find to improve task performance at the cost of
more inference steps.


The BLT-Entropy model outperforms the Llama 3 model on 4 out of 7 tasks while being trained on the same
number of bytes. This improvement is like due to a combination of (1) a better use of training compute via
dynamic patching, and (2) the direct modeling of byte-level information as opposed to tokens.


On the other hand, BLT-Space underperforms the Llama 3 tokenizer on all but one task, but it achieves a
significant reduction in inference flops with its larger average patch size of 6 bytes. In comparison, the bpe
and entropy-patching based models have roughly equivalent average patch size of approximately 4.5 bytes on
the training data mix. With the same training budget, the larger patch size model covers 30% more data
than the other two models which might push BLT further away from the compute-optimal point.


**5.3** **Patches Scale Better Than Tokens**


With BLT models, we can simultaneously increase model size and patch size while maintaining the same
training and inference flop budget and keeping the amount of training data constant. Arbitrarily increasing
the patch size is a unique feature of patch-based models which break free of the efficiency tradeoffs of
fixed-vocabulary token-based models, as discussed in Section 2.4. Longer patch sizes save compute, which can
be reallocated to grow the size of the global latent transformer, because it is run less often.


We conduct a fixed inference scaling study to test the hypothesis that larger models taking fewer steps on
larger patches might perform better than smaller models taking more steps. Starting from model sizes of 400m
and 3.6B parameters with the Llama 2 tokenizer, we find flop equivalent models with the Llama 3 tokenizer
and BLT-Entropy models with average patch sizes of 6 and 8 bytes on the training datamix (see Table 2 for
model details). For patch size 8 models, we use 3 encoder layers instead of 1. We train each model for various
training flop budgets.


12


Llama 3 Llama 3.1 BLT
(1T tokens) (16T tokens) (1T tokens)


**HellaSwag Original** 79.1 80.7 **80.6**
**HellaSwag Noise Avg.** 56.9 64.3 **64.3**

**- AntSpeak** 45.6 61.3 **57.9**

**- Drop** 53.8 57.3 **58.2**

**- RandomCase** 55.3 65.0 **65.7**

**- Repeat** 57.0 61.5 **66.6**

**- UpperCase** 72.9 76.5 **77.3**


**Phonology-G2P** 11.8 18.9 **13.0**


**CUTE** 27.5 20.0 **54.1**

**- Contains Char** 0.0 0.0 **55.9**

**- Contains Word** 55.1 21.6 **73.5**

**- Del Char** 34.6 34.3 **35.9**

**- Del Word** **75.5** 84.5 56.1

**- Ins Char** 7.5 0.0 **7.6**

**- Ins Word** **33.5** 63.3 31.2

**- Orthography** 43.1 0.0 **52.4**

**- Semantic** 65 0.0 **90.5**

**- Spelling** 1.1              - **99.9**

**- Spelling Inverse** 30.1 3.6 **99.9**

**- Substitute Char** 0.4 1.2 **48.7**

**- Substitute Word** 16.4 6.8 **72.8**

**- Swap Char** 2.6 2.4 **11.5**

**- Swap Word** 20.1 4.1 **21**


**Table 3** We compare our 8B BLT model to 8B BPE Llama 3 trained on 1T tokens on tasks that assess robustness to
noise and awareness of the constituents of language (best result bold). We also report the performance of Llama 3.1 on
the same tasks and underline best result overall. BLT outperforms the Llama 3 BPE model by a large margin and
even improves over Llama 3.1 in many tasks indicating that the byte-level awareness is not something that can easily
be obtained with more data.


Figure 1 shows that BLT models achieve better scaling trends than tokenization-based architectures for both
inference flop classes. In both cases, BPE models perform better with small training budgets and are quickly
surpassed by BLT, not far beyond the compute-optimal regime. In practice, it can be preferable to spend
more during the one-time pretraining to achieve a better performing model with a fixed inference budget. A
perfect example of this is the class of 8B models, like Llama 3.1, which has been trained on two orders of
magnitude more data than what is compute-optimal for that model size.


The crossover point where BLT improves over token-based models has shifted slightly closer to the computeoptimal point when moving to the larger flop class models (from 3x down to 2.5x the compute optimal
budget). Similarly, the larger patch size 8 model has steeper scaling trend in the larger flop class overtaking
the other models sooner. As discussed in Section 5.1, larger patch sizes appear to perform closer to BPE
models at larger model scales. We attribute this, in part, to the decreasing share of total flops used by the
byte-level Encoder and Decoder modules which seem to scale slower than the Latent Transformer. When
growing total parameters 20x from 400M to 8B, we only roughly double BLT’s local model parameters. This
is important as larger patch sizes only affect flops from the patch Latent Transformer and not the byte-level
modules. In fact, that is why the BLT-Entropy ps=8 went from 1.6x to 1.7x of the Llama 2 model size when
moving to the larger model scale.


In summary, our patch-length scaling study demonstrates that the BLT patch-based architecture can achieve
better scaling trends by simultaneously increasing both patch and model size. Such trends seem to persist
and even improve at larger model scales.


13


Language Language _→_ English English _→_ Language


Llama 3 BLT Llama 3 BLT


**Arabic** 22.3 24.6 10.4 8.8

**German** 41.3 42.0 29.8 31.2

**Hindi** 20.7 20.9 7.8 7.2

**Italian** 34.0 33.9 24.4 26.2

**Vietnamese** 31.2 31.0 28.4 23.7

**Thai** 17.9 18.1 10.5 7.7


**Armenian** 1.7 6.3 0.6 0.9

**Amharic** 1.3 3.1 0.4 0.5

**Assamese** 2.7 5.4 0.8 1.6

**Bengali** 4.7 12.7 1.7 4.1
**Bosnian** 36.0 37.3 16.9 19.6

**Cebuano** 18.2 20.6 5.8 9.1

**Georgian** 1.7 7.4 1.0 2.5
**Gujarati** 2.0 5.8 1.0 2.2
**Hausa** 5.75 5.9 1.2 1.3

**Icelandic** 16.1 17.9 4.8 5.3

**Kannada** 1.6 3.9 0.7 1.7

**Kazakh** 5.6 7.0 1.0 2.6

**Kabuverdianu** 20.3 20.9 5.1 6.8

**Khmer** 4.4 9.5 0.8 0.8

**Kyrgyz** 4.6 5.1 0.9 2.0
**Malayalam** 1.8 3.5 0.7 1.4
**Odia** 1.6 2.7 0.8 1.1

**Somali** 5.0 5.0 1.1 1.4

**Swahili** 10.1 12.0 1.4 2.3

**Urdu** 9.3 9.5 2.0 1.4

**Zulu** 4.7 5.0 0.6 0.5


**Overall Average** 12.1 **14.0** 5.9 **6.4**


**Table 4** Performance of 8B BLT and 8B Llama 3 trained for 1T tokens on translating into and from six widely-used
languages and twenty one lower resource languages with various scripts from the FLORES-101 benchmark (Goyal
et al., 2022).

### **6 Byte Modeling Improves Robustness**


We also measure the robustness of BLT compared to token-based models that lack direct byte-level information,
and present an approach to byte-ify pretrained token-based models.


**6.1** **Character-Level Tasks**


A very early motivation for training byte-level models was to take advantage of their robustness to byte
level noise in the input, and also to exploit their awareness of the constituents of tokens, which current
tokenizer-based models struggle with. To measure these phenomena, we perform additional evaluations on
benchmarks that evaluate both robustness to input noise as well as awareness of characters, both English and
multi-lingual, including digits and phonemes. We present these results in Table 3.


_Noisy Data_ We create noised versions of the benchmark classification tasks described in Section 5.2, to
compare the robustness of tokenizer-based models with that of BLT. We employ five distinct character-level
noising strategies to introduce variations in the text: (a) _AntSpeak_ : This strategy converts the entire text into
uppercase, space-separated characters. (b) _Drop_ : Randomly removes 10% of the characters from the text. (c)


14


|Task|Prompt|Llama 3|BLT|
|---|---|---|---|
|Substitute<br>Word|Question: Substitute " and " with " internet " in<br>" She went to the kitchen and saw two cereals. ".<br>Answer:|She<br>went<br>to<br>the<br>kitchen<br>and<br>saw<br>two<br>cereals.|She<br>went<br>to<br>the<br>kitchen<br>internet<br>saw<br>two cereals.|
|Swap Char|Question: Swap " h " and " a " in " that ". Answer:|that|taht|
|Substitute<br>Char|Question: Substitute " a " with " m " in " page ".<br>Answer:|-|pmge|
|Semantic<br>Similarity|Question: More semantically related to " are ": "<br>seem ", " acre ". Answer:|acre|seem|
|Orthographic<br>Similarity|Question: Closer in Levenshtein distance to " time<br>": " timber ", "period ". Answer:|period|timber|
|Insert Char|Question: Add an " z " after every " n " in " not ".<br>Answer:|znotz|nzot|



**Figure 7** Output responses from Llama 3 and BLT models for various tasks from CUTE benchmark. BLT model
performs better on sequence manipulation tasks compared to the tokenizer-based Llama 3 model. Note that few-shot
examples are not shown in the above prompts to maintain clarity.


_RandomCase_ : Converts 50% of the characters to uppercase and 50% to lowercase randomly throughout the
text. (d) _Repeat_ : Repeats 20% of the characters up to a maximum of four times. (e) _UpperCase_ : Transforms
all characters in the text to uppercase. During evaluation, we apply each noising strategy to either the prompt,
completion, or both as separate tasks and report the average scores. In Table 3 we report results on noised
HellaSwag (Zellers et al., 2019) and find that BLT indeed outperforms tokenizer-based models across the
board in terms of robustness, with an average advantage of 8 points over the model trained on the same data,
and even improves over the Llama 3.1 model trained on a much larger dataset.


_Phonology - Grapheme-to-Phoneme (G2P)_ We assess BLT’s capability to map a sequence of graphemes
(characters representing a word) into a transcription of that word’s pronunciation (phonemes). In Table 3, we
present the results of the G2P task in a 5-shot setting using Phonology Bench (Suvarna et al., 2024) and find
that BLT outperforms the baseline Llama