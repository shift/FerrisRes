## ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

Samyam Rajbhandari _[∗]_, Jeff Rasley _[∗]_, Olatunji Ruwase, Yuxiong He


_{_ `samyamr, jerasley, olruwase, yuxhe` _}_ `@microsoft.com`

### **Abstract**


Large deep learning models offer significant accuracy gains, but training billions to trillions of
parameters is challenging. Existing solutions such as data and model parallelisms exhibit fundamental limitations to fit these models into limited device memory, while obtaining computation,
communication and development efficiency. We develop a novel solution, Zero Redundancy
Optimizer ( _ZeRO_ ), to optimize memory, vastly improving training speed while increasing the
model size that can be efficiently trained. _ZeRO_ eliminates memory redundancies in data- and
model-parallel training while retaining low communication volume and high computational
granularity, allowing us to scale the model size proportional to the number of devices with
sustained high efficiency. Our analysis on memory requirements and communication volume
demonstrates: _ZeRO_ has the potential to scale beyond 1 _Trillion_ parameters using today’s
hardware.

We implement and evaluate ZeRO: it trains large models of over 100B parameter with
super-linear speedup on 400 GPUs, achieving throughput of 15 Petaflops. This represents an
8x increase in model size and 10x increase in achievable performance over state-of-the-art. In
terms of usability, _ZeRO_ can train large models of up to 13B parameters (e.g., larger than
Megatron GPT 8.3B and T5 11B) without requiring model parallelism which is harder for
scientists to apply. Last but not the least, researchers have used the system breakthroughs
of _ZeRO_ to create the world’s largest language model (17B parameters) with record breaking

accuracy.

### **1 Extended Introduction**


Deep Learning (DL) models are becoming larger, and the increase in model size offers significant
accuracy gain. In the area of Natural Language Processing (NLP), the transformers have paved
way for large models like Bert-large (0.3B) [1], GPT-2 (1.5B) [2], Megatron-LM (8.3B) [3], T5
(11B) [4]. To enable the continuation of model size growth from 10s of billions to trillions of
parameters, we experience the challenges of training them - they clearly do not fit within the
memory of a single device, e.g., GPU or TPU, and simply adding more devices will not help
scale the training.
Basic data parallelism (DP) does not reduce memory per device, and runs out of memory for models with more than 1.4B parameters on current generation of GPUs with 32 GB
memory. Other existing solutions such as Pipeline Parallelism (PP), Model Parallelism (MP),


_∗_ Equal Contributors


1


CPU-Offloading, etc, make trade-offs between functionality, usability, as well as memory and
compute/communication efficiency, but all of which are crucial to training with speed and scale.
Among different existing solution for training large models, MP is perhaps the most promising. The largest models in the current literature, the 11B T5 model [4], and Megatron-LM
8.3B [3], were both powered by model parallelism, implemented in Mesh-Tensorflow [5] and
Megatron-LM[3], respectively. However, MP cannot scale much further beyond these models
sizes. MP splits the model vertically, partitioning the computation and parameters in each layer
across multiple devices, requiring significant communication between each layer. As a result,
they work well within a single node where the inter-GPU communication bandwidth is high,
but the efficiency degrades quickly beyond a single node [3]. We tested a 40B parameter model
using Megatron-LM across two DGX-2 nodes and observe about 5 _Tflops_ per V100 GPU (less
than 5% of hardware peak).
So, how can we overcome the limitations of existing solutions and train large models more
efficiently? To answer this question, we first analyze the full spectrum of memory consumption
of the existing systems on model training and classify it into two parts: 1) For large models,
the majority of the memory is occupied by _model states_ which include the optimizer states
(such as momentum and variances in Adam [6]), gradients, and parameters. 2) The remaining
memory is consumed by activation, temporary buffers and unusable fragmented memory, which
we refer to collectively as _residual_ states. We develop _ZeRO_ — Zero Redundancy Optimizer
— to optimize memory efficiency on both while obtaining high compute and communication
efficiency. As these two parts face different challenges, we develop and discuss their solutions
correspondingly.
**Optimizing Model State Memory** Model states often consume the largest amount of
memory during training, but existing approaches such as DP and MP do not offer satisfying
solution. DP has good compute/communication efficiency but poor memory efficiency while
MP can have poor compute/communication efficiency. More specifically, DP replicates the
entire model states across all data parallel process resulting in redundant memory consumption;
while MP partition these states to obtain high memory efficiency, but often result in too finegrained computation and expensive communication that is less scaling efficient. Furthermore,
all of these approaches maintain all the model states required over the entire training process
statically, even though not all model states are required all the time during the training. Based
on these observations, we develop _ZeRO_ -DP, ZeRO-powered data parallelism, that achieves the
computation/communication efficiency of DP while achieving memory efficiency of MP. _ZeRO_ DP removes the memory state redundancies across data-parallel processes by _partitioning_ the
model states instead of replicating them, and it retains the compute/communication efficiency
by retaining the computational granularity and communication volume of DP using a dynamic
communication schedule during training.
_ZeRO_ -DP has three main optimization stages (as depicted in Figure 1), which correspond
to the partitioning of optimizer states, gradients, and parameters. When enabled cumulatively:
1) Optimizer State Partitioning ( _P_ _os_ ): 4x memory reduction, same communication volume
as DP;
2) Add Gradient Partitioning ( _P_ _os_ + _g_ ): 8x memory reduction, same communication volume
as DP;
3) Add Parameter Partitioning ( _P_ _os_ + _g_ + _p_ ): Memory reduction is linear with DP degree _N_ _d_ .
For example, splitting across 64 GPUs ( _N_ _d_ = 64) will yield a 64x memory reduction. There is
a modest 50% increase in communication volume.
ZeRO-DP eliminates memory redundancies and makes the full aggregate memory capacity
of a cluster available. With all three stages enabled, ZeRO can train a trillion-parameter model
on just 1024 NVIDIA GPUs. A trillion-parameter model with an optimizer like Adam [6] in 16

2


Figure 1: Comparing the per-device memory consumption of model states, with three stages of
_ZeRO_ -DP optimizations. Ψ denotes model size (number of parameters), _K_ denotes the memory
multiplier of optimizer states, and _N_ _d_ denotes DP degree. In the example, we assume a model
size of Ψ = 7 _._ 5 _B_ and DP of _N_ _d_ = 64 with _K_ = 12 based on mixed-precision training with
Adam optimizer.


bit precision requires approximately 16 terabytes (TB) of memory to hold the optimizer states,
gradients, and parameters. 16TB divided by 1024 is 16GB, which is well within a reasonable
bound for a GPU (e.g., with 32GB of on-device memory).
**Optimizing Residual State Memory** After _ZeRO_ -DP boosts memory efficiency for
model states, the rest of the memory consumed by activations, temporary buffers, and unusable memory fragments could become a secondary memory bottleneck. We develop _ZeRO_ -R
to optimize the residual memory consumed by these three factors respectively.
1) For activations (stored from forward pass in order to perform backward pass), we noticed
checkpointing [7] helps but not sufficient for large models. Thus _ZeRO_ -R optimizes activation
memory by identifying and removing activation replication in existing MP approaches through
activation partitioning. It also offloads activations to CPU when appropriate.
2) _ZeRO_ -R defines appropriate size for temporary buffers to strike for a balance of memory
and computation efficiency.
3) We observe fragmented memory during training due to variations in the lifetime of
different tensors. Lack of contiguous memory due to fragmentation can cause memory allocation
failure, even when enough free memory is available. _ZeRO_ -R proactively manages memory
based on the different lifetime of tensors, preventing memory fragmentation.
_ZeRO_ -DP and _ZeRO_ -R combined together forms a powerful system of memory optimizations for DL training that we collectively refer to as _ZeRO_ .
_**ZeRO**_ **and MP** : Since _ZeRO_ eliminates the memory inefficiency in DP, it is natural to ask:
Do we still need MP, and when? How does _ZeRO_ work with MP? With _ZeRO_, MP becomes
a less attractive option for the purpose of fitting large models alone. _ZeRO_ -DP is at least as
effective on reducing per-device memory footprint as MP, or more effective sometimes when MP
cannot divide the model evenly. It also has comparable or better scaling efficiency. Furthermore,
data parallelism is so easy to use that it is widely applicable across different workloads, while


3


Figure 2: _ZeRO_ training throughput and speedup w.r.t SOTA baseline for varying model sizes.
For _ZeRO_, the MP always fit in a node, while for baseline, models larger than 40B require MP
across nodes.


MP approaches today often need some work from model developers to revise their model,
system developers to work out distributed operators, and existing work like Megatron-LM only
supports a limited set of operators and models.
That being said, there are still cases where we want to leverage MP: i) When used with
_ZeRO_ -R, MP can reduce activation memory footprint for very large models. ii) For smaller
models where activation memory is not an issue, MP can also have benefits when aggregated
batch size using DP alone is too big to have good convergence. [1] In those case, one can combine
_ZeRO_ with MP to fit the model with an acceptable aggregated batch size.
We show that _ZeRO_ can be combined with MP, resulting in a max theoretical memory
reduction of _N_ _d_ _×_ _N_ _m_ times on each device with a DP degree of _N_ _d_ and MP degree of _N_ _m_ . This
could allow us to fit a trillion parameter model on 1024 GPUs with 16-way model parallelism
(within each DGX2 node) and 64-way data parallelism across nodes, and run it efficiently using
a modest batch size!

**Implementation & Evaluation** The complete set of optimizations in _ZeRO_ could allow
us to run models with trillion parameters on the high-end hardware cluster today (e.g., with 1K
V100 GPUs), however, the hardware compute capacity is still too limited and training time can
be impractically long ( _>_ 1 year). Therefore, our focus for this implementation is to efficiently

_∼_
support models with 10x parameters ( 100B parameters) than state-of-the-art (SOTA) while
still being within reach of the compute capabilities of current hardware. We implement and
evaluate a subset of optimizations in _ZeRO_ called _ZeRO_ -100B — _P_ _os_ + _g_ of _ZeRO_ -DP plus
ZeRO-R — that allow us to achieve this goal. The results show:
Model Size Combined with MP, _ZeRO_ -100B runs 170B parameter models efficiently, while
the existing system like using Megatron alone cannot scale efficiently beyond 40B parameters,
as shown in Figure 2. This is an over 8x increase in model size compared to SOTA.
Speed Improved memory efficiency powers higher throughput and faster training. As shown
in Figure 2, _ZeRO_ runs 100B parameter models on a 400 Nvidia V100 GPU cluster with over


1 Prior work [8] shows, very large batch size could slow down convergence. For given model and data, there
is a measure of critical-batch size, where increasing batch size further slows down convergence. The detailed
discussion of this topic is beyond the scope of the paper.


4


Figure 3: Superlinear scalability and per GPU training throughput of a 60B parameter model
using _ZeRO_ -100B.


38 TFlops per GPU, and aggregate performance over 15 Petaflops. This is more than 10x
improvement in training speed compared to SOTA for the same model size.
Scalability We observe super linear speedup in the regime of 64-400 GPUs, where the performance more than doubles when we double the number of GPUs. This is a property of _ZeRO_ -DP
which reduces the memory footprint of the model states as we increase the DP degree allowing
us to fit larger batch sizes per GPU resulting in better performance. We expect this behaviour
to continue further as we increase the number of GPUs beyond 400.
Democratization of Large Model Training _ZeRO_ -100B powers data scientist to train models
with up to 13B parameters without any MP or PP that requires model refactoring, where 13B
is more parameters than the largest model in literature (T5 with 11B parameters). Data
scientists can thus experiment freely with large models without worrying about parallelism. In
comparison, exist systems (e.g., PyTorch Distributed Data Parallel) runs out of memory with
1.4B parameter models.
New SOTA Model _ZeRO_ powers the largest language model with 17B parameters and
record-breaking accuracy, Turing-NLG [9].
We share _ZeRO_ as a part of our open source DL training optimization library called DeepSpeed [2] . We plan to release all implementations described in this paper by end of May 2020
and extend it further to support 1 trillion parameters by enabling _ZeRO_ -DP stage 3 partitioning parameters ( _P_ _os_ + _g_ + _p_ ). We plan to make _ZeRO_ fully accessible to the DL community to
catalyze the evolution and democratization of large model training at scale.


2 https://github.com/microsoft/deepspeed


5


### **2 Related Work**

**2.1** **Data, Model and Pipeline Parallelism**


Parallelization is a key strategy on training large models at scale. For a model that fits in the
device memory for training, data parallelism (DP) is used to scale training to multiple devices.
In DP, model parameters are replicated on each device. At each step, a mini-batch is divided
evenly across all the data parallel processes, such that each process executes the forward and
backward propagation on a different subset of data samples, and uses averaged gradients across
processes to update the model locally.
When a model does not fit in the device memory, model parallelism (MP) [5, 3] and pipeline
parallelism (PP) [10, 11] split the model among processes, in vertical and horizontal way respectively. Sec. 1 discussed how _ZeRO_ relates to DP and MP. We now discuss PP and how it
relates to reducing memory consumption.
PP splits a model horizontally across layers running each partition on a different device
and use micro-batching to hide the pipeline bubble [10, 11]. Model functionalities such as
tied-weights and batch-normalization are difficult to implement due to horizontal splitting and
micro-batching, respectively. Popular PP implementation such as G-pipe [10] partitions both
model parameters and total activations but requires a batch size proportional to number of
pipeline partitions to hide the pipeline bubble. The large batch size can affect the convergence
rate, while also requiring significant memory to store activations. A different implementation
of PP in PipeDream [12] keeps multiple copies of stale parameters to hide the pipeline bubble
without increasing the batch size significantly, making it less memory efficient. Additionally,
the implementation is not equivalent to the standard DL training and has implications on
training convergence. In contrast, _ZeRO_ obtains the same or better memory efficiency than
PP without incurring functionality, performance and convergence related restrictions of PP.


**2.2** **Non-parallelism based approach to reduce memory**


In addition to MP and PP, there are multiple lines of work that target reducing memory
overheads of DL training.


**2.2.1** **Reducing Activation Memory**


Multiple efforts have focused on reducing the memory footprint of activations through compression [13], activation checkpointing [7, 14], or live analysis [15]. These efforts are complimentary
and can work together with _ZeRO_ . In fact, activation memory reduction in _ZeRO_ -R works in
parallel with activation checkpointing.


**2.2.2** **CPU Offload**


[16, 17] exploit heterogeneous nature of today’s compute nodes, offloading model states to
CPU memory through algorithmic design or virtualized memory, respectively. Up to 50% of
training time can be spent on GPU-CPU-GPU transfers [16]. _ZeRO_ differs in that it reduces
the memory consumption significantly without storing the model states to CPU memory whose
bandwidth is severely constrained due to PCI-E. On rare cases, _ZeRO_ -R may offload just the
activation checkpoints for very large models to improve performance (see Sec. 6.1 for details).


6


**2.2.3** **Memory Efficient Optimizer**


[18, 19] focus on reducing memory consumption of adaptive optimization methods by maintaining coarser-grained statistics of model parameters and gradients, with potential impact on
model convergence guarantees. _ZeRO_ is orthogonal to these efforts, and its optimizations do
not change the model optimization method or affect model convergence, but effectively reduce
memory footprint of optimizer states and gradients per device.


**2.3** **Training Optimizers**


Adaptive optimization methods [20, 6, 21, 22] are crucial to achieving SOTA performance and
accuracy for effective model training of large models. Compared to SGD, by maintaining finegrained first-order and second-order statistics for each model parameter and gradient at the cost
of significant memory footprint. _ZeRO_ can reduce the memory footprint of these optimizers by
orders of magnitude, making these sophisticated optimization methods practical for training
large models on hardware with modest device memory. It also makes it possible to develop and
use even more complex and memory hungry optimizers that may have better convergence.

### **3 Where Did All the Memory Go?**


Let’s take a step back to examine the memory consumption of the current training system. For
example, a 1.5B parameter GPT-2 model requires 3GB of memory for its weights (or parameters) in 16-bit precision, yet, it cannot be trained on a single GPU with 32GB memory using
Tensorflow or PyTorch. One may wonder where all the memory goes. During model training,
most of the memory is consumed by _model states_, i.e., tensors comprising of pptimizer states,
gradients, and parameters. Besides these model states, the rest of the memory is consumed by
activations, temporary buffers and fragmented memory which we call _residual states_ . We look
at the memory consumption from both in details.


**3.1** **Model States: Optimizer States, Gradients and Parameters**


Majority of the device memory is consumed by model states during training. Consider for
instance, Adam [6], one of the most popular optimizers for DL training. Adam requires storing
two optimizer states, i) the time averaged momentum and ii) variance of the gradients to
compute the updates. Therefore, to train a model with ADAM, there has to be enough memory
to hold a copy of both the momentum and variance of the gradients. In addition, there needs to
be enough memory to store the gradients and the weights themselves. Of these three types of
the parameter-related tensors, the optimizer states usually consume the most memory, specially
when mixed-precision training is applied.
**Mixed-Precision Training** The state-of-the-art approach to train large models on the
current generation of NVIDIA GPUs is via mixed precision (fp16/32) training [23], where
parameters and activations are stored as fp16, enabling the use of the high throughput tensor
core units [24] on these GPUs. During mixed-precision training, both the forward and backward
propagation are performed using fp16 weights and activations. However, to effectively compute
and apply the updates at the end of the backward propagation, the mixed-precision optimizer
keeps an fp32 copy of the parameters as well as an fp32 copy of all the other optimizer states.
Let’s take Adam as a concrete example. Mixed precision training of a model with Ψ parameters using Adam requires enough memory to hold an _fp_ 16 copy of the parameters and the
gradients, with memory requirements of 2Ψ and 2Ψ bytes respectively. In addition, it needs


7


to hold the optimizer states: an _fp_ 32 copy of the parameters, momentum and variance, with
memory requirements of 4Ψ, 4Ψ, and 4Ψ bytes, respectively. Let’s use _K_ to denote the memory
multiplier of the optimizer states, i.e., the additional memory required to store them is _K_ Ψ
bytes. Mixed-precision Adam has _K_ = 12. In total, this results in 2Ψ + 2Ψ + _K_ Ψ = 16Ψ bytes
of memory requirement. For a model such as GPT-2 with 1 _._ 5 Billion parameters, this leads to
a memory requirement of at least 24 _GB_, which is significantly higher than the meager 3 _GB_
of memory required to hold the _fp_ 16 parameters alone.


**3.2** **Residual Memory Consumption**


**Activations** can take up a significant amount of memory [7] during training. As a concrete
example, the 1.5B parameter GPT-2 model trained with sequence length of 1K and batch size of
32 requires about 60 GB of memory [3] . Activation checkpointing (or activation recomputation)
is a common approach to reduce the activation memory by approximately the square root of
the total activations at the expense of 33% re-computation overhead [7]. This would reduce
the activation memory consumption of this model to about 8 GB.
Despite the significant reduction, the activation memory can grow quite large for bigger
models even with activation checkpointing. For example, a GPT-like model with 100 billion
parameters requires around 60 GB of memory for batch size 32, even when using activation
checkpointing.
**Temporary buffers** used for storing intermediate results consumes non-trivial amount of
memory for large models. Operations such as gradient all-reduce, or gradient norm computation
tend to fuse all the gradients into a single flattened buffer before applying the operation in an
effort to improve throughput. For example, the bandwidth of all-reduce across devices improves
with large message sizes. While the gradient themselves are usually stored as fp16 tensors, the
fused buffer can be an fp32 tensor depending on the operation. When the size of the model
is large, these temporary buffer sizes are non-trivial. For example, for a model with 1.5B
parameters, a flattened fp32 buffer would required 6 _GB_ of memory.
**Memory Fragmentation:** So far we have discussed the actual memory consumption
during training. Additionally, it is possible to run out of usable memory even when there
is plenty of available memory. This can happen with memory fragmentation. A request for a
memory will fail if there isn’t enough contiguous memory to satisfy it, even if the total available
memory is larger than requested. We observe significant memory fragmentation when training
very large models, resulting in out of memory issue with over 30% of memory still available in

some extreme cases.

### **4 ZeRO : Insights and Overview**


_ZeRO_ has two sets of optimizations: i) _ZeRO_ -DP aimed at reducing the memory footprint of
the model states, and ii) _ZeRO_ -R targeted towards reducing the residual memory consumption.
We present an overview of the optimizations and the insights behind, which allows _ZeRO_ to
reduce memory footprint while remaining efficient. Please note efficiency is a key here: without
this constraint, trivial solutions like moving all the parameter states to the CPU memory, or
increasing the MP degree arbitrarily can reduce memory footprint.


3 The activation memory of a transformer-based model is proportional to the number of transformer layers _×_
hidden dimensions _×_ sequence length _×_ batch size. For a GPT-2 like architecture the total activations is about
12 _× hidden_ ~~_d_~~ _im × batch × seq_ ~~_l_~~ _ength × transformer_ ~~_l_~~ _ayers_ .


8


**4.1** **Insights and Overview:** _**ZeRO**_ **-DP**


_ZeRO_ powered DP is based on three key insights:
_a)_ DP has better scaling efficiency than MP because MP reduces the granularity of the
computation while also increasing the communication overhead. Beyond a certain point, lower
computational granularity reduces the efficiency per GPU, while the increased communication
overhead, hiders the scalability across GPUs, especially when crossing node boundaries. On
the contrary, DP has both higher computational granularity and lower communication volume,
allowing for much higher efficiency.
_b)_ DP is memory inefficient as model states are stored redundantly across all data-parallel
processes. On the contrary, MP partitions the model states to obtain memory efficiency.
_c)_ Both DP and MP keep all the model states needed over the entire training process, but
not everything is required all the time. For example, parameters corresponding to each layer is
only needed during the forward propagation and backward propagation of the layer.
Based on these insights, _ZeRO_ -DP retains the training efficiency of DP while achieving
the memory efficiency of MP. _ZeRO_ -DP _partitions_ the model states instead of replicating them
(Section 5) and uses a dynamic communication schedule that exploits the intrinsically temporal
nature of the model states while minimizing the communication volume (Section 7). By doing
so, _ZeRO_ -DP reduces per-device memory footprint of a model _linearly_ with the increased DP
degree while maintaining the communication volume close to that of the default DP, retaining
the efficiency.


**4.2** **Insights and Overview:** _**ZeRO**_ **-R**


**4.2.1** **Reducing Activation Memory**


Two key insights are:
_a)_ MP partitions the model states but often requires replication of the activation memory.
For example, if we split the parameters of a linear layer vertically and compute them in parallel
across two GPUs, each GPU requires the entire activation to compute its partition
_b)_ For models such as GPT-2 or larger, the arithmetic intensity (ratio of the amount of computation per iteration to amount of activation checkpoints per iteration) is very large ( _≥_ 10 _K_ )
and increases linearly with hidden dimension making it possible to hide the data-movement
cost for the activation checkpoints, even when the bandwidth is low.
_ZeRO_ removes the memory redundancies in MP by _partitioning_ the activations checkpoints
across GPUs, and uses allgather to reconstruct them on demand. The activation memory footprint is reduced proportional to the MP degree. For very large models, _ZeRO_ can even choose
to move the activation partitions to the CPU memory, while still achieving good efficiency due
to large arithmetic intensity in these models.


**4.2.2** **Managing Temporary buffers**


_ZeRO_ -R uses constant size buffers to avoid temporary buffers from blowing up as the model
size increases, while making them large enough to remain efficient.


**4.2.3** **Managing fragmented Memory**


Memory fragmentation is a result of interleaving between short lived and long lived memory
objects. During the forward propagation activation checkpoints are long lived but the activations that recomputed are short lived. Similarly, the backward computation, the activation
gradients are short lived while the parameter gradients are long lived. Based on this insight,


9


_ZeRO_ performs on-the-fly memory defragmentation by moving activation checkpoints and gradients to pre-allocated contiguous memory buffers. This not only increases memory availability
but also improves efficiency by reducing the time it takes for the memory allocator to find free
contiguous memory.

### **5 Deep Dive into ZeRO -DP**


While the existing DP approach replicates the model states at each device and introduces
significant memory overhead, _ZeRO_ -DP eliminates this memory redundancy by partitioning
them — optimizer states, gradients and parameters — across data parallel processes. Figure 1
quantifies and visualizes the memory requirement with and without _ZeRO_ -DP. The figure shows
the memory footprint after partitioning (1) optimizer state, (2) gradient and (3) parameter
redundancies accumulatively. We refer to them as the three optimization phases of _ZeRO_ -DP:
_P_ _os_, _P_ _g_, and _P_ _p_, which we elaborate below.


**5.1** **P** _os_ **: Optimizer State Partitioning**


For a DP degree of _N_ _d_, we group the optimizer states into _N_ _d_ equal partitions, such that the
_i_ _[th]_ data parallel process only updates the optimizer states corresponding to the _i_ _[th]_ partition.
Thus, each data parallel process only needs to store and update _N_ 1 _d_ [of the total optimizer states]
and then only update _N_ 1 _d_ [of the parameters. We perform an all-gather across the data parallel]
process at the end of each training step to get the fully updated parameters across all data
parallel process.
**Memory Savings:** As shown in Figure 1, the memory consumption after optimizing state
partition reduces from 4Ψ + _K_ Ψ to 4Ψ + _[K]_ _N_ [Ψ] _d_ [. As the concrete example depicted in Figure 1, a]

7.5 B parameter model requires 31.4GB of memory using _P_ _os_ with 64-way DP ( _N_ _d_ = 64), while
requiring 120 GB with standard DP. Furthermore, when _N_ _d_ is large, the memory requirement
on model states reduces from 4Ψ + 12Ψ = 16Ψ bytes to 4Ψ + [12Ψ] _N_ _d_ _[≈]_ [4Ψ bytes, leading to a 4x]

reduction.


**5.2** **P** _g_ **: Gradient Partitioning**


As each data parallel process only updates its corresponding parameter partition, it only needs
the reduced gradients for the corresponding parameters. Therefore, as each gradient of each
layer becomes available during the backward propagation, we only reduce them on the data
parallel process responsible for updating the corresponding parameters. After the reduction
we no longer need the gradients and their memory can be released. This reduces the memory
footprint required to hold the gradients from 2Ψ bytes to _N_ [2Ψ] _d_ [.]

Effectively this is a Reduce-Scatter operation, where gradients corresponding to different
parameters are reduced to different process. To make this more efficient in practice, we use
a bucketization strategy, where we bucketize all the gradients corresponding to a particular
partition, and perform reduction on the entire bucket at once. This is similar in spirit to how
NVIDIA’s AMP [25] optimizer bucketizes the all-reduce gradient computation to overlap communication and computation. In our case we perform a reduction instead of an all-reduce at the
partition boundaries to reduce memory footprint and overlap computation and communication.
**Memory Savings:** By removing both gradient and optimizer state redundancy, we reduce
the memory footprint further down to 2Ψ + [14Ψ] _N_ _d_ _[≈]_ [2Ψ. As the example in Figure 1, a 7.5 B]

parameter model requires only 16.6 GB of memory using _P_ _os_ + _g_ with 64-way DP ( _N_ _d_ = 64),


10


|DP|7 .5B Model (GB)|Col3|Col4|128B Model (GB)|Col6|Col7|1T Model (GB)|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|DP|P_os_|P_os_+_g_|P_os_+_g_+_p_|P_os_|P_os_+_g_|P_os_+_g_+_p_|P_os_|P_os_+_g_|P_os_+_g_+_p_|
|1<br>4<br>16<br>64<br>256<br>1024|120<br>52.5<br>35.6<br>**31.4**<br>30.4<br>30.1|120<br>41.3<br>**21.6**<br>16.6<br>15.4<br>15.1|120<br>**30**<br>7.5<br>1.88<br>0.47<br>0.12|2048<br>896<br>608<br>536<br>518<br>513|2048<br>704<br>368<br>284<br>263<br>257|2048<br>512<br>128<br>**32**<br>8<br>2|16000<br>7000<br>4750<br>4187<br>4046<br>4011|16000<br>5500<br>2875<br>2218<br>2054<br>2013|16000<br>4000<br>1000<br>250<br>62.5<br>**15.6**|


Table 1: Per-device memory consumption of different optimizations in _ZeRO_ -DP as a function
of DP degree . Bold-faced text are the combinations for which the model can fit into a cluster
of 32GB V100 GPUs.


while requiring 120 GB with standard DP. When _N_ _d_ is large, the memory requirement of model
states reduces from 2Ψ + 14Ψ = 16Ψ bytes to 2Ψ + [14Ψ] _N_ _d_ _[≈]_ [2Ψ bytes, leading to a 8x reduction.]


**5.3** **P** _p_ **: Parameter Partitioning**


Just as with the optimizer states, and the gradients, each process only stores the parameters
corresponding to its partition. When the parameters outside of its partition are required for
forward and backward propagation, they are received from the appropriate data parallel process
through broadcast. While this may seem to incur significant communication overhead at first
glance, we show that this approach only increases the total communication volume of a baseline
DP system to 1 _._ 5x, while enabling memory reduction proportional to _N_ _d_ .
**Memory Savings:** With parameter partitioning, we reduce the memory consumption of
an Ψ parameter model from 16Ψ to 16Ψ _N_ _d_ [.] As the example in Figure 1, a 7.5 B parameter

model requires 1.9 GB of model-state memory using _P_ _os_ + _p_ + _g_ with 64-way DP ( _N_ _d_ = 64), while
requiring 120 GB with standard DP. This has a profound implication: ZeRO _powers DP to fit_
_models with arbitrary size as long as there are sufficient number of devices to share the model_

_states_ .


**5.4** **Implication on Model Size**


The three phases of partitioning _P_ _os_, _P_ _os_ + _g_, and _P_ _os_ + _g_ + _p_ reduces the memory consumption
of each data parallel process on model states by up to 4x, 8x, and _N_ _d_ respectively. Table
1 analyzes model-state memory consumption of a few example models under the 3 stages of
_ZeRO_ -DP optimizations for varying DP degree. Without _ZeRO_, the memory consumption is
equal to the first row in the table, regardless of the DP degree. Note that, with _N_ _d_ = 64,
_ZeRO_ can train models with up to 7.5B, 14B, and 128B parameters using _P_ _os_, _P_ _os_ + _g_, and
_P_ _os_ + _g_ + _p_, respectively. When _N_ _d_ = 1024, _ZeRO_ with all of its optimizations enabled ( _P_ _os_ + _g_ + _p_ )
could train models with 1 Trillion parameters! Or potentially, models with Arbitrary size!
Without _ZeRO_, the largest model DP alone can run has less than 1.5 Billion parameters.


11


### **6 Deep Dive into ZeRO -R**

**6.1** _P_ _a_ **: Partitioned Activation Checkpointing**


As discussed in 4.2, MP by design requires a replication of the activations, resulting in redundant
copies of the activations across model parallel GPUs. _ZeRO_ eliminates this redundancy by
partitioning the activations, and only materializes them in a replicated form one activation
layer at a time, right before the activation is used in computation. More specifically, once the
forward propagation for a layer of a model is computed, the input activations are partitioned
across all the model parallel process, until it is needed again during the backprogation. At this
point, _ZeRO_ uses an all-gather operation to re-materialize a replicated copy of the activations.
We refer to this optimization as _P_ _a_ . It works in conjunction with activation checkpointing

[7], storing partitioned activation checkpoints only instead of replicated copies. Furthermore,
in the case of very large models and very limited device memory, these partitioned activation
checkpoints can also be offloaded to the CPU reducing the activation memory overhead to
nearly zero at an additional communication cost, which we will discuss in 7. We refer to this
as _P_ _a_ + _cpu_ .
**Memory Saving** With partitioned activation checkpointing, _ZeRO_ reduces the activation
footprint by a factor proportional to the MP degree. Consider training a 100B model shown in
Table 4 with a batch size of 32, sequence length of 1024 and a MP degree of 16. If we checkpoint
a single activation for each transformer layer, it would require about 33 GB of memory per GPU
just to store the activation checkpoints. But with P _a_ in _ZeRO_, it can be reduced to about 2 GB
per GPU. Furthermore, this 2GB can be offloaded to the CPU reducing the memory footprint
for activations to nearly zero.


**6.2** **C** _B_ **: Constant Size Buffers**


_ZeRO_ carefully selects the sizes of the temporal-data buffers to balance memory and compute
efficiency. During training, the computational efficiency of some operations can be highly
dependent on the input size, with larger inputs achieving higher efficiency. For example, a
large all-reduce operation achieves much higher bandwidth than a smaller one. Hence, to
get better efficiency, high performance libraries such as NVIDIA Apex or Megatron fuses all
the parameters into a single buffer before applying these operations. However, the memory
overhead of the fused buffers is proportional to the model size, and can become inhibiting. For
example, for a 3B parameter model, a 32-bit fused buffer will require 12 GB of memory. To
address this issue, we simply use a performance-efficient constant-size fused buffer when the
model becomes too large. By doing so, the buffer size does not depend on the model size, and
by keeping the buffer size large enough, we can still achieve good efficiency.


**6.3** **M** _D_ **: Memory Defragmentation**


Memory fragmentation in model training occurs as a result of activation checkpointing and
gradient computation. During the forward propagation with activation checkpointing, only
selected activations are stored for back propagation while most activations are discarded as
they can be recomputed again during the back propagation. This creates an interleaving of
short lived memory (discarded activations) and long lived memory (checkpointed activation),
leading to memory fragmentation. Similarly, during the backward propagation, the parameter
gradients are long lived, while activation gradients and any other buffers required to compute
the parameter gradients are short lived. Once again, this interleaving of short term and long
term memory causes memory fragmentation.


12


|MP|GPUs|Max Theoretical Model Size|Col4|Col5|Col6|Measured Model Size|Col8|
|---|---|---|---|---|---|---|---|
|MP|GPUs|Baseline|P_os_|P_os_+_g_|P_os_+_g_+_p_|Baseline|_ZeRO_-DP(P_os_)|
|1|64|2B|**7.6B**|14.4B|128B|1.3B|**6.2B**|
|2|128|4B|**15.2B**|28.8B|256B|2.5B|**12.5B**|
|4|256|8B|**30.4B**|57.6B|0.5T|5B|**25B**|
|8|512|16B|**60.8B**|115.2B|1T|_10B_|**50B**|
|16|1024|32B|**121.6B**|230.4B|_2T_|20B|**100B**|


Table 2: Maximum model size through memory analysis (left) and the measured model size
when running with _ZeRO-OS_ (right). The measured model size with _P_ _os_ matches the theoretical
maximum, demonstrating that our memory analysis provides realistic upper bounds on model
sizes.


Limited memory fragmentation is generally not an issue, when there is plenty of memory to
spare, but for large model training running with limited memory, memory fragmentation leads
to two issues, i) OOM due to lack of contiguous memory even when there is enough available
memory, ii) poor efficiency as a result of the memory allocator spending significant time to
search for a contiguous piece of memory to satisfy a memory request.
_ZeRO_ does memory defragmentation on-the-fly by pre-allocating contiguous memory chunks
for activation checkpoints and gradients, and copying them over to the pre-allocated memory
as they are produced. M _D_ not only enables _ZeRO_ to train larger models with larger batch
sizes, but also improves efficiency when training with limited memory.

### **7 Communication Analysis of ZeRO -DP**


As _ZeRO_ boosts model size by removing memory redundancy, it is only natural to ask if we are
trading communication volume for memory efficiency. In other words, what is the communication volume of _ZeRO_ -powered DP approach compared to a baseline DP approach? The answer
is in two parts: i) _ZeRO_ -DP incurs no additional communication using _P_ _os_ and _P_ _g_, while enabling up to 8x memory reduction, ii) _ZeRO_ -DP incurs a maximum of 1 _._ 5x communication
when using _P_ _p_ in addition to _P_ _os_ and _P_ _g_, while further reducing the memory footprint by _N_ _d_
times. We present the analysis in this section. We begin by first presenting a brief overview of
the communication volume for standard DP.


**7.1** **Data Parallel Communication Volume**


During data parallel training, gradients across all data parallel processes are averaged at the end
of the backward propagation before computing the updates for the next step. The averaging
is performed using an all-reduce communication collective. For a large model size, the allreduce communication is entirely communication bandwidth bound, and therefore, we limit
our analysis to the total communication volume send to and from each data parallel process.
State-of-art implementation of all-reduce uses a two-step approach, where the first step is
a reduce-scatter operation, which reduces different part of the data on different process. The
next step is an all-gather operation where each process gathers the reduced data on all the
process. The result of these two steps is an all-reduce. Both reduce-scatter and all-gather are
implemented using a pipelined approach, that results in a total data movement of Ψ elements
(for a data with Ψ elements) for each. Therefore, the standard DP incurs 2Ψ data movement
during each training step.


13


**7.2** _**ZeRO**_ **-DP Communication Volume**


**7.2.1** **Communication Volume with** _P_ _os_ + _g_


With gradient partitioning, each process only stores the portion of the gradients, that is required
to update its corresponding parameter partition. As such, instead of an all-reduce, _ZeRO_ only
requires a scatter-reduce operation on the gradients, incurring communication volume of Ψ.
After each process updates the partition of the parameters that it is responsible for, an allgather is performed to collect all the updated parameters from all the data parallel process.
This also incurs a communication volume of Ψ. So the total communication volume per training
step is Ψ + Ψ = 2Ψ, exactly the same as the baseline DP.


**7.2.2** **Communication Volume with** _P_ _os_ + _g_ + _p_


After parameter partitioning, each data parallel process only stores the parameters that it
updates. Therefore, during the forward propagation it needs to receives the parameters for all
the other partitions. However, this can be pipelined to avoid the memory overhead. Before
computing the forward propagation on the part of the model corresponding to a particular
partition, the data parallel process responsible for that partition can broadcast the weights
to all the data parallel processes. Once the forward propagation for that partition is done,
the parameters can be discarded. The total communication volume is thus [Ψ] _[×]_ _N_ _[N]_ _d_ _[d]_ = Ψ. In

other words, we reschedule the parameter all-gather by spreading it across the entire forward
propagation, and discarding the parameters once they have been used. Note however that this
all-gather needs to happen once again for the backward propagation in the reverse order.
The total communication volume is therefore the sum of the communication volumes in
curred by these all-gathers in addition to the communication volume incurred by the reducescatter of the gradients. The total volume is therefore 3Ψ which is 1.5x compared to the
baseline. Both gradient and parameter partitioning leverage the insight that — not all states
of gradients and parameters are needed all the time — to optimize memory by communicating
the states judiciously.

### **8 Communication Analysis of ZeRO -R**


We compare the communication volume of partitioned activation checkpointing ( _P_ _a_ ) in _ZeRO_ -R
with baseline MP, and show that _P_ _a_ incurs a communication volume increase that is in general
less than one tenth of the baseline MP. Furthermore, we analyze the communication overhead
of _P_ _a_ in relation to DP communication volume to identify scenarios when _P_ _a_ improves efficiency
by allowing for a larger batch size and reducing DP communication. We leverage such analysis
to decide if and when to apply _P_ _a_ as well as _P_ _a_ + _cpu_ .
Communication volume trade-off of partitioning activation checkpoints depends on the
model size, checkpointing strategy and the MP strategy. To share concrete insights, we perform the analysis in the context of transformer based models implemented using SOTA MP
approach, Megatron-LM.
In Megatron-LM with activation checkpointing, each transformer block performs two allreduce operations of size _batch × seq_ ~~_l_~~ _ength × hidden_ ~~_d_~~ _im_ in the forward propagation, two
all-reduce for forward re-computation and two more in the backward propagation. The total
communication per block is 12 _× seq_ ~~_l_~~ _ength × hidden_ ~~_d_~~ _im_ since communication volume of an
all-reduce is 2 _× message_ ~~_s_~~ _ize_ .
When _ZeRO_ -R partitions activation checkpoints, it requires an additional all-gather operation before the forward recomputation of the back-propagation on each activation checkpoint. In


14


general, we checkpoint the input activation for each transformer block, requiring one all-gather
per transformer block. The communication overhead _P_ _a_ is therefore _seq_ ~~_l_~~ _ength ∗_ _hidden_ ~~_d_~~ _im_,
since the communication volume of an all-gather is _message_ ~~_s_~~ _ize_ . Therefore, the total communication overhead of _P_ _a_ is less than 10% of the original communication volume for model
parallelism.
When MP is used in conjunction with DP, _P_ _a_ can be used to reduce the data-parallel communication volume by an order of magnitude at the expense of a 10% increase in model-parallel
communication volume, and significantly boost efficiency when data-parallel communication is
a performance bottleneck. Notice that _P_ _a_ reduces the activation memory consumption by the
MP degree allowing for a proportional increase in batch size. For large models, MP can be
as large as 16 (#GPUs on a DGX-2 node), allowing for up to 16x increase in the batch size.
The communication volume of a data-parallel training is inversely proportional to the batch
size. Therefore, an order of magnitude increase in batch size due to _P_ _a_ could result in an
order-of-magnitude decrease in data-parallel communication volume.
Finally if _P_ _a_ + _cpu_ is applied, partitioned activation checkpoints are offloaded to CPU, reducing the activation memory requirement to nearly zero at the expense of 2x added data
movement to and from CPU memory compared to _P_ _a_ . In extreme cases where DP communication volume is the major bottleneck due to a small batch size even with _P_ _a_, _P_ _a_ + _cpu_ can improve
efficiency by increasing the batch size as long as the CPU data transfer overhead is less than
the DP communication volume overhead, which is generally true for small batch sizes.
Given model and hardware characteristics, we leverage the above analysis to decide if and
when to apply _P_ _a_ and _P_ _a_ + _cpu_ .

### **9 Step Towards 1 Trillion Parameters**


The largest published models today are in the range of 10 billion parameters, which are already challenging to train. Getting to a trillion parameters, 3-orders of magnitude larger, will
inevitably happen, but the road will be full of hurdles, surprises and innovations. While we
do not claim knowing or addressing all of them, _ZeRO_ addresses one of the most fundamental
challenges from a system perspective: the ability to fit a model of this scale on current hardware
while allowing it to train with good system scalability.
**A Leap from State-of-Art** The largest model that the state-of-art framework, Megatron,
can train with acceptable throughput is a 16 - 20B parameter model in a DGX-2 system.
Scaling further by having model parallelism across multiple DGX nodes results in significant
efficiency drop due to limited internode bandwidth.
_ZeRO_ vastly increase the efficiently-runnable model size. It enables the current generation
of hardware to run significantly larger models without requiring fine-grained model parallelism
to go across the node boundaries. As demonstrated in Table 1, _ZeRO_, with all optimizations
turned on (P _os_ + _g_ + _p_ ), could fit more than 1 _Trillion_ parameters on 1024 GPUs using DP only.
Alternatively, when combined with model parallelism (as shown in Table 2), _ZeRO_ could fit
more than 1 _Trillion_ parameters on 1024 GPUs with 16-way model parallelism (within each
DGX2 node) and 64-way data parallelism across nodes. Running a model with a trillion parameters efficiently is no longer impossible!
**Compute Power Gap** Training a trillion parameter model end-to-end within an acceptable
time range, however, could still require significant amount of compute power, which is lacking
in today’s AI clusters.
To understand the resource requirement, we present a brief comparison with Bert-Large.
Bert-Large can be trained in 67 minutes on a 1024 GPU DGX-2H cluster [26]. A 1 Trillion
Parameter model can easily contain 3000x (1 trillion / 330 million) more computation than


15


Figure 4: Max model throughput with _ZeRO_ -DP.



Figure 5: SOTA Turing-NLG enabled
by _ZeRO_ .



Figure 6: Max model size



Figure 7: Max cache allocated.



Figure 8: Throughput per
GPU.



.


a Bert-Large model for a data sample. Even if we assume the same sequence length and the
total number of samples required to train the model, training a 1T model would take 140
days, assuming the same hardware and similar computational efficiency. In practice, both
data samples and sequence length are likely to increase with the increased model size requiring
over a year to train. It would require an exa-flop system to train a 1T parameter model in
a reasonable time. But when such compute capacity becomes available, we h