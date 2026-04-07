# **Project Brief: FerrisRes**

## **Objective**

Develop **FerrisRes**, a distributed, high-performance runtime written entirely in Rust to train and infer Small Language Models (SLMs) and Large Language Models (LLMs) on commodity, heterogeneous hardware.

The core architectural feature will be **Block Attention Residuals (Block AttnRes)**, derived from the Kimi team's research. This approach replaces fixed uniform residual accumulation with learned, input-dependent softmax attention over depth. The compute backend will target **Vulkan** to ensure universal GPU/NPU compatibility across diverse consumer hardware (NVIDIA, AMD, Intel, Apple Silicon).

## **1\. Core Architectural Pillars**

### **1.1. The Neural Architecture: Block Attention Residuals**

Standard residual connections suffer from the *PreNorm dilution problem*, where hidden-state magnitudes grow monotonically with depth, diluting the contribution of earlier layers. FerrisRes will implement Block AttnRes to solve this without the extreme memory scaling of Full Attention Residuals:

* **Block Partitioning:** Layers are grouped into ![][image1] blocks. Each block is reduced to a single representation by summing its layer outputs.  
* **Memory Efficiency:** Memory and communication overhead scale at ![][image2] instead of ![][image3], making it viable for commodity hardware clusters with limited VRAM and interconnect bandwidth.  
* **Learned Aggregation:** Each layer uses a layer-specific learnable pseudo-query vector ![][image4] to compute softmax attention over preceding block representations and the local intra-block partial sum.

### **1.2. The Hardware Backend: Vulkan via Rust**

To support heterogeneous hardware environments reliably, the compute stack will bypass proprietary APIs like CUDA in favor of Vulkan.

* **Compute Graph:** Utilizing Rust crates (like vulkano or wgpu transitioning to WebGPU/Vulkan backends) to compile dynamic neural network graphs into Vulkan SPIR-V compute shaders.  
* **Heterogeneous Pipeline Parallelism:** Because commodity hardware varies wildly in TFLOPS and VRAM, pipeline stages will be dynamically partitioned based on device capabilities.

## **2\. Technical Implementation Phases**

### **Phase 1: Vulkan Tensor & Autodiff Foundation**

Before implementing attention residuals, the engine requires a robust, Vulkan-first automatic differentiation and tensor math foundation.

* Implement primitive compute shaders for Matrix Multiplication, RMSNorm, and Softmax.  
* **Crucial Detail:** Ensure all pseudo-query vectors ![][image5] are initialized to zero. This guarantees uniform attention weights at the start of training, reducing to an equal-weight average and preventing training volatility.

### **Phase 2: Training Engine & Cross-Stage Caching**

Training across network-connected commodity hardware (which typically suffers from high latency and low bandwidth) requires pipeline parallelism. Naïvely transmitting the full block history across stages causes massive communication overhead.

* **Implementation:** Build a cross-stage caching system. Blocks received during earlier virtual stages are kept in local memory. Stage transitions only transmit incremental blocks rather than the full history.  
* **Impact:** This reduces peak per-transition communication costs from ![][image6] to ![][image7] (where ![][image8] is total chunks and ![][image9] is physical stages), enabling overlap with computation.

### **Phase 3: Inference Engine & Two-Phase Computation**

For generation on consumer hardware, repeatedly accessing accumulated block representations degrades memory bandwidth and latency. FerrisRes will implement the two-phase computation strategy:

* **Phase 1 (Batched Inter-Block):** Batch the inter-block queries for an entire block. Since queries ![][image5] are decoupled from forward computation, memory access is amortized from ![][image10] reads to 1 per block. Read cost reduces to ![][image11].  
* **Phase 2 (Sequential Intra-Block):** Compute the intra-block attention sequentially using the evolving partial sum ![][image12], merging it with Phase 1 outputs via online softmax.

## **3\. System Design Visualization: Cross-Stage Caching**

Below is a representation of how the **Cross-Stage Caching** mechanism will operate across heterogeneous commodity hardware nodes during pipeline-parallel training.

sequenceDiagram  
    participant Node0 as Node 0 (e.g., AMD RX 7900 XTX)  
    participant Node1 as Node 1 (e.g., NVIDIA RTX 4090\)  
      
    Note over Node0, Node1: Virtual Stage 0 (v=1)  
    Node0-\>\>Node0: Accumulate partial block \[b\_0, b\_1\]  
    Node0-\>\>Node1: Transmit full history \[b\_0, b\_1\]  
    Node1-\>\>Node1: Cache \[b\_0, b\_1\] locally in VRAM  
    Node1-\>\>Node1: Accumulate local block \[b\_2\]  
      
    Note over Node0, Node1: Virtual Stage 1 (v=2)  
    Node0-\>\>Node0: Accumulate new block \[b\_2\]  
    Note right of Node0: Instead of sending \[b\_0, b\_1, b\_2\]...  
    Node0-\>\>Node1: Transmit ONLY incremental \[+b\_2\]  
    Node1-\>\>Node1: Combine with Cache \-\> \[b\_0, b\_1, b\_2\]  


[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAaCAYAAAC6nQw6AAABOElEQVR4Xu2SvUoDURSEQ0DrPMD+I4jbrpWVtsFKGwtB0SBWWgi+RCBFbLSITSK2FoKFpS8g2KppbCwtrPU74Wxy79nkBTQDw97MzJmc/anV/geSJNmP4/jRYdP1oygqjC/ccjMjpGkaU7aO2YY/8Mn1i6JYoGwT/UH9izzPF92MBwI92NfwmvVlcwrvrF4Bw0PCiRYNrE/JJTyxugcGD+i40vOXlDF0WvpZlkVon5OJGZDbYnBXzx3d6rn0+ZMdft9OJmaA0LvclpyDIFjSIuGG+l3sY2/IglAqRUa716LRFlxf4IqbqYDANWwZuY72KmVssse1b/wqCL3xLS1P0c90qwE8tL4HfR4fVhewSQPvW8okZ30PvKkjgjdWLyGfBP7Q6mOEYbiqa49J6bnNCfC2rTbHHH8bv1CDV/3u6RnvAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAaCAYAAADmF08eAAAEFElEQVR4Xu2XXWiWZRjHX92sKWWYLG0f7/3sI9Z2IMIksQ5mFnhgRAhDbRIxiFBwMPCkhpVihCKKBxWCgs7KDzooAkWGJwnhR9JB28Gsg8jEhrIWHY0x5u/ac91v16492yS2deD7h4v7vv//67q/nvvryeWKeMRQU1OzKkmS9Z7PAn4bQwg3Scu8Nleoqqp6jvZWitH2L16fEQSeIvBX7GtsD+XrpG96Pwv0P2i43vMCtH1YTzQm8AWrU/9aq4tZ3SOfz7fi04uNYe3Ckb5cWVlZ5X0zkaQ4R0UdDQ0NT0ae8ltSKennFBeakHHU19cvJe49z0cQ10T8Ge2Y2HmrNzc3L4K7GNuorq5+3eoedXV11bS3Xv1rI0/+kvXLhMwygUNYn9cE8MelYhrY4jW4A+Xl5U943oLYK9jpOFivU8fb8Bc8PxWY3Mfx/81ylEepZ6PlJgCHGmwA+1vyXhdQQYt28mfLV1RULIEbtFwW8BmSVaLtTBqofEna6PT8VKCODVi3477U/pVYvgAdwJ9kS70Wwf6rVD+xDZEn/114uIHullT2EfkRrM3pAyQLLOchyxVbqfnv/ZnAkn5G+sekfWj5ArTz+z1vwb5ZEwdKRa9EnvJ9yt9Y3yzQsbUxj/9Z4i5bnfIZW7aora19Cr2buCOkvaSfkI56PwF8v687Cut0AC96zQJ9Zxwo9qxwfJ3lWj7s/S3ksMqZ1RK3Aelqw+2IeQsGtQzfq0FXDV9xMfm7Eu99BSE91O56XoSD2F+5aZatAJ/zOihZ4uOgc88LR2c+sL4e6K95Tgd6zJQbrR4B/5X6bo0c+aNTDVRXy0jO3w6QVwj8dgLpgM8hHeQgM/p05Kn0Ve1E5teIwO+a54IeSlgX+jteFwQ96bETkaOtMsrDxPxufSPiJEy6U2VWwwzHekhPY/lyHZaP+zaZ5g6VicHnH8/DfayD6Cf+nNcFaDe1/m2Rk/NB+3LSuBaA1i26bpd/QSWdYfrnU6lWfDbnTsWQXkvSkQOWt0B7A5+Lnufehg6jOtjMZQh/T/XClUd9e5Vrx9qampoeczFyC0w+qPgqFRKYZL9pF4b0ku+RQ8CLMmsaW9hrHuifYl2eF+h+kvhbXhOg/ah6fD+XJulzdEzeuSHjcRPSh8mEu76AfPrEk9m7Q0Ufqd2SvQW3yftb5NMjf9jz0hlr8mW9jwDtRtZhJdDXzxfYDyE9J+QrNmJ9WLd9pgpoY7W0RX2tlp8AltIKnDbjtAvbjr3kfbKQpH8tmUtvtqBv2xZDLUjMvRwhZwh9uZ2b4Qb5rygJUy2VeQb9+Al73/OzBvnbYIYTz883GOSwvf7mBCE9Wad9q84l5OeCuX7X87MO+U1jsJ8FfR7OJ2S/hhmeoUUUUUQR/zseANsgMzrXVcVPAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAaCAYAAAD8K6+QAAADr0lEQVR4Xu2Xa2iOYRjHXzM5hEIjOz3PDkyKYs6HzJmUwgeSFB+U1VbKF83Q8GEmh5RDsZIPjoWED1vzQRIl5ZBCkawm2vg4pflde6777d7l2SHbfND7r6vnuf///3Wf3vvwvIlECv8Z8vLypoZhWGL5OOBbGQTBM55DrNZXoP5p8szNzd1POxet3i0kiUreEzeISspPeW62Ph/on7OzswstL0CrIupM3EdKs94YpOG9SzQRHzyuoYOrK4QRrjIj5UVFRSMcT3krFbXxPJOI6UxhYeFI8vZY3oG8yeTf1Do+EWU5OTkzrK8zUPci4gD5tY5jEidQnuf7YsGym4XxO/HaagL489IxGthoNbjqjIyM4Zb3Qe4jyScqrdYTkFdBbDfcY7/8BzDkEV+IH/JudYHMmnbspc9nZmYOg2v2uTjg+Sn5/FILrdYTkFufn58/0XC/4iY6Ce1wE6/pVnPgp89Sn8QSx/N+p4cDk7wHlu8MeOfJci0uLh4URgdTW4znFNEadnZgaaOHLO9DGlGf7JOljqf8jfIt3xuHnrThgG8dUU/UEm+Iy5JvfW7vB95EJwE5V8UuNyJ6qfokxguXlZU1RsvHrN+HzKj42McrrGaBd5t4eR0gZd53u3aNVQa2QLVSq0niEaIl0cUyFOC5ppXIkm0HnZgkHA3s870WeBYTP2U/Ws2H3JvaxnPHeZPnjvok2B5TVNtrNWn0IR28bXkfeI5qBc1UNtrxDGiZ8OTv9P0WeB4QVZYXwE+XpUwdobYhB0ympx9UvsOJKOCayRCN/LNWk1k/h3jP8j6C6LSUCsp93u27sIs7TK4BPK3kLreaIIj20XoNGcArozcIH3f5yympOTVWk4HtQnhneQ/pOqgrCV33DkF0TcjAqn3eB/pq8fgXvgN1FqM18pouk6advOB0vUpaA12GtLMomRyV52hOhc+3Q3527VyJ1RLRp8sloo4ZG2pF+eLQ3HNWcwiiPfzHxqfdAviP5J5Qn5yE0snDzsP7Bq2//buQwW9xmgB+leqbfD4JPTa/Eo2YDmi8hX8Ct8b6feA5jqfV8trJ7uJ9wlsFMrlB1I8a4joHRzbPw9IXKXvVt0M4tBeW7wBOpHEY19PRMpkZYr71xCHs5PL8W7AKBvNrzvS5goKCsXHflrTbEnZzcPUGAwPzmfWvQLvN3V0hvQKzuZaZCy3f32BgRy3X5wii/1cdTs3+BFtlBzHK8n0Ova9OB/q51Z9gcZwkZls+hRRSSKFX+A3ExB3Vf9AljwAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAaCAYAAADovjFxAAADhElEQVR4Xu2XW0iUQRTHNzW6EEXUluxt1l27UBGFdhEMspdKHyKCegh6KqiIyMAyKNHoCgZRYQ89Rb1UkGWCQkFBVkSgWA9hQUVBdqOLQiA9bL/jN6uzk+7FhNh2//BnzpwzZy5nz5z51uXKMLjd7klKqScwYtsyCsFgcHw2CMFgHUF4ZuvTAmy8jQM8DQQCb5F/wC/wFeyED2BVKBQK2H4CfOZjb6A9TvuIeU7bY9IGHGAzB6gWmQO1+nw+bzgcnoFuObZdEhT0NaYP/RD6DyL7/f4VchXgBnNMWoHNV8AjWj5l2jjsvsLCQjcBuQkro3rkK4xt0fIaCYLX65026PlvMIYN72czh0zag4aCDsJRLccEgf41aQsKChDVG5H1axCBO6XPuseQ2w23QWAs5T5NsfUmPB7PRCLos/XJgI3NYfFOfok9ti0VMEe5HETLdibckJYrMgFbm8jseTpyH+sv5CosQf7FuI3yI5i+LkkhjBfgT1gV1SOv10WohHvnp30Pv0t0Tf9koJy3ucLWpwqCuDpOEGqkPqBvtK5DJWxSTsbdgZdgvekrk12mydVBGEgV5BblpFIJUVxA2yt9iajhnhTwO2PrRgIJgjKug/wg7KeYAOyG59G9gw2Yc00/I4NzGDfPtEm1LMapmclX6gM3a1Oecp6hnqKiorGiYMx2HYQw7UEdlP6qmwhyHWzdSGAFQZ7Hl7ADfa1yMneq7ZM0mOAz7ONqTJY+k1bLgWkPGGPKYKPR7/rjbg2NHMbeTkTbaSgwbi08rOX+6yDprpxUvw7LYz1SgHKyoL+6Cjhcq+hYYFlUh1yHfpuW87VPSdQeD6P1JCnndYgJgsu5yvfIUA9t13AfTAmhD7zD6H+DPYg50pcrQf+5FFHpE4wt9HsR86I+8aCMgvs3kCDIj6HlgcKIbi5NLu069HddSe4rBjoIm8w+fB3tB516cNawy2tyK9pPBMZ2j0Zd0EGo1bL9nbBXt/VK142UgFMV/MQCJ+RwQacAdcOP8Bxv7yxrfIRDLTV1iYBPGfO+gBeRF8uzS2aNs8cNB3zuwz7l/FeQOhKBHXCr2PUL1i42MvUx65y050gISXkcFzFBqfTlo0O+tYX2WBb66nIK3irbFg9SvXVWPZSAKOcFksOk319bDtBEAZqt9Hd8RkLSjQBc5UrMtG0ZAzIhP6MDkEUWWWSRxX+O3x3x/hGB5XRtAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAbCAYAAACeA7ShAAABVUlEQVR4Xu2TPy8EURTF187iE4jC/M1MIVOIZAhRyX4GhUQnEpL1BUREpdCoJBoJEgmhUItEqaKi04mgFI2Igt+1702uZxsK1Zzk5t5zzn33vjfZrdUqVPgn+L4/EMfxXhRFL2EYTlkdbR3thrIunLolWnmwE2g6TpKkn/xB7Cv9lXhQfI1lp5b/AA0TxAExY4ZtKE/4kWr3GHZo6i68c+kpXRkWBMEI+UQMnjGmPBnWKpvb2rKidfiz4m0gvhG3jibDhx2taWvxvt3MwhzUH7eB9qj4l1YURbcl+Is8+043WEOGjSveJLadnguHPxGzWrOGDFsw1IOfEZfUDRHwpuGbth8+KGf43qnVSpjf1D3XXpUh5C3yO3FFvUNeyfO8x/bD5zo+0SLLsl42jUoY3seSSbmF28ugXVmSpmnger8GN7tmyTyx5Hp/Av+aIZLn6hUqOPgEoD9T7yp+qkoAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAaCAYAAADIUm6MAAADP0lEQVR4Xu2WXWjNcRjH9+ad5OV0dHZ23jp12i6QJdm8U8pceEtppawQF0MhynsukJZxw5XCSCYvmeVK8hLhQm6YySJaLExRLtZ8nu35r5/H/9jGudjF+dTT/+z7fX7P8/x/+/9/5+TkZBlgxGKxycQcq/sRjUYXEU/IH2o9Q24ymQxI3UgkknAN1pfpdR7x1PX6BIvOEK+IS8Ru4iFNVto8F/y38Xg8anUP/DHEXsmjXifxgGghGtHKuZkhfK7y8rmx/aWlpYPcGmlhYZwFFylUnUqlRjl6lTTDq3XzPXSo7VZ3YX2zDlxDFDt6hXoN4XA46SwpQNvm/O0PQ00j8Svx3HoC+lltvNTHOxoIBEZa3QN/GfG6qKhopvUEeu+Q2lZHa+Nmxlq9BxLixEeinSIx6wvs6Hwd/LdnLxQKDUf77GourFsh63iMUtbzwC9OM3gL4xyzeg86UCsfC6znoTcneZ0MM8PRb6QbHL1e8ml+zXou7Oowv8FZt0pvepb1hFwd6IA1XOQFSjN4Gw2uu7mqT/Hy8Rda30VeTGqus7o889pvj/WkQZk2mG49F5pv9gYJBoMjROMxGa9ajU/+YfW+8Wee9ftIntY4bw0Z/AjxhY/51nMh54oWeeNoXc+mHHNuroB2S/NvW68/sL5NalldjLsYV63uwu7V6hAf3GOSdQtEx9/g5guaL95667nIlxF596zugfcs6nfSUfgURoPVXfC/+w3B8TZV9Z2uLqC36/C7rOeCvzX6l/Ma7x1x3+oy+BaiyeoeJSUlg3WAOuux4wkd/JCPd1rX1VvPI5FIRPBfFBYWjrOeB34HcdPqsmshbT7bejndL8cFolHefGvSeLSsZciT1pN6OrgcZxOtr30fE3Ot5yGHgNY4br0uaLwa8xPxnob7NJqi3b8nKmy+C/4J4qfVPfA2Eq30uMP1IFFHNLPLYZtrYc0mcju48aD1ehCTpOUkV3OtjPZyPHqQv5jcP748XPQ/IzcgP9gqveO0N8i9TJyzeqbIp/hLK2YC6v5gY8qtnjF4rJYQMav/Lwz9yGoZh91p5JJr9X+FR3eS30udceRnrZwubPwE6/UX6qylzhqrZ8mSZYDzC6Su7WRjCKnfAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAaCAYAAADIUm6MAAADE0lEQVR4Xu2WW4iMYRjHZ+065BjZhml3v5lpajIKNQ45hJZSViKlpNQqyc1aF8TFslwhyXJjc+csySFsuBRRFLmyFBGJzeHCbtS2fo95Xz37zLd2tp2LvZh//ZuZ//95nvd5D9/7TSRSwhBDPB6fBZdaPQxBEKyET4gfZb2Bgjq18KnV+wVJp+EreBk2wUc1NTXrbZwG/rtEIhFYXcBkdlPjXhjx7sLW6urq2SZnfzabHa61PkGhBAmXaKIhnU6PU/pm2IPXouM9iJ8Id1ndAy8DL0oN+BzWyk46NsJ2tN8mrQJtp9HyQfI8Ar/DF9YToJ9xA68N8Y5UVlaOtbpGkFthyW+yHpNa7rypWud3R1VV1SSt9QIBCfgZ/mACcesLKL7MFe919mKx2Gi0r1oLAzFdkk/5Jdaj9nTxOC6rtY72lvhjWusF19AnvlZYz8NNTuJ6GGiR0m8W2Ljk3bG6AO+U+Hwt0zpNbxCdZ2ex1j3KXEMHrKHBoAv7aLyDAW7o2DBIHnHNVpcdlonDTutxTFJuvL3Wk4ILXEPzradB/UbfeDQaHSMax2Sy047aeINyF1frhVQqNZ7fdfA1fEZzWZ3gMMzlnbeGNH4YfuNrufU0iLnqirxR2jTRGHSfjrVgq+e6XM0uFuM6rM9kMiNsjgdxHaFHDOM+xjWra1C8xQ32UV+T/jbA36bjLYi5TcwhqxeCIHd95t90FGzFuGV1DfyfrsGtWucWmOP0PVrXkJeI5BOzwnqFgNz38IHVpfEdsN3qHrKNbrXPWY8VT7rGD1rPQ24EifHPxUBBbrfsmNVl1WJu8Lz7NZJ7OC7ANh6mkdZMJpMTJJcJnLSeh5x/ibF6IZDJukU7br2/oPgmzC/wAxNodpTX8ENYZ+M18E/AXyG6DGiZfzv8B/S1nZxudi1qvX8Qk6B1BDfwuTHo53r0IH6VNGX1YoC6V+BZqxcLcke/tGIxQN1OefFZvWjgWK2BcasPFjT92GpFB6vTFjH/NQYDju5MOMPqRYf8rZXbhYWfYr2BgjpbqFNv9RJKKGGI4w8Tl+oR+vfqqAAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAbCAYAAACjkdXHAAABIUlEQVR4XmNgGD5AQUFBQE5Orl5eXv4xEH8E4iNA/ByIF8vKyioD5R4pKyuLoetjAErqABXdAOL/QNwDFGKEyQH5OUB8FoivIWmBAKCN9kCJd0B8H2iILbo8CADl1gBtnoEuDpL4CcSfZWRkdNHlYABoQQJQTTSKIFCDEMipQFNDUCTQAFDeBahWGi4ACiCgxvdAnISkDisAalRFEQBqjgDZqqSkxI8iQQwAOqUdpBldnCgA1LiNEs1XiNEMVOMExH0ogkA/7yJS8yIgdkQXlJSHxLEnigQUAMNEGyh3AYgV0eXAABriz0GpDE08HYgvAeWskMUxAFDBPZDz5SEZoQWIlwHxZFACQleLAYA2cACxBxBnAjXVAulIdDWjYBSQBACyv0TlfdfnoQAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAbCAYAAACjkdXHAAAA+klEQVR4XmNgGPpAXl4+BYh348JycnI7gHQXuj4wkJWVVVZQUHAAKlgMxP+B+AyIDxVzAmquB9LvgHwNdL1wAFSwGaQZqLgci1wVEB9FF4cBJqDke6jNVuiS8hCv/QcymdHlGGRkZHShGt8BuSzo8kDXtAPl3qCLg4E81L9AfzVikXMCyamrq/Oiy4EBUPI+SIGioqIbTAxomyAQx4JcA8RbkNXDgZKSkhzUyej4E9AlC4Gx4YeuBw6ApsdBFV9ElyMIgJpmQTVPRpcjCICaroM0A10Qgi5HEJDrZBagJkOo5m3S0tLC6AqwAqBCGagmdLwUXe0oGAX0BQBWSVIl55U0vAAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAaCAYAAABsONZfAAABH0lEQVR4Xu2Rv0rDUBTGo5NTcahgQxITkkewOnVx0LXUoVuXUuzaUXBxVClFHNyK2Fco2LXdWnB1civU0Ueov5MmcO/B7EL94OOQ78+9J4nj/An4vl9LkqSUP0dRdGL6FjCPwCu8D4LghfkWhuEpnOlsCgJn8JvAY65xY4z2CftmNkUcx74UOH3C447pobc5qG5qKRDvMNesd649bqvi72tdTvuQErz5xatoLQXGMCsJl5zcLQznICCfbWUUhfKOHZ21wH85IPiky9x6obOFOxPuZcV37Tms0NKagK92nJXG2pObGloTiJ6td2UZnuclGA+WuMEuG0zhnNKe5VAYwUv4hbkg9My8ZQ6Yh1Y4B0ZTpuu6ZSnLy8NrnfvHFuAH6ahIt/FexjMAAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFMAAAAaCAYAAADL5WCkAAAEDElEQVR4Xu2XS6iNURTHj4sI5Xld3dc+9+Ed0SUTIgOPJO9H8oikSDIw8MozKSTkWZ4ZKEZIKdyBhIlEblwTE+QxYGIg6fqt86197W855zLxOUfnV6vz7f9a+9t7r7O/9e0vlSpSpCBoaGjo6Jy7Y/VCorq6ens6nb5g9bYoLy/vUlVVNS7UuMeI+vr6TqGWDfI10moZmMg5nMetnq9UVlZWWA1KWEOjJNU6LCSwnNjT2APsAHaLJE6pqakZyHUzISW2jyJj3MDeYa+sUzK8BvtQWlrazfryDXZSH5K1l4XPtD6BJPdnLd+sbpH1Yme47CBt7teP9hPsPXbFhMcgdjy2g7izMQeD90L8gm2JOfIM5ncfa2ERL/l9nSuZAv6LVgth95URcy+lifSgTZMx+LPWhXo2JF/YipjIpDYifuK3R8yRxzDfW9hsq3tIxlDWs8DqHikD2FyrS+nQP2yE9VmIu11bWzvAii3Y2pioSBHmxp1DjRt0T+WuJ4nwu2QK+L/auXvwPcL2Wb2srKxrGzu+HXV2lLyoiZksebMBmWQSNNrqUqDpdA3/ex6L4fxr9bSviqF95ne17ZMUf5hM2SQTrS6gH1O/vETOuBybyYN/FvYCO4s9xy5JfxuXGZQt3zOLfgSbqoM+xh46PQrw+wl7bfskhYuSOcvqITrvNVYXKioqKtXfamyOTTZOQF8ufnJ0Utpcb/B9bGxm0JR5bKV+uuh44Cf1lt3pvF982q99a6cEcVEy51g9ROe91eoeFx2HXmmct9gLRZ5I1R+ndK38Eb1Vy3osemu1EOlYV1dXZTUSfijULC56NGTRv7U/OReGaJ95Vg+ROfrd5JEdKeUr1DwSj73LokkZbO1De7fq8Te5OnOeyeRMh7/J6nqzrPUoCVyUzPlWD9E57g81+pxAWxlqHo2XHWi1Z0ZrFF3eIaHunS25DuvsvhnOfBVpCXjDZXuuOzPBpaE/CSSZ6TaOPoImInZ2dtHLY1GoeSSee+60mosO9hnk05P2V6ePOPHjf0anog72MfbgO2gnTfKWoB+Va3zLaG8L/UmgyVxo9RBNTmuM1HxNzpEwzoPebF/EGr8naM/R+2a+/4lf/DM6CviO7YqJCvpnq1E/6rjJTXxNbPVh1v+3YLwrurhs9kvJYcFPwzZzPk/cZX3a5AvqlNRqrh/JdRjrQZ+A/yO2H7uup4A96egr7NdPThx3cTRaXWCwIVYTZBCpp1bPI0rS5hxMe4EcuOVadiDt9ax7c9baFyAfLnIOJ36Q13iS+8oBPozLwA0XYd8ITltfoUKypkt9s3oiyFa3x4hChvU8tFpiMPhg7AMJHWt9hQZrWOXMceafwBtvktUKCSrVYWyM1YsUKVKkSJH/kh9ZDy72XvqJDgAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB8AAAAaCAYAAABPY4eKAAACE0lEQVR4Xu2WP2hTURTGk0CLllJFfYhJXl7+DRrEJV0sFCR0KAiCxcWmoHVoNwcdXK0k6KSlXToodOnQqQhZtBT/jOKk0EFE5y6lIB0SEPs78l45PbT6St7b8sHHvfc7557v/snrbSJxDGSz2bLVwqBQKFzxPK9m9dBwHGcwn8/ftfr/gOkGbDN3zsY0ksViMUdiwQa6BTV/wIbV/8J13csEd+EfuGDCqVwu9wJ9zeih8U/zcrk8RLDlm9d1jPEE5jO0k1o/DnzzJ1bfB3fyUcxp81rnVG5UKpV+s/MUadeOYjqdPqdyA/Om1g6AYAf+tLqAk3GI/Q7GFB+whposOK3nhzGXXd/hs/CkAOMLKraB9pj2gZ4TFmLO1T21+j7EnIRl+Ib+ItwOYjIR81U5fj0nDKizLrXhlvSpNXUggcInfPNHgcb4Ibypci4G/UiB6RhGLa0xrslJaC0WYPQeXjJaU75vrcWBJEa/rIj2Dt6zeqTgsTgp9220M2gd7vm01mMBRptm3LALig2lUslll58wfEn7DU6rsPw1ewbfytcAl8hbob2lcroDBUe8Q140TG/DcWKv4HU/tw6/29zIgclEtVrtYwFfAo1d30f/rPNiQyaTOat/AyzkNePnOic2yHHDHen7j0ybx2OYE6ja3MiByQd2Oyt9jK/Cr4znWcCozY0c9onk5TvPv1yntNZDD2GwB/1WgMOAPKLXAAAAAElFTkSuQmCC>