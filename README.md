# Small Language Model (SLM)

This is a small size language model approximately with 40M parameters develped and trained from scratch.

**Some sample generations from this model:**
- Prompt: "One day, Lily met a Shoggoth"

![Lilly met](generated_texts/Oneday_lilly_met.png)

- Prompt: "There was a girl called Rosy"

![there was a girl called rosy](generated_texts/there_was_girl_called_rosy.png)

**This model architecture and components as folows:**
- Custom tokenizer with with 4096 tokens
- 8 layers and 8 heads
- Rotary Positional Embeddings(RoPE)
- RMSNorm
- Multi Head attention (MHA)
- Grouped Query Attentions (GQA)
- Flash Attentions
- SwiGLU

**Training Details:**

- Model is trained usind pytorch distributed data parallel (DDP) on a HPC cluster with single node and multi GPU setup. 
- Dataset is partitioned with distributed sampling.
- Optimizer: adamW with initial learning rate 5e-4, betas (0.9, 0.95), weight decay 0.1
- learnign rate scheduler: consine scheduler with initial learning rate 5e-3 and 1000 warmup steps
- gradient accumalation steps: Optimize for 100k tokens approx. 

**Inference Details:**

This project explores the batch inferencing at scale or faster inferencign techniques to generate as many sequences as possible with in less time.
Code includes two options currently, in both the methods speed of inferencign is measures bases on number of tokens generated per second.

1) Distributed inferencing on singlenode and multi-gpu

In this mode, multiprocessing is used to run the inferencing in the on multiple processes. refer `inference.py` for the implementation details. 
some results of comparing inferencign usign single GPU and mullti gpu is below.

Inference/generation speed with single GPU inference:

![standalone inference](generated_texts/inference_speed_1gpu.png)

Distributed Inference/generation speed with multi GPU:

![distributed inference](generated_texts/inference_speed_distributed.png)

The token generation speed not linearly incresed with number of GPUs due to communicaiton overhead. Further epploring to remove the communication and inproving the inference speed linearly.

2) KV chaching

still under experimantation, expected to post results by May 1st, 2024

