# Small Language Model (SLM)

This is a small size language model approximately with 40M parameters develped and trained from scratch.

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

Model is trained on a HPC cluster with single node and multi GPU setup. 
