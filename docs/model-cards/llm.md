# MiniMind-Lab LLM Model Card

## Model

Dense decoder-only causal Transformer with 8 layers, hidden size 768, 8 query heads, 4 KV heads,
SwiGLU, RoPE, Pre-RMSNorm and tied input/output embeddings. The 6,400-token MiniMind BPE vocabulary
gives 63,912,192 trainable parameters.

## Training

The complete language model is initialized randomly and pretrained on the pinned MiniMind text corpus,
then instruction-tuned with assistant-only labels. Formal MPS configuration and seed are in
`configs/llm/pretrain-mps.yaml` and `configs/llm/sft-mps.yaml`.

## Evaluation and limitations

Final validation loss, perplexity, bits-per-byte, throughput and fixed generation samples are summarized in
`reports/final-results.md` after the run. This is a compact educational model, not a production assistant;
it can hallucinate, repeat text, produce unsafe content, and has not received a safety-alignment certification.

Training data and generated outputs can carry biases from their public sources. The checkpoint must not be
used for medical, legal, financial, security-critical, or other high-stakes decisions.
