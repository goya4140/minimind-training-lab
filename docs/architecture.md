# 模型架构

## 1. LLM

主线为约 64M 参数的 Dense Decoder-only Transformer：8 层、hidden size 768、8 个 Query head、4 个 KV head，使用 Pre-RMSNorm、GQA、RoPE、SwiGLU 和共享词嵌入。

正式模型词表为 MiniMind BPE 6400。仓库中的 `ByteTokenizer` 仅用于无外部依赖的流水线冒烟测试，不是正式训练架构。

## 2. VLM

VLM 在 LLM 上增加冻结的 SigLIP2 Base P32 视觉编码器。256×256 图像被编码为 64 个 patch token，经 `LayerNorm → Linear → GELU → Linear` 投影到 768 维，并替换文本序列中的 `<|image_pad|>` 占位 embedding。

视觉特征和文本 embedding 随后进入同一个 causal Transformer，不增加 cross-attention。

## 3. Omni

Omni 使用 Thinker–Talker 双路径：

- Thinker：复用 LLM，接收文本、SenseVoice 音频特征和 SigLIP2 图像特征；
- Bridge：默认提取 Thinker 中间层 hidden state；
- Talker：4 层 Decoder，以 Bridge 语义状态和历史 Mimi codes 为条件；
- MTP heads：同步预测 8 路 Mimi codebook；
- Mimi decoder：将离散 code 增量还原为 24 kHz 波形；
- CAM++ speaker embedding：提供音色条件。

SenseVoice、SigLIP2、Mimi 始终冻结。可训练主体约 113M，但运行时还需加载约 425M 的冻结外部模块。

