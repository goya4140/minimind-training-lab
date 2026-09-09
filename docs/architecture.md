# 模型架构

## 1. LLM

主线为约 64M 参数的 Dense Decoder-only Transformer：8 层、hidden size 768、8 个 Query head、4 个 KV head，使用 Pre-RMSNorm、GQA、RoPE、SwiGLU 和共享词嵌入。

正式模型词表为 MiniMind BPE 6400。仓库中的 `ByteTokenizer` 仅用于无外部依赖的流水线冒烟测试，不是正式训练架构。

## 2. VLM

VLM 在 LLM 上增加冻结的 SigLIP2 Base P32 视觉编码器。256×256 图像被编码为 64 个 patch token，经 `LayerNorm → Linear → GELU → Linear` 投影到 768 维，并替换文本序列中的 `<|image_pad|>` 占位 embedding。

视觉特征和文本 embedding 随后进入同一个 causal Transformer，不增加 cross-attention。
仓库实现位于 `src/minimind_lab/vlm/model.py`：既支持只训练 projector 的 alignment 阶段，
也支持解冻 LLM 首尾边界层的 SFT 阶段。

## 3. Video-Omni（Video → Text）

第三个模型关注视频理解，而不是语音生成。输入是均匀采样的 8 帧视频和一个文本问题，输出是文本答案：

```text
video.mp4
  → uniform frame sampler (8 frames, 保留首尾)
  → frozen SigLIP2 (per-frame patch features)
  → learned spatial query-attention (1 vector / frame)
  → learned frame-position embeddings
  → 2-layer temporal Transformer
  → 16 learned-query temporal tokens
  → MLP projector (768 → 768)
  → 替换 16 个 <|video_pad|> embedding
  → MiniMind causal LLM
  → text answer
```

每帧的 learned spatial query 会对全部 patch token 做 attention，避免简单均值过早抹去物体位置。
`TemporalVideoAdapter` 显式加入帧位置，因此交换帧序会改变视频表示；learned-query resampler
把可变的帧语义压缩为固定 16 个视频 token。视觉编码器始终冻结，第一阶段只训练 temporal adapter
和 projector；第二阶段额外解冻 LLM 第一层与最后一层。该设计让三条路线共享同一个语言主干，
同时能用“倒序帧”消融检验模型是否真正使用时间顺序。

正式配置共 178,569,984 参数：SigLIP2 94,552,320（冻结）、LLM 63,912,192、包含空间汇聚的
temporal adapter 18,922,752、projector 1,182,720。Alignment 阶段可训练 20,105,472 参数；
SFT 阶段加上 LLM 首尾层后可训练 34,854,528 参数。

这里的“从头训练”指 MiniMind LLM 主干及新增的多模态/时序模块从随机初始化训练；SigLIP2
作为明确标注的冻结感知器使用公开预训练权重。以 2,900 条视频从零训练视觉 backbone 不足以形成
有意义的视觉表征，因此不把它伪装成本项目已经完成的目标。
