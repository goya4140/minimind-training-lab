# 10 · 代码导览

## 先看目录职责

```text
configs/                 可复现超参数，不包含代码逻辑
src/minimind_lab/        模型、数据、训练公共能力
scripts/                 可执行训练、评估、下载和报告入口
tests/                   小而确定的结构与语义门禁
artifacts/               本机 checkpoint/原始日志/评估，Git 忽略
reports/                 提炼后可提交 GitHub 的实验结果
handbook/                面向初学者的解释与练习
```

## LLM 调用链

```text
scripts/train_pretrain.py
  ├─ load_config
  ├─ JsonlPretrainDataset.__getitem__
  ├─ DeterministicBatchStream.batch
  ├─ MiniMindForCausalLM.forward
  │    ├─ embed_tokens
  │    ├─ DecoderBlock × 8
  │    │    ├─ GroupedQueryAttention + RoPE
  │    │    └─ SwiGLU
  │    └─ shifted_cross_entropy
  ├─ loss.backward
  ├─ clip_grad_norm + AdamW.step
  └─ atomic checkpoint + JSON report
```

先读 [`src/minimind_lab/llm/model.py`](../src/minimind_lab/llm/model.py)，再读训练循环。这样不会被
checkpoint、日志等工程代码遮住 Transformer 主体。

## VLM 调用链

```text
scripts/train_vlm.py
  ├─ ParquetVLMDataset
  │    ├─ chat template + assistant labels
  │    └─ SiglipImageProcessor
  ├─ MiniMindVLM.forward
  │    ├─ frozen SigLIP2
  │    ├─ VisionProjector
  │    ├─ strict placeholder injection
  │    └─ MiniMindForCausalLM.forward_from_embeddings
  └─ projector-only / boundary-layer optimizer
```

重点比较 `set_alignment_trainable` 与 `set_instruction_trainable`，它们是“阶段策略”落实到具体参数的
地方。

## Video-Omni 调用链

```text
scripts/cache_video_features.py
  └─ MP4 → 8 frames → frozen SigLIP2 → local FP16 patch cache

scripts/train_video_omni.py
  ├─ QIVDVideoDataset → cached [8, 64, 768]
  ├─ MiniMindVideoOmni.forward_with_patch_features
  │    ├─ spatial query attention
  │    ├─ frame positions + temporal Transformer
  │    ├─ learned-query resampler → 16 tokens
  │    ├─ video projector
  │    └─ MiniMind LLM + assistant-only loss
  └─ temporal-alignment / video-instruction optimizer
```

最终 [`scripts/evaluate_video_omni.py`](../scripts/evaluate_video_omni.py) 不读取训练 feature cache，
而是重新从 MP4 解码，从而验证完整输入链。

## 修改实验时只改一层

| 想研究的问题 | 首选修改位置 | 必须保持不变的对照 |
|---|---|---|
| 学习率影响 | 对应 YAML `learning_rate` | 数据 split、seed、模型 |
| 帧数影响 | Video config + cache 名称 | 训练样本和 temporal layers |
| Projector 结构 | `VisionProjector` | frozen encoder 与 LLM checkpoint |
| 时间层数 | `temporal_layers` | 帧采样、训练步数 |
| Label mask | data module + tests | 模型架构 |

一次同时修改多个变量虽然可能得到更好结果，却不能告诉你是哪项改变起作用。

## 阅读测试的顺序

1. `test_llm.py`：基础张量与 causal model；
2. `test_sft_data.py`：监督 token 边界；
3. `test_vlm.py`：视觉注入与冻结；
4. `test_video_omni.py`：空间/时间 adapter 与 feature cache 等价性；
5. `test_temporal_benchmark.py`：生成视频的 ground truth；
6. `test_reporting.py`：最终展示只接受有限、合法指标。

测试通过表示这些约束在小输入上成立；正式训练结果仍必须由真实数据和 held-out evaluation 证明。
