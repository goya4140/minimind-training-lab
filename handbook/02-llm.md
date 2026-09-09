# 02 · LLM：从随机参数到文本生成

## 架构

正式 LLM 是 8 层 dense decoder-only Transformer：

| 部件 | 配置 | 作用 |
|---|---:|---|
| Vocabulary | 6,400 | token ID 空间 |
| Hidden size | 768 | 每个 token 的向量宽度 |
| Query / KV heads | 8 / 4 | GQA，减少 KV 计算与参数 |
| Intermediate size | 2,432 | SwiGLU 前馈层宽度 |
| Layers | 8 | 重复的 attention + MLP block |
| Parameters | 63,912,192 | 含共享 embedding / LM head |

每层采用 Pre-RMSNorm：先归一化再进入 attention/MLP；RoPE 把位置信息旋转进 query/key；causal
mask 确保当前位置看不到未来 token。实现见 [`src/minimind_lab/llm/model.py`](../src/minimind_lab/llm/model.py)。

## 一次 forward 发生什么

```text
input_ids [B, T]
  → embedding [B, T, 768]
  → 8 × (GQA attention + SwiGLU MLP)
  → RMSNorm
  → logits [B, T, 6400]
  → shifted cross-entropy
```

loss 是所有有效 label 的平均负对数似然。它下降表示模型更擅长预测训练分布的 token，但不自动代表
事实正确、会遵循指令或安全。

## Pretrain 与 SFT

Pretrain 从随机权重开始，让模型获得语言统计规律。SFT 从 pretrain checkpoint 初始化，用对话模板
训练 assistant 回答。两阶段分开有助于观察：“会续写文本”和“会按角色回答”不是同一种能力。

正式配置：

- [`configs/llm/pretrain-mps.yaml`](../configs/llm/pretrain-mps.yaml)
- [`configs/llm/sft-mps.yaml`](../configs/llm/sft-mps.yaml)

## 生成

生成是循环执行 next-token prediction：取最后位置 logits，选择一个 token，追加到序列，再继续。
温度为 0 时使用 argmax，可重复；温度大于 0 时按概率采样，多样但不稳定。本项目固定评估用温度 0，
确保不同时间运行可以直接比较。

## 学习检查点

运行：

```bash
uv run pytest tests/test_llm.py
```

然后回答：为什么 tied embedding 会减少参数？为什么 labels 要 shift？为什么 validation perplexity
比训练末尾的单个 batch loss 更可信？
