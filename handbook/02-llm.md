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

## Teacher forcing：训练时看到的是谁的前文

训练用真实序列一次并行计算多个位置的预测。假设样本符号序列为
`[BOS, 问题, assistant前缀, 答案A, 答案B, EOS]`（仅作教学示意，不是实际分词结果）：
预测“答案B”时，模型看到的是数据中的正确“答案A”，而不是自己刚才生成的一个可能错误的 token。
因果 attention 保证它仍看不到“答案B”及之后的内容，不是把未来答案泄漏给当前位置。

源码将 `logits[:, :-1]` 与 `labels[:, 1:]` 对齐。因此最后一个 prompt 位置的 logits 预测第一个
assistant token，回答中前一个位置预测后一个 token；不是让位置 t 直接抄输入的 token t。
`-100` 只是 labels 的忽略标记，不是加入词表或输入 Embedding 的 token ID。

SFT 不计算 user/prompt 的直接预测损失，但回答仍能通过注意力依赖它们。图像、视频位置同理：
这些输入可以影响答案，不需要作为目标 token 计分。“没有 label loss”不等于“没有参与计算”。

单个微批的交叉熵可写为 `L = -Σ log p(正确目标token | 前文) / 有效目标token数`。
有效目标仅包含 shift 后且 label 不为 -100 的位置，不能把 padding 或 user token 计入分母。
推理则逐 token 使用自己的输出继续生成，前面一旦答偏就可能进入复读或错误轨迹。
所以 teacher-forced 验证 loss 低，并不保证开放生成正确、简短或及时停止。

## Pretrain 与 SFT

Pretrain 从随机权重开始，让模型获得语言统计规律。SFT 从 pretrain checkpoint 初始化，用对话模板
训练 assistant 回答。两阶段分开有助于观察：“会续写文本”和“会按角色回答”不是同一种能力。

正式配置：

- 当前 CUDA：[pretrain-cuda.yaml](../configs/llm/pretrain-cuda.yaml)、[sft-cuda.yaml](../configs/llm/sft-cuda.yaml)。
- 以下 MPS 配置用于历史本机实验，不作为当前 CUDA 训练证据：
- [`configs/llm/pretrain-mps.yaml`](../configs/llm/pretrain-mps.yaml)
- [`configs/llm/sft-mps.yaml`](../configs/llm/sft-mps.yaml)

## 生成

生成是循环执行 next-token prediction：取最后位置 logits，选择一个 token，追加到序列，再继续。
温度为 0 时使用 argmax；温度大于 0 时按概率采样。本项目固定评估用温度 0 来控制解码规则，
但这不消除 CUDA 算法、精度或环境变化带来的差异，不承诺跨设备逐位一致。
当前实现没有 KV cache，每个新 token 都重算上下文；不能把它的推理速度与优化引擎等同。
原 raw-prefix 固定生成与单独冻结的 chat-template 诊断在提示格式和 EOS 处理上不同，结果分开解释。

## 用真实结果检验这些概念

[CUDA LLM 两阶段总览](../reports/cuda-llm-final-overview.md)中，SFT 自身 assistant-only 验证 loss
为 0.84844，但相同通用文本留出的 PPL 从 6.22272 升到 6.77964。不同数据、监督位置和统计口径
回答的是不同问题，不能把两列 loss 直接相减成“能力收益”。
补充对话诊断中，首都题要求仅输出城市名，SFT 实际回答“中国的首都是北京。”：事实内容正确，
输出约束仍失败。这正是语言建模、事实知识、指令遵循不能用单个 loss 互相代替的例子。
完整[逐题结果与真实截图](../reports/cuda-llm-final-chat-diagnostic.md)同时展示，不临时放宽评分。

## 学习检查点

运行：

```bash
uv run pytest tests/test_llm.py
```

然后回答：为什么 tied embedding 会减少参数？为什么 labels 要 shift？为什么 validation perplexity
比训练末尾的单个 batch loss 更可信？
