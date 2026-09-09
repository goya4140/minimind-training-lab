# LLM BPE Mini 001

## 结论

固定的 MiniMind BPE 6400 Tokenizer 已接入，并完成随机初始化、训练、checkpoint 重载和固定提示生成。与 ByteTokenizer smoke 相比，中文输出不再出现破碎的 UTF-8 字节。

本实验仅使用 12 条样本并重复 256 次，目的是验证正式词表的数据和生成路径。它严重过拟合，不代表通用语言能力。

## 环境与配置

| 项目 | 值 |
|---|---|
| 设备 | Apple M4 Pro MPS |
| Python | 3.13.5 |
| PyTorch | 2.12.0 |
| Tokenizer | MiniMind BPE 6400 |
| Tokenizer commit | `6fc918beb68a0d8c40452338df6319fe168014ba` |
| 模型参数 | 4,590,336 |
| Hidden/Layers | 256 / 4 |
| Sequence length | 128 |
| Batch size | 8 |
| Steps | 300 |
| 耗时 | 8.620 秒 |

## Loss

| Step | Loss |
|---:|---:|
| 1 | 8.8215 |
| 40 | 0.9144 |
| 80 | 0.0640 |
| 120 | 0.0383 |
| 160 | 0.0287 |
| 200 | 0.0326 |
| 240 | 0.0155 |
| 280 | 0.0140 |
| 300 | 0.0130 |

极低 loss 是记忆重复语料的结果，不是泛化证据。

## 固定提示生成

```text
Prompt: 语言模型
Completion: 通过预测下一个词元学习文本中的规律。Transformer 使用自注意力机制建模上下文关系。训练模型需要数据、模型、损失函数和优化器……

Prompt: Transformer
Completion: 使用自注意力机制建模上下文关系。训练模型需要数据、模型、损失函数和优化器……

Prompt: Question: 什么是训练？ Answer:
Completion: 语言模型用于估计一段词元序列的概率。Question: What is training? Answer: Training updates parameters to minimize an objective……
```

模型主要逐段复述训练语料，证明 tokenization 和自回归生成链路正确，同时证明此数据规模不能用于正式能力评估。

## 下一步

下载并校验 `pretrain_t2t_mini.jsonl`，改用可流式读取、固定 validation split 的正式数据管道，先在 MPS 上做少量 batch 验证，再迁移到 CUDA 运行 63,912,192 参数配置。

