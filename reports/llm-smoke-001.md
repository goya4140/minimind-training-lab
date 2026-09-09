# LLM Smoke Test 001

## 结论

原生 PyTorch LLM 的最小训练链路已经验证：随机初始化、causal loss、反向传播、梯度裁剪、AdamW 更新、checkpoint 保存、重新加载和自回归生成均可执行。

这不是模型能力评估。数据是重复的小样本文本，模型只有 771,840 参数，Tokenizer 是 smoke 专用 ByteTokenizer。

## 环境

| 项目 | 值 |
|---|---|
| 设备 | Apple M4 Pro MPS |
| 内存 | 48 GiB unified memory |
| Python | 3.13.5 |
| PyTorch | 2.12.0 |
| Seed | 42 |
| 模型参数 | 771,840 |
| Steps | 120 |
| 耗时 | 2.757 秒 |

## Loss

| Step | Loss |
|---:|---:|
| 1 | 5.6009 |
| 20 | 4.2078 |
| 40 | 3.1654 |
| 60 | 2.2907 |
| 80 | 1.8267 |
| 100 | 0.9499 |
| 120 | 0.5583 |

Loss 明显下降，证明优化链路有效。由于训练集被重复使用，这同时体现出严重过拟合，不能用来推断泛化能力。

## 固定提示生成

训练后模型能够输出局部训练语料模式，例如 `语言模型`、`Transformer`、`训练` 等片段，但存在大量 Unicode replacement characters：

```text
Prompt: Transformer
Output: er toble�樀数＼��Trainminins bler: �Ceainin��让捕�的�列的模型�低��
```

原因是 ByteTokenizer 的自回归采样可能在 UTF-8 多字节字符中间生成无效组合。这是预期的 smoke 局限，也是正式训练必须切换到固定 BPE Tokenizer 的直接证据。

## 验证项

- 4 个单元测试通过；
- Ruff 静态检查通过；
- loss 为有限值且持续下降；
- 非 embedding 参数在一步优化后发生变化；
- embedding 与 LM Head 确认权重共享；
- checkpoint 成功重新加载；
- 固定 seed 下完成三条提示生成；
- 失败输出如实保留。

## 下一步

接入 MiniMind BPE 6400 Tokenizer 和正式 JSONL 数据管道，然后运行 LLM mini/full Pretrain，而不是继续扩大 ByteTokenizer 实验。

