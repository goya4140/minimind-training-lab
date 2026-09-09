# LLM Pretrain 中期评估：step 12000

这是训练进行中的诊断快照，不是最终模型。快照仅保存模型权重，大小 255,728,629 bytes，
SHA-256 为 `b1a7fabdf4ae6686242b54914e18bd304a455dd3bbb8766123a600f2f954d203`。

## 训练状态

- 模型：63,912,192 参数，MiniMind BPE 6400；
- step：12,000 micro-steps / 375 optimizer updates；
- 当前 batch loss：`3.281565`；
- 初始 step-1 loss：`8.8969`；
- 设备：Apple M4 Pro / MPS；
- 样本顺序：确定性、可精确恢复。

## 小规模 held-out 诊断

为避免与后台训练争抢 MPS，本次在 CPU 上评估固定末尾 32 条 validation 样本：

| 指标 | 结果 |
|---|---:|
| Validation loss | 4.0441 |
| Perplexity | 57.06 |
| Bits per byte | 2.0264 |
| Predicted tokens | 10,495 |
| Validation throughput | 8,330 token/s |

最终报告会扩大到完整 2,048 条 validation 集；这里的数值只用于和后续 checkpoint 做同协议比较。

## 固定生成样例

贪心解码，每题 20 个新 token，CPU 约 40.7–41.8 token/s。

| Prompt | Completion | 判断 |
|---|---|---|
| `中国的首都是` | `哪个？中国的首都是哪个？中国的首都是哪个？中国的首都` | 失败：复读且未回答 |
| `请用三句话解释什么是机器学习：` | `机器学习是一种人工智能技术，它可以让计算机能够自主地学习和适应。机器学习是一种人工智能技术` | 部分成功：语义正确，但未遵循三句话且开始重复 |
| `用户：如何制定一个可执行的学习计划？\n助手：` | `首先，您需要了解哪些方面的因素？\n客户：首先，您需要了解哪些方面` | 失败：角色漂移、回答不完整 |
| `Transformer 模型中的注意力机制` | `是如何使用SOOO的？在SOOOOOOO中，` | 失败：明显退化 token 模式 |

结论：预训练已学到局部语言模式和部分知识表达，但事实回答、指令跟随、角色稳定性和抗复读均未达标。
这些问题需要剩余预训练与 assistant-only SFT 改善，不能把 loss 下降直接写成“模型已可用”。
