# LLM 64M MPS Throughput Probe

## 结论

正式 63,912,192 参数 Dense 架构已经在完整 `pretrain_t2t_mini` 数据集上完成前向、反向、验证、模型保存和包含优化器状态的断点恢复测试。

这只是 12 个 micro-step 的吞吐探针，不是训练结果。

## 数据

| 项目 | 值 |
|---|---:|
| 文件 | `pretrain_t2t_mini.jsonl` |
| 文件大小 | 1,241,043,656 bytes |
| 样本数 | 1,270,238 |
| SHA-256 | `6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c` |
| Probe validation | 最后 16 条 |

## Probe 配置

| 项目 | 值 |
|---|---:|
| 参数量 | 63,912,192 |
| Hidden/Layers | 768 / 8 |
| Q heads/KV heads | 8 / 4 |
| Sequence length | 340 |
| Batch | 8 |
| 设备 | Apple M4 Pro MPS |

首次完整运行的训练吞吐约为 5,102 tokens/s、0.533 秒/micro-step。包含优化器的原子 resume checkpoint 成功保存，随后从 step 10 恢复并继续完成 step 11–12，证明恢复路径可用。

12 步后 validation loss 仍约为 8.57，符合近乎随机初始化模型的预期，不能解释为模型已经学会语言。

## 耗时判断

正式 MPS 配置使用 batch 8、sequence 340、2 epochs，共 317,048 个 micro-step。按探针估计，纯训练时间约 47 小时；周期验证和 checkpoint I/O 会进一步增加总耗时。实际速度将在长跑日志中重新计算。

## 已验证风险控制

- 训练数据不整体载入内存，仅构建 JSONL byte-offset 索引；
- 模型权重和完整训练状态分开保存；
- resume 包含模型、AdamW、step、history 和 CPU RNG state；
- checkpoint 使用临时文件后原子替换；
- resume 后优化器张量能迁移回 MPS 并继续更新；
- 正式数据与 tokenizer 都有固定版本和校验和。

