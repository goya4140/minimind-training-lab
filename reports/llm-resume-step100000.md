# LLM step 100,000 恢复点审计

正式 64M LLM 预训练首次超过十万 micro-step 后，本项目对最近一个完整 optimizer 边界执行只读审计。
本报告不包含模型权重、optimizer 内容、训练数据或逐步原始日志。

| 字段 | step 100,000 |
|---|---:|
| 训练进度 | 31.54% |
| Checkpoint bytes | 767,423,085 |
| SHA-256 | `3f85a17cbad16f1acd8974464258559427e6ef2b4f10fe4022e1f9bfaeea2db0` |
| Model parameters | 63,912,192 |
| Model tensors finite | ✅ |
| Optimizer state entries / tensors | 90 / 270 |
| Optimizer tensors finite | ✅ |
| History records | 5,001 |
| Torch RNG state | 5,056 bytes |
| Cumulative active training time | 35,873.93s |
| Excluded suspended time | 33,228s |
| Config matches / optimizer boundary | ✅ / ✅ |

审计后训练继续运行，速度约 0.3495 秒/micro-step。累计活跃耗时与休眠耗时保持分离，说明
step 60,000 引入的计时修复已在后续原子 checkpoint 中持续保存。

checkpoint 文件会被后续恢复点覆盖；这里的 SHA-256 只记录本次里程碑，不代表最终模型哈希。
