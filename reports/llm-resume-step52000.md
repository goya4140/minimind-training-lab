# LLM step 52,000 恢复点审计

本报告只记录本机 resume checkpoint 的结构化审计结果，不包含模型权重、optimizer 内容、训练数据或
逐步原始日志。审计命令：

```bash
uv run python scripts/verify_resume_checkpoint.py --config configs/llm/pretrain-mps.yaml
```

## 两审计结果

| 字段 | step 48,000 | step 52,000 | 解释 |
|---|---:|---:|---|
| Checkpoint bytes | 767,246,887 | 767,246,701 | 两次原子快照均可完整读取 |
| SHA-256 | `84dea67c…47b7` | `bf409591…f1c4` | 参数或状态更新后文件内容确实变化 |
| Model parameters | 63,912,192 | 63,912,192 | 架构没有漂移 |
| Finite tensors | ✅ | ✅ | 模型参数中没有 NaN/Inf |
| Optimizer state entries | 90 | 90 | AdamW 恢复状态完整存在 |
| History records | 2,401 | 2,601 | 每 20 步记录一次，增加 200 条 |
| Torch RNG state | 5,056 bytes | 5,056 bytes | 随机状态可恢复 |
| Cumulative training time | 17,393.61s | 18,790.01s | 跨进程重载后继续累计 |
| Config matches | ✅ | ✅ | 快照与正式 YAML 一致 |
| Optimizer boundary | ✅ | ✅ | 两个步数均能被 accumulation 32 整除 |

从 step 48,000 到 52,000 共训练 4,000 个 micro-step、执行 125 次 optimizer update，累计时间增加
1,396.40 秒，即约 0.3491 秒/micro-step，与运行日志速度一致。这证明新版训练器没有在恢复后把计时
归零，也没有把 accumulation 中间状态误当成可恢复断点。

step 52,000 快照的完整 SHA-256 为
`bf409591d5bd1816d721d5f2ec679f84f151ee1e54789d601113d253da69f1c4`。该文件持续被后续快照覆盖，
哈希只作为本次审计证据，不代表最终模型哈希。
