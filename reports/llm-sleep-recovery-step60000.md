# LLM step 60,000 休眠恢复审计

2026-09-10，本机在电量 1% 时进入低电量休眠，并在接入电源后恢复。macOS 电源日志给出的休眠区间
为 01:11:11–10:24:59（Asia/Shanghai），共 33,228 秒。训练进程恢复后继续正常更新，但旧版
`time.time()` 累计方式把这段休眠误计入了训练耗时；loss、模型权重和 optimizer 状态不受影响。

为避免在 accumulation 中间截断，本项目等到 step 60,000 的完整 optimizer 边界原子保存后停止流水线。
修复前审计结果如下：

| 字段 | 结果 |
|---|---:|
| Resume step | 60,000 |
| Checkpoint bytes | 767,275,501 |
| SHA-256 | `aa62c960…750d` |
| Model parameters | 63,912,192 |
| Model tensors finite | ✅ |
| Optimizer state entries / tensors | 90 / 270 |
| Optimizer tensors finite | ✅ |
| History records | 3,001 |
| Cumulative training time | 55,131.5253s |
| Suspended time | 0s |

随后用 `scripts/repair_resume_timing.py` 原子修改计时元数据：从 active training time 中扣除电源日志明确
记录的 33,228 秒，并把同一时长计入 `suspended_seconds`。修复操作带唯一 reason，重复执行会被拒绝，
以防二次扣减。修复后 checkpoint 仍通过模型、optimizer、RNG、config 和 optimizer-boundary 全量审计：

| 字段 | 修复后结果 |
|---|---:|
| Cumulative active training time | 21,903.5253s |
| Suspended time | 33,228s |
| Checkpoint SHA-256 | `13687af0…413d` |
| Model / optimizer tensors finite | ✅ / ✅ |
| Resume config matches | ✅ |

流水线从 step 60,000 重启后，step 60,020 的速度为 0.3420 秒/micro-step，与休眠前约 0.35 秒的
速度一致。后续所有训练入口改用逐步活跃计时器：相邻步的 wall-clock gap 超过 60 秒时整段记入
`suspended_seconds`，不再污染吞吐与 ETA；checkpoint 写盘等正常短开销仍计入活跃耗时。

说明：checkpoint 文件持续被后续训练覆盖，上述哈希只证明本次修复边界的完整快照，不代表最终模型哈希。

## 后续加固

首次启用活跃计时器后，macOS 的 wall clock 在 step 62,300 后发生一次向后校正。初版实现为防止负耗时
直接终止了进程；由于下一原子恢复点是 step 64,000，流水线安全回退到已审计的 step 60,000，没有接受
未保存的中间状态。计时器随后加固为忽略单个负向区间并立即重置基准，既不记录负耗时，也不会因系统
校时终止训练。对应行为有独立单元测试覆盖。
