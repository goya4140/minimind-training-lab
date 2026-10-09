# LLM Pretrain 完成报告

正式预训练 `llm-64m-pretrain-mps` 已于 2026-09-11 16:05 (GMT+8) 跑完全部 317,048 个 micro-step，
并写出最终 checkpoint。训练器随后交棒，六阶段流水线已自动启动阶段 ② 的 SFT。

## 最终结果

| 指标 | 值 |
|---|---:|
| 总 micro-steps | 317,048 / 317,048 |
| Optimizer updates | 约 9,908 次（每 32 步一次） |
| Validation loss | **1.8326** |
| Validation perplexity | **6.2504** |
| 验证集 | 完整 2,048 条 held-out 样本 |
| 训练吞吐 | 7,682.7 token/s |
| 平均速度 | 0.3540 秒/micro-step |
| 模型参数 | 63,912,192（去重后），全部有限 |

## 耗时

| 项 | 秒 | 折合 |
|---|---:|---:|
| 纯训练时间 | 112,248.73 | 31 小时 11 分 |
| 系统休眠（已扣除） | 36,920.19 | 10 小时 15 分 |
| 合计墙钟 | 149,168.92 | 41 小时 26 分 |

训练跨多次运行完成（含三次恢复/计时审计），休眠时间由 `ActiveTrainingTimer` 自动识别并单列，
不计入吞吐统计。这是本项目在消费级笔记本上跑完 31 小时纯训练的关键工程保障。

## Checkpoint

- 路径：`artifacts/checkpoints/llm-64m-pretrain-mps.pt`
- 大小：256,849,685 bytes
- SHA-256：`2baf3b6a75fb2cd7decaff66be72134a9d9d5e8c21f938600ef772913c5e3075`
- 校验：91 个 state tensors、63,912,192 参数、全部有限（`all_finite: true`）

这是**最终模型**哈希，与运行期间持续覆盖的 `.resume.pt` 不同，可用于复现校验。

## 与中期诊断的关系

| 检查点 | 验证集 | Val loss | Perplexity |
|---|---|---:|---:|
| step 12,000 | 32 条 | 4.0441 | 57.06 |
| step 272,000 | 32 条 | 1.8560 | 6.398 |
| **step 317,048（最终）** | **2,048 条** | **1.8326** | **6.2504** |

最后两行的模型步数与验证集规模都不同（相差 45,048 步、32 条 vs 2,048 条），因此**不能相减**
来推算"最后 4.5 万步的收益"。但两者数值接近，说明训练末期 loss 已进入平台期，也印证了
32 条小样本诊断给出的量级是可信的——它适合判断趋势，不适合作为最终数字。

## 这一阶段完成了什么

模型从随机初始化开始，在 126.8 万条中文文本上训练了 2 轮（约 8.6 亿 token），
学会了中文的语言模式与部分世界知识。相对 step 12,000 的基线：

- 退化乱码完全消失；
- 角色漂移消失，能维持 `用户：/助手：` 结构；
- 能正确回答常识问题（如"中国的首都是北京"）；
- perplexity 从 57 降到 6.25。

## 这一阶段没有完成什么

预训练不教模型听指令。它不知道"请用三句话"是约束，不知道何时该停止，也没有稳定的助手身份。
这些问题由阶段 ② 的 assistant-only SFT 处理，届时**必须同时做语言能力回归测试**，
确认 SFT 没有损害这里达到的语言建模质量。

## 复现

```bash
uv run python scripts/fetch_tokenizer.py
uv run python scripts/fetch_data.py pretrain sft
uv run python scripts/train_pretrain.py --config configs/llm/pretrain-mps.yaml --resume
```

完成后结果写入 `artifacts/logs/llm-64m-pretrain-mps.json`（Git 忽略）。
