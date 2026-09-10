# 正式训练进度快照

生成时间：2026-09-10T02:44:10.284900+00:00。数据来自本机 Git 忽略日志；不包含权重或训练数据。

| Stage | Status | Micro-step | Progress | First loss | Recent mean loss | Seconds/step | ETA |
|---|---|---:|---:|---:|---:|---:|---:|
| llm-pretrain | active | 60,360 / 317,048 | 19.0% | 8.9057 | 2.3724 | 0.3516 | 25h 04m |
| llm-sft | pending | 0 / 40,000 | 0.0% | — | — | — | — |
| vlm-alignment | pending | 0 / 30,000 | 0.0% | — | — | — | — |
| vlm-sft | pending | 0 / 30,000 | 0.0% | — | — | — | — |
| video-omni-alignment | pending | 0 / 7,200 | 0.0% | — | — | — | — |
| video-omni-sft | pending | 0 / 4,800 | 0.0% | — | — | — | — |

`Recent mean loss` 是最近 20 条日志记录的均值，仅用于观察趋势；不同阶段的 loss mask 与数据不同，
不能把数值直接横向排名。ETA 按当前进程的累计 seconds/step 外推，不包含验证、评估和后续阶段。
