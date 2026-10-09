# 正式训练进度快照

生成时间：2026-10-09T06:46:34.229741+00:00。数据来自本机 Git 忽略日志；不包含权重或训练数据。

| Stage | Status | Micro-step | Progress | First loss | Recent mean loss | Seconds/step | ETA |
|---|---|---:|---:|---:|---:|---:|---:|
| llm-pretrain | complete | 317,040 / 317,048 | 100.0% | 8.9057 | 1.8253 | 0.3512 | — |
| llm-sft | complete | 40,000 / 40,000 | 100.0% | 2.5216 | 1.7009 | 0.4530 | — |
| vlm-alignment | active | 40 / 30,000 | 0.1% | 3.2635 | 3.2595 | 0.3314 | 2h 45m |
| vlm-sft | pending | 0 / 30,000 | 0.0% | — | — | — | — |
| video-omni-alignment | pending | 0 / 7,200 | 0.0% | — | — | — | — |
| video-omni-sft | pending | 0 / 4,800 | 0.0% | — | — | — | — |

`Recent mean loss` 是最近 20 条日志记录的均值，仅用于观察趋势；不同阶段的 loss mask 与数据不同，
不能把数值直接横向排名。ETA 按当前进程的累计 seconds/step 外推，不包含验证、评估和后续阶段。
