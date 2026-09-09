# 项目进度与证据

## 环境基线

- 主机：Apple M4 Pro，20-core GPU，48 GiB unified memory；
- 本地加速：PyTorch MPS；
- CUDA：不可用；
- 策略：本地执行 smoke，正式训练使用 NVIDIA CUDA 环境。

## 里程碑

| ID | 交付 | 状态 | 完成证据 |
|---|---|---|---|
| M0 | GitHub 仓库与复现规范 | 已完成 | `goya4140/minimind-training-lab` 与首次远端提交 |
| M1 | LLM 原生架构 | 已完成 | 4 个测试通过；正式配置参数量 63,912,192 |
| M2 | LLM 从零 Pretrain | 未开始 | 正式 checkpoint、训练曲线 |
| M3 | LLM SFT 与评估 | 未开始 | SFT checkpoint、评估报告 |
| M4 | VLM 对齐与 SFT | 未开始 | VLM checkpoint、视觉评估 |
| M5 | Omni T2A/A2A/I2T | 未开始 | Omni checkpoint、跨模态评估 |
| M6 | 最终 GitHub 展示 | 未开始 | README、报告、模型卡、release |

`artifacts/` 下生成的日志和评估输出默认不提交；经过审核的结果将提炼到 `reports/` 并提交。

## 已运行实验

| 实验 | 设备 | 结果 | 报告 |
|---|---|---|---|
| `llm-smoke-001` | Apple M4 Pro / MPS | 120 步，loss 5.6009 → 0.5583 | [`reports/llm-smoke-001.md`](../reports/llm-smoke-001.md) |
| `llm-bpe-mini-001` | Apple M4 Pro / MPS | 300 步，loss 8.8215 → 0.0130 | [`reports/llm-bpe-mini-001.md`](../reports/llm-bpe-mini-001.md) |
| `llm-64m-mps-probe` | Apple M4 Pro / MPS | 正式 63.9M 架构、真实数据、完整恢复通过 | [`reports/llm-64m-mps-probe.md`](../reports/llm-64m-mps-probe.md) |
