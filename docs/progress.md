# 项目进度与证据

## 环境基线

- 主机：Apple M4 Pro，20-core GPU，48 GiB unified memory；
- 本地加速：PyTorch MPS；
- CUDA：不可用；
- 策略：先在本机 MPS 完成 64M 正式训练与恢复验证；VLM/Omni 如本机成本不可接受，
  再在不改变配置语义和评估协议的前提下迁移到 NVIDIA CUDA。

## 里程碑

| ID | 交付 | 状态 | 完成证据 |
|---|---|---|---|
| M0 | GitHub 仓库与复现规范 | 已完成 | `goya4140/minimind-training-lab` 与首次远端提交 |
| M1 | LLM 原生架构 | 已完成 | 4 个测试通过；正式配置参数量 63,912,192 |
| M2 | LLM 从零 Pretrain | 进行中 | 已超过 step 15,000；step 12,000 中期评估已提交 |
| M3 | LLM SFT 与评估 | 已准备 | assistant-only 数据管线、配置和训练入口已测试 |
| M4 | VLM 对齐与 SFT | 管线就绪 | 真实 Parquet + SigLIP2 前向探针通过，等待 LLM SFT |
| M5 | Omni T2A/A2A/I2T | 管线就绪 | T2A/A2A 数据、SenseVoice、Mimi 与真实前向探针通过 |
| M6 | 最终 GitHub 展示 | 未开始 | README、报告、模型卡、release |

`artifacts/` 下生成的日志和评估输出默认不提交；经过审核的结果将提炼到 `reports/` 并提交。

## 已运行实验

| 实验 | 设备 | 结果 | 报告 |
|---|---|---|---|
| `llm-smoke-001` | Apple M4 Pro / MPS | 120 步，loss 5.6009 → 0.5583 | [`reports/llm-smoke-001.md`](../reports/llm-smoke-001.md) |
| `llm-bpe-mini-001` | Apple M4 Pro / MPS | 300 步，loss 8.8215 → 0.0130 | [`reports/llm-bpe-mini-001.md`](../reports/llm-bpe-mini-001.md) |
| `llm-64m-mps-probe` | Apple M4 Pro / MPS | 正式 63.9M 架构、真实数据、完整恢复通过 | [`reports/llm-64m-mps-probe.md`](../reports/llm-64m-mps-probe.md) |

## 当前运行

- `llm-64m-pretrain-mps`：从随机初始化训练 63,912,192 参数 LLM；
- 数据：`pretrain_t2t_mini.jsonl`，1,270,238 条；
- 配置：sequence length 340、batch size 8、gradient accumulation 32、2 epochs；
- checkpoint 采用原子替换写入，并保存 optimizer、随机状态、历史和全局步数；
- 样本顺序由 seed、epoch 与全局步数决定，恢复后不会从 DataLoader 起点重放。

首个恢复点已于 step 2000 读取验证：文件大小 767,066,797 bytes，SHA-256 为
`2496b17f64d428275920954ebfcbf936fd47ee5f945b48cd94c0b7b1d7fdd5fc`。该文件是持续覆盖的
运行恢复点，因此哈希只用于证明当时的原子快照，不作为最终模型哈希。

本节只陈述已验证的运行状态。最终步数、耗时、曲线和 checkpoint 哈希将在训练完成后写入正式报告。

step 12,000 已保留模型中期快照并按 32 条 held-out 样本完成诊断：validation loss 4.0441、
perplexity 57.06、BPB 2.0264。固定生成已有局部语义能力但仍存在复读、角色漂移和乱码，详见
[`reports/llm-pretrain-step12000.md`](../reports/llm-pretrain-step12000.md)。
