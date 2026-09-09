# MiniMind Training Lab

一个面向学习的、可复现的模型训练实验室。目标不是简单复制三个上游仓库，而是沿着同一条能力演进路线，从零训练并理解：

1. **LLM**：文本输入 → 文本输出；
2. **VLM**：图像 + 文本输入 → 文本输出；
3. **Omni**：文本 / 图像 / 语音输入 → 文本 / 流式语音输出。

最终交付包括模型代码、训练配置、实验日志、检查点来源、定量评估、定性样例和可复现命令。大体积数据和模型权重通过 release 或外部模型仓库存放，不直接提交到 Git。

## 当前状态

| 阶段 | 架构 | 训练 | 评估 |
|---|---|---|---|
| LLM | ✅ 原生 PyTorch 主干（63,912,192 参数正式配置） | 🚧 MPS smoke 已通过，正式训练待运行 | 🚧 smoke 生成已记录，正式评估待运行 |
| VLM | 📋 已定版 | ⏳ | ⏳ |
| Omni | 📋 已定版 | ⏳ | ⏳ |

状态以 [`docs/progress.md`](docs/progress.md) 中的证据为准。

## 目标架构

```text
MiniMind LLM (Decoder-only, 64M)
  ├── Pretrain → SFT
  ├── + Frozen SigLIP2 + Vision Projector → VLM
  └── + Frozen SenseVoice/SigLIP2 Projectors
      + Thinker/Bridge/Talker + Mimi codes → Omni
```

设计详情：[`docs/architecture.md`](docs/architecture.md)  
训练策略：[`docs/training-strategy.md`](docs/training-strategy.md)  
评估协议：[`docs/evaluation.md`](docs/evaluation.md)

## 快速验证 LLM 链路

```bash
uv sync --extra dev
uv run python scripts/train_llm.py --config configs/llm/smoke.yaml
uv run python scripts/evaluate_llm.py \
  --config configs/llm/smoke.yaml \
  --checkpoint artifacts/checkpoints/llm-smoke.pt
uv run pytest
```

`smoke.yaml` 是本机链路验证配置，不代表最终模型。正式复现使用 `configs/llm/full.yaml`。

首次本地运行的训练曲线和失败样例见 [`reports/llm-smoke-001.md`](reports/llm-smoke-001.md)。

## 上游基线

本项目学习并验证以下公开实现，实验必须固定 commit：

- [jingyaogong/minimind](https://github.com/jingyaogong/minimind) — LLM
- [jingyaogong/minimind-v](https://github.com/jingyaogong/minimind-v) — VLM
- [jingyaogong/minimind-o](https://github.com/jingyaogong/minimind-o) — Omni

具体版本记录在 [`docs/upstream.md`](docs/upstream.md)。

## 原则

- 所有正式结果必须能由仓库中的配置和命令复现；
- smoke test 与正式训练结果明确分开；
- 每个实验记录数据版本、Git commit、随机种子、硬件、耗时和指标；
- 不把训练 loss 下降等同于模型能力提升；
- VLM 和 Omni 必须同时做语言能力回归测试；
- 未完成或未验证的内容明确标记，不制造“训练成功”的结论。

## License

代码将沿用兼容的 Apache-2.0 许可；第三方模型和数据分别遵守其原始许可。
