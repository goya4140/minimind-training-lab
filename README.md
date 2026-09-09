# MiniMind Training Lab

一个面向学习的、可复现的模型训练实验室。目标不是简单复制三个上游仓库，而是沿着同一条能力演进路线，从零训练并理解：

1. **LLM**：文本输入 → 文本输出；
2. **VLM**：图像 + 文本输入 → 文本输出；
3. **Video-Omni**：视频 + 文本问题输入 → 文本输出，重点学习跨帧时序建模。

最终交付包括模型代码、训练配置、日志解读、定量评估、定性样例和可复现命令。数据集、模型权重、
optimizer state 和逐步原始日志均不上传 GitHub；仓库只记录足以学习和审阅训练全过程的内容。

第一次接触模型训练的读者，请从 [`handbook/README.md`](handbook/README.md) 开始。

## 当前状态

<!-- STATUS_START -->
| 阶段 | 架构 | 训练 | 评估 |
|---|---|---|---|
| LLM | ✅ 原生 PyTorch 主干（63,912,192 参数正式配置） | 🚧 正式 MPS 预训练进行中 | 🚧 step 12,000 中期评估已记录 |
| VLM | ✅ Early-fusion 与冻结策略已实现 | ⏳ 真实数据/SigLIP2 已验证，等待 LLM SFT | ✅ 固定 6 图评估入口就绪 |
| Video-Omni | ✅ 帧编码器 + 时序适配器 + LLM 已实现 | ⏳ QIVD 下载中，等待 LLM SFT | ✅ 留出集、倒序帧消融与分类型评估就绪 |
<!-- STATUS_END -->

状态以 [`docs/progress.md`](docs/progress.md) 中的证据为准。

模型说明：[`LLM`](docs/model-cards/llm.md) · [`VLM`](docs/model-cards/vlm.md) ·
[`Video-Omni`](docs/model-cards/video-omni.md)

## 目标架构

```text
MiniMind LLM (Decoder-only, 64M)
  ├── Pretrain → SFT
  ├── + Frozen SigLIP2 + Vision Projector → VLM
  └── + Frozen SigLIP2 + Temporal Adapter
      + Learned-query Resampler → Video-Omni (Video → Text)
```

设计详情：[`docs/architecture.md`](docs/architecture.md)  
训练策略：[`docs/training-strategy.md`](docs/training-strategy.md)  
评估协议：[`docs/evaluation.md`](docs/evaluation.md)
冻结组件：[`docs/external-components.md`](docs/external-components.md)

## 快速验证 LLM 链路

```bash
uv sync --extra dev
uv run python scripts/train_llm.py --config configs/llm/smoke.yaml
uv run python scripts/evaluate_llm.py \
  --config configs/llm/smoke.yaml \
  --checkpoint artifacts/checkpoints/llm-smoke.pt
uv run pytest
```

正式 MPS 预训练（可安全恢复）：

```bash
uv run python scripts/fetch_tokenizer.py
uv run python scripts/fetch_data.py pretrain sft
uv run python scripts/train_pretrain.py --config configs/llm/pretrain-mps.yaml --resume
```

`fetch_data.py all` 会进一步下载 VLM 数据。
冻结的多模态组件使用 `uv run python scripts/fetch_models.py all` 下载并校验。

正式 BPE 评估：

```bash
uv run python scripts/evaluate_bpe_llm.py \
  --config configs/llm/pretrain-mps.yaml \
  --checkpoint artifacts/checkpoints/llm-64m-pretrain-mps.pt
```

VLM 两阶段 MPS 训练入口（需先完成 LLM SFT）：

```bash
uv run python scripts/train_vlm.py --config configs/vlm/alignment-mps.yaml --resume
uv run python scripts/train_vlm.py --config configs/vlm/sft-mps.yaml --resume
uv run python scripts/evaluate_vlm.py \
  --config configs/vlm/sft-mps.yaml \
  --checkpoint artifacts/checkpoints/vlm-sft-mps.pt
```

Video-Omni 使用固定版本 QIVD，先训练时序适配器，再进行视频指令微调：

```bash
uv sync --extra multimodal --extra video --extra dev
uv run python scripts/fetch_qivd.py
uv run python scripts/preflight_mps_pipeline.py
uv run python scripts/train_video_omni.py --config configs/video/alignment-mps.yaml --resume
uv run python scripts/train_video_omni.py --config configs/video/sft-mps.yaml --resume
uv run python scripts/evaluate_video_omni.py \
  --config configs/video/sft-mps.yaml \
  --checkpoint artifacts/checkpoints/video-omni-sft-mps.pt
```

评估脚本会自动生成一个不参与训练的 160 条受控时序基准，分别测试水平/垂直移动、大小变化和
事件先后，并与 QIVD held-out test 一起进行正常帧/倒序帧对照。

若希望在每个上游 checkpoint 完成后自动接力，并在 LLM、VLM、Video-Omni 末端运行固定评估：

```bash
caffeinate -i uv run python scripts/run_mps_pipeline.py
```

执行器会跳过已有的最终 checkpoint、等待正在持锁的阶段、对中断阶段使用 `--resume`，并用
独立 pipeline lock 防止重复执行；任一阶段失败时会立即停止，保留已有 checkpoint 和日志。

`smoke.yaml` 是本机链路验证配置，不代表最终模型。当前正式本机复现使用
`configs/llm/pretrain-mps.yaml`，后续阶段严格从前一阶段 checkpoint 初始化。

首次本地运行的训练曲线和失败样例见 [`reports/llm-smoke-001.md`](reports/llm-smoke-001.md)，正式 BPE 词表的 mini 验证见 [`reports/llm-bpe-mini-001.md`](reports/llm-bpe-mini-001.md)。
已固定的数据版本、大小和校验和见 [`reports/data-manifest.md`](reports/data-manifest.md)。

## 上游基线

本项目学习并验证以下公开实现，实验必须固定 commit：

- [jingyaogong/minimind](https://github.com/jingyaogong/minimind) — LLM
- [jingyaogong/minimind-v](https://github.com/jingyaogong/minimind-v) — VLM

具体版本记录在 [`docs/upstream.md`](docs/upstream.md)。

## 原则

- 所有正式结果必须能由仓库中的配置和命令复现；
- smoke test 与正式训练结果明确分开；
- 每个实验记录数据版本、Git commit、随机种子、硬件、耗时和指标；
- 不把训练 loss 下降等同于模型能力提升；
- VLM 和 Video-Omni 必须同时做语言能力回归与模态消融测试；
- 未完成或未验证的内容明确标记，不制造“训练成功”的结论。

## License

代码将沿用兼容的 Apache-2.0 许可；第三方模型和数据分别遵守其原始许可。
