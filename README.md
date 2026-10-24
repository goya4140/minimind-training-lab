# MiniMind Training Lab

一个面向学习的、可复现的模型训练实验室。目标不是简单复制三个上游仓库，而是沿着同一条能力演进路线，从零训练并理解：

1. **LLM**：文本输入 → 文本输出；
2. **VLM**：图像 + 文本输入 → 文本输出；
3. **Video-Omni**：视频 + 文本问题输入 → 文本输出，重点学习跨帧时序建模。

最终交付包括模型代码、训练配置、日志解读、定量评估、定性样例、可复现命令和 GitHub 学习文档。
根据用户最新交付调整，不再制作或交付 DOCX/PPTX 文件；已有样稿仅作内部历史记录。数据集、模型权重、
optimizer state 和逐步原始日志均不上传 GitHub；仓库只记录足以学习和审阅训练全过程的内容。

第一次接触模型训练的读者，请从 [`handbook/README.md`](handbook/README.md) 开始。

## 当前状态

2026-10-24：因本地空间不足，旧权重已删除，项目迁至服务器
`/root/autodl-tmp/wlh_goya/MiniMind` 从零进行 CUDA 重训。正式 run `mm64-cuda-20261024T075332Z`
已完成 LLM Pretrain/SFT，随后 VLM 截断失败。独立
[修复运行](reports/cuda-vlm-repair-run.md)继承精确 LLM 归档，现已完成四个多模态阶段和固定评估，
并完成260条视频独立重放留存，无周期性监控。
请先读[三模型综合学习报告](reports/cuda-three-model-synthesis.md)：设计、实现、六阶段训练、
真实截图/曲线、评估和失败解读。工程完成不等于指令遵循、视觉理解或时序推理已经达标。
原始记录见[正式实验登记](reports/formal-run-mm64-cuda-20261024.md)。
环境、数据和测试流程见 [`docs/cuda-training.md`](docs/cuda-training.md)。
下表为当前CUDA实验；[旧MPS历史](docs/progress.md#历史-mps-环境基线非当前-cuda-服务器)单独保留，不混用结果。

<!-- STATUS_START -->
| 阶段 | 架构 | 训练 | 评估 |
|---|---|---|---|
| LLM | 原生63,912,192参数语言主干 | Pretrain 317,048＋SFT 40,000步完成 | 同语料PPL 6.2227→6.7796；严格诊断0/8 |
| VLM | 冻结SigLIP2＋Projector＋语言首尾层适配 | Alignment/SFT各30,000步完成 | 7题关键词差不代表语义正确；保留幻觉与图序失败 |
| Video-Omni | 空间池化＋时序Transformer＋16-query接口 | Alignment 7,200＋SFT 4,800步完成 | 250条test loss、100＋160条生成；正常/倒序回答未变 |
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
实现与指标口径解读：[`docs/design-implementation.md`](docs/design-implementation.md)
架构图与冻结策略：[`reports/figures/minimind-architecture/README.md`](reports/figures/minimind-architecture/README.md)
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
