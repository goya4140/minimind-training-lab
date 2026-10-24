# 实验记录、证据索引与报告交付

本次目标包含训练、评估、中文学习报告、可编辑 PPTX 和 DOCX。正式训练完成不等于项目交付完成。
报告按“设计问题 → 实现选择 → 实验设置 → 实际证据 → 结论与局限”展开，原始证据保留在服务器。

## 正式运行约束

每次正式运行使用独立 run ID。准备命令只允许在 Git 工作区干净、正式输出尚不存在时创建新 run，
不会把探针当作已完成的正式训练。恢复已有运行应沿用相同 run ID 和 checkpoint，不能静默改动设置。

```bash
source scripts/cuda_env.sh
.venv/bin/python scripts/prepare_formal_run.py --run-id <run-id>
bash scripts/start_cuda_training.sh --run-id <run-id>
```

`artifacts/runs/<run-id>/` 是服务器上的证据根目录：

| 内容 | 用途 |
|---|---|
| manifest.json | 初始代码 commit、源码/配置哈希、数据哈希、GPU 和运行约定 |
| source.tar、configs/ | 保存实际实验使用的代码与六阶段配置 |
| events.jsonl | 阶段开始、校验、评估、归档、完成和失败事件 |
| metrics/*.jsonl | run ID、UTC 时间、进程/segment、loss、LR、梯度、耗时、吞吐、显存和恢复起点 |
| pipeline.console.log | 完整实际命令输出及异常栈 |
| stages/ | 校验后归档的各阶段权重、完整恢复点、总结、实际日志与评估 |

现有源码、配置和评估素材在每个阶段开始前核对哈希；原始数据如尺寸或修改时间变化，会重新校验。
阶段权重归档采用独立复制并复核 SHA-256，后续新运行不能通过覆盖默认输出破坏已有归档。
不把终端环境变量整体写入记录；凭据不进入日志或仓库。

## 指标口径

- Micro-step 是一次 micro-batch 前后向；optimizer update 是梯度累积完成后的一次参数更新。
- `padded_tokens_per_second` 使用 batch × 配置序列长度计算，是包括 padding 的名义吞吐，不能当作
  实际有效监督 token 吞吐。SFT 的 prompt mask 和多模态占位 token 需要另行解释。
- CUDA 显存统计属于当前 PyTorch 进程，不代表整张卡全部使用量；终端资源截图提供全卡状态。
- `training_seconds` 与暂停时长分开；`resumed_from_step`、PID 和 segment ID 用于识别恢复造成的
  原始日志重叠。最终曲线以最终 checkpoint 的有效 history 为依据，不直接拼接重复微步。
- 同一测试集、相同解码设置和评估口径下比较结果；短程探针不能作为正式模型能力结论。

## 视觉证据要求

真实终端截图只从实际终端窗口捕获。事后查看日志的截图应注明查看时间和来源，不能伪称为当时实时截图。
由日志绘制的曲线和展示页应标为“日志生成图表／结果展示”，不能伪装成终端截图。

每项素材记录证据编号、run ID、文件路径、UTC 时间、源日志/评估文件、内容说明及 SHA-256。
只公开精简输出和可公开的固定评估素材，不公开完整原始日志、数据集、权重及 optimizer。

## 学习报告结构

PPTX 用于讲解，DOCX 提供更详细的原理、关键实现和证据：项目范围；训练基础；三模型架构；数据和
Tokenizer；loss mask；冻结策略；工程实现；训练过程与真实截图；固定评估和消融；失败与修复；
局限、改进和复现命令。两种文件使用同一事实与素材索引，并逐页渲染检查。

交付验收必须同时覆盖：六阶段正式训练、完整评估、可追溯证据、具体设计实现解读、真实结果截图，
以及已完成视觉 QA 的 PPTX 和 DOCX。未完成的环节明确标记，不以流水线结束替代报告完成。
