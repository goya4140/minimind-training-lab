# 训练策略

## 实验分级

- `smoke`：本机 MPS/CPU 上验证张量、反向传播、保存、加载和生成；结果不用于能力结论。
- `mini`：使用公开 mini 数据验证完整模型训练链路。
- `full`：使用固定数据版本与正式配置，产出最终权重和评估报告。

## LLM

1. 从随机初始化进行文本预训练，全 token causal language modeling；
2. 从预训练权重进行 SFT，只在 assistant token 上计算损失；
3. 可选 LoRA 与 GRPO 作为后训练专题，不作为三模型主线的完成条件。

## VLM

1. 从 LLM SFT 权重初始化；
2. 冻结 SigLIP2 和 LLM，只训练 Vision Projector 完成图文对齐；
3. 冻结 SigLIP2，训练 Projector 与 LLM 首尾层完成视觉指令微调；
4. 全参数 LLM 微调只作为消融实验。

## Omni

1. 从 LLM SFT 权重初始化 Thinker；Talker 从 Thinker 后四层复制初始化；
2. T2A：训练文本到文本+语音输出；
3. A2A projector alignment：只训练 Audio Projector；
4. A2A joint training：小学习率联合训练 Thinker、Talker 与 Projector；
5. I2T：只训练 Vision Projector，避免破坏语言和语音能力。

损失为文本交叉熵、8 路音频 code 交叉熵和可选 MoE 辅助损失之和。第一期只训练 Dense。

## 可复现要求

每个正式实验必须记录：

- 仓库 commit 与脏工作区状态；
- 数据名称、来源、许可、版本与校验和；
- 完整配置和随机种子；
- Python、PyTorch、CUDA、GPU；
- 参数量、可训练参数量与冻结模块；
- loss、梯度范数、吞吐、显存、耗时；
- checkpoint 哈希；
- 定量评估和固定定性样例。

## M4 Pro 正式配置

本机正式路线使用单进程 MPS。显存是统一内存，但仍用 micro-batch + gradient accumulation
控制激活峰值；下表保持原 CUDA 配置的 effective batch 语义：

| 阶段 | sequence length | micro-batch | accumulation | effective batch |
|---|---:|---:|---:|---:|
| LLM pretrain | 340 | 8 | 32 | 256 |
| LLM SFT | 768 | 4 | 4 | 16 |
| VLM alignment | 450 | 4 | 4 | 16 |
| VLM SFT | 768 | 2 | 8 | 16 |
| Omni T2A | 512 | 8 | 5 | 40 |
| Omni audio alignment | 640 | 8 | 5 | 40 |
| Omni A2A SFT | 768 | 4 | 4 | 16 |
| Omni I2T | 768 | 4 | 4 | 16 |

VLM/Omni 的 batch 是保守起点，正式启动前还会用完整模型做 MPS 峰值和吞吐探针；若需要
下调 micro-batch，将同比提高 accumulation，保持 effective batch 不变。运行
`python scripts/preflight_mps_pipeline.py` 可检查设备、数据、冻结组件、阶段 checkpoint 链与
每阶段 effective batch。`pending` 表示资源齐全但必须等待上游 checkpoint，并非错误。

`scripts/run_mps_pipeline.py` 是可恢复的阶段编排器：已有最终 checkpoint 会被跳过，正在运行的
阶段按锁等待，中断阶段通过 `--resume` 继续；LLM SFT、VLM SFT 和 Omni I2T 完成后分别触发
固定评估。pipeline 自身也持有独立锁，避免启动两个编排器。
