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

## Video-Omni

1. 从 LLM SFT 权重初始化语言主干；SigLIP2 使用冻结公开权重；
2. Temporal alignment：随机初始化并训练帧位置、2 层 temporal Transformer、query resampler 与 projector；
3. Video instruction tuning：保持 SigLIP2 冻结，继续训练时序模块，并以低学习率解冻 LLM 首尾层；
4. 训练目标仅为 assistant answer token 的 causal cross-entropy；问题、视频占位符和 padding 均不计 loss；
5. 训练/验证/测试采用 seed 48 对 QIVD ID 做稳定哈希划分，固定为 2400/250/250，无样本重叠。

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

| 阶段 | seq len | micro-steps | micro-batch | accumulation | 样本曝光数 |
|---|---:|---:|---:|---:|---:|
| LLM pretrain | 340 | 317,048 | 8 | 32 | 2,536,384 |
| LLM SFT | 768 | 40,000 | 4 | 4 | 160,000 |
| VLM alignment | 450 | 30,000 | 4 | 4 | 120,000 |
| VLM SFT | 768 | 30,000 | 2 | 8 | 60,000 |
| Video-Omni alignment | 384 | 7,200 | 1 | 8 | 7,200 |
| Video-Omni SFT | 384 | 4,800 | 1 | 8 | 4,800 |

VLM/Video-Omni 的 batch 是保守起点，正式启动前还会用完整模型做 MPS 峰值和吞吐探针；若需要
下调 micro-batch，将同比提高 accumulation，保持 effective batch 不变。运行
`python scripts/preflight_mps_pipeline.py` 可检查设备、数据、冻结组件、阶段 checkpoint 链与
每阶段 effective batch。`pending` 表示资源齐全但必须等待上游 checkpoint，并非错误。

`scripts/run_mps_pipeline.py` 是可恢复的阶段编排器：已有最终 checkpoint 会被跳过，正在运行的
阶段按锁等待，中断阶段通过 `--resume` 继续；LLM SFT、VLM SFT 和 Video-Omni SFT 完成后分别触发
固定评估。pipeline 自身也持有独立锁，避免启动两个编排器。

除 LLM pretrain 完整覆盖约 2 个 epoch 外，后续 MPS 阶段采用固定 micro-step 预算并从完整
数据池做确定性无放回轮转：LLM SFT 约覆盖 17.7% 数据，VLM alignment 约 9.4%，VLM SFT
约 2.1%；Video-Omni alignment 完整轮转 3 次训练集，Video-Omni SFT 完整轮转 2 次。这是为了在
单台 M4 Pro 上完成三模型训练—评估闭环；最终报告必须称为 fixed-budget reproduction，不能
声称完成上游 full-epoch 训练。
