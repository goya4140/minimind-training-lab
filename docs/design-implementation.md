# 三模型设计与实现解读

本文解释正式 CUDA 实验 `mm64-cuda-20261024T075332Z` 的固定实现，代码版本为
`53f9b9f46560bee446d1c74e89cac991b71a8053`。这是报告的原理与实现底稿，不是最终结果报告。
后续结果必须引用该 run 的真实评估，不能用历史 MPS 实验或短程探针填补。

## 1. 为什么是三种模型、六个训练阶段

三种模型共用语言架构，但不是六个阶段首尾串接的一条权重链。语言预训练学文本统计规律，
语言 SFT 学对话格式和回答行为；随后从同一个 LLM SFT 分叉，分别扩展图像和视频输入。
Video-Omni **不继承 VLM SFT 权重**，因此不能解释为“图像模型继续学视频”。

| 阶段 | 初始化来源 | 允许更新的组件 | 希望检验什么 |
|---|---|---|---|
| LLM Pretrain | 随机初始化 | 全部语言参数 | 能否学习可泛化的文本分布 |
| LLM SFT | LLM Pretrain | 全部语言参数 | 回答行为变化及通用文本能力代价 |
| VLM Alignment | LLM SFT + 公开 SigLIP2 | 新建图像 Projector | 冻结语言模型能否利用视觉表示 |
| VLM SFT | VLM Alignment | Projector + 第 0、7 个 Decoder Block | 有限语言适配能否改善图文回答 |
| Video Alignment | LLM SFT + 公开 SigLIP2 | 新建 Temporal Adapter + Projector | 视频表示能否接入冻结语言模型 |
| Video SFT | Video Alignment | Adapter、Projector + 第 0、7 个 Block | 回答与时间顺序利用是否改善 |

这里的 Omni 是项目内视频理解分支的名称，实际接口为“视频 + 问题 → 文本答案”；
没有音频输入、语音生成或任意模态输出。视觉编码器并非从零训练。

## 2. 语言核心：从 token 到下一个 token

实现见 [语言模型](../src/minimind_lab/llm/model.py)。固定结构：词表 6,400、隐藏维度 768、
8 层 Decoder、8 个 Query 头、4 个 Key/Value 头、每头 96 维、SwiGLU 中间维度 2,432。
独立参数为 63,912,192；输入 Embedding 与输出 LM Head 共享权重，不能按 state_dict
逐项相加后把同一块权重计数两次。该快照的 state_dict 合计 68,827,392 不是独立参数量。

数据流为 token IDs → Embedding → 8 个 Decoder Block → RMSNorm → 共享输出投影 → 词表 logits。
每个 Block 采用 Pre-Norm 残差结构，包含因果 GQA 和 SwiGLU。GQA 先生成 8 组 Q、4 组 K/V，
当前实现将 K/V repeat 到 8 头后调用 PyTorch SDPA；不应据此声称实现了专门优化的原生 GQA 内核。
Q/K 带 RMSNorm 和 RoPE。因果遮罩保证当前位置看不到未来文本。

配置的位置上限 32,768 只是接口容量；正式文本训练长度为 340/768，**不能证明长上下文能力**。
当前生成循环逐 token 重算上下文，没有 KV cache；吞吐只能代表本实现，不能直接代表同规模优化引擎。

## 3. 数据与 loss mask：有输入不等于有监督

预训练 [JsonlPretrainDataset](../src/minimind_lab/data/pretrain.py) 以文件字节偏移实现随机访问，
每条样本截断后加入 BOS/EOS，右侧 padding 到固定长度，不做多文档 packing。
labels 从 input IDs 复制，padding 位置变为 `-100`。模型以 `logits[:, :-1]` 预测
`labels[:, 1:]`，因此输入和目标错开一位。

[SFT 数据](../src/minimind_lab/data/sft.py) 使用同一个 Tokenizer 的 chat template；system/user
文本仍作为上下文，但只保留 assistant 回答及回合终止标记的 labels，其余均为 `-100`。
图像/视频占位 token 同样不作为回答监督。这并不意味着被 mask 的上下文“没有作用”：
回答预测仍依赖这些输入，梯度可以通过注意力传回允许训练的上游模块。

文本数据尾部 2,048 条为固定留出集，训练采样明确排除。它是同源留出，不是外部基准；
行级分离不自动证明语义去重。Video QIVD 按 `SHA256(seed:ID)` 排序划分 2,400/250/250，
避免文件重排改变划分，但仍需如实说明这不是按拍摄来源分组的严格泛化测试。

## 4. VLM：不是给语言模型多加一个字符串

实现见 [VLM](../src/minimind_lab/vlm/model.py)。冻结的 SigLIP2 将 256×256 图像编码为
64×768 patch 特征；Projector 使用 LayerNorm → Linear → GELU → Linear，将其映射到语言空间。
单图文本中预留 64 个 ID=12 的位置，实际送入 LLM 的是这些位置替换后的视觉 embedding，
不是让模型仅靠 `<image>` 文字猜图。双图依输入顺序替换 128 个位置，图像数量与占位数量不匹配即报错。

Alignment 只更新 Projector，目的是先建立输入接口，避免随机视觉映射直接扰动语言核心。
SFT 再开放首尾 Decoder Block；中间 6 层、Embedding、输出头及最终 Norm 仍冻结。
“首尾层解冻”不等于“全参数微调”，也不等于“只训练最后一层”。

## 5. Video-Omni：时间信息在哪里进入

实现见 [视频模型](../src/minimind_lab/video/model.py) 和 [视频数据](../src/minimind_lab/data/video.py)。
每段视频均匀抽取 8 帧，包含首尾；SigLIP2 得到 `[B,8,64,768]` patch 特征。
每帧用可训练 query 做空间注意力池化，得到 `[B,8,768]`，加入可训练帧位置，再经过
2 层非因果 Temporal Transformer。16 个可训练 query 通过交叉注意力重采样为 `[B,16,768]`，
经 Projector 替换语言序列中 16 个 ID=13 的视频位置。整个视频可被同时编码，因果限制作用于文本解码。

冻结编码器允许预缓存全部视频 patch 特征；缓存不是答案标签，也不训练视觉编码器。
当前缓存为 float16，读取后转 float32 并在 CUDA autocast 下参与训练，省去重复解码和视觉前向。
帧位置使网络具备表达时间顺序的可能，但**架构中存在时间模块不等于学会时间推理**，需倒序消融验证。
8 帧采样可能遗漏短暂事件，空间池化可能丢失小物体，均需作为局限。

## 6. 训练工程与可恢复性

六阶段 CUDA YAML 固定 micro-batch 与累积步数。日志中的 `step` 是 micro-step，不是 optimizer
update；有效 batch 为二者乘积。317,048 次预训练 micro-step 包含末尾不满 32 次的累积组，
代码会重标定该组梯度再更新，不能简单用向下取整代表最终更新数。

数据顺序由 seed + epoch 的确定性置换决定；恢复点包含模型、AdamW、配置、步数、history、
CPU/CUDA RNG 和活跃/暂停时间。仅在完整 optimizer 边界保存恢复点，临时文件通过原子替换落盘。
CUDA 正式实验使用 BF16 autocast，保留非有限 loss/gradient 检查；默认 CUDA 运算不承诺逐位确定性。
此前 deterministic 探针的逐位恢复一致，只证明该探针设置，不是所有正式运行的通用保证。

配置、源码、依赖和数据指纹由 [RunEvidence](../src/minimind_lab/training/evidence.py) 固定。
阶段开始验证未变更，阶段结束归档权重、恢复点、评估和 SHA-256。报告文稿可以继续编写，
不得借此静默修改本次训练实现或预算。

## 7. 结果口径与不能越过的结论边界

- 训练日志 loss 是被记录 micro-batch 的均值，不是整段区间或整个数据集的 token 加权均值。
- 最终语言 NLL 按有效预测 token 数加权，PPL 为其指数；BPB 的字节分母来自解码后的有效文本，
  并非原始 JSONL 全文件字节数。不同目标 mask 的训练 loss 不可直接横比。
- `padded_tokens_per_second` 为名义 batch×长度/时间，包含 padding 和不受监督上下文，不能写成有效监督 token/s。
- 模型分配显存、PyTorch 保留显存与 nvidia-smi 进程/设备占用不是同一口径。
- 当前 CUDA 文本生成计时在输出转 CPU 之前结束，未显式同步设备。该耗时只作实现内诊断；
  最终若讨论精确推理速度，需要另行带同步、预热和重复次数的计时实验，不用现有字段冒充严谨硬件基准。
- 固定基础提示并非全部使用 chat template，因此反映统一接口下的回归，不等价于完整 chat 指令遵循基准。
- 正确图与错图、正常帧与倒序帧的差异是条件利用的线索；重复率、幻觉、弱关键词评分和语言退化也须展示。

最终报告应逐项回答：设计预期是什么、实际改了哪些参数、用什么证据检验、结果支持什么、
结果不支持什么。完整结果尚待六阶段固定评估完成。
