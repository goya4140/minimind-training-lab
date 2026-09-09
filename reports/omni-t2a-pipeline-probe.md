# Omni T2A 真实数据管线探针

运行日期：2026-09-09  
目的：在正式 Omni 训练前，确认已下载的真实 T2A 数据能贯通 tokenizer、数据适配器和
Thinker–Bridge–Talker 前向。该探针使用随机初始化的小模型，只验证结构与数值稳定性，
不代表生成质量。

## 输入与完整性

- 数据：`data/raw/sft_t2a_mini.parquet`
- 固定 revision：`d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de`
- 文件大小：1,558,442,729 bytes
- SHA-256：`dfe44b8b263ecd0579627160cf258b363b4c18457ae03221691e2e1a85e60ab8`
- Parquet：515,415 行、1 个 row group
- tokenizer：本仓库 `assets/tokenizer`，词表 6,400；`<|audio_pad|>` 的 token id 为 16

## 数据适配结果

首条真实样本在 `sequence_length=256` 下产生：

- `text_ids`: `[1, 255]`
- `audio_ids`: `[1, 8, 255]`
- 有效文本监督 token：181
- 有效音频监督 token：1,772
- 音频目标 id 范围：1–2,050，包含 stop token 2,050

## 小模型前向

探针模型为 9,786,018 参数，语言 hidden size 64、2 层 Thinker、1 层 Talker、8 个音频码本。

| 指标 | 结果 |
|---|---:|
| text logits | `[1, 255, 6400]` |
| 每个 audio head logits | `[1, 255, 2112]` |
| text loss | 8.776473 |
| audio loss | 7.845820 |
| total loss | 16.622293 |
| finite | true |

结论：真实 T2A 样本的外层 `answer_audios` 按 assistant turn 组织，内部 Mimi code 为帧优先
交错；当前数据适配器能够正确选择末次 assistant 音频、拆分 8 码本并产生可训练的有限 loss。

## 完整尺寸装配复验

2026-09-09 使用 step 12,000 的真实 63.9M LLM 中期快照装配 8 层 Thinker、4 层 Talker，
并按正式策略从 Thinker 后四层复制初始化 Talker。完整 Omni 主体共 111,950,082 参数；T2A
阶段冻结两个 Projector 后，可训练参数为 109,781,762。

真实首条 T2A 样本在 `sequence_length=128` 下得到 104 个文本、748 个音频监督 token；前向
产生 `[1, 127, 6400]` 文本 logits 和 8 份 `[1, 127, 2112]` 音频 logits，text/audio/total
loss 分别为 4.765340、7.805069、12.570409，全部为有限值。由此确认正式 LLM state dict、
Talker 层复制和 8-codebook heads 在完整尺寸下兼容。正式训练仍从最终 LLM SFT 权重开始。
