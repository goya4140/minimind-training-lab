# Omni A2A 真实数据与冻结编码器探针

运行日期：2026-09-09  
目的：在正式 A2A alignment 前，验证真实音频输入、SenseVoice frontend/encoder、音色条件、
Mimi 延迟 targets 与 Thinker–Bridge–Talker 前向能够贯通。随机初始化的小模型仅用于结构检查。

## 数据完整性与结构

- 数据：`data/raw/sft_a2a_mini.parquet`
- 固定 revision：`d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de`
- 文件大小：881,313,734 bytes
- SHA-256：`fba0159e424ee106c9e5a732fe607875b3780d0c9f8b6806038879acd279782b`
- Parquet：76,797 行、1 个 row group
- 字段：`conversations`、`question_audios`、`answer_audios`、`ref_audios`、`spk_emb`

首条样本包含 6,861 bytes 的问题音频、1,536 个回答 code、1,344 个参考音色 code，
以及 192 维 speaker embedding。回答和参考 code 分别对应 192 与 168 个 8-codebook 帧。

## 真实冻结编码器与模型前向

使用真实首条音频、`sequence_length=256`、冻结 SenseVoiceSmall，以及 9.8M 参数的随机小型
Omni 主体：

| 检查项 | 结果 |
|---|---:|
| frontend features | `[1, 53, 560]` |
| SenseVoice encoded features | `[1, 53, 512]` |
| encoded feature length | 53 |
| speaker embedding | `[1, 192]` |
| speaker token position | 0 |
| 有效文本监督 token | 46 |
| 有效音频监督 token | 1,308 |
| text loss | 8.680292 |
| audio loss | 7.843444 |
| total loss | 16.523735 |
| finite | true |

结论：真实 A2A 数据的音频、参考音色和 8 码本监督均能进入模型，SenseVoice 的 512 维输出
与 Audio Projector 配置一致；当前 alignment/SFT 管线在数值上可运行。
