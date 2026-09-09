# 评估协议

## 通用门禁

所有模型必须通过：

1. 模型前向与 loss 单元测试；
2. 固定 seed 的训练冒烟测试；
3. checkpoint 保存、重新加载一致性；
4. 输入输出张量和 loss mask 检查；
5. 与训练集分离的评估集。

## LLM

- Validation loss、Perplexity、Bits Per Byte；
- 固定基础知识、指令跟随、重复生成测试；
- tokens/s、峰值显存与首次生成延迟；
- 生成样例必须同时展示成功和失败案例。

## VLM

- Caption、VQA、OCR 与视觉幻觉测试；
- 固定图像提示词记录关键词召回率，作为轻量、可复现的客观下限；
- 多图顺序和指代测试；
- 重新运行 LLM 文本评估，测量语言能力遗忘。

## Omni

- 语音输入理解 Accuracy、CER/WER；
- 语音输出 CER/WER 与 Speaker Similarity；
- VQA/Caption；
- 首文本 token 延迟、首音频帧延迟、实时系数；
- barge-in 成功率；
- LLM/VLM 能力回归测试。

仓库固定 `data/eval/omni/` 的中英文语音、图像和纯文本输入。`scripts/evaluate_omni.py`
对同一 checkpoint 运行 T2A、A2A 和 I2A，保存文本回答、音频 code 帧数、端到端耗时，
并通过冻结 Mimi 解码为 24 kHz WAV。语音输出 CER/WER 已自动量化；speaker similarity 和
barge-in 属于后续门禁，不能用主观试听替代。

`minimind_lab.evaluation` 已提供不依赖第三方库的 Levenshtein、CER 和 WER，供冻结 ASR
转写后统一计算；中文 CER 会忽略空白，英文 WER 不区分大小写。

当前 `evaluate_omni.py` 已把语音输出经 Mimi 解码，再用冻结 SenseVoiceSmall 转写，以模型的
文本回答为 reference 计算逐样本 CER/WER，同时记录音频时长与 real-time factor。转写前会去除
SenseVoice 控制标签、Unicode 规范化并移除标点。speaker similarity 和 barge-in 仍是未实现门禁，
最终报告不得把它们写成已通过。

最终报告必须注明模型规模、冻结外部模块规模、数据范围与硬件，不能只展示训练 loss。
