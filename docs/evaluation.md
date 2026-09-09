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
并通过冻结 Mimi 解码为 24 kHz WAV。CER/WER、speaker similarity 和 barge-in 属于后续自动量化门禁，
不能用主观试听替代。

最终报告必须注明模型规模、冻结外部模块规模、数据范围与硬件，不能只展示训练 loss。
