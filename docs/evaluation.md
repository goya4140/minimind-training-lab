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

## Video-Omni

- 固定 250 条 held-out QIVD test 的 assistant-only loss；
- 在固定前 100 条 test 样本上报告 normalized exact match、reference containment 与 token F1；
- 按 QIVD category 分组报告 token F1，避免总体均值掩盖弱项；
- 对相同视频倒序 8 帧，报告 loss 差值、token F1 差值和回答变化率；
- 保存至少 12 组问题、参考答案、正常帧回答、倒序帧回答作为定性样例；
- 报告单样本生成耗时，并重新运行 LLM 文本评估检查能力遗忘。

倒序评估是必要的时间建模门禁：正常输入优于倒序输入才构成模型利用帧序的证据。若两者无差异，
最终报告必须写成“时序敏感性未建立”，不能仅凭视频 QA 文本分数宣称视频理解成功。后续可使用
VisualBench 等专门要求跨帧推理的外部数据作第二重检验，但不与 QIVD 训练集混用。

最终报告必须注明模型规模、冻结外部模块规模、数据范围与硬件，不能只展示训练 loss。
