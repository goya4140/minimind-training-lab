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

Pretrain 与 SFT checkpoint 使用完全相同的文本留出集、tokenizer 和固定提示分别评估。最终报告并排
展示 PPL 变化与生成结果，避免只凭 SFT 训练 loss 判断指令微调是否有效。

## VLM

- 固定图像描述、VQA 与视觉幻觉检查；
- 固定图像提示词记录关键词召回率，作为轻量、可复现的客观下限；
- 固定双图顺序和指代测试；
- 单图案例替换为固定错图，双图案例反转图像顺序，报告关键词召回差与回答变化率；
- 重新运行 LLM 文本评估，测量语言能力遗忘。

VLM 和 Video-Omni 的纯文本回归使用与 LLM 相同的 held-out 文本、tokenizer、loss/PPL/BPB
计算和固定提示。最终报告同时展示 PPL 变化与同一提示的三模型回答，用于区分“多模态能力增加”
与“语言能力被破坏”。

## Video-Omni

- 固定 250 条 held-out QIVD test 的 assistant-only loss；
- 在固定前 100 条 test 样本上报告 normalized exact match、reference containment 与 token F1；
- 按 QIVD category 分组报告 token F1，避免总体均值掩盖弱项；
- 对相同视频倒序 8 帧，报告 loss 差值、token F1 差值和回答变化率；
- 保存至少 12 组问题、参考答案、正常帧回答、倒序帧回答作为定性样例；
- 报告单样本生成耗时，并重新运行 LLM 文本评估检查能力遗忘。

此外，`scripts/generate_temporal_benchmark.py` 固定 seed 生成 160 个无文字提示的视频，均分为
水平移动、垂直移动、尺寸变化和事件先后四类。该集合不参与训练；对 160 条全部计算正常/倒序 loss，
对 160 条全部生成正常/倒序答案。倒序会反转每道题的正确语义，因此 normal-minus-reversed 指标比
普通真实视频倒序更容易解释。

倒序评估是必要的时间建模门禁：正常输入优于倒序输入才构成模型利用帧序的证据。若两者无差异，
最终报告必须写成“时序敏感性未建立”，不能仅凭视频 QA 文本分数宣称视频理解成功。后续可使用
VisualBench 等专门要求跨帧推理的外部数据作第二重检验，但不与 QIVD 训练集混用。

最终报告必须注明模型规模、冻结外部模块规模、数据范围与硬件，不能只展示训练 loss。
