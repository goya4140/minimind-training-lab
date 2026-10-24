# 06 · 评估：从“loss 降了”到“模型真的会了”

## 三层证据

一次可信评估至少包含：

1. 结构证据：forward、backward、save/load、冻结策略测试通过；
2. 统计证据：在未训练数据上报告 loss 和任务指标；
3. 行为证据：固定输入的成功与失败生成、消融实验。

只有训练 loss 属于“优化过程证据”，不能代替后两层。

## LLM 指标

- Validation loss：held-out token 的平均负对数似然；越低越好；
- Perplexity：`exp(loss)`，可理解为平均不确定性的尺度；
- Bits per byte：按 UTF-8 字节归一，减少 tokenizer 粒度差异；
- Distinct-2：生成中不同二元 token 组合比例，辅助发现复读；
- tokens/s：效率指标，不是能力指标。

固定生成必须使用相同 prompt、tokenizer、temperature 和 max tokens，否则不能直接横向比较。
最终发布门禁会核对四组语言生成的提示文本与顺序完全一致，而不只比较样本数。

LLM 必须保留 Pretrain 与 SFT 两份独立评估 JSON，并用同一文本留出集和固定提示比较。这样读者能
直接观察“续写模型”变成“指令模型”时，PPL、回答格式与内容发生了什么变化。

## VLM 指标

VLM 同时报告 validation loss 和固定图像关键词召回。关键词召回只是轻量下限，不是完整 VQA benchmark；
因此还要保存原图问题、生成答案和失败样例，并重新检查纯文本能力是否退化。

## Video-Omni 指标

QIVD held-out test 报告：normalized exact match、reference containment、bag-of-token F1、按类别 F1。
主观长答案常有多种合理说法，所以 exact match 很严格；token F1 较宽松，但也会奖励只复述关键词的答案。

受控时序集让顺序扰动更容易解释：同一个视频倒序后，移动方向、尺寸变化和事件先后的正确答案会
反转。但当前脚本给倒序回答打分时仍使用**原视频答案**，所以两列不是各自正确答案下的准确率：

```text
normal score > reversed score  → 原答案在正常视频下更受支持；还需核对倒序回答是否正确
normal score ≈ reversed score  → 不能确认时序能力；也可能是两次都错或评分不敏感
normal score < reversed score  → 检查逐样本错误与偏置；不能仅凭差值判定工程故障
```

例如“red square”和“blue circle”的首事件对换后答案不同，但若回答同时包含两组词，词袋 F1 仍
可能相同。反过来，倒序后回答变化也可能是错误或不稳定，而非理解了事件先后。必须同时看正常输入
正确率、逐对回答、任务类别与参考含义，不能用单个分差替代行为验证。

原固定视频汇总只保存前 12 条回答。为满足完整复核要求，另有
[全量证据重放](../reporting/video-evidence-replay.md)，保留 100 条 QIVD 与 160 条受控样例全部回答；
其结果使用独立 eval ID，不能声称来自原先未留存的那次生成。

正式报告只接受完整协议：4 个固定语言提示、7 个 VLM 案例、250 条 QIVD test loss、前 100 条
QIVD 生成，以及 160 条受控时序正常/倒序生成。缩短样本数的调试运行不能冒充最终评估。

## 避免指标误读

- 不在 test set 上调学习率或选择 checkpoint；
- 不删掉难看的固定样例；
- 不把 frozen backbone 的能力归功于本次从零训练；
- 不把一个 smoke run 的低 loss 当成正式模型结果；
- 指标为 0 或倒序消融失败时如实记录。

完整协议在 [`docs/evaluation.md`](../docs/evaluation.md)，最终结果由
[`scripts/build_final_report.py`](../scripts/build_final_report.py) 从本地证据自动生成。
