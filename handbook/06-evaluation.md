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

### PPL 的分母和“变化”如何计算

固定文本 evaluator 先累计 `batch_mean_loss × 该批有效预测token数`，再除以全部有效 token 数，
得到全局 NLL；`PPL=exp(NLL)`。PPL 可理解为几何平均概率尺度上的不确定性，
不是“平均每次有多少个正确答案”，也不是事实正确率。可比性要求相同语料、Tokenizer、截断及 mask。

当前 CUDA 实测 Pretrain PPL=6.222716802、SFT PPL=6.779640879：
绝对差为约 0.556924077，相对增幅为 `100×(6.779640879/6.222716802−1)≈8.95%`。
别把绝对差写成百分比，也别把这一变化推广到所有任务。
训练结束摘要的 batch 平均 loss 与固定 evaluator 的 token 加权 NLL 不完全相同，
SFT 自身对话验证集与通用文本回归集更不是同一数据；见[真实结果总览](../reports/cuda-llm-final-overview.md)。

Distinct-2 低可能提示复读，但随机无意义输出也能有高多样性；严格短答 0 分可能来自多余解释，
也可能真的答错。必须保留原始回答，区分“内容错误”“内容相关但格式错误”“截断/未 EOS”。
项目另有[8 题固定 chat-template 诊断](../reporting/protocols/README.md)，不是外部 benchmark；
不能将其严格通过率当作总体智能分数。

## VLM 指标

VLM 同时报告 validation loss 和固定图像关键词召回。关键词召回只是轻量词面代理，不是语义正确率下限或完整 VQA benchmark；
因此还要保存原图问题、生成答案和失败样例，并重新检查纯文本能力是否退化。
例如双图题即使先后关系说反，只要包含全部关键词仍可能得满分；回答对错图发生变化也可能是两次都错。
图序理解必须结合对象与顺序的实际回答审阅，不由关键词差或回答变化率单独判定。

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
[`scripts/build_final_report.py`](../scripts/build_final_report.py) 从本地证据自动生成，但输出只是固定模板汇总，
需完成[独立归档及结论复核](../reporting/final-evidence-review.md)，不能以脚本成功取代完整验收。
