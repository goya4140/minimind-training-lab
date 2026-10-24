# CUDA Video-Omni SFT：训练完成，但未观察到帧序利用

六阶段训练和原固定评估已经完成，视频全量回答独立重放与逐样本评分复核也已完成。
这不等于完整项目或报告交付完成；视觉证据与综合解读仍需完善。

当前 run 为 `mm64-cuda-vlmfix-v3-20261024T170500Z`，训练代码
`9eafe5a548df7d5eae7ce2efe420e512a52d647e`。LLM 两阶段继承父 run
`mm64-cuda-20261024T075332Z`、代码 `53f9b9f` 的精确归档，不把继承伪装成重新计算。
[六阶段归档及父运行血缘核验](six-stage-final-lineage-20261025.json)已逐文件验证；
该核验仅证明归档内容与血缘，不证明能力、数据无泄漏或报告交付完成。

## 设计和实现

视频分支从 LLM SFT 独立扩展，先训练 Temporal Adapter 与 Video Projector，随后本阶段联合训练
这两个模块和语言主干的第0、7层，共34,854,528个可训练参数。其余语言层与公开预训练 SigLIP2
冻结。不继承 VLM SFT，不包含音频、视频生成或通用全模态能力。

输入为8帧视频和问题，输出为文本。训练复用已核验的冻结视觉 patch 缓存；固定评估从原MP4
解码并执行视觉前向。训练耗时不包含先前缓存构建，不是完整端到端复现成本。

| 训练项 | 实测或固定配置 |
|---|---|
| 训练／验证／独立QIVD test | 2,400 / 250 / 250 |
| microsteps／optimizer updates | 4,800 / 600 |
| microbatch／梯度累积／序列长度 | 1 / 8 / 384 |
| 活跃训练耗时 | 260.472036839秒，约4分20秒 |
| 正常／倒序验证loss | 1.042245714784 / 1.042245786071 |
| 阶段完成UTC | 2026-10-24 18:25:46.925430 |
| 原流水线完成事件UTC | 2026-10-24 18:25:50.927007 |

4,800步相当于2,400训练样本的2个epoch。验证loss倒序减正常仅约7.13e-8，不能因其为正就
宣称时序能力。下一节是独立固定test及生成协议，不与验证样例混作同一分数。

## 固定评估：保留负面结果

| 指标 | QIVD | 受控时序集 |
|---|---:|---:|
| loss评估条数 | 250 | 160 |
| 正常帧loss | 1.042471869409 | 0.655733847432 |
| 倒序帧loss | 1.042471861482 | 0.655733643845 |
| 实际生成条数 | 100 | 160 |
| 正常／倒序 normalized exact match | 0 / 0 | 0 / 0 |
| 正常／倒序 contains-reference | 0.18 / 0.18 | 0.2625 / 0.2625 |
| 正常／倒序 token F1 | 0.087945887446 / 相同 | 0.127777777778 / 相同 |
| 归一化回答变化率 | 0 | 0 |

loss为各batch标量loss的算术平均，不是全体监督token加权NLL。QIVD只对固定test前100条
生成回答，不能把生成指标写成250条。受控集包括水平运动、竖直运动、大小变化和事件先后，
每类40条；正常F1依次为0、0、0.166666666667、0.344444444444。

两套生成集合均未观察到归一化回答随倒序改变，F1也完全相同；loss差在数值微小尺度，且test
倒序loss略低。它们不支持本基线学会利用帧序的结论。也不应据此断言任何视频、任何解码设置下
模型都对帧序不敏感。EM为0与contains-reference非零可以同时出现：回答可能包含参考词，
但冗长、复读、格式或内容仍不匹配。必须展示原文，不能把关键词包含当作正确推理。

倒序评分仍沿用**原视频**参考答案，不是倒序视频自身正确答案；受控运动反向后语义真值会改变。
本次保留固定评分，不根据测试结果回调训练参数或改动接受答案。

## 语言回归和独立指令诊断

同一2,048条文本留出集、670,079个预测token、1,906,560 UTF-8字节上，NLL为
1.920499257203、PPL为6.824364732166、BPB为0.973787538987。
LLM SFT的同口径PPL为6.779640879465；视频扩展没有带来该通用文本指标的改善。
原视频脚本文本生成上限为48 token，而独立LLM脚本为80、VLM为96，不能直接比较复读比例或生成速度。

另已完成冻结 `chat-diagnostic-v1` 的CPU FP32、greedy、80-token纯语言诊断，运行ID为
`chat-v1-video-sft-final-20261025`，18:30:30.049140–18:30:49.355343 UTC。
严格参考匹配为0/8，2条EOS停止、6条达到80-token上限；所有评分和文件哈希已独立复核。
这是语言主干抽取后的8条项目自编诊断，不含视频输入，也不是外部泛化基准。
失败原因仍需结合逐条原文分析，不能把严格格式失败一概解释为完全缺乏相关知识。
逐样本文件SHA-256为 `72579ab7a7e86a04f148f7fa7c05c02a7d85b3ee304022a042b8fea391d5d37a`。

## 最终恢复状态与归档

[独立恢复审计](video-sft-final-integrity-20261025.json)确认step=4,800、配置一致、更新边界与RNG完整；
136个模型state tensors、84,017,664个去重存储参数全部有限；optimizer67个entries、201个张量
全部有限；history241条。保存参数包括冻结语言权重，不含独立外部SigLIP2，不能当作全系统总参数。

[最终权重关系审计](final-weight-relations-20261025.json)进一步确认该阶段136项model-only张量
与resume中的模型状态逐项相等。视频Alignment的91项语言权重与LLM SFT完全一致；视频SFT中
39项时序模块、6项投影器、22项语言首尾层改变，69项其他语言权重保持一致。
这补充实际冻结策略证据，不是逐步梯度追踪、外部视觉张量比较、实际重启或时序能力证明。

服务器工作目录：`/root/autodl-tmp/wlh_goya/MiniMind/experiments/vlm-slot-v3`。
下列文件保留于 `artifacts/runs/mm64-cuda-vlmfix-v3-20261024T170500Z/stages/video-omni-sft/`：

| 文件 | SHA-256 |
|---|---|
| video-omni-sft-cuda.pt | `2f320a04ecfee4cb46863b3698cb680cce41407bfb93990e28c028680a9876b8` |
| video-omni-sft-cuda.resume.pt | `53d67f18f4bb01b4dcdeadb7e2b9c9d048dc6523cb035d42dadf5dc792fcd32d` |
| video-omni-sft-cuda.json | `00073fc3ed0820f589d054c5fc4179f1e14e690486ac5529c6922c27d83be235` |
| video-omni-sft-cuda.console.log | `ded5af49e2868c86fe9701a26feeabef363a50d42f61df178bba5f1de54ac87a` |
| video-omni-final.json | `8c054eaa60a032c24c9fee56682bb610166ec948fcba91c94e0b24bd83b56d96` |

原评估只保留每套12条qualitative，不声称已留存其260条原执行回答。另注册独立重放
`video-full-output-replay-final-20261025`，只增加全部回答留存，不覆盖上述原评估。
该重放已完成，[260条全量输出核验](video-full-output-replay-integrity-20261025.json)逐条重新计算
EM、参考包含、bag-of-token F1和倒序变化标记，复核类别均值及汇总与原基线一致。
此次100条QIVD与160条受控样例不仅归一化后相同，正常／倒序的**原始回答文本也逐字相同**。
这是独立重放的证据，不能冒充原执行未保存的回答。

例如QIVD id6300问题“What object I'm holding?”，参考“A leaf”，模型两种帧序均答
“You are holding a pair of pairs.”；F1为0.2222主要来自英文冠词等重叠，不代表识别正确。
受控id146问圆形变大还是变小，参考“larger”，两种帧序均答“The circle is not smaller.”。
这里短答案词评分得0，但“not smaller”本身也不严格等于“larger”（允许不变），要结合真实帧检查，
不能为提升分数事后改参考或接受词。
[实际受控视频8帧及正常／倒序回答](evidence/mm64-cuda-vlmfix-v3-20261024T170500Z/E005-controlled-video-final/README.md)
现已按每类第一条固定规则展示并逐图检查，包含四类失败解读；不是生成图片或终端截图。

全量JSONL及其工具、manifest保留于本run的
`supplemental-evaluations/video-full-output-replay-final-20261025/`，不上传GitHub。
[最终5张曲线、CSV与来源](evidence/mm64-cuda-vlmfix-v3-20261024T170500Z/E002-video-sft-final/README.md)
已完成120桶独立复算和逐张视觉检查。受控视频输入输出展示已完成，最终综合解读与审计仍需完善。

[E011真实终端截图](evidence/mm64-cuda-vlmfix-v3-20261024T170500Z/README.md#e011最终视频评估真实终端截图)
展示归档评估指标和两套集合各第一条完整回答，已检查可读性及敏感信息。
它是只读查看原结果的实际窗口，不是重新运行推理，也不是输入视频帧的展示。
