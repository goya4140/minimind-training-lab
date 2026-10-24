# CUDA VLM SFT：训练、固定评估与失败解读

状态：VLM 两阶段与固定评估已完成，视频分支已启动；不是六阶段全部完成声明。
run：`mm64-cuda-vlmfix-v3-20261024T170500Z`；训练 commit：
`9eafe5a548df7d5eae7ce2efe420e512a52d647e`。

## 1. 实际设计与训练

本阶段继承已完成的 [VLM Alignment](cuda-vlm-alignment-final.md)，更新 Projector 与 Decoder
第 0、7 层，共 15,931,776 可训练参数；其余语言参数与公开预训练 SigLIP2 冻结。
语言主干最初来自父 run 的随机初始化预训练及 LLM SFT，并非本阶段重新从零训练。
单图有 64 个视觉位置，双图有 128 个；文本解码器预测答案，不生成图片。

| 固定配置／结果 | 数值 |
|---|---:|
| microsteps / optimizer updates | 30,000 / 3,750 |
| batch / accumulation / sequence length | 2 / 8 / 768 |
| 学习率 | 5e-6 → 5e-7 |
| seed / precision | 42 / BF16 autocast |
| 训练活跃秒数 | 2,105.572785（约35分06秒） |
| 阶段结束验证 loss（末尾512行） | 2.624279913 |
| 固定 evaluator 验证 loss（末尾128行） | 2.657603383 |
| 完成并归档事件 UTC | 2026-10-24 18:12:48.741131 |

原始图文 SFT 数据 2,904,511 行，按恰好一个图像标记筛选保留 2,654,827 行，排除 249,684 行。
末尾 512 行不参与训练；固定 evaluator 的 128 行是其中子集，不是另一份独立测试集。
本预算在第一 epoch 中访问 60,000 个训练行位置，约为训练行数的 2.26%，不是完整遍历全部数据。
行级划分不证明图像/文本语义去重；不能仅由覆盖率解释全部失败。

两种验证 loss 均是 batch 标量均值，不是全局 token 加权 NLL；验证器会跳过非有限 batch，
汇总不含跳过计数，不能据有限均值声称所有 batch 均有限。
也不能拿不同数据的 Alignment/SFT loss 差直接证明问答能力提升。

## 2. 固定图像与反事实评估

保持原固定协议：6 单图 + 1 双图，greedy，最多 96 个新 token；单图替换为固定错图，
双图反转输入顺序。全部正常/反事实回答保存在归档 `vlm-final.json`，没有删去差的案例。
已独立复算 7 题的全部关键词分数、回答变化标记和 macro 平均，与原汇总一致。

| 指标 | 正常输入 | 错图／倒序输入 |
|---|---:|---:|
| 全7题 macro 关键词召回 | 0.321429 | 0.178571 |
| 仅6道单图 macro 关键词召回 | 0.333333 | 0.166667 |
| 双图题关键词召回 | 0.250000 | 0.250000 |

全7题差值为 0.142857（14.29 个百分点），回答变化率为 7/7。
这是小型固定样例上的词面指标差异，不是语义 VQA 准确率或稳定视觉 grounding 证明。
双图两次都得 0.25，原图回答遗漏第二幅伞图，倒序后仍把第一幅描述为汽车，图序能力未通过实际检查。
全部[固定输入与逐题摘录/解释](evidence/mm64-cuda-vlmfix-v3-20261024T170500Z/E004-vlm-fixed-cases/README.md)
同时展示正确信息、遗漏、幻觉与重复。

尤其要警惕两个方向的评分偏差：自行车/赛车等词已在问题中出现，复述题目也能得分；
反之，模型说“黄色轿车”而关键词要求“黄色、汽车”，同义词未被收录会少计分。
关键词正差不能消除伞图认成小熊猫等明显错误，回答变化率也可能来自两次都错。
不根据本次输出修改关键词、提示词、训练参数或最终检查点选择。

## 3. 语言回归与独立对话诊断

相同通用文本 2,048 条、670,079 个预测 token、1,906,560 UTF-8 bytes：

| 模型 | token加权NLL | PPL | BPB |
|---|---:|---:|---:|
| 父 LLM SFT | 1.913924133 | 6.779640879 | 0.970453628 |
| VLM SFT 语言主干 | 1.935256764 | 6.925822130 | 0.981270321 |

PPL 相对父 LLM SFT 上升约 2.156%，说明在该固定通用文本回归上退化，不代表所有语言任务都退化。
原 raw-prefix 生成仍见复读、角色漂移及 Transformer/TensorFlow 混淆；VLM 原生成上限 96、
父 LLM 为80，不能把长度/复读率/速度当作严格同条件比较。

另使用此前冻结的 [chat-diagnostic-v1](../reporting/protocols/README.md)，只抽取语言主干，
不输入图片：CPU FP32 greedy、最多80新token、EOS停止、无KV cache，与已完成 LLM 补充诊断一致。
eval ID `chat-v1-vlm-sft-final-20261025`，UTC 18:13:36–18:13:54 完成，严格通过率仍为 0/8；
3/8 遇 EOS，5/8 到80-token上限。它是项目助手构造的小型诊断，不是外部通用 benchmark。

| 题号 | 实际回答概要（定性阅读，不改评分） |
|---|---|
| C01 首都 | “中国的首都是北京。”事实正确，但违反仅城市名 |
| C02 2+3 | 给出 `2 + 3 = 5`，又说需“再加上3”，解释冗长且矛盾 |
| C03 复制 | 转为选择和调整横幅的步骤，未原样输出 |
| C04 翻译 | 译出“猫”，但附整句解释，违反只输出译文 |
| C05 排序 | 给 Python sorted 教程，未按要求输出数字列表 |
| C06 JSON | 复读“只输出一个字段”，不是合法目标 JSON |
| C07 情感 | 判断正面并附解释，违反仅标签 |
| C08 号码 | 混用73与26且复述步骤，未给出简短可靠答案 |

完整预测仅留服务器 `supplemental-evaluations/<eval-id>/`，含真实渲染 prompt、token IDs、
参考、EOS与评分。预测 SHA `1fb75c61aa6187d4b2cd0970474e3bd04d0b32e8d83586d1ddca5bb0209e95a5`；
manifest SHA `8f93fa769060d27fae80b0f9f07d96a7204e47807f031da5cbee88897672489e`；
summary SHA `bd6a2a13c0182f0eacd8b0f09e512600786b924d8ac44003f6f5470055314fa0`。
与 LLM summary 同哈希是严格评分相同，不是预测相同；协议与 evaluator 指纹保持不变。

## 4. 完整性、冻结策略、曲线与保留

[最终恢复审计与分数复算](vlm-sft-final-integrity-20261025.json)：step=30,000，97 个模型 state
tensors、optimizer 28 个 entries / 84 个张量全部有限，配置、RNG与更新边界完整，history 1,501条。
[最终权重冻结对照](vlm-freezing-final-20261025.json)再次确认：Projector 6 项及首尾层22项变化，
其余69项与 Alignment 完全一致；Alignment 的91项语言权重与父 LLM SFT 完全一致。
这是最终快照证据，不是逐步梯度追踪或外部视觉权重比较。

[最终5张曲线与CSV](evidence/mm64-cuda-vlmfix-v3-20261024T170500Z/E002-vlm-sft-final/README.md)
来自最终归档恢复文件：150个200-step桶全部复算，5图逐张检查且哈希核对通过。
首末桶训练loss均值为2.853958390→2.618599463，不把下降当作视觉能力验证。
固定评估的[真实终端截图E010](evidence/mm64-cuda-vlmfix-v3-20261024T170500Z/README.md#e010固定-vlm-评估真实终端截图)
已补齐，不以图表或输入图片冒充运行截图。

服务器根目录：`/root/autodl-tmp/wlh_goya/MiniMind/experiments/vlm-slot-v3`。
阶段归档目录：`artifacts/runs/mm64-cuda-vlmfix-v3-20261024T170500Z/stages/vlm-sft/`。

| 文件 | SHA-256 |
|---|---|
| vlm-sft-cuda.pt | `13c23d6aa16c789d3910cf7a46de12be35e0fd2bc94bab908899f3f3f47ee332` |
| vlm-sft-cuda.resume.pt | `a50869adec6322d31ff1117b14a2ab43b8c971f7ef933408d63f95eb6933835c` |
| vlm-sft-cuda.json | `05c2a792654b830a22352dde81df7a7f145745933405cf2c1cb897eb02509e14` |
| vlm-sft-cuda.console.log | `efa4bd318c561c058f50198fcf29adfabe848c432085c985dced0e26a8927173` |
| vlm-final.json | `28d86765fa195ba4d9a55693b0060d8da944ddf7183b270858ccb3d829431c38` |

固定评估原始命令、配置和源码在 run manifest、source.tar 和 pipeline console 中，
完整预测、原始history、工具快照及权重仅留服务器。此次实验完成了 VLM 工程训练与固定评估，
但尚未建立可靠的视觉问答或图序理解。视频分支及三模型最终汇总仍待完成。
