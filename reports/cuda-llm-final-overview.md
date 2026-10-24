# CUDA LLM 两阶段结果与学习解读

这是一份已完成 LLM 分支的阶段总结，不是三模型最终报告。VLM/Video-Omni 使用独立修复运行继续，
不能用本页证明它们已经完成。下列数字均来自 CUDA run `mm64-cuda-20261024T075332Z`，
训练代码 `53f9b9f46560bee446d1c74e89cac991b71a8053`，不混用历史 MPS 结果。

## 1. 我们实际训练了什么

语言核心是本项目实现的 Decoder-only Transformer：63,912,192 个参数，8 层，hidden 768，
8 个 Query 头、4 个 KV 头、SwiGLU 中间维度 2,432，词表 6,400；使用 RoPE、RMSNorm、GQA。
“64M”是规模简称。语言权重随机初始化后预训练，SFT 从该次最终预训练权重继续训练。
Tokenizer 是复用的固定资产，不是这两个阶段重新学习得到。配置允许的最大位置长度为 32,768，
但本次训练序列长度分别只有 340 和 768，不能因此宣称已经验证 32k 长上下文能力。

具体接口与 mask 实现见[设计实现说明](../docs/design-implementation.md)，训练入口为
`scripts/train_pretrain.py`、`scripts/train_sft.py`，配置为
[pretrain-cuda.yaml](../configs/llm/pretrain-cuda.yaml) 与 [sft-cuda.yaml](../configs/llm/sft-cuda.yaml)。

| 项目 | Pretrain | Assistant-only SFT |
|---|---:|---:|
| seed | 42 | 43 |
| 总数据行 | 1,270,238 | 905,718 |
| 训练行／末尾验证行 | 1,268,190 / 2,048 | 903,670 / 2,048 |
| microsteps | 317,048 | 40,000 |
| batch / 累积次数 | 8 / 32 | 4 / 4 |
| 常规完整更新的名义样本数 | 256 | 16 |
| optimizer 更新数（含尾部更新） | 9,908 | 10,000 |
| 序列长度 | 340 | 768 |
| 基础学习率 → 末端学习率 | 5e-4 → 5e-5 | 1e-5 → 1e-6 |
| weight decay / grad clip | 0.1 / 1.0 | 0.1 / 1.0 |
| 训练精度 | BF16 autocast | BF16 autocast |
| 累计活跃训练秒数 | 19,355.405830 | 2,208.060883 |
| 活跃训练时间约 | 5小时22分35秒 | 36分48秒 |

microstep 不等于 optimizer update：每次累积若干微批再更新，尾部不足完整累积窗口时单独归一化。
有效监督 token 数因 padding、截断和 assistant-only mask 变化，不能用名义样本数直接推算。
Pretrain 的 317,048 步恰好是 2 个确定性置换 epoch；SFT 的 40,000 步仅抽取第一 epoch 的
160,000 个训练行位置，约为训练行数的 17.7%，并非把 903,670 行完整训练了一遍。
行数不是唯一内容数，未证明文本去重或语义无重叠。

硬件为 RTX PRO 6000 Blackwell Server Edition，Python 3.12.4、Torch 2.8.0+cu128、
CUDA 12.8。活跃训练时间来自训练计时器，包含循环中的数据/记录/中间保存开销，不是纯 GPU
kernel 时间；不含阶段结束后的完整验证、最终归档及后续独立评估，不等于端到端墙钟耗时。
未记录系统休眠；共享服务器结果不当作独占硬件速度基准。

## 2. 三种 loss 要分开读

| 指标与来源 | Pretrain | SFT | 可以说明什么 |
|---|---:|---:|---|
| 各自阶段结束验证 loss | 1.827967831 | 0.848441020 | 各自数据分布和监督 mask 下的 batch 平均 loss |
| 各自阶段结束验证 PPL | 6.221231210 | 2.336002232 | 不能跨这两列断言通用语言能力改善 |
| 相同通用文本留出集 token 加权 NLL | 1.828206596 | 1.913924133 | 固定语料语言建模回归 |
| 相同留出集 PPL | 6.222716802 | 6.779640879 | SFT 比本次 Pretrain 上升约 8.95% |
| 相同留出集 BPB | 0.926990623 | 0.970453628 | 同 tokenizer/语料/字节口径下的编码代价变化 |

阶段结束验证的 batch 平均与独立 evaluator 的 token 加权平均不是同一统计量，
因此 Pretrain 两个 loss 略有差异并不是检查点错误。SFT 分布内验证监督的是 assistant response，
其数据不是通用文本回归集，不能拿 0.848 与 1.828 直接比较成“能力提高一半”。
SFT 验证器会跳过非有限 batch、但没有给出跳过计数；仅凭这一汇总不能证明每批验证均有限。

相同通用文本评估使用 2,048 条、670,079 个预测 token、1,906,560 UTF-8 bytes。
NLL 上升 0.085717537，PPL 相对上升 8.95%，支持“本次 SFT 后在该固定通用文本留出上的
语言建模表现退化”，不证明所有语言能力都退化，也不能单凭一次运行归因到具体训练机制。
这一集合是末尾行留出，不是已经证明与 SFT 文本语义去重的外部独立 benchmark。

SFT 固定评估 JSON 的 `experiment` 字段仍显示 pretrain 名称，因为 evaluator 复用语言配置；
必须按 checkpoint SHA、阶段归档及调用命令识别对象，不能只看这个字符串。

## 3. 训练曲线与实际回答相互补充

[Pretrain 最终曲线](evidence/mm64-cuda-20261024T075332Z/E002-llm-pretrain-final/README.md)
含 15,853 条日志记录、159 个固定桶；[SFT 最终曲线](evidence/mm64-cuda-20261024T075332Z/E002-llm-sft-final/README.md)
含 2,001 条记录、154 个桶。两者都附 loss、LR、梯度、padded throughput、PyTorch 显存 5 张图。
曲线上的 loss 是已记录 microbatch 的均值，不是每个 batch 的完整轨迹或验证 NLL。

原固定四条 raw-prefix 生成各用 80 新 token，两个阶段都暴露复读与错误：Pretrain 把注意力机制
称作 Precision，SFT 则混淆 Transformer 与 TensorFlow；SFT 还会自演重复的用户/助手回合。
这些输入并非完整 chat template，不能据此单独评判 SFT 的对话能力。

因此另行冻结 chat-template 的 8 题补充诊断，两个最终模型严格评分均为 0/8。
Pretrain 多处离题；SFT 更像中文对话，但常输出教程或多余解释，存在事实矛盾与复读。
例如首都题回答“中国的首都是北京。”事实正确，只是不满足“仅城市名”的严格约束。
完整[逐题解读](cuda-llm-final-chat-diagnostic.md)和[真实截图](evidence/mm64-cuda-20261024T075332Z/E003-chat-final/README.md)
一并保存，不用严格 0 分抹去内容差异，也不根据结果临时放宽评分。

## 4. 检查点与复现证据

服务器：`/root/autodl-tmp/wlh_goya/MiniMind/artifacts/runs/mm64-cuda-20261024T075332Z/`。
完整源代码快照、manifest、配置、输入指纹、事件、optimizer/RNG 恢复文件及原始评估均在该 run 下。
阶段归档：`stages/llm-pretrain/`、`stages/llm-sft/`；补充输出：`supplemental-evaluations/`；
曲线源数据：`curves/final-pretrain-source/`、`curves/final-sft-source/`。
模型 91 个 state tensors、63,912,192 参数已核验有限；恢复点还核验 optimizer 的 90 个 state
entries / 270 个张量有限，步数、配置、RNG 和更新边界完整，不能只靠 model-only 文件声称可恢复。

| 文件（相对阶段目录） | SHA-256 |
|---|---|
| llm-64m-pretrain-cuda.pt | `5d93e0b34336130dc7395e6fb9fc0028c969cdce0c417b92aa8e1818d8724d60` |
| llm-64m-pretrain-cuda.resume.pt | `a087b33de93ccaea52bb18748c3893e2df9313c5b79ac1f848e77fc6d8eafa79` |
| llm-pretrain-final.json | `3b50f4d17c98e5f4eafc7505bc4ded226d7ff1c4aef068cf2a191480bf59739b` |
| llm-64m-sft-cuda.pt | `61aff7c70d0e6d110b07d5a92f82e2ba0e7667aba6468404a94be585861910a4` |
| llm-64m-sft-cuda.resume.pt | `88afc98991c4364d1b8b932bffdc1c9fff4f71ef35075c457d869deca45a00ab` |
| llm-sft-final.json | `23392375dbff726f87ae697d1a8b4044b0f1856c75b4f57e24fc71380d3bf235` |

归档摘要是 `llm-64m-pretrain-cuda.json` 与 `llm-64m-sft-cuda.json`，包含本页分布内验证和计时。
两个摘要 SHA-256 分别为 `5b555a8ac2c9d9d575c43b048ee7deeb2cffff599b3d5252dc0646fe9c331d25`
和 `45afee1ef0e2e1d5752f11e469ca4915cd6e642d9e7af95cedfccbd345daf2d4`；
发布前重新读取摘要、核对上表全部 6 个产物指纹、训练步数/更新数算术和源文件保护检查均通过。
进一步复现命令见[CUDA 操作说明](../docs/cuda-training.md)、[正式 run 登记](formal-run-mm64-cuda-20261024.md)
及各曲线/对话索引。不重新启动已完成的语言训练，不改写原归档。

## 5. 阶段结论

我们已验证随机初始化语言核心 → 预训练 → assistant-only SFT → 可恢复检查点 → 固定回归与生成
的完整 LLM 工程链路；但模型的简短格式遵循、事实一致性和停止行为尚不可靠。
本次结果说明“工程链路完成”“训练 loss 下降”“通用建模改善”“指令遵循改善”是四个不同问题。
后续如研究更长 SFT、数据组成、格式约束或其他学习率，需另开对照 run、固定验证方法与预算，
不能在当前正式运行中静默修改。当前只继续已约定的多模态阶段，不依据这 8 题调参。
