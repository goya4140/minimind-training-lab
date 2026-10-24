# 固定补充对话诊断协议

`chat-diagnostic-v1.json` 冻结了 8 条项目构造的简单问题、参考答案、严格评分规则及生成参数。
它与原有 2,048 条文本留出和 4 条统一基础提示**分开记录、分开解释**，不替换原协议。
该补充集没有外部基准代表性，也未证明与训练文本语义去重，不能据此估计通用能力。

## 冻结与来源

协议于代码 `7554b8b` 固定，SHA-256 为
`af230c71f939b0c663a6dbfa681d1046c6d34c9fe89c5e8c7f94e4c2b6ccb98c`，
随后在正式 run 的 `supplemental-protocols/` 留存并登记事件，早于首次模型输出。
固定问题、参考答案、规则与参数不得在看到结果后继续修改而仍称为同一协议。

协议原始描述中的 `Human-authored` 对来源表述不准确：这些题目实际由助手为本项目构造，
不是人类受试数据或外部人工标注测试集。为保持冻结指纹，原文件不覆盖；
[来源勘误](chat-diagnostic-v1-provenance.json)单独更正此描述，题目、参考、分数和生成设置均不变。
已完成的早期测试 manifest 保留原样并附勘误，后续 manifest 使用更正后的来源说明。

## 与统一基础提示的区别

本协议使用与 SFT 同一个 Tokenizer 的 chat template，并开启 assistant generation prompt。
FP32 greedy 生成最多 80 token，遇 Tokenizer EOS 停止，不使用 KV cache。
原固定语言回归继续保持其自身提示与生成规则，因此不能把两套协议的分数混在一起。

VLM 和视频模型的此项评估仅抽取语言主干，**不输入图片或视频**；它检查多模态训练后语言
回答行为是否变化，不是视觉能力评分。四种正式比较对象为 LLM Pretrain、LLM SFT、VLM SFT
与 Video-Omni SFT，均必须使用对应阶段完成后归档的精确 checkpoint。

## 评分解释

- 每题仅计 0 或 1，8 题 macro mean 是“该固定诊断的参考约束通过率”。
- exact 仅去除首尾空白，不去掉解释、标点、内部空格或复读；表达正确但未遵守输出约束也会失败。
- JSON 要求键和值与参考一致，键顺序不影响结果；拒绝重复键、额外键、NaN、非法 JSON 和代码块。
- 参考集很小，未收录的合理措辞也可能失败；逐样本原始输出必须与分数一起保留。
- 单次请求耗时有显式 CUDA 同步，但没有预热和重复测量，不作为严格硬件速度基准。

## 调用与保留

工具为 [chat_diagnostic.py](../chat_diagnostic.py)。完成阶段评估示例（阶段归档文件真实存在后使用）：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python reporting/chat_diagnostic.py \
  --run-dir artifacts/runs/mm64-cuda-20261024T075332Z \
  --protocol reporting/protocols/chat-diagnostic-v1.json \
  --checkpoint artifacts/runs/mm64-cuda-20261024T075332Z/stages/llm-sft/llm-64m-sft-cuda.pt \
  --stage llm-sft --eval-id chat-v1-llm-sft-final --device cpu
```

工具检查协议与勘误的注册指纹、阶段完成事件、归档路径和 checkpoint SHA-256，拒绝将探针或
正在训练的全局权重冒充正式阶段结果。`--probe` 只能使用 run 下 `diagnostics/` 的已审计模型快照，
强制 CPU，并明确标为 probe。已存在的 eval ID 不覆盖；失败保留 manifest、已有逐样本记录和事件。

每次输出保留在 `supplemental-evaluations/<eval-id>/`，包括 protocol、来源勘误、执行器副本、
manifest、逐样本 JSONL、参考答案、实际渲染提示、token IDs、EOS 行为和 summary。
完整输出只留服务器，GitHub 只提交精简案例与结果说明。

LLM Pretrain 与 SFT 最终归档权重已完成该诊断，见[逐题解读](../../reports/cuda-llm-final-chat-diagnostic.md)，
两者严格参考约束通过率均为 0/8；真实截图和完整服务器输出均已留存。
VLM SFT 与 Video-Omni SFT 尚待阶段完成，不能用工具测试或早期探针替代最终结果。

2026-10-25 在修复 run `mm64-cuda-vlmfix-v3-20261024T170500Z` 的首次补充推理之前，
逐字节继承父 run 的同一协议与来源勘误；`evaluation_protocol_registered` 与
`evaluation_protocol_provenance_corrected` 事件同时记录父 run、原注册时间和文件指纹。
这不是看到 LLM 结果后重新出题。继承过程中核验父事件、父文件、当前工具目录文件与新 run
留存文件的 SHA 全部一致，且新 run 源码指纹检查通过；没有更改训练、原固定评估或生成设置。
