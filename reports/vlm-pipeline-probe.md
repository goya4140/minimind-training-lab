# VLM 真实链路探针

## 目的

在正式 VLM 训练前，用真实 tokenizer、真实 Parquet 图文样本和冻结的 SigLIP2 验证数据格式、
图像预处理、占位 token、projector、LLM forward 与 assistant-only loss 能连通。

## 固定输入

- 数据：`pretrain_i2t.parquet` 前两条，抽取为 `data/sample/vlm.parquet`；
- tokenizer：MiniMind BPE 6400；
- vision encoder：`jingyaogong/siglip2-base-p32-256-ve` revision `9465d1dc…`；
- LLM：2 层、hidden 64 的随机初始化探针，不是正式权重；
- sequence length：128；image tokens：64。

## 结果

| 检查项 | 结果 |
|---|---|
| 数据行数 | 2 |
| `input_ids` | `[2, 128]` |
| `pixel_values` | `[2, 1, 3, 256, 256]` |
| 每条图像数 | `[1, 1]` |
| assistant 监督 token | `[36, 44]` |
| vision encoder | 全参数冻结 |
| forward loss | `8.715042`，finite |

结论仅限于真实数据链路可运行。随机初始化 loss 不代表 VLM 能力；能力结论必须等待 LLM SFT、
vision alignment 与 VLM SFT 完成后再按固定图集评估。

## 完整尺寸装配复验

2026-09-09 进一步使用 step 12,000 的真实 63,912,192 参数 LLM 中期快照、完整冻结
SigLIP2 和 `pretrain_i2t.parquet` 首条样本进行 CPU 前向：

| 检查项 | 结果 |
|---|---:|
| VLM 总参数（含冻结 SigLIP2） | 159,647,232 |
| alignment 可训练参数 | 1,182,720 |
| 输入 / 像素 | `[1, 128]` / `[1, 1, 3, 256, 256]` |
| assistant 监督 token | 36 |
| logits | `[1, 128, 6400]` |
| loss | 5.951065（finite） |

该复验确认正式 768-hidden LLM 权重、768-hidden SigLIP2 与 Projector 的维度和 state dict
完全兼容；使用中期预训练权重只是为了提前做装配检查，正式 VLM 仍必须从最终 LLM SFT 初始化。
