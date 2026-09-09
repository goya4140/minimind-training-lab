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
