# 数据清单

本项目不把训练数据提交到 Git。`scripts/fetch_data.py` 从固定 URL 下载并在使用前验证
字节数与 SHA-256；任一项不一致都会中止。

| 阶段 | 文件 | 条目数 | 字节数 | SHA-256 |
|---|---|---:|---:|---|
| LLM Pretrain | `pretrain_t2t_mini.jsonl` | 1,270,238 | 1,241,043,656 | `6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c` |
| LLM SFT | `sft_t2t_mini.jsonl` | 905,718 | 1,739,201,170 | `abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518` |

来源：[jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset)。

## SFT 结构验证

SFT 样本使用 ChatML 模板，只对 assistant 片段计算 loss。真实数据包含普通单轮、多轮及
tool-call 样本；部分工具元数据以 JSON 字符串存储，因此数据层会先规范化 `tools` 和
`tool_calls`，再调用 tokenizer 的 chat template。

固定随机抽取 1,000 条真实样本进行检查：

- 无监督 token 的空样本：0；
- 每条 assistant 监督 token：最少 20，最多 745，平均 410.685；
- 数据集首条、第二条与最后一条（tool-call）均可编码并生成有效 labels。

这项检查证明 loss mask 与已下载数据格式兼容，不代表 SFT 模型质量。
