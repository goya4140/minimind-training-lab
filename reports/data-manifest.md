# 数据清单

本项目不把训练数据提交到 Git。`scripts/fetch_data.py` 从固定 URL 下载并在使用前验证
字节数与 SHA-256；任一项不一致都会中止。

| 阶段 | 文件 | 条目数 | 字节数 | SHA-256 |
|---|---|---:|---:|---|
| LLM Pretrain | `pretrain_t2t_mini.jsonl` | 1,270,238 | 1,241,043,656 | `6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c` |
| LLM SFT | `sft_t2t_mini.jsonl` | 905,718 | 1,739,201,170 | `abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518` |
| VLM alignment | `pretrain_i2t.parquet` | 1,274,698 | 4,326,415,097 | `65761f37d1947d54a1d85457ff70938275e4ef58ba5cedcd02463a3a247c93fd` |
| VLM SFT / Omni I2T | `sft_i2t.parquet` | 2,904,511 | 4,934,887,104 | `712f4026cd0e21b369feddca7334b1e465cb8182b5f298006f3f4f877f926643` |
| Omni T2A mini | `sft_t2a_mini.parquet` | 515,415 | 1,558,442,729 | `dfe44b8b263ecd0579627160cf258b363b4c18457ae03221691e2e1a85e60ab8` |
| Omni A2A mini | `sft_a2a_mini.parquet` | 待本地验证 | 881,313,734 | `fba0159e424ee106c9e5a732fe607875b3780d0c9f8b6806038879acd279782b` |

来源：[jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset)。

固定 revisions：文本 `312afb4f…`、VLM `1e279a8b…`、Omni `d6588e12…`。数据卡声明的许可分别为
`Apache-2.0 / CC-BY-NC-2.0`、`Apache-2.0`、`Apache-2.0 / GPL-3.0`；模型与数据产物发布时
必须分别保留适用的来源与许可说明。

Omni mini 数据按上游定义只覆盖英文、无视觉的低成本闭环验证；中文语音能力不能由该数据
推出。Omni 的视觉阶段复用完整 `sft_i2t.parquet`。

## SFT 结构验证

SFT 样本使用 ChatML 模板，只对 assistant 片段计算 loss。真实数据包含普通单轮、多轮及
tool-call 样本；部分工具元数据以 JSON 字符串存储，因此数据层会先规范化 `tools` 和
`tool_calls`，再调用 tokenizer 的 chat template。

固定随机抽取 1,000 条真实样本进行检查：

- 无监督 token 的空样本：0；
- 每条 assistant 监督 token：最少 20，最多 745，平均 410.685；
- 数据集首条、第二条与最后一条（tool-call）均可编码并生成有效 labels。

这项检查证明 loss mask 与已下载数据格式兼容，不代表 SFT 模型质量。

## Omni T2A 结构验证

本地 Parquet 元数据验证为 515,415 行，字段为 `conversations: large_string`、
`question_audios: list<binary>`、`answer_audios: list<list<int64>>`。真实首条样本完成了
ChatML 编码、Mimi 帧优先 code 的 8 码本拆分以及延迟 teacher-forcing 标签构造；详细结果见
`reports/omni-t2a-pipeline-probe.md`。
