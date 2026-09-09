# 数据清单

本项目不把训练数据提交到 Git。`scripts/fetch_data.py` 从固定 URL 下载并在使用前验证
字节数与 SHA-256；任一项不一致都会中止。

| 阶段 | 文件 | 条目数 | 字节数 | SHA-256 |
|---|---|---:|---:|---|
| LLM Pretrain | `pretrain_t2t_mini.jsonl` | 1,270,238 | 1,241,043,656 | `6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c` |
| LLM SFT | `sft_t2t_mini.jsonl` | 905,718 | 1,739,201,170 | `abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518` |
| VLM alignment | `pretrain_i2t.parquet` | 1,274,698 | 4,326,415,097 | `65761f37d1947d54a1d85457ff70938275e4ef58ba5cedcd02463a3a247c93fd` |
| VLM SFT | `sft_i2t.parquet` | 2,904,511 | 4,934,887,104 | `712f4026cd0e21b369feddca7334b1e465cb8182b5f298006f3f4f877f926643` |
| Video-Omni | `QIVD/metadata.parquet + videos` | 2,900 | 下载完成后生成 | 下载完成后生成 |

来源：[jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset)。

固定 revisions：文本 `312afb4f…`、VLM `1e279a8b…`。Video-Omni 使用
[Qualcomm AI Research QIVD](https://huggingface.co/datasets/Qualcomm-AI-Research/QIVD)，固定 revision
`c5376ab0b9fd3643545a1503413aee64f26ba22a`。数据卡声明的许可分别为
`Apache-2.0 / CC-BY-NC-2.0`、`Apache-2.0`；QIVD 为 research-only。模型与数据产物发布时
必须分别保留适用的来源与许可说明。

QIVD 视频不提交到 Git，也不重新分发。`scripts/fetch_qivd.py` 以单并发断点下载，并在全部 2,900
个文件完成后生成逐文件 SHA-256 与聚合 SHA-256。ID 经过 seed 48 稳定哈希后，固定划分为
2,400 train / 250 validation / 250 test。

下载器还会读取固定 revision 的 Hugging Face LFS tree，对每个视频核验上游声明的字节数与
SHA-256；本地自洽但不同于上游的截断或损坏文件不会通过，manifest 只有在全部匹配后才写入
`upstream_lfs_verified: true`。

## SFT 结构验证

SFT 样本使用 ChatML 模板，只对 assistant 片段计算 loss。真实数据包含普通单轮、多轮及
tool-call 样本；部分工具元数据以 JSON 字符串存储，因此数据层会先规范化 `tools` 和
`tool_calls`，再调用 tokenizer 的 chat template。

固定随机抽取 1,000 条真实样本进行检查：

- 无监督 token 的空样本：0；
- 每条 assistant 监督 token：最少 20，最多 745，平均 410.685；
- 数据集首条、第二条与最后一条（tool-call）均可编码并生成有效 labels。

这项检查证明 loss mask 与已下载数据格式兼容，不代表 SFT 模型质量。

## Video-Omni 结构验证

QIVD 元数据验证为 2,900 行，包含视频路径、问题、长答案、短答案、类别与时间戳。真实 MP4 已完成
PyAV 解码、8 帧均匀采样、SigLIP2 预处理及 Video-Omni 前向；详细结果见
[`reports/video-omni-pipeline-probe.md`](video-omni-pipeline-probe.md)。
