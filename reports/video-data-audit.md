# QIVD 训练前数据审计

在 Video-Omni 正式训练前，对固定 revision 的全部 2,900 条 metadata 运行与训练完全相同的
chat template、MiniMind BPE tokenizer、16 个视频占位符和 assistant-only label mask。

## 结果

| 检查 | 结果 |
|---|---:|
| 样本数 | 2,900 |
| 每条 `<|video_pad|>` 数量 | 16（全部一致） |
| token length P50 / P90 / P95 / P99 | 57 / 66 / 71 / 81 |
| 最大 token length | 138 |
| 超过正式 `sequence_length=384` | 0 |
| assistant 监督 token 最少 / 最多 | 9 / 102 |
| assistant 监督为空 | 0 |

因此正式配置不会截断任何 QIVD 样本，也不会在进入模型时触发视频 token 数量错配；每条样本都有
有效 assistant loss。该审计只证明数据结构可训练，不证明问题/答案本身完全无噪声。
