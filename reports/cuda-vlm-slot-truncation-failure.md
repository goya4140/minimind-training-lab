# VLM Alignment 中断：图像槽位被文本截断

Run：`mm64-cuda-20261024T075332Z`；固定训练代码：`53f9b9f`。
本记录基于 2026-10-24 16:37 UTC 后的只读核验，不能沿用此前“预训练进行中”的实时状态。

## 已完成与未完成

LLM Pretrain 于 13:18:30 UTC、LLM SFT 于 13:55:56 UTC 通过验证并归档。其余四阶段尚未完成。
流水线在 VLM Alignment 于 14:26:57 UTC 失败；进程已退出，不是仍在训练或仅观察超时。

同一固定文本留出集的已归档评估如下；两者均预测 670,079 token，UTF-8 字节数 1,906,560：

| checkpoint 阶段 | 留出 NLL | PPL | BPB |
|---|---:|---:|---:|
| LLM Pretrain | 1.828207 | 6.222717 | 0.926991 |
| LLM SFT | 1.913924 | 6.779641 | 0.970454 |

这表明该文本测试口径下 SFT 后的 loss/PPL 上升，不代表所有指令任务均退化。原固定续写提示上
仍有复读、事实自相矛盾和概念错误；不能将 SFT 训练完成等同于指令能力验证成功。chat-template
补充诊断尚待执行。SFT 评估 JSON 的 `experiment` 名称来自共用 pretrain 评估配置，阶段身份应按
checkpoint 路径、归档 SHA 和事件确定，不应仅按该字符串误标为第二次预训练。

完整原始结果保留在服务器 run 的 `stages/llm-pretrain/llm-pretrain-final.json` 与
`stages/llm-sft/llm-sft-final.json`，归档 SHA 分别为
`3b50f4d17c98e5f4eafc7505bc4ded226d7ff1c4aef068cf2a191480bf59739b` 与
`23392375dbff726f87ae697d1a8b4044b0f1856c75b4f57e24fc71380d3bf235`。

## 错误与精确复现

错误为 `sample 1 has 33 image placeholders; expected 64 for 1 image(s)`。
固定 seed 42、batch 4，按原数据过滤及 DeterministicBatchStream 重建采样顺序，在 micro-step
26,896 获得 filtered indices `[606220, 1246578, 584084, 1175311]`，四条样本图像槽位数为
`[64, 33, 64, 64]`，与异常完全对应。最后一条已打印的日志是 26,880，不应直接把它当作失败步。

失败样本 filtered index 为 1,246,578（不是原 Parquet 行号）。其 user/assistant 内容分别为
451/447 字符，`<image>` 起始于 user 内容的字符偏移 444。字符偏移不是 token 偏移；这里只用于
说明标记在长问题尾部，精确槽位计数来自实际 Tokenizer 输出。

`ParquetVLMDataset.__getitem__` 先把 `<image>` 展开为 64 个 image-pad token，再直接取整个
chat 序列的前 450 token。该截断边界穿过视觉槽位块，模型的严格槽位检查因此拒绝输入。
这是输入构造与截断策略的边界缺陷，不是 CUDA OOM、图像解码损坏或模型数值发散。

原始单图标记过滤仅保证语义标记数量为一，不能保证 Tokenizer 截断后仍保留全部图像槽位：
1,274,698 原始行中保留 1,272,607、排除 2,091 个多标记行，仍会遇到这个问题。此前冒烟测试
只覆盖少量样本，未覆盖此长前缀边界，不能替代完整的数据边界检查。

## 保留与修复约束

step 26,000 恢复点通过只读审计：97 个模型张量、65,094,912 个独立参数有限，6 个 optimizer
entries、18 个 optimizer 张量有限；配置匹配、完整更新边界、history 1,301 条。
恢复文件 SHA 为 `f93e8e9dfeca6b8cda53951e7dc06c05f0798e98a45e5f65a7d64fd52156b01a`。
这不是已完成的 Alignment 权重。

服务器 run 下的 `failures/vlm-image-slot-truncation-20261025/diagnosis.json` 保存复现细节，
`stages/vlm-alignment-failure-26896/` 独立归档恢复点、实际 console 和流水线输出；原失败不覆盖。
固定源码和依赖 guard 仍通过。当前没有重新启动训练。

后续应保留完整视觉块并检查有效 assistant 监督，增加长前缀、边界切片、双图和无有效标签的
回归测试，再明确受影响样本的处理策略。实质数据构造变化须使用新 commit 和独立 run/继承记录，
保留已完成 LLM 阶段的来源，不能静默修改原 run 的 manifest 或用“重新跑通”替代故障说明。
