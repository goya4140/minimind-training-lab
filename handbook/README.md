# 从零学习模型训练：MiniMind Training Handbook

这不是“运行一个现成模型”的教程，而是一条可以实际执行的训练路线。我们只研究三个模型：

1. LLM：文本 → 文本；
2. VLM：图像 + 文本 → 文本；
3. Video-Omni：视频 + 文本 → 文本。

三者不是三个互不相关的项目，而是同一个语言主干逐步获得视觉和时序能力。完成这条路线后，你应该能
解释一个训练系统中的数据、token、forward、loss、gradient、optimizer、checkpoint 和 evaluation
分别在做什么，也能根据日志判断一次实验是否可信。

## 推荐阅读顺序

| 章节 | 你会解决的问题 | 对应实践 |
|---|---|---|
| [00 · 训练全景](00-training-map.md) | “训练模型”到底包含哪些环节？ | 六阶段依赖图 |
| [01 · 数据与 Token](01-data-and-tokenizer.md) | 原始文本、图像、视频如何变成张量？ | BPE、label mask、帧采样 |
| [02 · LLM](02-llm.md) | Transformer 如何预测下一个 token？ | 从随机初始化 Pretrain → SFT |
| [03 · VLM](03-vlm.md) | 图像如何进入语言模型？ | SigLIP2、projector、early fusion |
| [04 · Video-Omni](04-video-omni.md) | 视频比图像多出的“时间”怎么建模？ | spatial pool、temporal adapter、倒序消融 |
| [05 · 训练工程](05-training-engineering.md) | 如何稳定训练、断点恢复且不重复跑？ | accumulation、lock、atomic checkpoint |
| [06 · 评估](06-evaluation.md) | loss 降低为什么不等于模型有效？ | PPL、VQA、Video-QA、受控时序测试 |
| [07 · 完整复现](07-reproduction-runbook.md) | 如何在一台机器上从头跑完？ | 安装、下载、preflight、训练、报告 |
| [08 · 读日志与排错](08-reading-logs.md) | loss、梯度、吞吐异常意味着什么？ | 真实日志字段与诊断树 |

## 学习方式

每一章按同一种方法阅读：

1. 先看“输入和输出”，确认张量形状；
2. 再看“哪些参数会更新”，避免把冻结模型误认为从零训练；
3. 找到 loss 的监督范围；
4. 运行章末验证命令；
5. 最后阅读失败条件，而不只看成功样例。

本仓库不会上传训练数据和权重。代码、配置、实验摘要与生成样例在 GitHub；数据、checkpoint、
optimizer state 和完整逐步日志保留在本机。这样既遵守数据许可，也让学习材料保持轻量。
