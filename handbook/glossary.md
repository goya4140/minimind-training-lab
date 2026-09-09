# 术语表

| 术语 | 本 handbook 中的含义 |
|---|---|
| Backbone | 提供主要表征能力的基础网络；这里包括 LLM 与冻结 SigLIP2 |
| Batch | 一次共同进入模型的多条样本 |
| BPE | Byte Pair Encoding，一种从文本学习子词词表的方法 |
| Causal LM | 只能利用当前位置之前 token 预测下一个 token 的语言模型 |
| Checkpoint | 模型参数或完整恢复状态的持久化文件 |
| Cross-entropy | 对正确类别负对数概率求平均的分类损失 |
| Dataset | 从索引读取、解析并张量化一条样本的对象 |
| Early fusion | 在进入共享主干前，把不同模态都变成同一 token 序列 |
| Embedding | token ID 对应的连续向量 |
| Epoch | 训练样本总曝光数约等于完整训练集一次的范围 |
| Fine-tuning | 从已有 checkpoint 继续针对特定数据训练 |
| Frozen | `requires_grad=False`，forward 可参与但 optimizer 不更新 |
| Gradient | loss 对参数的导数，指示局部更新方向 |
| Gradient accumulation | 多个 micro-batch 累积梯度后再执行一次 optimizer step |
| GQA | Grouped-Query Attention，多个 query head 共享较少的 KV head |
| Held-out | 训练过程中不参与参数更新的数据 |
| Label mask | 用 `-100` 等 ignore index 排除不应计算 loss 的 token |
| Logits | softmax 前、未归一化的类别分数 |
| MLP / FFN | Transformer block 中逐 token 的非线性前馈网络 |
| MPS | Apple Metal Performance Shaders，PyTorch 在 Apple GPU 上的设备后端 |
| Optimizer | 根据 gradient 和状态更新参数的算法，如 AdamW |
| Overfitting | 训练数据表现继续改善，但未见数据能力不再改善或变差 |
| Perplexity | `exp(language loss)`，语言模型不确定性的一种表达 |
| Projector | 把一种模态的 hidden vectors 映射到 LLM hidden space 的小网络 |
| Resampler | 将可变/较长特征压缩为固定数量 token 的模块 |
| RoPE | Rotary Position Embedding，把相对位置信息编码进 attention query/key |
| SFT | Supervised Fine-Tuning，使用指令—回答样本做监督微调 |
| SigLIP2 | 本项目冻结使用的视觉 encoder |
| SwiGLU | 带门控分支的 Transformer 前馈激活结构 |
| Temporal ablation | 改变帧序等时间信息，观察能力是否随之改变 |
| Token | tokenizer 词表中的最小离散单位 |
| Token F1 | 预测与参考答案的 bag-of-token precision/recall 调和平均 |
| Validation | 训练期间选超参数、监控泛化的数据；不同于最终 test |
| Video-QA | 给定视频与问题，生成文本答案 |
