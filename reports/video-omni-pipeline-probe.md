# Video-Omni 真实链路探针

## 目的

在正式训练前，用真实 QIVD MP4 和冻结 SigLIP2 验证 Video → Text 的全部张量边界，而不是只用
随机视觉特征通过单元测试。

## 输入与结果

- 数据：QIVD `videos/00000146.mp4`；
- 问题：`How many times did I snap my fingers?`；
- 采样：8 帧均匀采样，预处理张量 `[1, 8, 3, 256, 256]`；
- 语言探针：2 层、hidden size 64，仅用于降低架构探针成本；
- 输出 logits：`[1, 24, 259]`；
- assistant-only causal loss：`5.58540`，有限值；
- 正常帧与倒序帧 video embedding 的 mean absolute delta：`7.7198e-05`。

该结果证明真实视频解码、逐帧视觉编码、时间位置编码、learned-query resampling、embedding 注入
和语言 loss 可以端到端运行，也证明倒序会改变初始视频表示。它不是训练后能力结果；最终结论必须
由固定 250 条 held-out test、正常/倒序消融和定性生成共同给出。
