# 01 · 数据与 Token

## 文本为什么要 Tokenize

神经网络接收数字，不直接接收字符串。BPE tokenizer 把文本拆成有限词表中的 token ID：

```text
"模型训练" → [token_1, token_2, ...] → [整数 ID]
```

正式实验使用 6,400 词表的 MiniMind tokenizer。`input_ids` 是模型输入；`labels` 是希望模型预测的
下一个 token。Causal LM 会把 logits 与 labels 错开一位计算交叉熵。

## Pretrain 与 SFT 的 label 不同

Pretrain 的目标是预测样本中几乎所有后续 token：

```text
input : [BOS, 今, 天, 天, 气]
target: [今,  天, 天, 气, EOS]
```

SFT 则只训练 assistant 回答。system、user、padding 的 label 都设为 `-100`，PyTorch
cross-entropy 会忽略这些位置：

```text
tokens: system ... user ... assistant 答 案 EOS PAD
labels:  -100 ... -100 ...      答 案 EOS -100
```

实现位于 [`src/minimind_lab/data/sft.py`](../src/minimind_lab/data/sft.py)，测试会检查多轮 assistant
范围，而不是假设每个样本只有一个回答。

## 图像如何变成 token

SigLIP2 把 256×256 图像编码成 patch features。VLM projector 将视觉 hidden size 映射到 LLM
hidden size，再严格替换 64 个 `<|image_pad|>` embedding。占位符数量错误会立即报错，防止视觉
特征静默错位。

## 视频如何变成 token

[`decode_uniform_video`](../src/minimind_lab/data/video.py) 从视频首尾之间均匀抽取 8 帧。若视频少于
8 帧会确定性重复索引；若容器没有可靠帧数，则回退到短视频完整解码。

```text
MP4 → 8 RGB frames → [8, 3, 256, 256] → SigLIP2 patch features
    → temporal adapter → 16 vectors → 16 <|video_pad|> positions
```

QIVD 的 2,900 条记录使用 seed 48 对 ID 做 SHA-256 排序，固定切成 2400/250/250。哈希划分避免
依赖源文件排列，同时保证每次运行完全相同。

## 数据泄漏检查

- train、validation、test 索引两两不重叠；
- 受控时序基准只用于 evaluation，`training_overlap=0`；
- evaluation 不调用 `backward()` 或 `optimizer.step()`；
- QIVD 视频不提交 GitHub，且遵循 research-only 许可。

验证：

```bash
uv run pytest tests/test_sft_data.py tests/test_vlm_data.py tests/test_video_data.py
```
