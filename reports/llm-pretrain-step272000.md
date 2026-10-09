# LLM Pretrain 中期评估：step 272000

这是训练进行中的诊断快照，不是最终模型。快照仅保存模型权重，大小 255,681,813 bytes，
SHA-256 为 `199178119abc6573ae4ecf71c7255bfcffc71fc5567377346aa9521e6a1bda4d`。该哈希标识
这一个快照文件；`torch.save` 的 zip 存档带有时间戳，重跑提取会得到不同字节，因此哈希不能
作为权重内容的复现校验，只能用于确认归档文件本身未被改动。

## 训练状态

- 模型：63,912,192 参数（去重后），MiniMind BPE 6400；
- step：272,000 micro-steps / 8,500 optimizer updates；
- 采样时训练已推进到约 step 274,900，最终目标 317,048；
- 初始 step-1 loss：`8.8969`，当前 batch loss 约 `1.9`；
- 设备：Apple M4 Pro / MPS 训练，**CPU 评估**（避免与后台训练争抢 MPS）；
- 样本顺序：确定性、可精确恢复。

## 与 step 12000 的同协议对比

两次评估使用完全相同的协议：CPU、固定末尾 32 条 held-out validation 样本、
20 个新 token 贪心解码、同一组 4 条 prompt。差异只来自训练步数。

| 指标 | step 12,000 | step 272,000 | 变化 |
|---|---:|---:|---:|
| Validation loss | 4.0441 | **1.8560** | −54.1% |
| Perplexity | 57.06 | **6.398** | −88.8% |
| Bits per byte | 2.0264 | **0.9300** | −54.1% |
| Predicted tokens | 10,495 | 10,495 | 同集 |
| 训练 batch loss | 3.2816 | ≈ 1.9 | — |
| Optimizer updates | 375 | 8,500 | ×22.7 |

`bits_per_byte` 跌破 1.0 是一个有意义的门槛：模型对 held-out 中文文本的压缩效率已经
超过 1 字节 1 bit 的朴素基线。

## 固定生成样例（同 prompt，20 新 token 贪心解码）

| Prompt | step 12,000 | step 272,000 | 判断 |
|---|---|---|---|
| `中国的首都是` | `哪个？中国的首都是哪个？…`（复读、未回答） | `北京，中国的首都是北京，中国的首都是北京，` | **部分成功：答对事实，但仍复读** |
| `请用三句话解释什么是机器学习：` | `机器学习是一种人工智能技术，它可以让计算机…机器学习是一种人工智能技术`（重复） | `机器学习是一种人工智能技术，它使用算法和统计模型来让计算机系统能够从数据中学习` | **进步：无重复、定义更准确；仍未遵循"三句话"** |
| `用户：如何制定一个可执行的学习计划？\n助手：` | `首先，您需要了解哪些方面的因素？\n客户：首先…`（角色漂移） | `制定一个可执行的学习计划，首先要明确学习目标，然后制定一个可行的学习计划` | **成功：角色稳定、内容相关** |
| `Transformer 模型中的注意力机制` | `是如何使用SOOO的？在SOOOOOOO中，`（乱码退化） | `是什么？在Transformer模型中，的注意力机制是基于预训练的，` | **部分成功：乱码消失，但事实表述错误** |

重复度指标 `distinct_2`（越高越不重复）同步改善：0.368 → 0.316（首题仍在复读）、
0.842 → 1.000、0.684 → 0.842、0.579 → 1.000。

## 结论

**已解决（相对 step 12,000）：**

- 退化 token 模式（`SOOOOOOO` 之类乱码）完全消失；
- 角色漂移消失，模型能维持 `用户：/助手：` 的对话结构；
- 事实知识开始出现（正确回答"中国的首都是北京"）；
- perplexity 从 57 降到 6.4，语言建模质量是量级变化。

**仍未解决：**

- 复读：第一题在 20 个 token 内把同一句式重复了 3 次，贪心解码下未触发停止；
- 指令遵循：要求"三句话"时只产出一句未闭合的话——这是预训练模型的预期表现，
  需 assistant-only SFT 才能改善；
- 事实准确性不稳定：能答对常识题，但对 Transformer 注意力机制给出了错误描述。

不能把 loss 与 perplexity 的下降直接写成"模型已可用"。剩余预训练（约 45,000 micro-step）
主要用于继续压低 perplexity 与缓解复读；指令遵循、角色一致性和事实可靠性需要
M3 的 SFT 阶段处理，届时必须同时做语言能力回归测试，确认 SFT 没有损害这里的语言建模质量。

## 复现命令

先从持续覆盖的 resume checkpoint 提取权重快照，再在 CPU 上评估：

```bash
uv run python scripts/extract_model_snapshot.py \
  --source artifacts/checkpoints/llm-64m-pretrain-mps.resume.pt \
  --target artifacts/checkpoints/llm-64m-pretrain-step272000.pt

uv run python scripts/evaluate_bpe_llm.py \
  --config configs/llm/pretrain-mps.yaml \
  --checkpoint artifacts/checkpoints/llm-64m-pretrain-step272000.pt \
  --validation-samples 32 --max-new-tokens 20 --device cpu \
  --output artifacts/eval/llm-pretrain-step272000.json
```

评估在 CPU 上进行，是为了避免与后台 M4 Pro MPS 训练争抢算力。

原始输出写入 `artifacts/eval/llm-pretrain-step272000.json`（Git 忽略，不入库）。
权重快照为 `artifacts/checkpoints/llm-64m-pretrain-step272000.pt`。
