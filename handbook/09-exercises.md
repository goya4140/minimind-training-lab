# 09 · 动手练习

这些练习不写入正式 checkpoint；先读问题，预测结果，再运行命令。能解释“为什么”比记住数字更重要。

## 练习 1：数参数

阅读 `MiniMindConfig`，先估计 embedding、attention、MLP 各占多少参数，再运行：

```bash
uv run python - <<'PY'
from minimind_lab.llm import MiniMindConfig, MiniMindForCausalLM
model = MiniMindForCausalLM(MiniMindConfig())
print(sum(p.numel() for p in model.parameters()))
PY
```

思考：为什么开启 GQA 会减少参数？为什么 tied LM head 不再新增一份 `[vocab, hidden]` 权重？

## 练习 2：观察 assistant-only mask

在 [`tests/test_sft_data.py`](../tests/test_sft_data.py) 增加一个两轮对话，打印 `input_ids` 与 `labels`。
确认 user token 对应 `-100`，两个 assistant span 都有监督，padding 仍为 `-100`。

## 练习 3：跑最小 LLM

```bash
uv run python scripts/train_llm.py --config configs/llm/smoke.yaml
uv run python scripts/evaluate_llm.py \
  --config configs/llm/smoke.yaml \
  --checkpoint artifacts/checkpoints/llm-smoke.pt
```

比较第一条与最后一条训练 loss，再看生成。解释为什么训练数据很小的时候 loss 很低但生成仍可能很差。

## 练习 4：验证冻结策略

```bash
uv run pytest tests/test_vlm.py::test_alignment_freezes_language_and_vision
uv run pytest tests/test_video_omni.py::test_alignment_and_instruction_freeze_policies
```

列出 VLM alignment、VLM SFT、Video alignment、Video SFT 四阶段分别更新的 module。

## 练习 5：帧采样

手算 `total_frames=9, num_frames=5` 和 `total_frames=2, num_frames=4` 的均匀索引，再运行：

```bash
uv run python - <<'PY'
from minimind_lab.data import uniform_frame_indices
print(uniform_frame_indices(9, 5))
print(uniform_frame_indices(2, 4))
PY
```

思考短视频为何允许重复帧，而不是填充全黑图。

## 练习 6：看得见的时间反转

```bash
uv run python scripts/generate_temporal_benchmark.py
```

任选一个生成视频，对其 8 帧反向排列。判断移动方向、大小变化或 first event 的正确答案如何改变。

## 练习 7：确认视频不是图像集合

运行 `test_video_adapter_preserves_frame_order_information`，再临时删除 frame position embedding，观察
测试是否仍能发现倒序差异。注意 spatial/temporal attention 的非线性可能仍产生细小差异；一个单元测试
只证明计算图敏感，不证明训练后能力。

## 练习 8：理解 accumulation

选择 `batch_size=2, gradient_accumulation_steps=8`。回答：

- effective batch 是多少？
- 100 个 micro-step 执行多少次 optimizer step？
- 为什么每个 micro loss 要除以 8？
- save interval 不是 8 的倍数会给恢复语义带来什么麻烦？

## 练习 9：做一次诚实评估

拿到最终结果后，从 `reports/final-results.md` 找一项最弱指标和一个失败样例，写下三个可能原因，并设计
一个只改变单一变量的后续实验。不要同时换数据、结构和学习率，否则无法判断改善来自哪里。
