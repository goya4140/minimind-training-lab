# 05 · 训练工程：让实验稳定、可恢复、可解释

## Micro-step 与 Optimizer step

显存放不下大 batch 时，可以累积多个 micro-batch 的 gradient：

```text
loss_1 / N → backward
loss_2 / N → backward
...
loss_N / N → backward
clip_grad_norm → optimizer.step → zero_grad
```

`effective batch size = micro batch × gradient accumulation steps`。除以 `N` 是为了让 gradient
尺度与一次处理完整 effective batch 接近。本项目配置表明确区分 micro-step 与 optimizer step，避免
把“训练 10,000 步”理解错一个 accumulation 倍数。

若总 micro-step 数不能被 `N` 整除，最后不足 `N` 个的梯度仍需要执行一次 optimizer update，
并按实际累积数重新缩放成均值。本项目的 trainer 显式处理这个尾部 batch。

## 学习率与梯度裁剪

AdamW 更新参数，同时把 weight decay 与 gradient update 分开。Cosine schedule 让学习率从初始值
逐渐衰减到 10%；global gradient norm 超过阈值时裁剪，降低异常 batch 造成的破坏。

日志中的 `grad_norm` 需要结合 loss 观察：持续为 0 可能是 label mask 或冻结策略错误；突然变成极大值
可能是数值不稳定；单次波动不等于训练失败。

## 可恢复 checkpoint

只保存 model weights 不足以无缝恢复训练。本项目 resume checkpoint 同时保存：

- model state；
- optimizer state；
- 当前 micro-step；
- 已记录 history；
- PyTorch RNG state；
- 已完成训练片段的累计耗时；
- 完整训练 config。

保存先写 `.tmp`，完成后用 `os.replace` 原子替换。进程中途退出不会留下一个看似存在但只写了一半的
正式 checkpoint。

Resume checkpoint 只在 optimizer update 之后写入。如果在 accumulation 中间保存却不保存
尚未提交的 gradient，恢复时就会静默丢掉部分 micro-batch；边界保存避免了这个问题。

可以在训练进行中安全地审计最近一个原子快照：

```bash
uv run python scripts/verify_resume_checkpoint.py --config configs/llm/pretrain-mps.yaml
```

审计器会读取但不修改文件，并核对 optimizer state、config、RNG state、history、累计耗时、有限参数与
optimizer 边界。输出只有尺寸、SHA-256 和汇总计数，不会把权重内容写入报告。六阶段流水线在每个阶段
完成后还会加上 `--require-complete`，阻止未到最终步数的 resume 文件通过完成门禁。

## 防止重复训练

每个阶段持有 advisory lock，锁文件记录 owner PID。第二个同 checkpoint 任务会拒绝启动；preflight
会区分：

- `active`：该阶段正在运行；
- `complete`：最终 checkpoint 已存在；
- `pending`：资源齐全，但上游 checkpoint 未完成；
- `missing`：数据或组件缺失；
- `ready`：可以立即开始。

[`scripts/run_mps_pipeline.py`](../scripts/run_mps_pipeline.py) 只运行缺失阶段，等待 active stage，并对
中断阶段加 `--resume`。这使长时间本地训练不需要人工守在终端旁。

## 冻结特征缓存

冻结 encoder 的输出对同一预处理输入是确定的。`cache_video_features.py` 用 completion bitmap 与
memory-mapped FP16 数组逐批缓存 QIVD patch features；中断后只补未完成视频。训练仍更新 spatial/
temporal adapter 和 projector，最终评估仍走原始 MP4，因此缓存不是“跳过视频输入”。

## 数值门禁

每次训练检查 loss 是否 finite；最终 checkpoint 用
[`scripts/verify_stage_artifact.py`](../scripts/verify_stage_artifact.py) 遍历 tensor，记录参数数量、字节数、
SHA-256 与 NaN/Inf 结果。通过这个门禁只说明 artifact 完整，不说明模型能力合格。

最终报告生成器会再次计算六个 checkpoint 的字节数和 SHA-256，并要求与各阶段完成日志中的审计值
完全一致。这样可以证明报告引用的正是通过门禁的文件，而不是验证完成后被替换的同名文件。
