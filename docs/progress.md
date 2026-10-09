# 项目进度与证据

## 环境基线

- 主机：Apple M4 Pro，20-core GPU，48 GiB unified memory；
- 本地加速：PyTorch MPS；
- CUDA：不可用；
- 策略：先在本机 MPS 完成 64M 正式训练与恢复验证；VLM/Video-Omni 如本机成本不可接受，
  再在不改变配置语义和评估协议的前提下迁移到 NVIDIA CUDA。

## 里程碑

| ID | 交付 | 状态 | 完成证据 |
|---|---|---|---|
| M0 | GitHub 仓库与复现规范 | 已完成 | `goya4140/minimind-training-lab` 与首次远端提交 |
| M1 | LLM 原生架构 | 已完成 | 4 个测试通过；正式配置参数量 63,912,192 |
| M2 | LLM 从零 Pretrain | **已完成** | 317,048 步全部跑完；val loss 1.8326 / PPL 6.2504（完整 2,048 条 held-out）；见 [`reports/llm-pretrain-final.md`](../reports/llm-pretrain-final.md) |
| M3 | LLM SFT 与评估 | **已完成** | 40,000 步全部跑完；分布内 val loss 0.8632 / PPL 2.3707；同语料语言回归 1.8328 → **1.9089**（PPL 6.2515 → 6.7457，轻微反弹）；见 [`reports/llm-sft-final.md`](../reports/llm-sft-final.md) |
| M4 | VLM 对齐与 SFT | 恢复推进 | 2026-10-09 已修复单图数据标记筛选；见 [`reports/vlm-data-filter.md`](../reports/vlm-data-filter.md) |
| M5 | Video-Omni Video→Text | 管线就绪 | 时序 Transformer、倒序消融、QIVD 划分与训练/评估入口已测试；2,900 个视频已完成上游 LFS 哈希核验 |
| M6 | 最终 GitHub Handbook | 进行中 | 入门章节、README、报告生成器、模型卡；等待最终实测结果 |

`artifacts/` 下生成的日志和评估输出默认不提交；经过审核的结果将提炼到 `reports/` 并提交。

流水线长期无人值守运行，状态观测与告警规则见 [`docs/monitoring.md`](monitoring.md)：
`scripts/monitor_pipeline.py` 每小时检测一次阶段切换、日志停滞、进程存活与 loss 合法性，
事件历史写入 `artifacts/monitor-events.jsonl`。

## 已运行实验

| 实验 | 设备 | 结果 | 报告 |
|---|---|---|---|
| `llm-smoke-001` | Apple M4 Pro / MPS | 120 步，loss 5.6009 → 0.5583 | [`reports/llm-smoke-001.md`](../reports/llm-smoke-001.md) |
| `llm-bpe-mini-001` | Apple M4 Pro / MPS | 300 步，loss 8.8215 → 0.0130 | [`reports/llm-bpe-mini-001.md`](../reports/llm-bpe-mini-001.md) |
| `llm-64m-mps-probe` | Apple M4 Pro / MPS | 正式 63.9M 架构、真实数据、完整恢复通过 | [`reports/llm-64m-mps-probe.md`](../reports/llm-64m-mps-probe.md) |
| `llm-64m-pretrain-mps` step 12,000 诊断 | CPU 评估 | val loss 4.0441 / PPL 57.06 / BPB 2.0264 | [`reports/llm-pretrain-step12000.md`](../reports/llm-pretrain-step12000.md) |
| `llm-64m-pretrain-mps` step 272,000 诊断 | CPU 评估 | val loss 1.8560 / PPL 6.398 / BPB 0.9300 | [`reports/llm-pretrain-step272000.md`](../reports/llm-pretrain-step272000.md) |
| `llm-64m-pretrain-mps` **最终** | Apple M4 Pro / MPS | 317,048 步；val loss 1.8326 / PPL 6.2504；纯训练 31 小时 11 分 | [`reports/llm-pretrain-final.md`](../reports/llm-pretrain-final.md) |
| `llm-64m-sft-mps` **最终** | Apple M4 Pro / MPS | 40,000 步；分布内 val loss 0.8632 / PPL 2.3707；耗时 5 小时 1 分 58 秒 | [`reports/llm-sft-final.md`](../reports/llm-sft-final.md) |
| `llm-64m-sft-mps` 语言回归 | CPU + MPS 各一次 | 预训练同语料：val loss 1.8328 → 1.9089，PPL 6.2515 → 6.7457（+7.94%） | [`reports/llm-sft-final.md`](../reports/llm-sft-final.md) |

## 当前运行

阶段 ① 与 ② 已完成。2026-10-09 修复 VLM 标记异常后恢复阶段 ③，后续阶段由流水线接力。

- `llm-64m-sft-mps`：从 `llm-64m-pretrain-mps.pt` 初始化，做 assistant-only 指令微调；
- 数据：`sft_t2t_mini.jsonl`；
- 配置：sequence length **768**、batch size 4、gradient accumulation 4、40,000 步；
- 学习率 1e-5（比预训练的 5e-4 小 50 倍），同样用余弦退火；
- 只在新一轮 held-out 的 2,048 条上评估，不复用预训练验证集。

### 阶段 ② 结论（2026-09-11 21:08 完成）

`llm-64m-sft-mps` 跑完 40,000 步，分布内 val loss **0.8632**、PPL **2.3707**，耗时 18,118.29 秒
（5 小时 1 分 58 秒，无休眠）。最终 checkpoint 255,827,029 bytes，SHA-256
`8107602a1a171d8e06aa71f47bf2528c50282ef2440231a3bd8b5fdd73e78b10`，91 个 state tensors 全部有限。
完整报告见 [`reports/llm-sft-final.md`](../reports/llm-sft-final.md)。

**语言能力回归（M3 的硬性验收项）已在与预训练完全相同的 held-out 语料上完成**，MPS 与 CPU 各跑一次，
结果一致：

| 模型 | Val loss | Perplexity |
|---|---:|---:|
| 预训练最终 | 1.8328 | 6.2515 |
| SFT 最终 | 1.9089 | 6.7457 |
| 变化 | +4.15% | **+7.94%** |

判定：**存在轻微反弹，但不构成灾难性遗忘**。这是 63.9M 小模型上可预期的对齐税——用通用文本预测能力
的小幅下降换取指令格式的稳定性。按项目原则，不把 loss 下降等同于能力提升，同样也不把这次退隐去不报。

### 阶段 ③ 阻塞（2026-09-11 21:09）

SFT 交棒后 ③ `vlm-vision-alignment-mps` 于第 20 步崩溃退出，流水线当前**停止**：

```
ValueError: sample 3 has 128 image placeholders; expected 64 for 1 image(s)
```

只读核查已定位根因：`data/raw/pretrain_i2t.parquet`（1,274,698 行）中 **2,091 行（0.16%）** 的
非 system 消息含 2 个 `<image>` 标记，而 `image_bytes` 只有 1 张图（抽查行 1152 / 1645 / 1737 确认）。
数据管线把每个 `<image>` 展开为 64 个 `<|image_pad|>`，导致占位符数与图像数不匹配并触发硬断言。
该问题确定性可复现。2026-10-09 已在 `ParquetVLMDataset` 中确定性排除不一致样本，保留原始文件；
完整统计见 [`reports/vlm-data-filter.md`](../reports/vlm-data-filter.md)。锁采用进程 advisory lock，进程退出
后内核自动释放，残留的 PID 文本文件不需要删除即可重新获取锁。

以下为阶段 ① 的运行记录。

首个恢复点已于 step 2000 读取验证：文件大小 767,066,797 bytes，SHA-256 为
`2496b17f64d428275920954ebfcbf936fd47ee5f945b48cd94c0b7b1d7fdd5fc`。该文件是持续覆盖的
运行恢复点，因此哈希只用于证明当时的原子快照，不作为最终模型哈希。

step 40,000 在完整的 32-micro-batch optimizer 边界保存后，已由后台六阶段流水线成功重载。
step 48,000 再次于完整边界切换至包含累计耗时和 Pretrain/SFT 对照评估的新版流水线；step 52,000
由新版训练器自动写出后，完整恢复审计通过并继续训练。累计耗时由 `17,393.61s` 增至
`18,790.01s`，与新增 4,000 个 micro-step 的实测速度一致，证明恢复后计时没有归零。完整字段与哈希见
[`reports/llm-resume-step52000.md`](../reports/llm-resume-step52000.md)。这些证据验证了修正后的边界
checkpoint 恢复路径；训练器也会对最后不足一个 accumulation 的 micro-batch 执行尾部 optimizer update。

step 60,000 时又验证了一次完整恢复，并处理了低电量休眠导致的计时污染：模型与 270 个 optimizer
张量全部有限，33,228 秒系统休眠已从 active training time 中扣除并单列，恢复后的 step 60,020
回到 0.3420 秒/micro-step。修复依据、前后 SHA-256 与防重复机制见
[`reports/llm-sleep-recovery-step60000.md`](../reports/llm-sleep-recovery-step60000.md)。后续训练器会自动识别
超过 60 秒的步间隔，防止休眠再次污染吞吐统计。

step 100,000 的只读审计再次确认 63,912,192 个模型参数与 270 个 optimizer 张量全部有限，config、
RNG 和 optimizer 边界完整，累计 active / suspended time 分别为 35,873.93 / 33,228 秒；详见
[`reports/llm-resume-step100000.md`](../reports/llm-resume-step100000.md)。

阶段 ① 已于 2026-09-11 16:05 跑完 317,048 步并写出最终 checkpoint，流水线随即自动启动阶段 ②。
最终步数、耗时与 checkpoint 哈希见下。

**`llm-64m-pretrain-mps` 最终结果**：validation loss **1.8326**、perplexity **6.2504**，
评估使用完整的 2,048 条 held-out 验证集。纯训练耗时 112,248.73 秒（31 小时 11 分），
系统休眠 36,920.19 秒（10 小时 15 分）已单独扣除，合计墙钟约 41 小时 26 分。
最终 checkpoint 为 `artifacts/checkpoints/llm-64m-pretrain-mps.pt`，256,849,685 bytes，
SHA-256 `2baf3b6a75fb2cd7decaff66be72134a9d9d5e8c21f938600ef772913c5e3075`，
91 个 state tensors、63,912,192 参数全部有限。详见
[`reports/llm-pretrain-final.md`](../reports/llm-pretrain-final.md)。

注意：此最终数值（2,048 条）与 step 272,000 的中期诊断（32 条、1.8560）在**验证集规模和
模型步数上都不相同**，不能相减来推算最后 4.5 万步的收益。两者接近只说明训练末期 loss 已
进入平台期，也印证 32 条小样本诊断足以判断趋势、但不足以作为最终数字。

step 12,000 已保留模型中期快照并按 32 条 held-out 样本完成诊断：validation loss 4.0441、
perplexity 57.06、BPB 2.0264。固定生成已有局部语义能力但仍存在复读、角色漂移和乱码，详见
[`reports/llm-pretrain-step12000.md`](../reports/llm-pretrain-step12000.md)。

step 272,000 用完全相同的协议（CPU、同 32 条样本、同 4 条 prompt、20 新 token 贪心解码）
重做诊断并保留权重快照（`llm-64m-pretrain-step272000.pt`，255,681,813 bytes，SHA-256
`199178119abc6573ae4ecf71c7255bfcffc71fc5567377346aa9521e6a1bda4d`）：validation loss 降到
1.8560、perplexity 6.398、BPB 0.9300，相对 step 12,000 分别改善 54.1% / 88.8% / 54.1%。
定性上，退化乱码与角色漂移已消失，并能正确回答"中国的首都是北京"；复读和指令遵循仍未
达标，需 SFT 阶段处理。详见
[`reports/llm-pretrain-step272000.md`](../reports/llm-pretrain-step272000.md)。
