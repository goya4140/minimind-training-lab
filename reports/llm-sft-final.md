# LLM SFT 完成报告（含语言能力回归）

阶段 ② `llm-64m-sft-mps` 已于 2026-09-11 21:08 (GMT+8) 跑完全部 40,000 个 micro-step 并写出最终
checkpoint。训练器随后交棒到阶段 ③ VLM 对齐，但该阶段在第 20 步崩溃（见文末「流水线中断」）。

## 最终结果

| 指标 | 值 |
|---|---:|
| 总 micro-steps | 40,000 / 40,000 |
| Optimizer updates | 约 10,000 次（每 4 步一次） |
| 训练 loss | 2.5216（step 1）→ 1.1410（step 40,000，末步单点） |
| 末期 loss（step 39,960–40,000 区间） | 1.68 ~ 1.92 |
| 分布内 validation loss | **0.8632** |
| 分布内 validation perplexity | **2.3707** |
| 训练吞吐 | 0.4530 秒/micro-step |
| 耗时 | 18,118.29 秒（5 小时 1 分 58 秒），无休眠扣除 |
| 模型参数 | 63,912,192（去重后），全部有限 |

分布内验证集是 `sft_t2t_mini.jsonl` 的 2,048 条 held-out 样本，与预训练的验证集**不是同一批数据**，
因此 0.8632 不能与预训练的 1.8326 直接相减——后者是无条件语言建模，前者只监督 assistant 回复片段
（prompt 部分被 mask）。两者语义不同。

## 语言能力回归（关键结论：有轻微反弹）

按流水线约定，用 `scripts/evaluate_bpe_llm.py` 在**与预训练完全相同的 held-out 语料**
（`pretrain_t2t_mini.jsonl` 末 2,048 条，670,079 个预测 token、1,906,560 UTF-8 字节）上重跑评估，
并分别在本机 MPS 与 CPU 上各跑一次，结果一致：

| 模型 | Val loss | Perplexity | BPB |
|---|---:|---:|---:|
| 预训练 `llm-64m-pretrain-mps` | **1.8328** | **6.2515** | 0.9293 |
| SFT `llm-64m-sft-mps` | **1.9089** | **6.7457** | 0.9679 |
| 变化 | **+0.0761（+4.15%）** | **+7.94%** | +0.0386 |

**这是真实的、可复现的反弹，但幅度很小。** 结论需要两面看：

- 不是崩溃：perplexity 6.25 → 6.75 仍在同一量级，模型没有忘记中文；
- 但方向确实是退化的：assistant-only 微调（学习率 1e-5、40,000 步）让语言建模分布向"指令-回复"
  格式偏移，代价是通用文本上的预测能力小幅下降。这是小模型上典型的**对齐税（alignment tax）**，
  属于有价值的失败模式，如实记录。

定性观察也支持这一判断：SFT 后生成的句式明显更规整、会分点作答（"1. 确定目标：…2. 制定计划：…"），
但 `distinct_2` 多样性指标在"机器学习"这一条上从 1.0 掉到 0.329 —— 复读加重了。模型用语言多样性
换来了格式稳定性。

证据文件：
- `artifacts/eval/llm-sft-final.json`（流水线自评，MPS）
- `artifacts/eval/llm-sft-regression-cpu.json`（独立复跑，CPU，本报告生成）

## Checkpoint

- 路径：`artifacts/checkpoints/llm-64m-sft-mps.pt`
- 大小：255,827,029 bytes
- SHA-256：`8107602a1a171d8e06aa71f47bf2528c50282ef2440231a3bd8b5fdd73e78b10`
- 校验：91 个 state tensors、63,912,192 参数、全部有限（`all_finite: true`）

这是**最终模型**哈希，与运行期间持续覆盖的 `.resume.pt` 不同，可用于复现校验。
它同时是阶段 ③ VLM 与阶段 ⑤ Video-Omni 两条分支的共同初始化点。

## 这一阶段完成了什么

模型学会了指令遵循的"形"：能维持 `用户：/助手：` 结构、会用序号分点、会针对问题给出结构化回答。
分布内 val loss 0.8632 说明它对 SFT 数据的拟合相当充分。

## 这一阶段没有完成什么

- 语言建模能力小幅退化（见上）；
- 复读没有根除，只是换了形式——从预训练期的整句复读变成模板化句式复用；
- 事实准确性没有改善，也不该期待 63.9M 参数做到这一点；
- **尚未做模态消融**——VLM / Video-Omni 阶段完成后必须回过头来同时做语言回归与模态消融。

## 流水线中断（阶段 ③）

SFT 完成后流水线自动启动 ③ `vlm-vision-alignment-mps`，但在第 20 步抛错退出：

```
ValueError: sample 3 has 128 image placeholders; expected 64 for 1 image(s)
  at src/minimind_lab/vlm/model.py:75 (inject_image_features)
```

已定位到数据根因（只读核查，未修改任何数据或代码）：`data/raw/pretrain_i2t.parquet` 共 1,274,698 行，
其中 **2,091 行（0.16%）** 的非 system 消息里含有 2 个 `<image>` 标记，但 `image_bytes` 只有 1 张图
（抽查行 1152 / 1645 / 1737 确认）。数据管线把每个 `<image>` 展开成 64 个 `<|image_pad|>`，
于是出现 128 个占位符 vs 1 张图像特征的不匹配，触发硬断言。

这不是随机故障，而是**确定性会在约 0.16% 的批次上必然复现**的问题：不修数据或加校验，重跑同样会挂。
修复方向二选一——清洗这 2,091 行（去重标记），或在 `ParquetVLMDataset` 里跳过/修正占位符数与图像数
不一致的样本。目前流水线处于停止状态，`artifacts/checkpoints/vlm-alignment-mps.lock` 为崩溃残留的锁文件。

## 复现

```bash
uv run python scripts/train_sft.py --config configs/llm/sft-mps.yaml
uv run python scripts/evaluate_bpe_llm.py \
  --config configs/llm/pretrain-mps.yaml \
  --checkpoint artifacts/checkpoints/llm-64m-sft-mps.pt \
  --device cpu --output artifacts/eval/llm-sft-regression-cpu.json
```

完成后结果写入 `artifacts/logs/llm-64m-sft-mps.json`（Git 忽略）。
