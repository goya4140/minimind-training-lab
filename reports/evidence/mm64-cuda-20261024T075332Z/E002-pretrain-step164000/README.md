# E002b：预训练前 164,000 步阶段性曲线

这是**由真实训练记录生成的静态 PNG，不是终端截图或最终评估**。Run ID 为
`mm64-cuda-20261024T075332Z`，训练 commit 为 `53f9b9f`。覆盖 2026-10-24 07:55:29 至
10:41:27 UTC 的 step 1–164,000，即该预训练阶段 317,048 micro-step 预算的 51.73%，不是整个
六阶段项目的完成度。

## 图表与可复核来源

| 图表 | 口径 |
|---|---|
| [训练 loss](loss_mean.png) | 每桶已记录 micro-batch loss 的算术均值 |
| [学习率](learning_rate_last.png) | 每桶最后记录的 optimizer LR |
| [梯度范数](grad_norm_median.png) | 每桶最近 update 梯度范数记录的中位数 |
| [名义吞吐](padded_tokens_per_second_median.png) | 每桶进程段累计平均 padded token/s 记录的中位数 |
| [CUDA 显存](cuda-memory.png) | 每桶末尾 PyTorch allocated、reserved 与进程 peak，GiB |

[精简 CSV](curve-data.csv)包含 82 桶，每桶宽度 2,000 micro-step，与早期 E002a 使用相同预定
分桶规则。[来源 JSON](curve-source.json)关联 8,201 条原始 history 和检查点 SHA-256；
[图表清单](figure-manifest.json)关联每张 PNG、绘图器、来源与 CSV 的校验和。
完整 history 和模型仅保留服务器，不发布到 GitHub。

来源是已经[通过完整性审计](../../../pretrain-midpoint-integrity-20261024.json)并独立保留的
164,000 步快照，SHA-256 为
`d0a6ac554c6a2bcd4d156bb96f87406ec0e27c6df62d0d271b6879e41c46db6b`。
后台训练继续更新默认恢复点，不影响这里的快照或图表来源。

## 可以读出什么

| 采样分桶统计 | 第一桶 1–2,000 | 最后一桶 162,001–164,000 |
|---|---:|---:|
| micro-batch loss 均值 | 7.040278 | 2.036304 |
| 最后记录 LR | 0.000499957 | 0.000262797 |
| 最近 update 梯度范数中位数 | 0.515390 | 0.344387 |
| 名义 padded token/s 中位数 | 44,456.95 | 44,793.86 |

loss 图显示采样到的优化目标下降，不证明知识、指令遵循或泛化提升；最后一桶单条 loss 范围仍有
1.310804–2.596234，均值曲线不代表每个 batch 都单调改善。它不是 token 加权验证 NLL，不能
与早期留出诊断的 loss 混为同一指标。该范围是观察值范围，不是置信区间。

吞吐包含 padding，且是进程段累计平均而非瞬时值，不代表有效监督 token/s 或推理速度。显存图
的 reserved 约 2.848 GiB、进程 allocated peak 约 2.594 GiB；它们不是整卡占用。allocated
锯齿只反映记录时刻，不由图形断言泄漏或某个 CUDA 内核行为。

## 生成与检查

使用项目现有 [training_curves.py](../../../../reporting/training_curves.py)，在服务器已有
Matplotlib 3.10.5 上导出独立静态 PNG，无新增依赖或训练设置变更。原导出器副本与新绘图器副本
分开保留，manifest 分别记录来源与绘图器哈希；不要求二者来自同一文档工具版本。

服务器 run 下保留：

```text
curves/intermediate-003-source/       # 原始 history、导出器副本、CSV 和来源
curves/intermediate-003-figures/      # 首次渲染稿，内部保留
curves/intermediate-003-figures-v2/   # 修正纵轴留白后的发布 PNG 和绘图器副本
```

82 桶的计数、loss 均值及范围、LR、梯度中位数、吞吐中位数、显存末值和活跃耗时均从原始 history
独立重算。逐张检查最终 5 张 PNG 的标题、轴、单位、脚注和图例。初稿吞吐曲线贴近上沿，已保持
零基线并增加 8% 上界留白，数值和分桶不变。显存线分别使用实线、虚线、点线，未做软件色盲模拟。

数据可视化审阅促使本次显式区分采样均值、观察值范围和能力评估，避免将平滑趋势误读为泛化证据。
全部六阶段最终曲线、固定评估与多模态输出仍待完成。
