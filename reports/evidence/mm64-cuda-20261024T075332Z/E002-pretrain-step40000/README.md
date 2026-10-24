# E002a 预训练前 40000 步的日志生成曲线

本组是**阶段性图表，不是最终曲线，也不是真实终端截图**。Run ID 为
`mm64-cuda-20261024T075332Z`，训练预算为 317,048 micro-step。覆盖日志时间为
2026-10-24 07:55:29 至 08:36:25 UTC，对应 checkpoint history 的 step 1 至 40,000。

## 图表与来源

| 证据 | 图表 | 口径 |
|---|---|---|
| E002a-1 | [训练 loss](loss_mean.png) | 每 2,000 micro-step 桶内已记录 micro-batch loss 的均值 |
| E002a-2 | [学习率](learning_rate_last.png) | 每桶最后一条已记录的 optimizer LR |
| E002a-3 | [梯度范数](grad_norm_median.png) | 每桶记录中最近一次 optimizer update 梯度范数的中位数 |
| E002a-4 | [名义吞吐](padded_tokens_per_second_median.png) | 每桶已记录的同进程段累计平均 padded token/s 的中位数 |
| E002a-5 | [CUDA 显存](cuda-memory.png) | 每桶最后记录的 PyTorch allocated、reserved、进程 peak，GiB |

图表基于[精简分桶 CSV](curve-data.csv)，来源见[数据指纹与定义](curve-source.json)，
各 PNG 哈希见[图表清单](figure-manifest.json)。训练代码固定为 `53f9b9f`。
分桶宽度由完整预算预先确定，不随当前训练进度改变；原始 2,001 条记录聚合为 20 桶，
没有把未记录的训练 batch 当作已观察值，也没有给缺失桶补零。

服务器保留完整导出 history、精确绘图工具副本、CSV、metadata 和图表：

```text
artifacts/runs/mm64-cuda-20261024T075332Z/curves/intermediate-002-source/
artifacts/runs/mm64-cuda-20261024T075332Z/curves/intermediate-002-figures/
```

`history-records.json` 只在服务器保存，不提交 GitHub。恢复文件会继续原子覆盖，因此 metadata
中的 checkpoint SHA-256 代表导出时读到的历史恢复点，不代表当前或最终权重。导出时对同一
打开的文件描述符完成哈希和加载，避免后台训练替换文件名导致来源错配。

## 目前能解释什么

第一桶已记录 loss 均值为 7.040278，最后一桶为 2.477700。这说明本次优化目标在该区间内下降，
不能据此证明事实知识、指令遵循或泛化能力提升。它不是全训练样本的 token 加权均值，
也不是此前 32 条留出诊断的 NLL；这两种 loss 不应放在同一条线中直接比较。

最后一桶名义吞吐中位数约 44,304 padded token/s，含 padding 和未监督上下文，且来自同进程段
累计平均，不是瞬时吞吐或精确推理速度。显存 reserved 约 2.848 GiB、已记录进程 allocated
peak 约 2.594 GiB，与设备总占用不是同一概念。allocated 的锯齿仅是日志采样时刻的观察，
不据此断言显存泄漏或特定内核行为。

## 生成与检查

入口为 [training_curves.py](../../../../reporting/training_curves.py)，使用服务器已有 Torch、
PyYAML 与 Matplotlib 3.10.5，无新增安装、无 GPU 训练任务变更。仅本组独立 PNG 使用静态绘图；
后续 PPTX 可从同一精简 CSV 生成原生可编辑图表，不把 PNG 中的数据线声称为可编辑对象。

已逐张检查 5 张 PNG 的标题、坐标轴、单位、脚注及图例；20 桶 loss 均值、LR、吞吐中位数和
显存末值均从服务器原始 history 独立重算，2,001 条计数对齐。另有 8 项合成单元测试，覆盖
重复/乱序、错误 run、越界、非有限值、缺失指标和不可能的 peak。合成数据只用于单元测试，
不进入实验图表。`--final` 还在本次未完成 checkpoint 上实测拒绝发布，未留下伪最终输出。

灰度可区分的显存线使用实线、虚线和点线；未进行软件色盲模拟，不声称已完成该项测试。
六阶段最终曲线与评估尚待完成。
