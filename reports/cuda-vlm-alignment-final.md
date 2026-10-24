# CUDA VLM Projector Alignment：最终阶段结果

状态：该阶段已完成，VLM SFT 已自动启动；六阶段完整实验与视觉能力评估仍未完成。
run：`mm64-cuda-vlmfix-v3-20261024T170500Z`，训练 commit：
`9eafe5a548df7d5eae7ce2efe420e512a52d647e`。

## 训练与验证

| 项目 | 实测或固定配置 |
|---|---|
| 阶段完成事件 UTC | 2026-10-24 17:36:06.995219 |
| 预算 | 30,000 microsteps，7,500 optimizer updates |
| 可训练组件 | 1,182,720 参数的 Projector；语言主干和外部 SigLIP2 冻结 |
| 数据 | 原 1,274,698 行，单图标记筛选后 1,272,607 行；末尾 512 行留作验证 |
| batch / accumulation / length | 4 / 4 / 450 |
| 训练活跃时间 | 2,125.526780 秒，约 35分26秒；不含阶段结束验证和归档 |
| 阶段结束验证 loss | 2.781993867829442 |
| 新 VLM SFT 启动 UTC | 2026-10-24 17:36:07.067436 |

输入采用保留完整视觉槽位与回答窗口的修复策略，见[故障与修复说明](cuda-vlm-repair-run.md)。
数据摘要里的 `single-image-exactly-one-marker-v1` 指原行筛选，不是 token 截断策略；两者作用不同。
旧失败运行的 26,000 步恢复状态未继承，本次 Projector 从第 0 步训练。语言主干继承父 run 的
精确 LLM SFT，外部视觉编码器复用公开预训练资产，不是所有模块从零训练。

验证 loss 是各 batch 标量 assistant-token loss 的算术平均，不是跨 batch token 加权 NLL。
验证器会跳过非有限 batch，但汇总没有提供跳过计数；不能只凭有限均值声称每批都有限。
该数值与日志曲线只能描述本阶段目标，不能证明图像回答正确、视觉依赖或图像顺序能力。
正式固定图像问答、反事实和语言回归在 VLM SFT 完成后执行。

## 最终恢复完整性与归档

流水线完成训练验证后归档 4 个必要文件，独立 CPU 审计又核对了最终恢复文件：
97 个模型 tensors / 65,094,912 去重存储参数有限；optimizer 6 个 state entries / 18 个 tensors
有限；step=30,000，更新边界、配置、RNG 完整，history 1,501 条，`training_complete=true`。
保存参数包括冻结语言主干，不包括外部 SigLIP2，不能把保存参数量当可训练参数量或模型总量。
详见[最终审计 JSON](vlm-alignment-final-integrity-20261025.json)。

服务器工作目录：`/root/autodl-tmp/wlh_goya/MiniMind/experiments/vlm-slot-v3`。
归档相对路径：`artifacts/runs/mm64-cuda-vlmfix-v3-20261024T170500Z/stages/vlm-alignment/`。

| 归档文件 | Bytes | SHA-256 |
|---|---:|---|
| vlm-alignment-cuda.pt | 260,830,949 | `a621ced8c2a10f69e90ad2553533bfd1b7a81fca042ad05e81b43742f4227934` |
| vlm-alignment-cuda.resume.pt | 270,305,369 | `07f163cedbb320928a8a3ece33334af11215ea476ed705bd16e182379b1df6f4` |
| vlm-vision-alignment-cuda.json | 1,348 | `24719969478973e90d27bdee4c8e05655c642e2b2b96f74be9320c5168c08472` |
| vlm-vision-alignment-cuda.console.log | 878,983 | `b3ce083d2c408adb8d90e094759ab7b672d3b51d3efb02f9f6f79c44cd149c39` |

模型、恢复文件和完整 console 留服务器，不上传 GitHub。

## 曲线与真实截图

[最终 5 张曲线、分桶 CSV 与来源](evidence/mm64-cuda-vlmfix-v3-20261024T170500Z/E002-vlm-alignment-final/README.md)
从最终归档 resume 导出。1,501 条 history 按完整预算固定为 150 个 200-step 桶，全部记录数、
loss 均值、吞吐中位数与末步号均独立复算；PNG 逐张视觉检查通过，来源和图像哈希已核对。
首桶 loss 均值 3.134775、末桶 2.826270；后期仍波动，不把下降解读为能力达标。

[E009 实际终端截图](evidence/mm64-cuda-vlmfix-v3-20261024T170500Z/README.md#e009alignment-末步与-sft-启动记录)
显示 Alignment 30,000 步日志以及新 SFT 4,600 步日志；它是实际日志查看画面，不是生成图。
阶段完成以归档和完成事件为准，不仅依靠截图中的步数。

曲线导出时发现工具把该阶段的 experiment 名误写为 `vlm-alignment-cuda`，
实际固定 YAML 名为 `vlm-vision-alignment-cuda`。只修正独立 reporting 工具的名称映射，
并新增覆盖六份实际 CUDA YAML 的测试，没有修改训练代码、配置、记录或权重。
现有与新增报告工具共 34 项测试通过。原 LLM 曲线的旧工具快照与指纹保持原样。
