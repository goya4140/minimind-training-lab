# CUDA Video-Omni Temporal Alignment：阶段结果

该阶段已完成并归档；后续Video SFT及原固定评估也已完成，见[最终视频结果](cuda-video-sft-final.md)。
run `mm64-cuda-vlmfix-v3-20261024T170500Z`，训练 commit
`9eafe5a548df7d5eae7ce2efe420e512a52d647e`。

## 设计与训练

视频分支从父 run 的 LLM SFT 单独扩展，不继承 VLM SFT。冻结公开预训练 SigLIP2 与语言主干，
只训练随机初始化的 Temporal Adapter + Video Projector，共20,105,472参数。
接口是8帧视频＋问题→文本答案，不含音频或视频生成。
每帧64×768 patch features 经空间池化、帧位置、2层时序Transformer和query resampling，
得到16个视频向量，再投影并替换语言序列的16个视频位置；架构存在时序模块不等于已学会时序推理。

| 配置／实测 | 数值 |
|---|---|
| QIVD训练／验证 | 2,400 / 250；最终test另有250，不用于本阶段验证 |
| 划分seed／训练seed | 48 / 48 |
| microsteps／更新数 | 7,200 / 900 |
| batch／累积／文本长度 | 1 / 8 / 384 |
| 学习率／精度 | 2e-4→2e-5 / BF16 autocast |
| 活跃训练时间 | 379.286327秒，约6分19秒 |
| 阶段完成UTC | 2026-10-24 18:19:31.500274 |
| Video SFT启动UTC | 2026-10-24 18:19:31.559132 |

7,200步对应2,400训练样本的3个epoch。训练复用已完成并核验的FP16 patch缓存，
形状 `[2900,8,64,768]`，本阶段没有重新计算全部视觉特征；上述时间不包含先前缓存构建或
完整端到端数据准备，不应与原始MP4实时视觉前向的速度直接比较。最终视频评估仍从MP4解码。

## 验证：微小正差不等于时序能力

250条验证上的正常帧loss为 **1.1996735234260558**，倒序为 **1.1996735411286354**，
倒序减正常为 **1.770257962441235e-08**。这是各batch标量loss的算术平均，
不是跨batch监督token加权NLL；倒序仍沿用原视频答案，不是倒序视频自身真值准确率。

单次均值差几乎为零，不能因其符号为正就称时序敏感性已建立；也不能反过来据均值近零断言
所有样本输出完全相同。尚需最终SFT后的正常/倒序逐样本回答、受控时序集与类别结果。
当前只能说本阶段没有提供有意义的总体帧序利用证据，不调整正式训练参数或提前选择检查点。

## 最终恢复审计与来源

[独立最终审计](video-alignment-final-integrity-20261025.json)确认step=7,200、配置一致、更新边界与RNG完整，
136个模型state tensors/84,017,664去重存储参数有限；optimizer45个entries/135个张量有限，
history361条。保存参数含冻结语言主干，不含独立外部视觉资产，不能等同于可训练或全模型参数总量。

服务器工作目录：`/root/autodl-tmp/wlh_goya/MiniMind/experiments/vlm-slot-v3`。
以下文件位于 `artifacts/runs/mm64-cuda-vlmfix-v3-20261024T170500Z/stages/video-omni-alignment/`：

| 文件 | SHA-256 |
|---|---|
| video-omni-alignment-cuda.pt | `35737d932697cfd1ab1b7f733af1212f0ffbbfaaf621da6dc743e59ade807176` |
| video-omni-alignment-cuda.resume.pt | `31cb03a83f862191da9a5ca0f7d0baddac3a50159462c75833222e25bfddafba` |
| video-omni-alignment-cuda.json | `89e60fef420aa486724fe7ad3a33b7b9faf94348e8d4cc4c88c8f94d49ac26a4` |
| video-omni-alignment-cuda.console.log | `f606a2e7e50077a95d3be1ca0345196523df5214bfcdf7bbe412ac189e95ca91` |

[最终5图与CSV](evidence/mm64-cuda-vlmfix-v3-20261024T170500Z/E002-video-alignment-final/README.md)
绑定最终归档resume，120个60-step桶全部独立复算，5图已逐张查看且来源/PNG指纹核验一致。
首末桶loss均值1.779987931→0.636228164，中间波动很大；每桶仅少量日志，不能挑末点当总体收益。
完整history、工具副本、模型和optimizer保留服务器，不上传GitHub。
