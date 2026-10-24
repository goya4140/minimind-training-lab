# CUDA 修复运行：VLM 视觉槽位与只读阶段继承

状态：**修复运行已进入 VLM Alignment，后续完整训练与评估尚未完成。**

| 项目 | 记录 |
|---|---|
| 当前 run | `mm64-cuda-vlmfix-v3-20261024T170500Z` |
| 训练代码 | `9eafe5a548df7d5eae7ce2efe420e512a52d647e` |
| 创建时间 | 2026-10-24 16:59:30 UTC |
| 启动时间 | 2026-10-24 17:00:06 UTC（Asia/Shanghai 2026-10-25 01:00:06） |
| 工作目录 | `/root/autodl-tmp/wlh_goya/MiniMind/experiments/vlm-slot-v3` |
| 物理 GPU | 1；显式 `CUDA_VISIBLE_DEVICES=1` |
| 原始父 run | `mm64-cuda-20261024T075332Z`，训练代码 `53f9b9f` |
| 输入指纹 | 2,916 个；原始数据、Tokenizer、冻结编码器与视频缓存复用 |
| 截断策略 | `full-visual-slots-answer-window-v2` |

## 继承与重训边界

LLM Pretrain/SFT 使用父 run 已完成、已评估且归档的精确权重，不重复训练；父 run 的语言模型
确实从随机初始化训练。继承事件保留原 run ID、模型/optimizer SHA-256 和原评估身份，不把旧
history 改标为新 run 的计算。模型与恢复状态均只读复核，评估文件保留原始时间与内容。

四个多模态阶段在新 run 中执行：VLM Alignment 30,000、VLM SFT 30,000、Video Alignment
7,200、Video SFT 4,800 micro-step。六阶段 YAML 与父 run 一致，批量、累积、学习率、随机种子、
冻结策略和数据划分不变。VLM 和视频分支分别继承 LLM SFT，不串联。

旧失败 Alignment 的 step 26,000 optimizer **没有继承**；新 Alignment 从 step 0 的新增
Projector 开始，避免把不同数据构造策略的轨迹混作一次无变化恢复。

## 两次故障与修复

第一处故障是[视觉槽位截断](cuda-vlm-slot-truncation-failure.md)：在长问题尾部的 `<image>`
展开为 64 槽位后，被 450-token 前缀截断成 33 槽位。修复保留所有视觉槽位，并在必要时选取回答
附近的非视觉文本窗口，原位置 labels 随同选择，视觉槽位和 padding 不计预测损失。修复可能丢弃
早期上下文，不是无损压缩；原图像顺序与严格槽位检查不放宽，也不删除出错数据行。

第二处发生在启动集成：`mm64-cuda-vlmfix-20261024T170000Z`（代码 `9d99b05`）的继承阶段
被旧 verifier 重写了摘要 `verified_at`，触发 immutable archive 防覆盖保护，在进入新 VLM
训练前退出。没有覆盖归档或绕过保护。该运行和其错误栈留在 `experiments/vlm-slot-v2-offline/`。

当前流水线对已完成/继承阶段逐文件检查当前输出与归档 SHA、完整恢复状态及配置，然后记录
`preserved_stage_verified`、`evaluation_reused` 与带 `inherited=true` 的完成事件，不重写
训练摘要。新增测试明确验证 bytes/mtime 不变，并拒绝被篡改或缺失的证据。

## 启动前实际验证

- 独立 checkout 中全部 **105 项 tests 通过**，含原模型、数据、恢复、报告与新增边界/继承测试。
- 原失败 batch 的四条 filtered index 为 `[606220, 1246578, 584084, 1175311]`；实际 Tokenizer
  与图像处理后，槽位数 `[64,64,64,64]`，有效回答 target 数 `[248,193,240,161]`。
- 在 GPU 1 上执行一次 BF16 前向、反向、梯度裁剪与 Projector 更新：loss `3.364625`、裁剪前
  norm `3.200339`，均有限；Projector 参数确有变化，冻结的语言与视觉编码器没有梯度。
  探针不保存正式权重，不能当作模型能力结果。
- CUDA 探针在 `9d99b05` 的 VLM 实现上执行；当前 commit 只另修继承控制流，VLM 数据实现哈希
  仍为 `d88b1c48b587e95959a90979ef5d5883ee2e43a91647a8a2f56e3a8c030daf06`。准备工具核对
  该哈希和 GPU 绑定后才接受探针，副本保留在新 run 的 `cuda-repair-probe.json`。

17:04:55 UTC 的启动核验记录为 step 3,540 / 30,000、loss `2.891114`、有效 optimizer updates
885、`resumed_from_step=0`。这只是当时的运行记录，不是持续更新的进度或最终能力评估。
真实终端截图与哈希见 [E008 证据索引](evidence/mm64-cuda-vlmfix-v3-20261024T170500Z/README.md)。

### 新训练恢复点完整性核验（非最终结果）

2026-10-24 17:27:32 UTC 对 step 23,000 恢复点执行 CPU 只读审计，
先硬链接固定训练器原子保存的 inode，避免加载期间路径被新检查点替换。
[完整审计摘要](vlm-repair-progress-integrity-20261025.json)记录：

- 97 个模型 state tensors、65,094,912 个去重存储参数全部有限；此处是保存的语言主干与投影组件，
  不包括单独复用的冻结 SigLIP2 外部资产，不能当作包含视觉编码器的总模型参数量。
- optimizer 的 6 个 state entries / 18 个张量全部有限，处于完整更新边界；配置与固定 YAML 一致，
  RNG 存在，恢复文件含 1,151 条有序 history，活跃训练累计 1,606.394 秒。
- history 只属于新 run、单一训练 segment / PID，`resumed_from_step` 全部为 0；
  第 23,000 步记录 5,750 次更新，未从旧失败运行的 optimizer 接续。
- 恢复文件 270,208,409 bytes，SHA-256：
  `38221de510972ed49ab1241a7df74014a646b902fbbebcb598fb4adb2f114ab6`。
  原始快照与 audit 保留本 run 的 `audits/alignment-progress-20261024T172800Z/`。

这一核验没有实际重启训练，也不是 CUDA 逐位恢复一致性测试或阶段完成证据，未改变训练进程和 GPU。
当时尚未到旧失败的 microstep 26,896，因此不能凭此次审计声称已经跨过历史故障点；
精确旧故障批次的修复由前述独立 CUDA 探针证明，完整新阶段仍待训练结束。

## 资源与保留

服务器直连 GitHub 克隆未完成，因此使用本地 Git bundle 经 SSH 部署同一 commit，核对 HEAD
后再运行，没有更换代码版本。数据和外部资产通过忽略目录内的链接复用，没有重新下载。

诊断产生的默认 Arrow 缓存副本逐文件核对内容后改为引用项目缓存；初次硬链接因 builder 原子
重建缓存会保留旧 inode，随后改为符号链接。相同的继承 LLM 文件也按哈希合并存储，16 个逻辑
路径仍存在并共享父归档 inode。没有删除模型、原始数据、optimizer、日志或失败证据；未触碰正在
更新的 VLM 检查点。所有这些操作在父 run 事件中记录。

新 run 的证据根目录为工作目录下 `artifacts/runs/mm64-cuda-vlmfix-v3-20261024T170500Z/`。
保留 manifest、source.tar、六阶段配置、继承事件、原始日志、逐阶段归档和实际评估。继承 LLM
的曲线与补充评估必须引用父 run 的原始身份；新多模态阶段使用本 run 的身份。

已在本 run 的 `supplemental-protocols/` 继承父 run 冻结的 `chat-diagnostic-v1.json` 和来源勘误，
文件与父注册事件的 SHA 一致，在首次多模态补充对话推理前登记。两个 LLM 最终结果仍引用
[父 run 评估](cuda-llm-final-chat-diagnostic.md)，不重新标成修复 run 的产物；
后续 VLM/Video 对话诊断抽取语言主干，不输入视觉内容，不替代视觉性能评估。
固定 VLM 验证切分、关键词代理与图序指标的限制见[评估口径](../docs/evaluation.md)。

不设置周期监控，不停止其他项目。PyTorch 进程显存、整卡显存与其他进程占用不是同一口径；
截图中的整卡数字不能全部归因于本模型，也不能用于严格的独占硬件吞吐对比。

## 可复核入口

新 checkout 的准备脚本为 `scripts/prepare_vlm_repair.py`，CUDA 探针为
`scripts/probe_vlm_repair_cuda.py`。准备需要干净代码、独立目录、精确父归档、相同配置、成功探针
与足够空间；已有输出不覆盖。实际准备使用的探针文件位于前一隔离 checkout 的忽略日志目录。

后续恢复**现有**修复 run 仅使用当前固定工作目录与 run ID，不重复执行准备/导入：

```bash
cd /root/autodl-tmp/wlh_goya/MiniMind/experiments/vlm-slot-v3
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD/src"
bash scripts/start_cuda_training.sh --run-id mm64-cuda-vlmfix-v3-20261024T170500Z
```

已有同一运行进程时不要再次启动。该命令不是已完成所有阶段的声明。完整多模态训练、固定评估、
全量视频输出重放、最终曲线和结果解读均仍待完成。
