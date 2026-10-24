# 64M CUDA 正式实验登记

状态（2026-10-24 16:37 UTC 后核验）：**LLM Pretrain/SFT 已完成；VLM Alignment 因图像槽位
截断失败，流水线已停止。完整实验与报告尚未完成。** 复现原因、恢复点和已归档 LLM 指标见
[VLM 截断故障记录](cuda-vlm-slot-truncation-failure.md)。下文早期记录保留其原时间与解释范围。

| 项目 | 记录 |
|---|---|
| Run ID | `mm64-cuda-20261024T075332Z` |
| 代码 commit | `53f9b9f46560bee446d1c74e89cac991b71a8053` |
| 快照创建时间 | 2026-10-24 07:54:30 UTC |
| 流水线启动时间 | 2026-10-24 07:55:23 UTC（15:55:23 Asia/Shanghai） |
| 硬件 | 物理 GPU 1，RTX PRO 6000 Blackwell Server Edition |
| 输入指纹 | 2,916 个原始数据、感知组件、Tokenizer 和视频缓存文件 |
| 目标阶段 | LLM Pretrain/SFT，VLM Alignment/SFT，Video-Omni Alignment/SFT |
| 初始化 | 随机初始化语言主干；冻结公开预训练 SigLIP2 |
| 模型能力结论 | 尚无最终证据，不作完成或能力提升声明 |

六阶段配置固定在服务器 run 目录 `configs/`，完整源码固定在 `source.tar`。已通过 96 项测试，
包含源码与依赖变更检测、同尺寸数据篡改检测、归档覆盖保护、指标口径及已有模型测试；该代码提交
的 GitHub CI 也已通过。

## 已留存证据

- 服务器 run 目录保存 `manifest.json`、`events.jsonl`、`metrics/` 和完整 `pipeline.console.log`。
- 公开截图与来源见[本次视觉证据索引](evidence/mm64-cuda-20261024T075332Z/README.md)。
- 启动后原始指标已包含 UTC、segment、PID、恢复起点、有效 optimizer update 次数、名义 padded-token
  吞吐、活跃耗时和当前进程 CUDA 显存。初期 loss 下降只作优化过程记录，不作模型能力结论。
- 阶段完成后归档校验后的最终权重、恢复文件、阶段总结、实际日志和固定完整评估。

## 首个恢复点审计

step 4,000 已于完整 optimizer 边界落盘并通过只读审计：63,912,192 参数、91 个模型张量均有限；
90 个 optimizer state entries、270 个 optimizer 张量均有限；配置匹配、history 201 条，累计活跃
训练 245.09 秒，无暂停。快照 SHA-256 为
`5d2271dadb21e51c42c9f94e52fcd719de39b41240434112e3fcc42838948cf6`。
完整审计位于服务器 run 目录 `audits/first-resume.json`，并已写入事件记录。
运行期间恢复文件持续覆盖，因此该哈希仅代表本次审计快照，不代表最终权重。

## 预训练过半的恢复点审计

2026-10-24 10:41:52 UTC 对 step **164,000 / 317,048**（51.73% micro-step 预算）的恢复点
执行只读审计；不是六阶段整体完成度，也不是模型能力进度。结果见
[原始审计 JSON](pretrain-midpoint-integrity-20261024.json)。

- 63,912,192 个独立参数、91 个模型张量均有限；90 个 optimizer entries、270 个 optimizer 张量均有限。
- 配置匹配，处于完整 optimizer 更新边界，history 8,201 条；累计活跃 9,958.22 秒，暂停 0 秒。
- 恢复文件 769,305,387 bytes，SHA-256 为
  `d0a6ac554c6a2bcd4d156bb96f87406ec0e27c6df62d0d271b6879e41c46db6b`。
- 模型与 optimizer 快照仅留服务器 run 下 `audits/pretrain-midpoint-20261024T104200Z/`，
  GitHub 仅保存审计 JSON，不上传权重。

训练器通过原子替换保存恢复点；审计先对当时已完成保存的 inode 建立硬链接，再读取张量与计算哈希，
因此不会在训练刚好保存时把一个版本的张量和另一个版本的哈希写入同一记录。训练器后续替换原路径
不会修改本次留存快照。保留该 inode 在下一次保存后会额外占用约 0.77 GB 数据盘空间。

该检查验证配置、训练状态结构、数值有限性与文件身份，不执行暂停/恢复，不证明 CUDA RNG 全覆盖、
恢复后的逐位轨迹一致或最终能力。`training_complete=false`；正式阶段结束后仍须按原门禁重新审计。

## 早期诊断

已从 step 12,000 恢复点提取只读快照，在 CPU 上完成 32 条留出文本与 4 条固定提示的短生成诊断，
NLL 3.988987、PPL 54.000153。存在首都事实错误、复读和概念混淆；不能据此宣称能力提升。
原始结果保留在 run 的 `diagnostics/`，协议与实际失败案例见
[早期诊断记录](cuda-pretrain-step12000-diagnostic.md)。完整固定评估仍未完成。

## 仍需完成

六阶段正式训练、固定完整评估、图像/视频消融、最终曲线与真实结果截图、具体设计实现解读，以及
GitHub 中文实验解读。用户最新要求已取消 DOCX/PPTX 制作与交付，已有样稿只作内部历史保留。
实验记录规则见[实验与证据说明](../docs/experiment-evidence.md)。

不启用周期性监控。训练期间保留后台流水线及其自身日志；后续工作以实际进程和落盘证据判断进度。
