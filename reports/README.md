# Reports

这里存放经过验证并适合提交 Git 的实验摘要、训练曲线和评估结果。原始日志、数据和大体积 checkpoint 位于被忽略的 `artifacts/` 或外部模型仓库。

六阶段训练与三模型评估全部完成后，运行 `python scripts/build_final_report.py` 会严格检查所需
checkpoint、日志、评估 JSON 和 QIVD manifest，再生成 `final-results.md` 与本地 artifact manifest；
缺少任一证据时命令会失败，避免发布半成品结果。

仓库提交经过提炼的结果、公开截图与图表，以及 GitHub 学习报告。用户最新要求已取消 DOCX/PPTX
制作与交付；已有样稿仅作内部历史记录，不作为后续任务或交付文件。
原始 checkpoint、数据、optimizer state、逐步 console log 与完整评估 JSON 保持在服务器忽略目录，
不发布到 GitHub。六阶段完整实验与仓库结果解读尚未完成，不以本索引代替完成证据。

最终本地产物表会记录六阶段 checkpoint 的文件大小与 SHA-256，方便确认本机证据对应关系；表中出现
哈希不表示 checkpoint 文件会被提交或发布。

## 当前 CUDA 实验证据

- [正式 run 登记](formal-run-mm64-cuda-20261024.md)
- [CUDA LLM 两阶段结果总览与学习解读](cuda-llm-final-overview.md)（训练预算、分布内验证、统一文本回归与真实回答分开解释）
- [真实截图与待补证据索引](evidence/mm64-cuda-20261024T075332Z/README.md)
- [step 12,000 早期诊断与失败案例](cuda-pretrain-step12000-diagnostic.md)
- [LLM 最终检查点的固定对话诊断](cuda-llm-final-chat-diagnostic.md)（两阶段严格 0/8，包含格式失败与事实错误的区别）
- [LLM Pretrain 最终训练曲线](evidence/mm64-cuda-20261024T075332Z/E002-llm-pretrain-final/README.md)与 [LLM SFT 最终训练曲线](evidence/mm64-cuda-20261024T075332Z/E002-llm-sft-final/README.md)（真实 history 导出，各 5 图，附 CSV 和校验信息）
- [三模型设计、实现与结论边界](../docs/design-implementation.md)
- [预训练 step 164,000 恢复点完整性审计](pretrain-midpoint-integrity-20261024.json)（中间状态，非能力评估）
- [VLM Alignment 截断故障与精确复现](cuda-vlm-slot-truncation-failure.md)（原运行失败证据）
- [当前 CUDA 修复运行与继承关系](cuda-vlm-repair-run.md)（VLM Alignment 已启动，完整实验未完成）
- [历史文件样稿与仍有效的曲线和对话诊断工具](../reporting/README.md)（DOCX/PPTX 已取消）

其他 MPS 报告属于历史运行，不作为本次 CUDA 六阶段完成的证据。
