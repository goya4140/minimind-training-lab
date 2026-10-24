# Reports

这里存放经过验证并适合提交 Git 的实验摘要、训练曲线和评估结果。原始日志、数据和大体积 checkpoint 位于被忽略的 `artifacts/` 或外部模型仓库。

六阶段训练与三模型评估全部完成后，运行 `python scripts/build_final_report.py` 会严格检查所需
checkpoint、日志、评估 JSON 和 QIVD manifest，再生成 `final-results.md` 与本地 artifact manifest；
缺少任一证据时命令会失败，避免发布半成品结果。

仓库提交经过提炼的结果、公开截图与图表，以及完成后的可编辑 PPTX、DOCX 学习报告。
原始 checkpoint、数据、optimizer state、逐步 console log 与完整评估 JSON 保持在服务器忽略目录，
不发布到 GitHub。报告文件尚未完成，不以本索引代替实际交付。

最终本地产物表会记录六阶段 checkpoint 的文件大小与 SHA-256，方便确认本机证据对应关系；表中出现
哈希不表示 checkpoint 文件会被提交或发布。

## 当前 CUDA 实验证据

- [正式 run 登记](formal-run-mm64-cuda-20261024.md)
- [真实截图与待补证据索引](evidence/mm64-cuda-20261024T075332Z/README.md)
- [step 12,000 早期诊断与失败案例](cuda-pretrain-step12000-diagnostic.md)
- [三模型设计、实现与结论边界](../docs/design-implementation.md)

其他 MPS 报告属于历史运行，不作为本次 CUDA 六阶段完成的证据。
