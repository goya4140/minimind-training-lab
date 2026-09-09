# Reports

这里存放经过验证并适合提交 Git 的实验摘要、训练曲线和评估结果。原始日志、数据和大体积 checkpoint 位于被忽略的 `artifacts/` 或外部模型仓库。

六阶段训练与三模型评估全部完成后，运行 `python scripts/build_final_report.py` 会严格检查所需
checkpoint、日志、评估 JSON 和 QIVD manifest，再生成 `final-results.md` 与 release manifest；
缺少任一证据时命令会失败，避免发布半成品结果。
