# E008：修复运行启动后的真实终端证据

![修复运行实际记录](E008-repair-startup-terminal.png)

- Run：`mm64-cuda-vlmfix-v3-20261024T170500Z`；代码 `9eafe5a548df7d5eae7ce2efe420e512a52d647e`。
- 类型：实际 macOS Terminal 窗口截图，未重绘终端或修改像素，不是模型架构图。
- 查看时间：2026-10-24 17:04:11.437412 UTC；截图保存时间：17:04:17 UTC。
- 原始指标：step 2,980、`recorded_at_utc=2026-10-24T17:04:10.334252+00:00`、
  segment `16e48445d4ae`，来自新 run 的 `metrics/vlm-vision-alignment-cuda.jsonl`。
- 内容：2,916 个输入指纹、GPU 型号、745 次 optimizer update、loss/LR/梯度、活跃耗时、名义
  padded 吞吐、PyTorch 进程 peak 和实际整卡资源查询。全卡占用不能全部归因于本 run。
- PNG SHA-256：`677c3d2ab56cf20b9b5c7c175c0bfe4819884931f0dfd2e8ad7facacdcecf469`。
- 捕获入口：[capture_run_evidence.applescript](../../../reporting/capture_run_evidence.applescript)。
  SSH 完成后才捕获窗口，避免活动进程标题泄露连接参数；没有密码或 token。

已视觉检查：运行身份、时间、数字、脚注和 GPU 表格可读，未发现遮挡或敏感凭据。
这是启动后的中间记录，不证明已经越过原故障步、完成训练或建立视觉能力。
完整继承关系与故障说明见[修复运行登记](../../cuda-vlm-repair-run.md)。
