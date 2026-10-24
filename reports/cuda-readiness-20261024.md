# CUDA 重训准备验收（2026-10-24）

结论：环境、固定数据、六阶段真实数据短程训练及断点恢复均已验证，正式重训等待用户确认。
项目位于服务器 `/root/autodl-tmp/wlh_goya/MiniMind`，无周期性监控。

## 环境与资源

- Python 3.12.4；PyTorch 2.8.0+cu128；CUDA runtime 12.8；GPU capability 12.0。
- Transformers 4.57.6，Datasets 3.6.0，PyAV 16.1.0；`pip check` 无依赖冲突。
- 使用物理 GPU 1：NVIDIA RTX PRO 6000 Blackwell Server Edition；测试结束时可用显存约 16,090 MiB。
- 数据盘剩余约 55 GiB。项目约含 15 GiB 数据及特征、31 GiB 缓存、8.3 GiB 探针 checkpoint。
  正式训练沿用最新恢复点覆盖保存策略；额外正式 checkpoint 空间仍有余量。
- 项目虚拟环境只读复用基础环境的 torch，冲突包装在项目环境内；未停止既有 GPU 服务。

## 数据验收

四份文本/图像语料全部按原固定版本和 SHA-256 验证，详见
[`data-manifest.md`](data-manifest.md)。SigLIP2 三个文件及 Tokenizer 同样通过固定哈希检查。
QIVD 全部 2,900 个视频与元数据、许可文件匹配固定上游对象；离线重算聚合 SHA-256 为
`648b60445d59f1e445d0437ddd93f8ad97c09bb7e6ded0d46a996c98428ad7a3`。

冻结视觉特征缓存已完整生成：shape `[2900, 8, 64, 768]`，float16，2,280,652,928 bytes，完成位图全部为真。
VLM 数据加载沿用恰好一个图像标记的确定性筛选，不改变原始数据文件。

## 六阶段真实数据探针

所有探针使用正式模型、正式 micro-batch 和 BF16，单阶段 8 个 micro-step，validation 缩小到 2 条。
每阶段在 step 4 原子保存并退出，再恢复到 step 8；模型、optimizer、配置与 CUDA RNG 均通过检查。

| 阶段 | Micro-batch | 恢复路径 | 模型/optimizer 有限 | 峰值 PyTorch allocated 显存（GiB） |
|---|---:|---|---|---:|
| LLM Pretrain | 8 | 4 → 8 | ✅ / ✅ | 2.621 |
| LLM SFT | 4 | 4 → 8 | ✅ / ✅ | 2.760 |
| VLM Alignment | 4 | 4 → 8 | ✅ / ✅ | 1.419 |
| VLM SFT | 2 | 4 → 8 | ✅ / ✅ | 1.574 |
| Video Alignment | 1 | 4 → 8 | ✅ / ✅ | 1.272 |
| Video SFT | 1 | 4 → 8 | ✅ / ✅ | 1.458 |

LLM 另做相同配置的连续 8 步对照，确定性模式下最大参数绝对差为 **0.0**。初次默认 BF16 内核对照
在中断前就出现运行间数值差异，因此严格参数对照使用确定性探针，不把内核数值差异归为恢复失败。
另用已生成的正式视频特征缓存训练 4 步，最终模型与 135 个 optimizer 张量均为有限值。

这些结果验证训练链路、冻结策略、精度适配与恢复能力，不是模型质量或正式训练结果。
成功产物带 `cuda-probe-v2`，不会被正式流水线识别为已训练 checkpoint。

## 检查与正式入口

GPU 环境中 **90 项测试通过**，Ruff、依赖检查、Markdown 链接检查均通过。
CUDA preflight 的 `all_resources_present` 为真；正式六阶段均为未训练状态。

用户确认后使用：

```bash
cd /root/autodl-tmp/wlh_goya/MiniMind
bash scripts/start_cuda_training.sh
```

配置、环境说明及全部验证命令见 [`docs/cuda-training.md`](../docs/cuda-training.md)。
完整原始测试日志、数据、缓存、权重和 optimizer state 留在服务器 Git 忽略目录。
