# CUDA 训练环境与准备流程

服务器项目目录：`/root/autodl-tmp/wlh_goya/MiniMind`。所有环境、下载缓存、数据和 checkpoint 都放在
这个数据盘目录内，避免占用仅剩约 4 GB 的系统盘。原 Mac 的训练结果保留为历史实验，CUDA 重训使用
独立的配置与 `-cuda` checkpoint 名称。

## 环境

2026-10-24 实测为两张 NVIDIA RTX PRO 6000 Blackwell Server Edition，各约 96 GB 显存。
两张卡都有既有业务占用；本次短程测试指定物理 GPU 1，保留原服务进程。

项目 `.venv` 使用 Python 3.12.4，启用 `system-site-packages`，只读复用服务器基础环境中的
PyTorch 2.8.0+cu128（包含 sm_120）。Transformers、Datasets 和其他冲突依赖安装在项目环境中，
不修改基础环境。该模式依赖基础环境继续存在；迁移时可新建完全独立环境，安装 `requirements-cuda.txt`。
该文件固定核心依赖版本，不代表全量传递依赖锁；完整环境快照保存在本机忽略的安装日志中。

```bash
cd /root/autodl-tmp/wlh_goya/MiniMind
source scripts/cuda_env.sh
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
```

`cuda_env.sh` 默认指定物理 GPU 1，并把 HF、Datasets、UV、临时文件缓存指向项目 `.cache/`。
如需其他卡，可先设置 `CUDA_VISIBLE_DEVICES`。CPU 线程默认 4，不会设置周期性监控。

## 数据

所有原始数据沿用 [`reports/data-manifest.md`](../reports/data-manifest.md) 中固定版本、大小与 SHA-256。
单图训练沿用 [`reports/vlm-data-filter.md`](../reports/vlm-data-filter.md) 的一致性筛选。
下载可在 shell 中启用平台提供的 `/etc/network_turbo`；仓库不保存代理凭据。

大语料可使用 `scripts/fetch_cuda_segmented.py`：按 CDN Content-Range 校验分段，再组装并核验整个
文件的 SHA-256。验证成功才发布到正式数据文件路径；成功后的分段缓存自动移除。QIVD 支持
`scripts/fetch_qivd.py --workers 16`，仍逐视频对照固定上游 LFS 哈希，并核验元数据和许可文件。

## 六阶段配置和测试

`scripts/prepare_cuda_configs.py` 从既有 MPS 配置生成 `*-cuda.yaml` 及独立的 `*-cuda-probe.yaml`。
正式配置保持原模型、batch、步数、种子和冻结策略，启用 BF16。
训练器保存 CPU 与 CUDA RNG 状态；RoPE 保持输入 dtype，图像/视频特征注入时匹配文本 embedding
的 dtype，同时保留梯度传递。探针启用确定性算法和 cuBLAS workspace 设置，正式配置保留默认内核策略。

```bash
.venv/bin/python scripts/run_cuda_probes.py
.venv/bin/python scripts/preflight_mps_pipeline.py --profile cuda
```

本次成功运行使用 `cuda-probe-v2` 产物名。如需重跑，先用
`scripts/prepare_cuda_configs.py --probe-tag <新的名称>` 生成新路径；探针会拒绝覆盖已有 checkpoint。

探针对六阶段分别执行 8 步真实数据训练，在 step 4 保存后退出、重载并完成 step 8，验证模型与
optimizer 有限性及 CUDA RNG。LLM 另跑连续 8 步对照，比较恢复后的模型参数。探针使用正式模型和
正式 micro-batch，但缩小验证集与总步数；视频探针直接编码真实帧，正式训练使用冻结视觉特征缓存。
这些测试只验证工程链路，不用于评估模型能力。探针文件名带 `-probe`，不会满足正式阶段的完成门禁。

完整 QIVD 下载后可预先生成缓存：

```bash
.venv/bin/python scripts/cache_video_features.py --config configs/video/alignment-cuda.yaml
.venv/bin/python scripts/run_cuda_cache_probe.py
```

CUDA 正式流水线会先使用 `verify_qivd_local.py` 离线重算文件哈希和固定聚合哈希，再复用已完成缓存，
不依赖训练中途再次访问 Hugging Face。

## 正式启动

只有用户确认后才执行：

```bash
.venv/bin/python scripts/prepare_formal_run.py --run-id <run-id>
bash scripts/start_cuda_training.sh --run-id <run-id>
```

入口执行既有流水线的 `--profile cuda`，按顺序训练六阶段，校验恢复文件并运行固定完整评估，最后
生成报告。它不创建自动守护或定时监控。长任务可在 tmux 中启动；任一阶段失败会停止并保留恢复点。
正式运行的记录与代码固定规则见[实验与证据说明](experiment-evidence.md)。
