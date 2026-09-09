# 07 · 完整复现 Runbook

## 0. 硬件预期

本次正式运行使用 Apple M4 Pro、48 GiB unified memory 和 PyTorch MPS。其他 Apple Silicon 可以
运行，但耗时和可用 micro-batch 可能不同；CUDA 机器需要把配置中的 `device` 改为 `cuda`，并重新记录
环境与吞吐，不能直接冒充同一次实验。

## 1. 安装

```bash
git clone https://github.com/goya4140/minimind-training-lab.git
cd minimind-training-lab
uv sync --extra multimodal --extra video --extra dev
uv run pytest
```

## 2. 下载固定资源

```bash
uv run python scripts/fetch_tokenizer.py
uv run python scripts/fetch_data.py all
uv run python scripts/fetch_models.py all
uv run python scripts/fetch_qivd.py
```

下载脚本固定 repository revision；大文件支持断点恢复。QIVD 会逐文件比对固定 revision 的 LFS
字节数与 SHA-256，全部通过后才生成逐文件与聚合 manifest。数据全部位于 Git 忽略目录，不应提交或
重新分发。自动六阶段流水线在 Video-Omni 开始前会再次执行这项完整性检查。

## 3. Preflight

```bash
uv run python scripts/preflight_mps_pipeline.py
```

只有 `mps_available=true`、所有资源存在，且第一未完成阶段为 `ready`/`active` 时才继续。后续阶段显示
`pending` 是正常的，因为它们必须等待上游 checkpoint。

## 4. 自动运行六阶段

```bash
caffeinate -i uv run python scripts/run_mps_pipeline.py
```

顺序为：

```text
LLM Pretrain → LLM SFT → LLM eval
                         ├→ VLM alignment → VLM SFT → VLM eval
                         └→ Video alignment → Video SFT → Video eval
                                                        → final report
```

脚本会跳过已完成 checkpoint、等待正在运行的阶段、恢复 `.resume.pt`，并在每个最终 checkpoint 后
做 finite/hash 验证。

## 5. 手工运行单阶段

```bash
uv run python scripts/train_pretrain.py --config configs/llm/pretrain-mps.yaml --resume
uv run python scripts/train_sft.py --config configs/llm/sft-mps.yaml --resume
uv run python scripts/train_vlm.py --config configs/vlm/alignment-mps.yaml --resume
uv run python scripts/train_vlm.py --config configs/vlm/sft-mps.yaml --resume
uv run python scripts/train_video_omni.py --config configs/video/alignment-mps.yaml --resume
uv run python scripts/train_video_omni.py --config configs/video/sft-mps.yaml --resume
```

不要同时启动两个写入同一 checkpoint 的进程。

## 6. 最终报告

```bash
uv run python scripts/build_final_report.py --check
uv run python scripts/build_final_report.py
```

第一条命令缺少任何 checkpoint、日志、评估或数据 manifest 都会失败。第二条只在证据完整时生成
`reports/final-results.md` 和本地 artifact hash manifest。权重、数据和完整 JSON 不上传 GitHub；只提交
经过提炼的 handbook 报告。
