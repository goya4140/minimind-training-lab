# Tokenizer Asset

正式实验使用上游 MiniMind commit `6fc918beb68a0d8c40452338df6319fe168014ba` 的 BPE 6400 Tokenizer。

下载并校验：

```bash
uv run python scripts/fetch_tokenizer.py
```

文件不会重复提交到本仓库；脚本固定 URL 和 SHA-256，确保实验拿到相同内容。

