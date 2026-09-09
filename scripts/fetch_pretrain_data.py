#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/pretrain_t2t_mini.jsonl"
EXPECTED_SIZE = 1_241_043_656
EXPECTED_SHA256 = "6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    target = ROOT / "data/raw/pretrain_t2t_mini.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == EXPECTED_SIZE and sha256(target) == EXPECTED_SHA256:
        print(f"already verified: {target}")
        return
    temporary = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("wb") as output:
        downloaded = 0
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            print(f"downloaded {downloaded}/{EXPECTED_SIZE}", flush=True)
    if temporary.stat().st_size != EXPECTED_SIZE:
        raise RuntimeError(f"size mismatch: {temporary.stat().st_size} != {EXPECTED_SIZE}")
    actual = sha256(temporary)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"SHA-256 mismatch: {actual}")
    temporary.replace(target)
    print(f"verified {target}: {actual}")


if __name__ == "__main__":
    main()

