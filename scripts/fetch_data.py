#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main"
FILES = {
    "pretrain": {
        "name": "pretrain_t2t_mini.jsonl",
        "size": 1_241_043_656,
        "sha256": "6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c",
    },
    "sft": {
        "name": "sft_t2t_mini.jsonl",
        "size": 1_739_201_170,
        "sha256": "abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(spec: dict) -> None:
    target = ROOT / "data/raw" / spec["name"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == spec["size"] and sha256(target) == spec["sha256"]:
        print(f"already verified: {target}")
        return
    temporary = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(f"{BASE_URL}/{spec['name']}", timeout=120) as response, temporary.open("wb") as output:
        downloaded = 0
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            print(f"{spec['name']}: downloaded {downloaded}/{spec['size']}", flush=True)
    if temporary.stat().st_size != spec["size"]:
        raise RuntimeError(f"size mismatch: {temporary.stat().st_size} != {spec['size']}")
    actual = sha256(temporary)
    if actual != spec["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch: {actual}")
    temporary.replace(target)
    print(f"verified {target}: {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="*", choices=[*FILES, "all"], default=["all"])
    args = parser.parse_args()
    stages = list(FILES) if "all" in args.stages else args.stages
    for stage in stages:
        fetch(FILES[stage])


if __name__ == "__main__":
    main()
