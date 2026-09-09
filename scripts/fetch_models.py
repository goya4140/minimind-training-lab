#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "siglip2": {
        "repo": "jingyaogong/siglip2-base-p32-256-ve",
        "revision": "9465d1dc89db6bc6227c5b6b0e0ca9b940325d62",
        "directory": "siglip2-base-p32-256-ve",
        "files": {
            "config.json": (410, None),
            "preprocessor_config.json": (394, None),
            "model.safetensors": (
                189_129_296,
                "c1e9cc19ed6704b87353ee00b9ff5d6191886d741898339984364f789c62810d",
            ),
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid(path: Path, size: int, checksum: str | None) -> bool:
    return path.exists() and path.stat().st_size == size and (checksum is None or sha256(path) == checksum)


def download_file(repo: str, revision: str, name: str, target: Path, size: int, checksum: str | None) -> None:
    if valid(target, size, checksum):
        print(f"already verified: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
    if temporary.exists() and temporary.stat().st_size > size:
        temporary.unlink()
    attempts = 0
    while not temporary.exists() or temporary.stat().st_size < size:
        downloaded = temporary.stat().st_size if temporary.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={downloaded}-"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                mode = "ab" if downloaded and response.status == 206 else "wb"
                if mode == "wb":
                    downloaded = 0
                with temporary.open(mode) as output:
                    while chunk := response.read(8 * 1024 * 1024):
                        output.write(chunk)
                        downloaded += len(chunk)
                        print(f"{name}: downloaded {downloaded}/{size}", flush=True)
            attempts = 0
        except OSError:
            attempts += 1
            if attempts >= 10:
                raise
            time.sleep(min(2**attempts, 30))
    if not valid(temporary, size, checksum):
        actual = sha256(temporary) if temporary.exists() else "missing"
        raise RuntimeError(f"verification failed for {name}: size={temporary.stat().st_size}, sha256={actual}")
    temporary.replace(target)
    print(f"verified: {target}")


def fetch_model(spec: dict) -> None:
    directory = ROOT / "assets/models" / spec["directory"]
    for name, (size, checksum) in spec["files"].items():
        download_file(spec["repo"], spec["revision"], name, directory / name, size, checksum)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", choices=[*MODELS, "all"], default=["all"])
    args = parser.parse_args()
    selected = list(MODELS) if "all" in args.models else args.models
    for model in selected:
        fetch_model(MODELS[model])


if __name__ == "__main__":
    main()
