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
    "sensevoice": {
        "repo": "jingyaogong/SenseVoiceSmall",
        "revision": "964f07d8f0e48309a5f37770ac9b5ed2cdca6b91",
        "directory": "SenseVoiceSmall",
        "files": {
            "am.mvn": (11_203, None),
            "chn_jpn_yue_eng_ko_spectok.bpe.model": (
                377_341,
                "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8",
            ),
            "config.yaml": (1_855, None),
            "configuration.json": (396, None),
            "model.pt": (468_291_478, "218811976815c1673e1b852dc383d78987b229268f78e7ebd0a1fe67229c83dd"),
        },
    },
    "mimi": {
        "repo": "jingyaogong/mimi",
        "revision": "b4e362bbfbba9444b162de486da40af6639e0b98",
        "directory": "mimi",
        "files": {
            "config.json": (1_117, None),
            "preprocessor_config.json": (234, None),
            "model.safetensors": (
                192_346_842,
                "7542ee039d3025d5089cf227d21df64b6b8eff08fcd376a11a1fbd178dd9d3f5",
            ),
        },
    },
    "campplus": {
        "repo": "jingyaogong/campplus",
        "revision": "77bb7d92872bdbd66586cdcc6a7148dfd111683c",
        "directory": "campplus",
        "files": {
            "config.yaml": (537, None),
            "configuration.json": (581, None),
            "campplus_cn_common.pt": (
                14_173_135,
                "55ffb1a55d04bac4a9b7ed80497cb731909985c9b55d120fce04b851a697886c",
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
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
            break
        except OSError:
            if attempt == 5:
                raise
            time.sleep(2**attempt)
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
