#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "pretrain": {
        "repo": "jingyaogong/minimind_dataset",
        "revision": "312afb4f76391145c6902f765bb51691c09a12f5",
        "name": "pretrain_t2t_mini.jsonl",
        "size": 1_241_043_656,
        "sha256": "6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c",
    },
    "sft": {
        "repo": "jingyaogong/minimind_dataset",
        "revision": "312afb4f76391145c6902f765bb51691c09a12f5",
        "name": "sft_t2t_mini.jsonl",
        "size": 1_739_201_170,
        "sha256": "abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518",
    },
    "vlm-pretrain": {
        "repo": "jingyaogong/minimind-v_dataset",
        "revision": "1e279a8b665cb10383451a6af6fd62b9f35bdd79",
        "name": "pretrain_i2t.parquet",
        "size": 4_326_415_097,
        "sha256": "65761f37d1947d54a1d85457ff70938275e4ef58ba5cedcd02463a3a247c93fd",
    },
    "vlm-sft": {
        "repo": "jingyaogong/minimind-v_dataset",
        "revision": "1e279a8b665cb10383451a6af6fd62b9f35bdd79",
        "name": "sft_i2t.parquet",
        "size": 4_934_887_104,
        "sha256": "712f4026cd0e21b369feddca7334b1e465cb8182b5f298006f3f4f877f926643",
    },
    "omni-t2a-mini": {
        "repo": "jingyaogong/minimind-o_dataset",
        "revision": "d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de",
        "name": "sft_t2a_mini.parquet",
        "size": 1_558_442_729,
        "sha256": "dfe44b8b263ecd0579627160cf258b363b4c18457ae03221691e2e1a85e60ab8",
    },
    "omni-a2a-mini": {
        "repo": "jingyaogong/minimind-o_dataset",
        "revision": "d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de",
        "name": "sft_a2a_mini.parquet",
        "size": 881_313_734,
        "sha256": "fba0159e424ee106c9e5a732fe607875b3780d0c9f8b6806038879acd279782b",
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
    url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['name']}"
    if temporary.exists() and temporary.stat().st_size > spec["size"]:
        temporary.unlink()
    attempts = 0
    while not temporary.exists() or temporary.stat().st_size < spec["size"]:
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
                        print(f"{spec['name']}: downloaded {downloaded}/{spec['size']}", flush=True)
            attempts = 0
        except OSError:
            attempts += 1
            if attempts >= 10:
                raise
            time.sleep(min(2**attempts, 30))
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
