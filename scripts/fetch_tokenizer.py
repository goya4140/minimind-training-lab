#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "6fc918beb68a0d8c40452338df6319fe168014ba"
FILES = {
    "tokenizer.json": "71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668",
    "tokenizer_config.json": "d7cd6a60c9f191c4f9ffec69b2eb289ad2f960366724f328b681ae1e46e4c110",
}


def main() -> None:
    target_dir = ROOT / "assets/tokenizer"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in FILES.items():
        url = f"https://raw.githubusercontent.com/jingyaogong/minimind/{COMMIT}/model/{name}"
        target = target_dir / name
        data = urllib.request.urlopen(url, timeout=60).read()
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual_hash}")
        target.write_bytes(data)
        print(f"verified {name}: {actual_hash}")


if __name__ == "__main__":
    main()
