#!/usr/bin/env python3
"""Download pinned corpora via parallel validated CDN ranges, then verify full SHA-256."""

from __future__ import annotations

import argparse
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from fetch_data import FILES, ROOT, sha256

CHUNK = 8 * 1024 * 1024


def fetch(spec, base_dir=None):
    target = (base_dir or ROOT / "data/raw") / spec["name"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == spec["size"] and sha256(target) == spec["sha256"]:
        print("verified existing", spec["name"], flush=True)
        return
    source = (
        spec.get("url") or f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['name']}"
    )
    with urllib.request.urlopen(urllib.request.Request(source, method="HEAD"), timeout=60) as response:
        destination = response.url
    directory = ROOT / ".cache/download-segments" / spec["name"]
    directory.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    count = (spec["size"] + CHUNK - 1) // CHUNK

    def segment(index):
        start = index * CHUNK
        end = min(spec["size"], start + CHUNK) - 1
        expected = end - start + 1
        path = directory / f"{index:05d}.chunk"
        if path.is_file() and path.stat().st_size == expected:
            return path
        for attempt in range(5):
            temporary = path.with_suffix(".partial")
            try:
                request = urllib.request.Request(destination, headers={"Range": f"bytes={start}-{end}"})
                with opener.open(request, timeout=60) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != f"bytes {start}-{end}/{spec['size']}"
                    ):
                        raise RuntimeError("CDN range response does not match requested bytes")
                    with temporary.open("wb") as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                if temporary.stat().st_size != expected:
                    raise RuntimeError("incomplete CDN range")
                os.replace(temporary, path)
                if index % 20 == 0:
                    print(spec["name"], "range", index, "/", count, flush=True)
                return path
            except (OSError, RuntimeError):
                if attempt == 4:
                    raise
                time.sleep(2**attempt)

    with ThreadPoolExecutor(max_workers=16) as executor:
        paths = list(executor.map(segment, range(count)))
    temporary = target.with_suffix(target.suffix + ".assembled")
    with temporary.open("wb") as output:
        for path in paths:
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    output.write(chunk)
    if temporary.stat().st_size != spec["size"] or sha256(temporary) != spec["sha256"]:
        raise RuntimeError(f"assembled file failed pinned SHA-256: {spec['name']}")
    os.replace(temporary, target)
    for path in paths:
        path.unlink()
    print("VERIFIED", spec["name"], spec["sha256"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision", action="store_true")
    args = parser.parse_args()
    if args.vision:
        from fetch_models import MODELS, download_file

        model = MODELS["siglip2"]
        directory = ROOT / "assets/models" / model["directory"]
        for name, (size, digest) in model["files"].items():
            if size < CHUNK:
                download_file(model["repo"], model["revision"], name, directory / name, size, digest)
            else:
                fetch(
                    {
                        "name": name,
                        "size": size,
                        "sha256": digest,
                        "url": f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{name}",
                    },
                    directory,
                )
        return
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(fetch, FILES.values()):
            pass


if __name__ == "__main__":
    main()
