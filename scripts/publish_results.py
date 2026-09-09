#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHANGES = {
    "README.md",
    "data/manifests/qivd.json",
    "reports/final-results.md",
    "reports/release-manifest.json",
}


def run(command: list[str], capture: bool = False) -> str:
    result = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def changed_paths() -> set[str]:
    output = run(["git", "status", "--porcelain=v1"], capture=True)
    return {line[3:] for line in output.splitlines() if line}


def main() -> None:
    manifest_path = ROOT / "reports/release-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("run scripts/build_final_report.py before publishing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = manifest["assets"]
    for asset in assets:
        if not (ROOT / asset["path"]).is_file():
            raise FileNotFoundError(asset["path"])

    unexpected = changed_paths() - EXPECTED_CHANGES
    if unexpected:
        raise RuntimeError(f"refusing to publish with unrelated worktree changes: {sorted(unexpected)}")
    run(["git", "add", *sorted(EXPECTED_CHANGES)])
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False).returncode != 0
    if staged:
        run(["git", "commit", "-m", "docs: publish final trained model results"])
    run(["git", "push", "origin", "main"])

    tag = manifest["tag"]
    release_exists = subprocess.run(
        ["gh", "release", "view", tag],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    upload_arguments = [f"{asset['path']}#{asset['name']}" for asset in assets]
    if release_exists:
        run(["gh", "release", "edit", tag, "--notes-file", "reports/final-results.md"])
        run(["gh", "release", "upload", tag, *upload_arguments, "--clobber"])
    else:
        run(
            [
                "gh",
                "release",
                "create",
                tag,
                *upload_arguments,
                "--title",
                "MiniMind Training Lab — trained MPS checkpoints",
                "--notes-file",
                "reports/final-results.md",
                "--target",
                "main",
            ]
        )
    print(json.dumps({"published": True, "tag": tag, "assets": len(assets)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"publish failed: {error}", file=sys.stderr)
        sys.exit(1)
