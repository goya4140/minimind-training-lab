#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^]]*]\(([^)]+)\)")


def main() -> None:
    failures = []
    for markdown in sorted(ROOT.rglob("*.md")):
        if any(part.startswith(".") or part in {"artifacts", "data"} for part in markdown.relative_to(ROOT).parts):
            continue
        content = markdown.read_text(encoding="utf-8")
        for line_number, line in enumerate(content.splitlines(), start=1):
            for match in LINK.finditer(line):
                raw_target = match.group(1).strip().strip("<>")
                target = raw_target.split("#", 1)[0]
                if not target or "://" in target or target.startswith(("mailto:", "codex:")):
                    continue
                resolved = (markdown.parent / target).resolve()
                if not resolved.exists():
                    failures.append(f"{markdown.relative_to(ROOT)}:{line_number}: missing {raw_target}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        sys.exit(1)
    print("all local Markdown links resolve")


if __name__ == "__main__":
    main()
