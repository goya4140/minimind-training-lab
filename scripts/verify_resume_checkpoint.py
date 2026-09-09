#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.training import load_config, verify_resume_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a local resume checkpoint without modifying it.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Also require the resume step to equal the configured final step.",
    )
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    resume_path = (ROOT / config["training"]["checkpoint_path"]).with_suffix(".resume.pt")
    verification = verify_resume_checkpoint(
        resume_path,
        expected_config=config,
        require_complete=args.require_complete,
    )
    print(
        json.dumps(
            {"checkpoint": str(resume_path.relative_to(ROOT)), **verification},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
