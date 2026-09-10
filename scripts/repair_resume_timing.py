#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.training import load_config, repair_resume_timing, verify_resume_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atomically move a verified interruption from active to suspended checkpoint time."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-step", required=True, type=int)
    parser.add_argument("--excluded-seconds", required=True, type=float)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    resume_path = (ROOT / config["training"]["checkpoint_path"]).with_suffix(".resume.pt")
    verify_resume_checkpoint(resume_path, expected_config=config)
    repair = repair_resume_timing(
        resume_path,
        expected_step=args.expected_step,
        excluded_seconds=args.excluded_seconds,
        reason=args.reason,
    )
    verification = verify_resume_checkpoint(resume_path, expected_config=config)
    print(
        json.dumps(
            {
                "checkpoint": str(resume_path.relative_to(ROOT)),
                "repair": repair,
                "verification": verification,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
