#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.training import load_config, verify_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a final training checkpoint and enrich its JSON report.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    checkpoint_path = ROOT / config["training"]["checkpoint_path"]
    verification = verify_checkpoint(checkpoint_path)
    if not verification["all_finite"]:
        raise RuntimeError(f"checkpoint contains non-finite parameters: {checkpoint_path}")

    report_path = ROOT / "artifacts/logs" / f"{config['experiment']['name']}.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report.update(
        {
            "experiment": config["experiment"]["name"],
            "status": "complete",
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "artifact_verification": verification,
            "verified_at": datetime.now(UTC).isoformat(),
        }
    )
    temporary = report_path.with_suffix(".json.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
