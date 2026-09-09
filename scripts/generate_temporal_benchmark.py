#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.data.temporal_benchmark import generate_temporal_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the controlled temporal Video-QA evaluation set.")
    parser.add_argument("--output", default="data/eval/temporal-video")
    parser.add_argument("--samples-per-family", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    report = generate_temporal_benchmark(
        ROOT / args.output, samples_per_family=args.samples_per_family, seed=args.seed
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
