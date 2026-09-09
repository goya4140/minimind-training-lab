#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.training import acquire_run_lock


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Hold a training lock for a run started before locking was enabled.")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if not process_exists(args.pid):
        raise ProcessLookupError(f"target training process does not exist: {args.pid}")
    checkpoint = ROOT / args.checkpoint
    lock = acquire_run_lock(checkpoint.with_suffix(".lock"))
    print(f"guarding pid {args.pid} with {checkpoint.with_suffix('.lock')}", flush=True)
    try:
        while process_exists(args.pid):
            time.sleep(args.poll_seconds)
    finally:
        lock.close()
    print(f"training pid {args.pid} ended; lock released", flush=True)


if __name__ == "__main__":
    main()
