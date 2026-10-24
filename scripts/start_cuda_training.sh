#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/cuda_env.sh"
cd "$MINIMIND_ROOT"
exec .venv/bin/python scripts/run_mps_pipeline.py --profile cuda "$@"
