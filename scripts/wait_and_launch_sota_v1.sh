#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "[sota-v1-queue] waiting for any core/train.py on GPU..."
while pgrep -f "core/train.py" >/dev/null 2>&1; do sleep 30; done
echo "[sota-v1-queue] GPU free, launching sota-v1..."
bash scripts/launch_sota_v1.sh
