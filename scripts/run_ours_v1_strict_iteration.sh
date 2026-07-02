#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v1_strict.yaml}"
RUN_NAME="$(python - "$CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print((cfg.get("output") or {}).get("run_name") or cfg.get("experiment_name") or "ours_v1_strict_iteration")
PY
)"
LOG_DIR="$ROOT/results/logs"
mkdir -p "$LOG_DIR"

echo "Starting ours v1 strict iteration: $CONFIG"
python -m core.train --config "$CONFIG" 2>&1 | tee "$LOG_DIR/${RUN_NAME}.log"
