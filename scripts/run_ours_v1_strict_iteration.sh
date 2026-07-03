#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v1_strict.yaml}"
if [[ "$CONFIG" == *"standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v62_formal.yaml" && "${ALLOW_V62_FORMAL:-0}" != "1" ]]; then
  echo "Refusing v62 formal launch while eval-exit diagnostic is the active gate. Set ALLOW_V62_FORMAL=1 only after diagnostic review." >&2
  exit 64
fi
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
EXIT_STATUS_PATH="$LOG_DIR/${RUN_NAME}.exit.json"
LAUNCHER_PID="$$"
STARTED_AT="$(date -Is)"

write_exit_json() {
  local event="$1"
  local launcher_status="$2"
  local train_status="${3:-}"
  local tee_status="${4:-}"
  python - "$RUN_NAME" "$CONFIG" "$EXIT_STATUS_PATH" "$event" "$launcher_status" "$train_status" "$tee_status" "$LAUNCHER_PID" "$STARTED_AT" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

run_name, config, output_path, event, launcher_status, train_status, tee_status, launcher_pid, started_at = sys.argv[1:]

def _maybe_int(value):
    if value == "":
        return None
    return int(value)

payload = {
    "started_at": started_at,
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "event": event,
    "run_name": run_name,
    "config": config,
    "launcher_pid": int(launcher_pid),
    "launcher_ppid": os.getppid(),
    "launcher_exit_status": int(launcher_status),
    "train_exit_status": _maybe_int(train_status),
    "tee_exit_status": _maybe_int(tee_status),
}
Path(output_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

trap 'status=$?; if [[ ! -f "$EXIT_STATUS_PATH" ]]; then write_exit_json launcher_exit_trap "$status"; fi' EXIT
trap 'write_exit_json signal:SIGHUP 129; exit 129' HUP
trap 'write_exit_json signal:SIGINT 130; exit 130' INT
trap 'write_exit_json signal:SIGTERM 143; exit 143' TERM

echo "Starting ours v1 strict iteration: $CONFIG"
set +e
python -m core.train --config "$CONFIG" 2>&1 | tee "$LOG_DIR/${RUN_NAME}.log"
PIPE_STATUSES=("${PIPESTATUS[@]}")
TRAIN_STATUS="${PIPE_STATUSES[0]}"
TEE_STATUS="${PIPE_STATUSES[1]:-0}"
set -e
write_exit_json pipeline_exit "$TRAIN_STATUS" "$TRAIN_STATUS" "$TEE_STATUS"
exit "$TRAIN_STATUS"
