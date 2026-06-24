#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MANIFEST="${MANIFEST:-results/tables/lora_run_v10_manifest.csv}"
STATUS_CSV="${STATUS_CSV:-results/tables/lora_run_v10_status.csv}"
LOG_DIR="${LOG_DIR:-results/logs/lora_run_v10}"
QUEUE_LOG="${QUEUE_LOG:-results/logs/lora_run_v10_queue.log}"
mkdir -p "$LOG_DIR" "$(dirname "$STATUS_CSV")"
export WANDB_PROJECT="${WANDB_PROJECT:-lora-run_v10}"
export WANDB_MODE="${WANDB_MODE:-online}"
exec > >(tee -a "$QUEUE_LOG") 2>&1
echo "[lora-run_v10-start] $(date -Iseconds) WANDB_PROJECT=$WANDB_PROJECT"
update_status() {
  local run_name="$1" status_value="$2" exit_code="$3" log_file="$4"
  export MANIFEST STATUS_CSV
  python - "$run_name" "$status_value" "$exit_code" "$log_file" <<'PY2'
import csv, os, sys
from datetime import datetime
from pathlib import Path
run_name, status_value, exit_code, log_file = sys.argv[1:5]
for env_key in ["STATUS_CSV", "MANIFEST"]:
    path = Path(os.environ[env_key])
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    fields = list(rows[0].keys()) if rows else []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in rows:
        if row.get("run_name") == run_name:
            row["status"] = status_value
            row["exit_code"] = exit_code
            row["log_file"] = log_file
            row["updated_at"] = now
            row["notes"] = f"lora-run_v10 {status_value} {now}"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
PY2
}
mapfile -t RUN_ROWS < <(python - <<'PY2'
import csv
from pathlib import Path
skip = {"completed", "running", "skipped_existing", "blocked"}
with Path("results/tables/lora_run_v10_manifest.csv").open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("status") not in skip:
            print("\t".join([row["run_name"], row["config_path"], row.get("runner", "core_train")]))
PY2
)
for row in "${RUN_ROWS[@]}"; do
  IFS=$'\t' read -r run_name config_path runner <<< "$row"
  log_file="$LOG_DIR/${run_name}.log"
  if [[ -f "results/runs/${run_name}/final_metrics.json" ]]; then
    echo "[skip] $run_name already has final_metrics.json"
    update_status "$run_name" "completed" "0" "$log_file"
    continue
  fi
  while python - <<'PY2'
import subprocess
import sys
out = subprocess.check_output(["pgrep", "-af", "core/train.py|citb_bridge/run_continual.py|run_lfpt5_published_setting.py"], text=True)
active = [
    line for line in out.splitlines()
    if "run_lora_run_v10_queue.sh" not in line and "pgrep -af" not in line
]
sys.exit(0 if active else 1)
PY2
  do
    echo "[wait] another training process is running; sleep 60s ($(date))"
    sleep 60
  done
  echo "[run] $run_name <- $config_path (runner=$runner)"
  update_status "$run_name" "running" "" "$log_file"
  set +e
  if [[ "$runner" == "lfpt5_external" ]]; then
    python scripts/run_lfpt5_published_setting.py --config "$config_path" 2>&1 | tee "$log_file"
    code="${PIPESTATUS[0]}"
  else
    python core/train.py --config "$config_path" 2>&1 | tee "$log_file"
    code="${PIPESTATUS[0]}"
  fi
  set -e
  if [[ "$code" -eq 0 ]]; then
    update_status "$run_name" "completed" "$code" "$log_file"
    echo "[done] $run_name"
  else
    update_status "$run_name" "failed" "$code" "$log_file"
    echo "[failed] $run_name exit=$code"
  fi
done
echo "[lora-run_v10-all-done] $(date -Iseconds)"
