#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STATUS_CSV="${STATUS_CSV:-results/tables/published_setting_single_seed_status.csv}"
LOG_DIR="${LOG_DIR:-results/logs/published_setting_rerun}"
mkdir -p "$LOG_DIR" "$(dirname "$STATUS_CSV")"

export WANDB_PROJECT="${WANDB_PROJECT:-lora-published-setting}"
export WANDB_MODE="${WANDB_MODE:-online}"

mapfile -t FAILED_ROWS < <(python - <<'PY'
import csv
from pathlib import Path

path = Path("results/tables/published_setting_single_seed_status.csv")
if not path.exists():
    raise SystemExit(0)
with path.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row.get("status") == "failed":
            print("\t".join([row["run_name"], row["config_path"]]))
PY
)

update_status() {
  local run_name="$1"
  local status_value="$2"
  local exit_code="$3"
  local log_file="$4"
  python - "$run_name" "$status_value" "$exit_code" "$log_file" <<'PY'
import csv
import sys
from pathlib import Path

run_name, status_value, exit_code, log_file = sys.argv[1:5]
path = Path("results/tables/published_setting_single_seed_status.csv")
rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
fieldnames = list(rows[0].keys()) if rows else []
for key in ["status", "exit_code", "log_file"]:
    if key not in fieldnames:
        fieldnames.append(key)
for row in rows:
    if row.get("run_name") == run_name:
        row["status"] = status_value
        row["exit_code"] = exit_code
        row["log_file"] = log_file
with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
}

if [[ "${#FAILED_ROWS[@]}" -eq 0 ]]; then
  echo "[rerun] no failed published-setting runs"
  exit 0
fi

for row in "${FAILED_ROWS[@]}"; do
  run_name="${row%%$'\t'*}"
  config_path="${row#*$'\t'}"
  log_file="$LOG_DIR/${run_name}.log"
  echo "[rerun] $run_name <- $config_path"
  update_status "$run_name" "rerunning" "" "$log_file"
  set +e
  python core/train.py --config "$config_path" 2>&1 | tee "$log_file"
  code="${PIPESTATUS[0]}"
  set -e
  if [[ "$code" -eq 0 ]]; then
    update_status "$run_name" "completed" "$code" "$log_file"
    echo "[rerun-done] $run_name"
  else
    update_status "$run_name" "failed" "$code" "$log_file"
    echo "[rerun-failed] $run_name; see $log_file"
  fi
done
