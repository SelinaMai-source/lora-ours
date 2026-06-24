#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MANIFEST="${MANIFEST:-results/tables/four_benchmark_single_seed_manifest.csv}"
STATUS_CSV="${STATUS_CSV:-results/tables/four_benchmark_single_seed_status.csv}"
LOG_DIR="${LOG_DIR:-results/logs/four_benchmark_single_seed}"
WAIT_FOR_PIDS="${WAIT_FOR_PIDS:-}"

mkdir -p "$LOG_DIR" "$(dirname "$STATUS_CSV")"

export WANDB_PROJECT="${WANDB_PROJECT:-lora-four-benchmark-single-seed}"
export WANDB_MODE="${WANDB_MODE:-online}"

if [[ -n "$WAIT_FOR_PIDS" ]]; then
  echo "[wait] Waiting for existing training PIDs: $WAIT_FOR_PIDS"
  for pid in $WAIT_FOR_PIDS; do
    while kill -0 "$pid" 2>/dev/null; do
      echo "[wait] PID $pid still running; sleep 60s ($(date))"
      sleep 60
    done
  done
  echo "[wait] All PIDs finished."
fi

python - <<'PY'
from pathlib import Path
import csv

manifest = Path("results/tables/four_benchmark_single_seed_manifest.csv")
status = Path("results/tables/four_benchmark_single_seed_status.csv")
rows = list(csv.DictReader(manifest.open("r", encoding="utf-8")))
fieldnames = list(rows[0].keys()) + ["exit_code", "log_file", "updated_at"]
for row in rows:
    row.setdefault("exit_code", "")
    row.setdefault("log_file", "")
    row.setdefault("updated_at", "")
status.parent.mkdir(parents=True, exist_ok=True)
with status.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY

update_status() {
  local run_name="$1"
  local status_value="$2"
  local exit_code="$3"
  local log_file="$4"
  python - "$run_name" "$status_value" "$exit_code" "$log_file" <<'PY'
import csv
import sys
from datetime import datetime
from pathlib import Path

run_name, status_value, exit_code, log_file = sys.argv[1:5]
path = Path("results/tables/four_benchmark_single_seed_status.csv")
rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
fieldnames = list(rows[0].keys()) if rows else []
for key in ["status", "exit_code", "log_file", "updated_at"]:
    if key not in fieldnames:
        fieldnames.append(key)
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
for row in rows:
    if row.get("run_name") == run_name:
        row["status"] = status_value
        row["exit_code"] = exit_code
        row["log_file"] = log_file
        row["updated_at"] = now
with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
manifest = Path("results/tables/four_benchmark_single_seed_manifest.csv")
if manifest.is_file():
    mrows = list(csv.DictReader(manifest.open("r", encoding="utf-8")))
    mfields = list(mrows[0].keys()) if mrows else []
    for mrow in mrows:
        if mrow.get("run_name") == run_name:
            mrow["status"] = status_value
            if status_value in {"completed", "failed", "early_stopped", "running"}:
                mrow["notes"] = f"queue {status_value} {now}"
            if "updated_at" in mfields:
                mrow["updated_at"] = now
    with manifest.open("w", encoding="utf-8", newline="") as mf:
        w = csv.DictWriter(mf, fieldnames=mfields)
        w.writeheader()
        w.writerows(mrows)
PY
}

mapfile -t RUN_ROWS < <(python - <<'PY'
import csv
from pathlib import Path

with Path("results/tables/four_benchmark_single_seed_manifest.csv").open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("status") in {"completed", "running", "blocked"}:
            continue
        print("\t".join([
            row["run_name"],
            row.get("config_path", ""),
            row.get("status", "queued"),
        ]))
PY
)

for row in "${RUN_ROWS[@]}"; do
  run_name="${row%%$'\t'*}"
  rest="${row#*$'\t'}"
  config_path="${rest%%$'\t'*}"
  prior_status="${rest#*$'\t'}"
  log_file="$LOG_DIR/${run_name}.log"

  if [[ -z "$config_path" ]] || [[ ! -f "$config_path" ]]; then
    echo "[blocked] $run_name: missing config ($config_path)"
    update_status "$run_name" "blocked" "" "$log_file"
    continue
  fi

  if [[ -f "results/runs/${run_name}/final_metrics.json" ]]; then
    echo "[skip] $run_name already has final_metrics.json"
    update_status "$run_name" "skipped_existing" "0" "$log_file"
    continue
  fi

  if [[ -d "results/runs/${run_name}" ]] && [[ -n "$(ls -A "results/runs/${run_name}/segment_"* 2>/dev/null)" ]]; then
    echo "[skip] $run_name has partial segment checkpoints; manual review needed"
    update_status "$run_name" "skipped_partial" "" "$log_file"
    continue
  fi

  echo "[run] $run_name <- $config_path"
  update_status "$run_name" "running" "" "$log_file"
  set +e
  python core/train.py --config "$config_path" 2>&1 | tee "$log_file"
  code="${PIPESTATUS[0]}"
  set -e
  if [[ "$code" -eq 0 ]]; then
    update_status "$run_name" "completed" "$code" "$log_file"
    echo "[done] $run_name"
  else
    update_status "$run_name" "failed" "$code" "$log_file"
    echo "[failed] $run_name; see $log_file"
  fi
done

echo "[all-done] status: $STATUS_CSV"
