#!/usr/bin/env bash
# Serial full-run queue for lora-published-setting-run_v2 (seed=123, published_setting configs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MANIFEST="${MANIFEST:-results/tables/published_setting_run_v2_manifest.csv}"
STATUS_CSV="${STATUS_CSV:-results/tables/published_setting_run_v2_status.csv}"
LOG_DIR="${LOG_DIR:-results/logs/published_setting_run_v2}"
QUEUE_LOG="${QUEUE_LOG:-results/logs/published_setting_run_v2_queue.log}"

mkdir -p "$LOG_DIR" "$(dirname "$STATUS_CSV")"

export WANDB_PROJECT="${WANDB_PROJECT:-lora-published-setting-run_v2}"
export WANDB_MODE="${WANDB_MODE:-online}"

exec > >(tee -a "$QUEUE_LOG") 2>&1
echo "[queue-start] $(date -Iseconds) WANDB_PROJECT=$WANDB_PROJECT"

export MANIFEST STATUS_CSV
python - <<'PY'
import csv
import os
from pathlib import Path

manifest = Path(os.environ["MANIFEST"])
status = Path(os.environ["STATUS_CSV"])
rows = list(csv.DictReader(manifest.open("r", encoding="utf-8")))
fieldnames = list(rows[0].keys()) + ["exit_code", "log_file", "updated_at"]
for row in rows:
    row.setdefault("exit_code", "")
    row.setdefault("log_file", "")
    row.setdefault("updated_at", "")
status.parent.mkdir(parents=True, exist_ok=True)
with status.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
PY

update_status() {
  local run_name="$1"
  local status_value="$2"
  local exit_code="$3"
  local log_file="$4"
  export MANIFEST STATUS_CSV
  python - "$run_name" "$status_value" "$exit_code" "$log_file" <<'PY'
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

run_name, status_value, exit_code, log_file = sys.argv[1:5]
manifest_path = Path(os.environ["MANIFEST"])
status_path = Path(os.environ["STATUS_CSV"])
rows = list(csv.DictReader(status_path.open("r", encoding="utf-8")))
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
with status_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
if manifest_path.is_file():
    mrows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    mfields = list(mrows[0].keys()) if mrows else []
    for mrow in mrows:
        if mrow.get("run_name") == run_name:
            mrow["status"] = status_value
            if status_value in {"completed", "failed", "early_stopped", "running"}:
                mrow["notes"] = f"queue {status_value} {now}"
            if "updated_at" in mfields:
                mrow["updated_at"] = now
    with manifest_path.open("w", encoding="utf-8", newline="") as mf:
        w = csv.DictWriter(mf, fieldnames=mfields)
        w.writeheader()
        w.writerows(mrows)
PY
}

mapfile -t RUN_ROWS < <(python - <<'PY'
import csv
import os
from pathlib import Path

skip = {"completed", "running", "blocked", "skipped_existing", "skipped_partial", "early_stopped", "failed"}
with Path(os.environ["MANIFEST"]).open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("status") in skip:
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

  if [[ -d "results/runs/${run_name}" ]] && compgen -G "results/runs/${run_name}/segment_*" > /dev/null; then
    echo "[skip] $run_name has partial segment checkpoints; manual review needed"
    update_status "$run_name" "skipped_partial" "" "$log_file"
    continue
  fi

  while pgrep -f "core/train.py" > /dev/null; do
    echo "[wait] another core/train.py running; sleep 30s ($(date))"
    sleep 30
  done

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
    echo "[failed] $run_name exit=$code; see $log_file"
  fi
done

echo "[all-done] status: $STATUS_CSV ($(date -Iseconds))"
