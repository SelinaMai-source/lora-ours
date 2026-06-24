#!/usr/bin/env bash
# Gap queue: 8×5 matrix missing cells only (LFPT5 external + Seq-GLUE core/train).
# W&B project/group for restarted gap runs: lora-run_v10.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MANIFEST="${MANIFEST:-results/tables/published_setting_run_v2_gap_manifest.csv}"
STATUS_CSV="${STATUS_CSV:-results/tables/published_setting_run_v2_gap_status.csv}"
LOG_DIR="${LOG_DIR:-results/logs/published_setting_run_v2_gap}"
QUEUE_LOG="${QUEUE_LOG:-results/logs/published_setting_run_v2_gap_queue.log}"

mkdir -p "$LOG_DIR" "$(dirname "$STATUS_CSV")"

export WANDB_PROJECT="${WANDB_PROJECT:-lora-run_v10}"
export WANDB_GROUP="${WANDB_GROUP:-lora-run_v10}"
export WANDB_MODE="${WANDB_MODE:-online}"

exec > >(tee -a "$QUEUE_LOG") 2>&1
echo "[gap-queue-start] $(date -Iseconds) WANDB_PROJECT=$WANDB_PROJECT"

if [[ ! -f "$MANIFEST" ]]; then
  echo "[error] gap manifest missing: $MANIFEST"
  echo "Run: python scripts/gen_full_matrix_gap_audit_s123.py && python scripts/gen_published_setting_run_v2_gap_manifest.py"
  exit 1
fi

export MANIFEST STATUS_CSV
python - <<'PY'
import csv
import os
from pathlib import Path

manifest = Path(os.environ["MANIFEST"])
status = Path(os.environ["STATUS_CSV"])
rows = list(csv.DictReader(manifest.open("r", encoding="utf-8")))
if not rows:
    print("[info] gap manifest empty — nothing to run")
fieldnames = list(rows[0].keys()) if rows else []
for row in rows:
    row.setdefault("exit_code", "")
    row.setdefault("updated_at", "")
status.parent.mkdir(parents=True, exist_ok=True)
with status.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
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
status_path = Path(os.environ["STATUS_CSV"])
manifest_path = Path(os.environ["MANIFEST"])
rows = list(csv.DictReader(status_path.open("r", encoding="utf-8")))
fieldnames = list(rows[0].keys()) if rows else []
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
for row in rows:
    if row.get("run_name") == run_name:
        row["status"] = status_value
        row["exit_code"] = exit_code
        row["log_file"] = log_file
        row["updated_at"] = now
with status_path.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
if manifest_path.is_file():
    mrows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    mfields = list(mrows[0].keys()) if mrows else []
    for mrow in mrows:
        if mrow.get("run_name") == run_name:
            mrow["status"] = status_value
            if status_value in {"completed", "failed", "running", "blocked"}:
                mrow["notes"] = f"gap queue {status_value} {now}"
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

skip = {"completed", "running", "skipped_existing", "skipped_partial"}
only_methods = os.environ.get("GAP_ONLY_METHODS", "").strip()
only_methods_set = {m.strip() for m in only_methods.split(",") if m.strip()} if only_methods else None
only_runners = os.environ.get("GAP_ONLY_RUNNERS", "").strip()
only_runners_set = {r.strip() for r in only_runners.split(",") if r.strip()} if only_runners else None
with Path(os.environ["MANIFEST"]).open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("status") in skip:
            continue
        if only_methods_set and row.get("method") not in only_methods_set:
            continue
        if only_runners_set and row.get("runner") not in only_runners_set:
            continue
        print("\t".join([
            row["run_name"],
            row.get("config_path", ""),
            row.get("runner", "core_train"),
            row.get("status", "queued"),
        ]))
PY
)

for row in "${RUN_ROWS[@]}"; do
  IFS=$'\t' read -r run_name config_path runner manifest_status <<< "$row"
  log_file="$LOG_DIR/${run_name}.log"

  if [[ -f "results/runs/${run_name}/final_metrics.json" ]]; then
    echo "[skip] $run_name already has final_metrics.json"
    update_status "$run_name" "completed" "0" "$log_file"
    continue
  fi

  if [[ "$manifest_status" == "blocked" ]]; then
    echo "[blocked] $run_name (manifest status=blocked)"
    update_status "$run_name" "blocked" "" "$log_file"
    continue
  fi

  if [[ -z "$config_path" ]] || [[ ! -f "$config_path" ]]; then
    echo "[blocked] $run_name: missing config ($config_path)"
    update_status "$run_name" "blocked" "" "$log_file"
    continue
  fi

  if [[ -d "results/runs/${run_name}" ]] && compgen -G "results/runs/${run_name}/segment_*" > /dev/null; then
    echo "[skip] $run_name has partial segment checkpoints"
    update_status "$run_name" "skipped_partial" "" "$log_file"
    continue
  fi

  while pgrep -f "core/train.py" > /dev/null; do
    echo "[wait] another core/train.py running; sleep 30s ($(date))"
    sleep 30
  done

  echo "[run] $run_name <- $config_path (runner=$runner)"
  update_status "$run_name" "running" "" "$log_file"
  set +e
  if [[ "$runner" == "lfpt5_external" ]]; then
    python scripts/run_lfpt5_published_setting.py --config "$config_path" 2>&1 | tee "$log_file"
    code="${PIPESTATUS[0]}"
    if [[ "$code" -eq 0 ]]; then
      update_status "$run_name" "completed" "$code" "$log_file"
    else
      update_status "$run_name" "blocked" "$code" "$log_file"
      echo "[blocked] LFPT5 $run_name exit=$code; see $log_file"
    fi
  else
    python core/train.py --config "$config_path" 2>&1 | tee "$log_file"
    code="${PIPESTATUS[0]}"
    if [[ "$code" -eq 0 ]]; then
      update_status "$run_name" "completed" "$code" "$log_file"
      echo "[done] $run_name"
    else
      update_status "$run_name" "failed" "$code" "$log_file"
      echo "[failed] $run_name exit=$code; see $log_file"
    fi
  fi
  set -e
done

echo "[gap-all-done] status: $STATUS_CSV ($(date -Iseconds))"
