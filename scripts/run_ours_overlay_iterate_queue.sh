#!/usr/bin/env bash
# Serial GPU job queue for ours overlay iteration (one training job at a time).
# Respects early-stop rules from sota-24h-campaign plan; documents each launch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

QUEUE_LOG="${QUEUE_LOG:-results/logs/ours_overlay_iterate_queue_20260706.log}"
STATE_JSON="${STATE_JSON:-results/logs/ours_overlay_iterate_queue_20260706.json}"
POLL_SEC="${POLL_SEC:-60}"

mkdir -p results/logs

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$QUEUE_LOG"
}

gpu_empty() {
  local out
  out="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
  [[ -z "${out// /}" ]]
}

wait_gpu() {
  while ! gpu_empty; do
    log "GPU busy; waiting ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
}

launch_tmux() {
  local session="$1"
  local cmd="$2"
  if tmux has-session -t "$session" 2>/dev/null; then
    log "skip existing session ${session}"
    return 0
  fi
  tmux new-session -d -s "$session" "bash -lc $(printf '%q' "$cmd")"
  log "launched ${session}"
}

# Ordered queue: baselines first, then ours overlays
JOBS=(
  "citb-replay50-smoke|lora-ours-citb-replay50-smoke|cd ${REPO_ROOT} && DRY_RUN=0 SMOKE=1 RUN_NAME=citb_replay50_official_base_smoke_20260706 bash scripts/run_citb_instrdialog_replay50_official_base_repro.sh"
  "standard-v85-smoke|lora-ours-standard-v85-smoke|cd ${REPO_ROOT} && DRY_RUN=0 bash scripts/run_olora_standard_order1_official_base_ours_overlay_v85.sh"
  "arper-v86-formal|lora-ours-arper-v86-formal|cd ${REPO_ROOT} && FORCE=0 bash scripts/run_arper_woz3_official_sclstm_formal_v86.sh"
  "todcl-adapter-anchor|lora-ours-todcl-adapter-anchor|cd ${REPO_ROOT} && bash scripts/run_todcl_adapter_nlg_official_anchor.sh"
  "citb-ours-v85-smoke|lora-ours-citb-ours-v85-smoke|cd ${REPO_ROOT} && python -m core.train --config configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v85_smoke_strict.yaml"
)

completed=()
failed=()

for entry in "${JOBS[@]}"; do
  IFS='|' read -r job_id session cmd <<< "$entry"
  if tmux has-session -t "$session" 2>/dev/null; then
    log "job ${job_id}: session ${session} exists; waiting for GPU release"
    while ! gpu_empty; do sleep "$POLL_SEC"; done
    completed+=("$job_id:existing")
    continue
  fi
  wait_gpu
  if ! launch_tmux "$session" "$cmd"; then
    failed+=("$job_id")
    log "FAILED launch ${job_id}"
    continue
  fi
  completed+=("$job_id:launched")
  # Block until GPU frees (job finished or failed) before next serial job
  log "waiting for ${session} to release GPU..."
  while ! gpu_empty; do
    sleep "$POLL_SEC"
  done
  log "GPU idle after ${job_id}"
done

python3 - "$STATE_JSON" "${completed[@]}" "${failed[@]}" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
out, *rest = sys.argv[1:]
completed = [x for x in rest if ":launched" in x or ":existing" in x]
failed = [x for x in rest if x and ":" not in x]
Path(out).write_text(json.dumps({
    "updated_at": datetime.now().isoformat(),
    "completed": completed,
    "failed": failed,
    "policy": "serial GPU mutual exclusion",
}, indent=2) + "\n")
PY

log "queue complete; state=${STATE_JSON}"
