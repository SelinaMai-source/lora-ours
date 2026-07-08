#!/usr/bin/env bash
# Serial GPU queue for ours-v1-20260708 overlay iteration.
# smoke → gate → formal protocol; one training job at a time.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

QUEUE_LOG="${QUEUE_LOG:-results/logs/ours_v1_20260708_iterate_queue.log}"
STATE_JSON="${STATE_JSON:-results/logs/ours_v1_20260708_iterate_queue.json}"
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

# v1 queue: wait for ToDCL anchor, then serial overlay smokes
JOBS=(
  "todcl-anchor-wait|lora-ours-todcl-adapter-anchor|echo waiting for anchor"
  "standard-v1-smoke|lora-ours-standard-v1-smoke|cd ${REPO_ROOT} && bash scripts/run_olora_standard_order1_official_base_ours_overlay_v1_20260708.sh"
  "citb-v1-smoke|lora-ours-citb-ours-v1-smoke|cd ${REPO_ROOT} && bash scripts/run_citb_ours_v1_20260708_smoke.sh"
  "arper-v1-ssrg-smoke|lora-ours-arper-v1-ssrg-smoke|cd ${REPO_ROOT} && BOUNDED_SMOKE=1 bash scripts/run_arper_woz3_sclstm_plus_ssrg_overlay_v1_20260708.sh"
  "todcl-v1-assess-smoke|lora-ours-todcl-v1-assess-overlay|cd ${REPO_ROOT} && BOUNDED_SMOKE=1 bash scripts/run_todcl_adapter_nlg_ours_assess_overlay_v1_20260708.sh"
)

completed=()
skipped=()

for entry in "${JOBS[@]}"; do
  IFS='|' read -r job_id session cmd <<< "$entry"

  if [[ "$job_id" == "todcl-anchor-wait" ]]; then
    if tmux has-session -t "lora-ours-todcl-adapter-anchor" 2>/dev/null; then
      log "ToDCL anchor running; waiting for GPU release before v1 overlays"
      while ! gpu_empty; do sleep "$POLL_SEC"; done
      skipped+=("$job_id:anchor_was_running")
    else
      skipped+=("$job_id:no_anchor_session")
    fi
    continue
  fi

  if ! gpu_empty; then
    log "GPU not empty; skip launch ${job_id} (queue for manual retry)"
    skipped+=("$job_id:gpu_busy")
    continue
  fi

  wait_gpu
  if launch_tmux "$session" "$cmd"; then
    completed+=("$job_id:launched")
    log "waiting for ${session} to release GPU..."
    while ! gpu_empty; do sleep "$POLL_SEC"; done
    log "GPU idle after ${job_id}"
  else
    skipped+=("$job_id:launch_failed")
  fi
done

python3 - "$STATE_JSON" "${completed[@]}" "${skipped[@]}" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
out, *rest = sys.argv[1:]
completed = [x for x in rest if ":launched" in x or ":existing" in x]
skipped = [x for x in rest if x and ":" in x and ":launched" not in x and ":existing" not in x]
Path(out).write_text(json.dumps({
    "updated_at": datetime.now().isoformat(),
    "branch": "ours-v1-20260708",
    "completed": completed,
    "skipped": skipped,
    "policy": "serial GPU; smoke before formal",
    "failure_path": "docs/experiments/{suite}_failure_v1_20260708.md -> ours-v2-YYYYMMDD",
}, indent=2) + "\n")
PY

log "v1 iterate queue done; state=${STATE_JSON}"
