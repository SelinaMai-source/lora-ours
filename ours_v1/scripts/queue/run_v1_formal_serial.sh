#!/usr/bin/env bash
# Serial GPU queue: smoke -> formal for Standard, CITB (ID + ID++), ARPER; ToDCL if checkpoint gate passes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

QUEUE_LOG="${QUEUE_LOG:-results/logs/ours_v1_formal_serial_queue.log}"
STATE_JSON="${STATE_JSON:-results/logs/ours_v1_formal_serial_queue.json}"
POLL_SEC="${POLL_SEC:-120}"
mkdir -p results/logs results/manifests

log() { echo "[$(date -Iseconds)] $*" | tee -a "$QUEUE_LOG"; }

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

wait_tmux_done() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep "$POLL_SEC"
  done
}

run_job() {
  local job_id="$1" session="$2" cmd="$3" allow_fail="${4:-0}"
  log "JOB START ${job_id}"
  wait_gpu
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux kill-session -t "$session" 2>/dev/null || true
    sleep 2
  fi
  set +e
  eval "$cmd"
  local rc=$?
  set -e
  if [[ $rc -ne 0 && "$allow_fail" != "1" ]]; then
    log "JOB FAIL ${job_id} exit=${rc}; stopping queue"
    echo "{\"job\":\"${job_id}\",\"exit\":${rc},\"stopped_at\":\"$(date -Iseconds)\"}" >> "$STATE_JSON"
    exit "$rc"
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    wait_tmux_done "$session"
  fi
  wait_gpu
  log "JOB DONE ${job_id}"
}

log "=== ours-v1 formal serial queue start ==="
python3 ours_v1/scripts/freeze/freeze_v1_formal_matrix.py
python3 ours_v1/scripts/preflight/run_three_suite_preflight.py | tee -a "$QUEUE_LOG" || true

declare -a RESULTS=()

# 1 Standard smoke -> formal
run_job "standard-smoke" "lora-ours-standard-v1-smoke" \
  "bash ours_v1/scripts/launchers/run_standard_v1.sh" || exit 1
RESULTS+=("standard-smoke:launched")
run_job "standard-formal" "lora-ours-standard-v1-formal" \
  "FORMAL=1 TMUX_SESSION=lora-ours-standard-v1-formal bash ours_v1/scripts/launchers/run_standard_v1.sh" || exit 1
RESULTS+=("standard-formal:launched")

# 2 CITB InstrDialog smoke -> formal
run_job "citb-id-smoke" "lora-ours-citb-ours-v1-smoke" \
  "bash ours_v1/scripts/launchers/run_citb_v1.sh" || exit 1
RESULTS+=("citb-id-smoke:launched")
run_job "citb-id-formal" "lora-ours-citb-ours-v1-formal" \
  "FORMAL=1 bash ours_v1/scripts/launchers/run_citb_v1.sh" || exit 1
RESULTS+=("citb-id-formal:launched")

# 3 CITB InstrDialog++ smoke (formal blocked if split gate fails)
run_job "citb-idpp-smoke" "lora-ours-citb-pp-ours-v1-smoke" \
  "bash ours_v1/scripts/launchers/run_citb_pp_v1.sh" || exit 1
RESULTS+=("citb-idpp-smoke:launched")
set +e
FORMAL=1 bash ours_v1/scripts/launchers/run_citb_pp_v1.sh
idpp_rc=$?
set -e
if [[ $idpp_rc -eq 78 ]]; then
  log "citb-idpp-formal BLOCKED by split gate (exit 78)"
  RESULTS+=("citb-idpp-formal:blocked_split_gate")
elif [[ $idpp_rc -ne 0 ]]; then
  log "citb-idpp-formal failed exit=${idpp_rc}; stopping"
  exit "$idpp_rc"
else
  run_job "citb-idpp-formal-wait" "lora-ours-citb-pp-ours-v1-formal" "true" 1
  RESULTS+=("citb-idpp-formal:launched")
fi

# 4 ARPER smoke -> formal (no Fisher CPU)
run_job "arper-smoke" "lora-ours-arper-v1-ssrg-smoke" \
  "BOUNDED_SMOKE=1 bash ours_v1/scripts/launchers/run_arper_v1.sh" || exit 1
RESULTS+=("arper-smoke:launched")
run_job "arper-formal" "lora-ours-arper-v1-ssrg-formal" \
  "BOUNDED_SMOKE=0 RUN_ID=arper_woz3_sclstm_v89_ssrg_exemplar_overlay_v1_20260708_formal TMUX_SESSION=lora-ours-arper-v1-ssrg-formal bash ours_v1/scripts/launchers/run_arper_v1.sh" || exit 1
RESULTS+=("arper-formal:launched")

# 5 ToDCL — checkpoint gate
set +e
DRY_RUN=1 bash ours_v1/scripts/launchers/run_todcl_v1.sh 2>/dev/null
BOUNDED_SMOKE=1 bash ours_v1/scripts/launchers/run_todcl_v1.sh
todcl_rc=$?
set -e
if [[ $todcl_rc -eq 77 ]]; then
  log "todcl BLOCKED: no anchor checkpoint (exit 77)"
  RESULTS+=("todcl-smoke:blocked_no_ckpt")
  RESULTS+=("todcl-formal:blocked_no_ckpt")
else
  RESULTS+=("todcl-smoke:launched")
  run_job "todcl-formal" "lora-ours-todcl-v1-assess-overlay-formal" \
    "BOUNDED_SMOKE=0 TMUX_SESSION=lora-ours-todcl-v1-assess-overlay-formal bash ours_v1/scripts/launchers/run_todcl_v1.sh" || exit 1
  RESULTS+=("todcl-formal:launched")
fi

python3 ours_v1/scripts/publish/publish_formal_results.py || true

python3 - "$STATE_JSON" "${RESULTS[@]}" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
out, *rest = sys.argv[1:]
Path(out).write_text(json.dumps({
    "updated_at": datetime.now().isoformat(),
    "branch": "ours-v1",
    "results": rest,
}, indent=2) + "\n")
PY

log "=== queue complete ==="
