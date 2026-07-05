#!/usr/bin/env bash
# SOTA 24h campaign GPU queue: CITB -> ARPER v86 -> ToDCL adapter anchor.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
POLL_SEC="${POLL_SEC:-60}"
STATUS_FILE="${STATUS_FILE:-results/logs/gpu_queue_status_20260706.md}"
LOG_FILE="${LOG_FILE:-results/logs/lora_ours_gpu_queue_20260706.log}"
ARPER_SESSION="lora-ours-arper-v86-formal"
TODCL_SESSION="lora-ours-todcl-adapter-anchor"

mkdir -p results/logs
chmod +x scripts/run_arper_woz3_official_sclstm_formal_v86.sh \
  scripts/run_todcl_adapter_nlg_official_anchor.sh

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"; }

gpu_compute_apps() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
}

gpu_idle() {
  local apps
  apps="$(gpu_compute_apps | tr -d '[:space:]')"
  [[ -z "$apps" ]]
}

citb_running() {
  pgrep -f 'run_citb_|continual_learning/run_continual_instruct_tuning|lora_v10_citb' >/dev/null 2>&1
}

core_train_running() {
  pgrep -f 'core\.train' >/dev/null 2>&1
}

gpu_owner_summary() {
  if citb_running; then echo "CITB (continual_instruct_tuning / run_citb_*)"
  elif tmux has-session -t "$ARPER_SESSION" 2>/dev/null; then echo "ARPER v86 ($ARPER_SESSION)"
  elif tmux has-session -t "$TODCL_SESSION" 2>/dev/null; then echo "ToDCL adapter anchor ($TODCL_SESSION)"
  elif ! gpu_idle; then echo "GPU compute: $(gpu_compute_apps | head -1)"
  else echo "idle"
  fi
}

write_status() {
  local phase="$1" owner queue gpu
  owner="$(gpu_owner_summary)"
  queue="1. (wait) CITB jobs finish
2. ARPER v86: scripts/run_arper_woz3_official_sclstm_formal_v86.sh (tmux $ARPER_SESSION)
3. ToDCL: scripts/run_todcl_adapter_nlg_official_anchor.sh (tmux $TODCL_SESSION)"
  gpu="$(gpu_compute_apps)"
  [[ -z "${gpu// }" ]] && gpu="(none)"
  cat > "$STATUS_FILE" <<EOF
# GPU queue status — 2026-07-06

Updated: $(date -Iseconds)

## Current owner
${owner}

## Phase
${phase}

## Queue (dialogue worker order)
${queue}

## nvidia-smi compute apps
\`\`\`
${gpu}
\`\`\`

## Flags
- citb_running: $(citb_running && echo yes || echo no)
- core.train_running: $(core_train_running && echo yes || echo no)
- gpu_idle: $(gpu_idle && echo yes || echo no)
EOF
}

wait_for_gpu_and_citb() {
  write_status "waiting for CITB + idle GPU"
  while citb_running || core_train_running || ! gpu_idle; do
    log "GPU busy or CITB/core.train active; owner=$(gpu_owner_summary); waiting ${POLL_SEC}s"
    write_status "waiting for CITB + idle GPU"
    sleep "$POLL_SEC"
  done
}

launch_arper() {
  if tmux has-session -t "$ARPER_SESSION" 2>/dev/null; then
    log "ARPER session $ARPER_SESSION already exists; waiting for completion"
    return 0
  fi
  log "Launching ARPER v86"
  write_status "launching ARPER v86"
  bash scripts/run_arper_woz3_official_sclstm_formal_v86.sh
}

wait_session() {
  local sess="$1"
  while tmux has-session -t "$sess" 2>/dev/null; do
    log "Waiting for tmux session $sess"
    write_status "running: $sess"
    sleep "$POLL_SEC"
  done
}

launch_todcl() {
  if tmux has-session -t "$TODCL_SESSION" 2>/dev/null; then
    log "ToDCL session $TODCL_SESSION already exists"
    return 0
  fi
  wait_for_gpu_and_citb
  log "Launching ToDCL adapter anchor"
  write_status "launching ToDCL"
  bash scripts/run_todcl_adapter_nlg_official_anchor.sh
}

log "SOTA GPU queue started"
write_status "started"

wait_for_gpu_and_citb
launch_arper
wait_session "$ARPER_SESSION"
wait_for_gpu_and_citb
launch_todcl
wait_session "$TODCL_SESSION"
write_status "queue complete"
log "SOTA GPU queue finished"
