#!/usr/bin/env bash
# Continue SOTA GPU queue after ARPER v86: wait for Standard v85, then ToDCL anchor.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
POLL_SEC="${POLL_SEC:-60}"
LOG_FILE="${LOG_FILE:-results/logs/lora_ours_gpu_queue_continue_20260706.log}"
TODCL_SESSION="lora-ours-todcl-adapter-anchor"
V85_SESSION="lora-ours-standard-v85-smoke"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"; }

gpu_idle() {
  local apps
  apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]' || true)"
  [[ -z "$apps" ]]
}

v85_running() {
  tmux has-session -t "$V85_SESSION" 2>/dev/null || pgrep -f 'run_olora_standard.*v85' >/dev/null 2>&1
}

log "continue queue: waiting for ${V85_SESSION} + idle GPU"
while v85_running || ! gpu_idle; do
  log "waiting v85=$(v85_running && echo yes || echo no) gpu_idle=$(gpu_idle && echo yes || echo no)"
  sleep "$POLL_SEC"
done
log "Standard v85 released GPU; launching ToDCL"
if ! tmux has-session -t "$TODCL_SESSION" 2>/dev/null; then
  bash scripts/run_todcl_adapter_nlg_official_anchor.sh || log "ToDCL launch exit=$?"
fi
while tmux has-session -t "$TODCL_SESSION" 2>/dev/null || ! gpu_idle; do
  log "waiting ToDCL session / GPU"
  sleep "$POLL_SEC"
done
log "continue queue complete"
