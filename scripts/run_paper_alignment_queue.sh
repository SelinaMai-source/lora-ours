#!/usr/bin/env bash
# Serial GPU queue for paper-alignment retries (after current healthy runs complete).
# Order: ToDCL ADAPTER retry → CITB v2 → ARPER v88
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

POLL_SEC="${POLL_SEC:-60}"
LOG="${LOG:-results/logs/paper_alignment_queue_20260707.log}"
STATUS="${STATUS:-results/logs/paper_alignment_queue_status_20260707.md}"

mkdir -p results/logs
chmod +x scripts/run_citb_replay50_paper_aligned_v2.sh \
  scripts/run_arper_woz3_paper_aligned_formal_v88.sh \
  scripts/run_todcl_adapter_nlg_official_anchor.sh

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

gpu_idle() {
  [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]
}

wait_gpu() {
  local reason="$1"
  while ! gpu_idle; do
    log "waiting GPU (${reason})"
    sleep "$POLL_SEC"
  done
}

write_status() {
  local phase="$1"
  cat > "$STATUS" <<EOF
# Paper alignment queue — 20260707

Updated: $(date -Iseconds)
Phase: ${phase}

Priority:
1. ToDCL ADAPTER (local GPT-2 fixed)
2. CITB Replay(50) v2 (official Stage-1 path)
3. ARPER v88 (config.cfg exact: exemplar 250, batch 128)

Standard O-LoRA v57: PASS — no relaunch.

Do not kill: lora-ours-arper-v87-formal while running.
EOF
}

arper_v87_done() {
  ! pgrep -f 'run_woz3.*formal_v87' >/dev/null 2>&1 && \
    ! tmux has-session -t lora-ours-arper-v87-formal 2>/dev/null
}

log "Paper alignment queue started"
write_status "waiting for ARPER v87"

while ! arper_v87_done; do
  log "ARPER v87 still running — waiting"
  write_status "waiting: ARPER v87"
  sleep "$POLL_SEC"
done

log "ARPER v87 complete — starting ToDCL"
write_status "ToDCL ADAPTER retry"
wait_gpu "before ToDCL"
bash scripts/run_todcl_adapter_nlg_official_anchor.sh || {
  ec=$?; [[ "$ec" -eq 75 ]] && log "ToDCL queued (GPU busy)" || exit "$ec"
}
while tmux has-session -t lora-ours-todcl-adapter-anchor 2>/dev/null || ! gpu_idle; do
  sleep "$POLL_SEC"
done

log "Starting CITB v2"
write_status "CITB v2 formal"
wait_gpu "before CITB v2"
if ALLOW_FALLBACK_STAGE1=1 DRY_RUN=1 bash scripts/run_citb_replay50_paper_aligned_v2.sh; then
  log "CITB v2 dry-run OK (fallback Stage-1)"
  tmux new-session -d -s lora-ours-citb-replay50-v2 \
    "bash -lc 'cd ${REPO_ROOT} && ALLOW_FALLBACK_STAGE1=1 DRY_RUN=0 bash scripts/run_citb_replay50_paper_aligned_v2.sh' > results/logs/citb_replay50_paper_aligned_v2_formal.log 2>&1; echo EXIT_CODE=\$? >> results/logs/citb_replay50_paper_aligned_v2_formal.log"
else
  log "CITB v2 blocked — need official checkpoint-14000 (see root cause doc)"
fi
while tmux has-session -t lora-ours-citb-replay50-v2 2>/dev/null || ! gpu_idle; do
  sleep "$POLL_SEC"
done

log "Starting ARPER v88"
write_status "ARPER v88"
wait_gpu "before ARPER v88"
bash scripts/run_arper_woz3_paper_aligned_formal_v88.sh || {
  ec=$?; [[ "$ec" -eq 75 ]] && log "ARPER v88 queued" || exit "$ec"
}
while tmux has-session -t lora-ours-arper-v88-formal 2>/dev/null || ! gpu_idle; do
  sleep "$POLL_SEC"
done

write_status "complete"
log "Paper alignment queue finished"
