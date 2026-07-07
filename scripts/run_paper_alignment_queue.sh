#!/usr/bin/env bash
# Serial GPU queue for paper-alignment retries.
# Order: wait ToDCL (if running) → ARPER v88 → optional CITB Stage-1/v2
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

POLL_SEC="${POLL_SEC:-60}"
LOG="${LOG:-results/logs/paper_alignment_queue_20260707.log}"
STATUS="${STATUS:-results/logs/paper_alignment_queue_status_20260707.md}"

mkdir -p results/logs
chmod +x scripts/run_citb_replay50_paper_aligned_v2.sh \
  scripts/run_citb_stage1_seed50_train.sh \
  scripts/run_arper_woz3_paper_aligned_formal_v88.sh \
  scripts/run_todcl_adapter_nlg_official_anchor.sh \
  scripts/run_strict_paper_repro_iteration.sh

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
1. ToDCL ADAPTER (if not already running)
2. ARPER v88 (config.cfg exact: exemplar 250, batch 128)
3. CITB Stage-1 seed50 + Replay v2 (optional strict parity; AR already ±1 on v56)

Standard O-LoRA v57: PASS — no relaunch.
CITB v56 AR matrix: PASS ±1 — Stage-1 optional.

Orchestrator: scripts/run_strict_paper_repro_iteration.sh
EOF
}

log "Paper alignment queue started"
write_status "monitor in-flight jobs"

# Respect healthy ToDCL
while pgrep -f 'todcl.*train.py.*ADAPTER' >/dev/null 2>&1; do
  log "ToDCL ADAPTER running — waiting"
  write_status "running: ToDCL ADAPTER"
  sleep "$POLL_SEC"
done

while tmux has-session -t lora-ours-todcl-adapter-anchor 2>/dev/null; do
  log "ToDCL tmux active — waiting"
  sleep "$POLL_SEC"
done

log "Starting ARPER v88"
write_status "ARPER v88"
wait_gpu "before ARPER v88"
bash scripts/run_arper_woz3_paper_aligned_formal_v88.sh || {
  ec=$?; [[ "$ec" -eq 75 ]] && log "ARPER v88 queued (GPU busy)" || exit "$ec"
}
while tmux has-session -t lora-ours-arper-v88-formal 2>/dev/null || ! gpu_idle; do
  sleep "$POLL_SEC"
done

if [[ "${FORCE_CITB_STAGE1:-0}" == "1" ]]; then
  OFFICIAL_CKPT="/root/autodl-tmp/Lora-code/external_baselines/citb_official/output/initial_multitask_model/base_epoch15_lr1e-05_seed50/checkpoint-14000"
  if [[ ! -d "$OFFICIAL_CKPT" ]]; then
    log "Starting CITB Stage-1 seed50"
    write_status "CITB Stage-1 seed50"
    wait_gpu "before CITB Stage-1"
    tmux new-session -d -s lora-ours-citb-stage1-seed50 \
      "bash -lc 'cd ${REPO_ROOT} && bash scripts/run_citb_stage1_seed50_train.sh' > results/logs/citb_stage1_seed50_train_formal.log 2>&1; echo EXIT=\$? >> results/logs/citb_stage1_seed50_train_formal.log"
    while tmux has-session -t lora-ours-citb-stage1-seed50 2>/dev/null || ! gpu_idle; do
      sleep "$POLL_SEC"
    done
  fi

  log "Starting CITB v2"
  write_status "CITB v2 formal"
  wait_gpu "before CITB v2"
  if ALLOW_FALLBACK_STAGE1=0 DRY_RUN=0 bash scripts/run_citb_replay50_paper_aligned_v2.sh; then
    tmux new-session -d -s lora-ours-citb-replay50-v2 \
      "bash -lc 'cd ${REPO_ROOT} && ALLOW_FALLBACK_STAGE1=0 DRY_RUN=0 bash scripts/run_citb_replay50_paper_aligned_v2.sh' > results/logs/citb_replay50_paper_aligned_v2_formal.log 2>&1; echo EXIT_CODE=\$? >> results/logs/citb_replay50_paper_aligned_v2_formal.log"
  else
    log "CITB v2 blocked — need official checkpoint-14000"
  fi
  while tmux has-session -t lora-ours-citb-replay50-v2 2>/dev/null || ! gpu_idle; do
    sleep "$POLL_SEC"
  done
else
  log "Skipping CITB Stage-1/v2 (AR PASS on v56 matrix; set FORCE_CITB_STAGE1=1 to enable)"
fi

write_status "complete"
log "Paper alignment queue finished"
