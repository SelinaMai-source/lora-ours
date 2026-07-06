#!/usr/bin/env bash
# Baseline reproduction serial GPU queue — best-method-only scope (2026-07-06).
# Per suite: CITB Replay(50), Standard O-LoRA v57 (done), ToDCL ADAPTER, ARPER v87.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
POLL_SEC="${POLL_SEC:-60}"
STATUS_FILE="${STATUS_FILE:-results/logs/baseline_reproduction_queue_status_20260706.md}"
LOG_FILE="${LOG_FILE:-results/logs/baseline_reproduction_queue_20260706.log}"
TRACKER="${TRACKER:-results/tables/baseline_reproduction_tracker_20260706.md}"
SCOPE_NOTE="best-method-only: CITB Replay(50) → ToDCL ADAPTER → ARPER v87 (domain/exemplar500)"

mkdir -p results/logs results/tables
chmod +x scripts/run_citb_instrdialog_all_baselines_repro.sh \
  scripts/run_citb_instrdialog_replay50_official_base_repro.sh \
  scripts/run_todcl_adapter_nlg_official_anchor.sh \
  scripts/run_arper_woz3_paper_aligned_formal_v87.sh

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
  pgrep -f '[p]ython.*continual_learning/run_continual_instruct_tuning\.py' >/dev/null 2>&1
}

core_train_running() {
  pgrep -f 'core\.train' >/dev/null 2>&1
}

gpu_owner_summary() {
  if citb_running; then echo "CITB continual_instruct_tuning"
  elif pgrep -f 'train\.py.*--CL ADAPTER' >/dev/null 2>&1; then echo "ToDCL ADAPTER anchor"
  elif pgrep -f 'run_woz3\.py' >/dev/null 2>&1; then echo "ARPER SCLSTM formal"
  elif pgrep -f 'run_uie_lora\.py' >/dev/null 2>&1; then echo "Standard/O-LoRA runtime"
  elif ! gpu_idle; then echo "GPU: $(gpu_compute_apps | head -1)"
  else echo "idle"
  fi
}

write_status() {
  local phase="$1" current="$2" next="$3"
  cat > "$STATUS_FILE" <<EOF
# Baseline reproduction queue — 2026-07-06 (best-method-only)

Updated: $(date -Iseconds)

## Scope
${SCOPE_NOTE}

**Skipped (out of scope):** CITB L2/EWC/AGEM/Replay(10), Standard LFPT5/ProgPrompts/SeqLoRA/IncLoRA/Replay, LB-CL (paper_only; O-LoRA v57 completed).

## Phase
${phase}

## Current GPU owner
${current}

## Next queued job
${next}

## Full priority queue
1. CITB Replay(50) formal 19-task — \`lora-ours-citb-replay50-formal\` (if not already done)
2. ToDCL ADAPTER NLG 37-domain anchor — \`lora-ours-todcl-adapter-anchor\`
3. ARPER paper-aligned repro (domain-wise, exemplar 500) — \`lora-ours-arper-v87-formal\`

## Tracker
${TRACKER}

## Flags
- citb_running: $(citb_running && echo yes || echo no)
- core.train_running: $(core_train_running && echo yes || echo no)
- gpu_idle: $(gpu_idle && echo yes || echo no)
EOF
}

wait_for_gpu() {
  local reason="$1"
  write_status "waiting: ${reason}" "$(gpu_owner_summary)" "${2:-}"
  while ! gpu_idle; do
    log "GPU busy (${reason}); owner=$(gpu_owner_summary); wait ${POLL_SEC}s"
    write_status "waiting: ${reason}" "$(gpu_owner_summary)" "${2:-}"
    sleep "$POLL_SEC"
  done
}

launch_tmux_job() {
  local session="$1"
  local log_path="$2"
  shift 2
  local cmd="$*"
  if tmux has-session -t "$session" 2>/dev/null; then
    log "session ${session} already exists"
    return 0
  fi
  mkdir -p "$(dirname "$log_path")"
  tmux new-session -d -s "$session" "bash -lc $(printf '%q' "${cmd}") > $(printf '%q' "${log_path}") 2>&1; echo EXIT_CODE=\$? >> $(printf '%q' "${log_path}")"
  log "launched ${session} log=${log_path}"
}

wait_tmux_and_gpu() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null || ! gpu_idle; do
    log "waiting completion: ${session} gpu_owner=$(gpu_owner_summary)"
    write_status "running: ${session}" "$(gpu_owner_summary)" ""
    sleep "$POLL_SEC"
  done
  log "completed: ${session}"
}

replay50_done() {
  [[ -f results/logs/citb_replay50_formal_20260706.log ]] && \
    grep -q 'EXIT_CODE=0' results/logs/citb_replay50_formal_20260706.log 2>/dev/null
}

# --- job definitions ---
launch_citb_replay50_formal() {
  launch_tmux_job "lora-ours-citb-replay50-formal" \
    "results/logs/citb_replay50_formal_20260706.log" \
    "cd ${REPO_ROOT} && DRY_RUN=0 METHOD=replay50 RUN_NAME=citb_instrdialog_order1_seed1_official_script_500_50_50_tie_fixed_replay50_formal_v56 WANDB_RUN_GROUP=citb_instrdialog_order1_official_replay50_base_repro bash scripts/run_citb_instrdialog_all_baselines_repro.sh"
}

launch_todcl_adapter() {
  bash scripts/run_todcl_adapter_nlg_official_anchor.sh || {
    local ec=$?
    if [[ "$ec" -eq 75 ]]; then log "ToDCL queued (GPU busy at launch)"; return 0; fi
    return "$ec"
  }
}

launch_arper_paper_aligned() {
  bash scripts/run_arper_woz3_paper_aligned_formal_v87.sh || {
    local ec=$?
    if [[ "$ec" -eq 75 ]]; then log "ARPER v87 queued (GPU busy at launch)"; return 0; fi
    return "$ec"
  }
}

log "Baseline reproduction queue started (${SCOPE_NOTE})"
write_status "started" "$(gpu_owner_summary)" "CITB Replay(50) formal or ToDCL ADAPTER"

# Priority 1: CITB Replay(50) formal — idempotent (do not relaunch healthy run)
if replay50_done; then
  log "CITB Replay(50) formal already completed; skipping launch"
elif tmux has-session -t "lora-ours-citb-replay50-formal" 2>/dev/null || citb_running; then
  log "CITB Replay(50) formal already running; waiting without relaunch"
  wait_tmux_and_gpu "lora-ours-citb-replay50-formal"
else
  wait_for_gpu "before CITB Replay(50) formal" "CITB Replay(50) formal"
  launch_citb_replay50_formal
  wait_tmux_and_gpu "lora-ours-citb-replay50-formal"
fi

# Priority 2: ToDCL ADAPTER anchor
wait_for_gpu "before ToDCL ADAPTER" "ToDCL ADAPTER anchor"
launch_todcl_adapter
wait_tmux_and_gpu "lora-ours-todcl-adapter-anchor"

# Priority 3: ARPER paper-aligned (domain-wise exemplar 500)
wait_for_gpu "before ARPER v87" "ARPER v87 paper-aligned"
launch_arper_paper_aligned
wait_tmux_and_gpu "lora-ours-arper-v87-formal"

write_status "queue complete (best-method-only)" "idle" "(none)"
log "Baseline reproduction queue finished (${SCOPE_NOTE})"
