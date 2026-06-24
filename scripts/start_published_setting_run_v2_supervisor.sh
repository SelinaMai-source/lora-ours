#!/usr/bin/env bash
# Start tmux sessions for published_setting_run_v2 queue + supervisor.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QUEUE_SESSION="lora_published_setting_run_v2"
SUPERVISOR_SESSION="lora_published_setting_run_v2_supervisor"
QUEUE_SCRIPT="scripts/run_published_setting_run_v2_queue.sh"
SUPERVISOR_SCRIPT="scripts/supervisor_published_setting_run_v2.py"

# Regenerate manifest if missing
if [[ ! -f results/tables/published_setting_run_v2_manifest.csv ]]; then
  python scripts/gen_published_setting_run_v2_manifest.py
fi

start_session() {
  local session="$1"
  local cmd="$2"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[start] tmux session already exists: $session"
  else
    tmux new-session -d -s "$session" "bash -lc 'cd $ROOT && $cmd'"
    echo "[start] created tmux session: $session"
  fi
}

export WANDB_PROJECT="${WANDB_PROJECT:-lora-published-setting-run_v2}"
export WANDB_MODE="${WANDB_MODE:-online}"

start_session "$QUEUE_SESSION" \
  "export WANDB_PROJECT=$WANDB_PROJECT WANDB_MODE=$WANDB_MODE && bash $QUEUE_SCRIPT"

start_session "$SUPERVISOR_SESSION" \
  "python $SUPERVISOR_SCRIPT --loop --interval-min 90 --interval-max 120"

echo ""
echo "Sessions:"
tmux ls 2>/dev/null | grep -E "${QUEUE_SESSION}|${SUPERVISOR_SESSION}" || true
echo ""
echo "Manifest: results/tables/published_setting_run_v2_manifest.csv"
echo "Supervisor state: results/logs/published_setting_run_v2_supervisor_state.json"
echo "Status report: results/logs/published_setting_run_v2_status_report.md"
