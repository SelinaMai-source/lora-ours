#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f experiments/sota_campaign/PAUSE_SOTA_V3 ]]; then
  echo "PAUSE_SOTA_V3 set; refusing to launch sota-v3"
  exit 0
fi
LOG="results/logs/paper_instrdialogpp_sota_v3_ours_s123.log"
SESSION="sota-v3"
WATCHDOG="$ROOT/scripts/sota_v3_watchdog.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists"
  exit 1
fi

mkdir -p results/logs
chmod +x "$WATCHDOG"
tmux new-session -d -s "$SESSION" "bash '$WATCHDOG'"
echo "Started $SESSION (watchdog) -> $LOG"
