#!/usr/bin/env bash
# Launch sota-v1 single-seed experiment in tmux session sota-v1 (with auto-restart)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="results/logs/paper_instrdialogpp_sota_v1_ours_s123.log"
SESSION="sota-v1"
WATCHDOG="$ROOT/scripts/sota_v1_watchdog.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists"
  exit 1
fi

mkdir -p results/logs
chmod +x "$WATCHDOG"
tmux new-session -d -s "$SESSION" "bash '$WATCHDOG'"
echo "Started $SESSION (watchdog) -> $LOG"
