#!/usr/bin/env bash
# Watchdog loop for sota-v1 — keeps tmux alive across train.py restarts
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONFIG="configs/paper/sota_campaign/sota_v1_instrdialogpp_s123.yaml"
LOG="results/logs/paper_instrdialogpp_sota_v1_ours_s123.log"

while true; do
  echo "[$(date '+%F %T')] sota-v1 train start" | tee -a "$LOG"
  set +e
  python3 core/train.py --config "$CONFIG" 2>&1 | tee -a "$LOG"
  ec=${PIPESTATUS[0]}
  set -e
  echo "[$(date '+%F %T')] sota-v1 exited code=$ec" | tee -a "$LOG"
  if [[ "$ec" -eq 1 ]]; then
    echo "[$(date '+%F %T')] sota-v1 intentional stop (early-stop); watchdog exiting" | tee -a "$LOG"
    break
  fi
  sleep 10
done
