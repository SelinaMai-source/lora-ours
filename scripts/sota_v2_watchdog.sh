#!/usr/bin/env bash
# Watchdog for sota-v2 — restart on crash only, stop on trajectory early-stop (exit 1)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONFIG="configs/paper/sota_campaign/sota_v2_instrdialogpp_s123.yaml"
LOG="results/logs/paper_instrdialogpp_sota_v2_ours_s123.log"

while true; do
  echo "[$(date '+%F %T')] sota-v2 train start" | tee -a "$LOG"
  set +e
  python3 core/train.py --config "$CONFIG" 2>&1 | tee -a "$LOG"
  ec=${PIPESTATUS[0]}
  set -e
  echo "[$(date '+%F %T')] sota-v2 exited code=$ec" | tee -a "$LOG"
  if [[ "$ec" -eq 1 ]]; then
    echo "[$(date '+%F %T')] sota-v2 intentional stop (early-stop); watchdog exiting" | tee -a "$LOG"
    break
  fi
  sleep 10
done
