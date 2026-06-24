#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f experiments/sota_campaign/PAUSE_SOTA_V3 ]]; then
  echo "[$(date '+%F %T')] PAUSE_SOTA_V3 set; watchdog exiting"
  exit 0
fi
CONFIG="configs/paper/sota_campaign/sota_v3_instrdialogpp_s123.yaml"
LOG="results/logs/paper_instrdialogpp_sota_v3_ours_s123.log"

while true; do
  if [[ -f experiments/sota_campaign/PAUSE_SOTA_V3 ]]; then
    echo "[$(date '+%F %T')] sota-v3 paused (PAUSE_SOTA_V3); watchdog exiting" | tee -a "$LOG"
    break
  fi
  echo "[$(date '+%F %T')] sota-v3 train start" | tee -a "$LOG"
  set +e
  python3 core/train.py --config "$CONFIG" 2>&1 | tee -a "$LOG"
  ec=${PIPESTATUS[0]}
  set -e
  echo "[$(date '+%F %T')] sota-v3 exited code=$ec" | tee -a "$LOG"
  if [[ "$ec" -eq 1 ]]; then
    echo "[$(date '+%F %T')] sota-v3 intentional stop (early-stop); watchdog exiting" | tee -a "$LOG"
    break
  fi
  sleep 10
done
