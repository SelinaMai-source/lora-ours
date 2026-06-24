#!/bin/bash
# Continuous SOTA experiment monitor for InstrDialog ours chase.
# Launch: tmux new-session -d -s sota_monitor 'bash /root/autodl-tmp/Lora-code/scripts/sota_monitor_loop.sh'
set -euo pipefail
cd /root/autodl-tmp/Lora-code
INTERVAL_SEC="${SOTA_MONITOR_INTERVAL_SEC:-150}"  # 2.5 min default

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] sota_monitor_loop started (interval=${INTERVAL_SEC}s, workflow=failure_analysis->implement->launch)" | tee -a SOTA_MONITOR.log

while true; do
  python3 scripts/sota_monitor_tick.py || echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] tick failed exit=$?" >> SOTA_MONITOR.log
  sleep "${INTERVAL_SEC}"
done
