#!/bin/bash
# Quick check: should Cursor agent wake? Exit 0 = yes, 1 = no.
# Usage: bash scripts/sota_agent_check.sh
set -euo pipefail
cd /root/autodl-tmp/Lora-code

if [[ -f SOTA_AGENT_WAKE.flag ]]; then
  echo "WAKE: flag present"
  cat SOTA_AGENT_WAKE.flag
  exit 0
fi

if grep -q "AWAITING_AGENT:" SOTA_MONITOR.log 2>/dev/null; then
  last=$(grep "AWAITING_AGENT:" SOTA_MONITOR.log | tail -1)
  echo "WAKE: ${last}"
  exit 0
fi

if [[ -f SOTA_MONITOR_STATE.json ]]; then
  action=$(python3 -c "import json;print(json.load(open('SOTA_MONITOR_STATE.json')).get('pending_action') or '')" 2>/dev/null || true)
  if [[ -n "${action}" && "${action}" != "None" ]]; then
    echo "WAKE: pending_action=${action}"
    exit 0
  fi
fi

echo "IDLE: no agent action needed"
exit 1
