#!/bin/bash
# Quick check: should Cursor agent wake? Exit 0 = yes, 1 = no.
# Usage: bash scripts/sota_agent_check.sh
set -euo pipefail
cd /root/lora-ours

if [[ -f SOTA_AGENT_WAKE.flag ]]; then
  echo "WAKE: flag present"
  cat SOTA_AGENT_WAKE.flag
  exit 0
fi

if [[ -f results/logs/lora_ours_sentinel_status.json ]]; then
  pending=$(python3 - <<'PY'
import json
from pathlib import Path
p = Path("results/logs/lora_ours_sentinel_status.json")
data = json.loads(p.read_text(encoding="utf-8"))
print(data.get("pending_action") or "")
PY
)
  if [[ -n "${pending}" && "${pending}" != "None" ]]; then
    echo "WAKE: sentinel pending_action=${pending}"
    exit 0
  fi
fi

if grep -q "Needs Agent Attention" results/logs/lora_ours_sentinel_status.md 2>/dev/null; then
  echo "WAKE: sentinel status lists agent attention items"
  exit 0
fi

echo "IDLE: no agent action needed"
exit 1
