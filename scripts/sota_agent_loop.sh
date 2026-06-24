#!/bin/bash
# Cursor agent wake loop for SOTA closed-loop (see /root/.cursor/skills-cursor/loop/SKILL.md).
#
# Launch:
#   tmux new-session -d -s sota_agent_loop 'bash /root/autodl-tmp/Lora-code/scripts/sota_agent_loop.sh'
#
# Or in Cursor chat: /loop 5m Check SOTA_MONITOR.log ...
set -euo pipefail
cd /root/autodl-tmp/Lora-code

POLL_SEC="${SOTA_AGENT_POLL_SEC:-30}"
HEARTBEAT_SEC="${SOTA_AGENT_LOOP_SEC:-300}"
FLAG="SOTA_AGENT_WAKE.flag"
WAKE_LOG="SOTA_AGENT_WAKE.log"

PROMPT='Check SOTA_MONITOR.log and SOTA_MONITOR_STATE.json. If AWAITING_AGENT or pending_action needs work, execute failure analysis → implement → launch per SOTA_PROGRESS.md Iteration Protocol. Do not kill healthy training. Update SOTA_PROGRESS.md.'

emit_json() {
  python3 - "$1" "$2" <<'PY'
import json, sys
kind, prompt = sys.argv[1], sys.argv[2]
payload = {"prompt": prompt, "kind": kind, "repo": "/root/autodl-tmp/Lora-code"}
print(f'AGENT_LOOP_{kind}_SOTA ' + json.dumps(payload, ensure_ascii=False))
PY
}

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] sota_agent_loop started poll=${POLL_SEC}s heartbeat=${HEARTBEAT_SEC}s" | tee -a "${WAKE_LOG}"

last_heartbeat=$(date +%s)
# Per loop skill: first sentinel after initial sleep (no double-run on startup).
sleep "${POLL_SEC}"

while true; do
  now=$(date +%s)

  if [[ -f "${FLAG}" ]]; then
  payload=$(cat "${FLAG}")
  python3 - "${payload}" <<'PY'
import json, sys
flag = json.loads(sys.argv[1])
flag["wake_reason"] = "SOTA_AGENT_WAKE.flag"
flag["kind"] = "wake"
print("AGENT_LOOP_WAKE_SOTA " + json.dumps(flag, ensure_ascii=False))
PY
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] emitted AGENT_LOOP_WAKE_SOTA (flag present)" >> "${WAKE_LOG}"
  fi

  if (( now - last_heartbeat >= HEARTBEAT_SEC )); then
    emit_json "TICK" "${PROMPT}"
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] emitted AGENT_LOOP_TICK_SOTA (heartbeat)" >> "${WAKE_LOG}"
    last_heartbeat=$now
  fi

  sleep "${POLL_SEC}"
done
