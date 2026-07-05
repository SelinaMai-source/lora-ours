#!/bin/bash
# Cursor agent wake loop for lora-ours SOTA closed-loop (see /root/.cursor/skills-cursor/loop/SKILL.md).
#
# Launch:
#   tmux new-session -d -s lora-ours-agent-loop 'bash /root/lora-ours/scripts/sota_agent_loop.sh'
# Or:
#   bash scripts/launch_lora_ours_agent_stack.sh
set -euo pipefail
cd /root/lora-ours

POLL_SEC="${SOTA_AGENT_POLL_SEC:-30}"
HEARTBEAT_SEC="${SOTA_AGENT_LOOP_SEC:-300}"
FLAG="SOTA_AGENT_WAKE.flag"
WAKE_LOG="SOTA_AGENT_WAKE.log"

PROMPT='Read results/logs/lora_ours_sentinel_status.md and lora_ours_sentinel_status.json. If pending_action or wake_items need work, execute failure analysis → single-mechanism vN+1 patch → smoke gate → formal per docs/official_alignment/ccfa_experiment_gate.md. Do not kill healthy training. Update docs/official_alignment/status.md.'

emit_json() {
  python3 - "$1" "$2" <<'PY'
import json, sys
kind, prompt = sys.argv[1], sys.argv[2]
payload = {"prompt": prompt, "kind": kind, "repo": "/root/lora-ours"}
print(f'AGENT_LOOP_{kind}_LORA_OURS ' + json.dumps(payload, ensure_ascii=False))
PY
}

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] sota_agent_loop started poll=${POLL_SEC}s heartbeat=${HEARTBEAT_SEC}s repo=lora-ours" | tee -a "${WAKE_LOG}"

last_heartbeat=$(date +%s)
sleep "${POLL_SEC}"

while true; do
  now=$(date +%s)

  if [[ -f "${FLAG}" ]]; then
    payload=$(cat "${FLAG}")
    python3 - "${payload}" <<'PY'
import json, sys
flag = json.loads(sys.argv[1])
flag["wake_reason"] = flag.get("wake_reason") or "SOTA_AGENT_WAKE.flag"
flag["kind"] = "wake"
print("AGENT_LOOP_WAKE_LORA_OURS " + json.dumps(flag, ensure_ascii=False))
PY
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] emitted AGENT_LOOP_WAKE_LORA_OURS (flag present)" >> "${WAKE_LOG}"
  fi

  if (( now - last_heartbeat >= HEARTBEAT_SEC )); then
    emit_json "TICK" "${PROMPT}"
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] emitted AGENT_LOOP_TICK_LORA_OURS (heartbeat)" >> "${WAKE_LOG}"
    last_heartbeat=$now
  fi

  sleep "${POLL_SEC}"
done
