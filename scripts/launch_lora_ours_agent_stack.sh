#!/usr/bin/env bash
# Start lora-ours sentinel monitor + Cursor agent wake loop in tmux.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SENTINEL_SESSION="${SENTINEL_SESSION:-lora-ours-sentinel}"
AGENT_SESSION="${AGENT_SESSION:-lora-ours-agent-loop}"
SENTINEL_POLL_SEC="${SENTINEL_POLL_SEC:-120}"
SENTINEL_EMIT_WAKE="${SENTINEL_EMIT_WAKE:-1}"

for session in "$SENTINEL_SESSION" "$AGENT_SESSION"; do
  if [[ "$session" != lora-ours-* ]]; then
    echo "[blocked] tmux session must start with lora-ours-; got: $session" >&2
    exit 2
  fi
done

launch_sentinel() {
  if tmux has-session -t "$SENTINEL_SESSION" 2>/dev/null; then
    echo "[skip] sentinel already running: $SENTINEL_SESSION"
    return 0
  fi
  local emit_flag=""
  if [[ "$SENTINEL_EMIT_WAKE" == "1" ]]; then
    emit_flag="--emit-wake"
  fi
  local cmd="cd '$ROOT' && while true; do python3 scripts/lora_ours_sentinel.py ${emit_flag}; sleep ${SENTINEL_POLL_SEC}; done"
  tmux new-session -d -s "$SENTINEL_SESSION" "bash -lc $(printf '%q' "$cmd")"
  echo "[launched] $SENTINEL_SESSION (poll=${SENTINEL_POLL_SEC}s emit_wake=${SENTINEL_EMIT_WAKE})"
}

launch_agent_loop() {
  if tmux has-session -t "$AGENT_SESSION" 2>/dev/null; then
    echo "[skip] agent loop already running: $AGENT_SESSION"
    return 0
  fi
  tmux new-session -d -s "$AGENT_SESSION" "bash '$ROOT/scripts/sota_agent_loop.sh'"
  echo "[launched] $AGENT_SESSION"
}

launch_sentinel
launch_agent_loop

python3 scripts/lora_ours_sentinel.py ${SENTINEL_EMIT_WAKE:+--emit-wake} || true

cat <<EOF
[agent-stack-ready]
  repo: $ROOT
  sentinel: tmux attach -t $SENTINEL_SESSION
  agent loop: tmux attach -t $AGENT_SESSION
  status md: results/logs/lora_ours_sentinel_status.md
  status json: results/logs/lora_ours_sentinel_status.json
  wake flag: SOTA_AGENT_WAKE.flag
  wake log: SOTA_AGENT_WAKE.log
EOF
