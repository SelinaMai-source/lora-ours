#!/usr/bin/env bash
# Launch controlled phase-1 jobs in tmux. Defaults to a local debug smoke run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSION="${SESSION:-lora-ours-v0-smoke}"
CONFIG="${CONFIG:-configs/phase1/ours_v0_debug_smoke.yaml}"
LOG_DIR="${LOG_DIR:-results/logs/phase1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
REMOTE_TIMEOUT_SEC="${REMOTE_TIMEOUT_SEC:-8}"
WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$LOG_DIR" results/preflight

if [[ "$SESSION" != lora-ours-* ]]; then
  echo "[blocked] SESSION must start with lora-ours-; got: $SESSION" >&2
  exit 2
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[blocked] tmux session already exists: $SESSION" >&2
  echo "Inspect with: tmux attach -t $SESSION" >&2
  exit 2
fi

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  set +e
  python scripts/phase1_preflight.py \
    --remote-timeout-sec "$REMOTE_TIMEOUT_SEC" \
    --out "results/preflight/${SESSION}.json" \
    2>&1 | tee "$LOG_DIR/${SESSION}_preflight.log"
  PREFLIGHT_CODE="${PIPESTATUS[0]}"
  set -e
  if [[ "$PREFLIGHT_CODE" -ne 0 ]]; then
    echo "[blocked] preflight returned $PREFLIGHT_CODE; not launching training." >&2
    exit "$PREFLIGHT_CODE"
  fi
fi

COMMAND="export WANDB_MODE='$WANDB_MODE'; python scripts/phase1_prepare_data.py --write-toy-smoke; python core/train.py --config '$CONFIG' 2>&1 | tee '$LOG_DIR/${SESSION}.log'"
tmux new-session -d -s "$SESSION" "$COMMAND"

echo "[launched] $SESSION"
echo "Log: $LOG_DIR/${SESSION}.log"
echo "Attach: tmux attach -t $SESSION"
