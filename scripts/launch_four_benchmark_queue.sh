#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SESSION="lora_four_benchmark_single_seed"
LOG="results/logs/four_benchmark_single_seed_queue.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists"
  exit 1
fi

mkdir -p results/logs
chmod +x scripts/run_four_benchmark_single_seed_queue.sh

# Wait for active sota-v3 InstrDialog++ training if running
WAIT_PIDS=""
if pgrep -f "sota_v3_instrdialogpp_s123.yaml" >/dev/null 2>&1; then
  WAIT_PIDS="$(pgrep -f 'sota_v3_instrdialogpp_s123.yaml' | tr '\n' ' ')"
  echo "Will wait for sota-v3 InstrDialog++ PIDs: $WAIT_PIDS"
fi

tmux new-session -d -s "$SESSION" \
  "export WAIT_FOR_PIDS='$WAIT_PIDS'; export WANDB_PROJECT=lora-four-benchmark-single-seed; export WANDB_MODE=online; bash scripts/run_four_benchmark_single_seed_queue.sh 2>&1 | tee '$LOG'"
echo "Started tmux session $SESSION -> $LOG"
