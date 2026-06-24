#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_FILE="$ROOT/results/logs/queue_ours_instrdialog.log"
mkdir -p "$ROOT/results/logs"

echo "Waiting for PID 92186 to finish..." | tee -a "$LOG_FILE"
while kill -0 92186 2>/dev/null; do
    sleep 60
done
echo "PID 92186 finished. Starting ours runs for instrdialog..." | tee -a "$LOG_FILE"

# Run ALL ours variants for instrdialog (main and ablations)
python3 scripts/run_paper_matrix.py --benchmarks instrdialog --modes ours 2>&1 | tee -a "$LOG_FILE"

echo "Runs completed. Rebuilding artifacts and PDF..." | tee -a "$LOG_FILE"
python3 scripts/build_paper_artifacts.py 2>&1 | tee -a "$LOG_FILE"
python3 scripts/build_rp_lora_v3.py 2>&1 | tee -a "$LOG_FILE"

echo "All queued tasks finished successfully!" | tee -a "$LOG_FILE"
