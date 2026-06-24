#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_FILE="$ROOT/results/logs/priority_run.log"
mkdir -p "$ROOT/results/logs"

echo "=== 1. Starting priority run for instrdialog ours ===" | tee -a "$LOG_FILE"
python3 scripts/run_paper_matrix.py --benchmarks instrdialog --modes ours 2>&1 | tee -a "$LOG_FILE"

echo "=== 2. Resuming interrupted instrdialog++ baseline runs ===" | tee -a "$LOG_FILE"
python3 scripts/run_paper_matrix.py --benchmarks instrdialog++ --categories main --skip-existing 2>&1 | tee -a "$LOG_FILE"

echo "=== 3. Building artifacts and PDF ===" | tee -a "$LOG_FILE"
python3 scripts/build_paper_artifacts.py 2>&1 | tee -a "$LOG_FILE"
python3 scripts/build_rp_lora_v3.py 2>&1 | tee -a "$LOG_FILE"

echo "=== All tasks finished successfully! ===" | tee -a "$LOG_FILE"
