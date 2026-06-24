#!/usr/bin/env bash
# Run all behavior-gate overfit-8 configs (manual vs completion-only, LoRA ablations).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIGS=(
  "configs/behavior_gate_manual_qv.yaml"
  "configs/behavior_gate_completion_qv.yaml"
  "configs/behavior_gate_manual_qkvo.yaml"
  "configs/behavior_gate_manual_alllinear.yaml"
)

for c in "${CONFIGS[@]}"; do
  echo "========== $c =========="
  python core/train.py --config "$c"
done

echo "Done. See results/runs/<experiment>/debug/overfit8/ and results/debug_report_behavior_overfit_gate.md"
