#!/usr/bin/env bash
# ACL/ARR staged experiment runner.
#
# Stage 1: short anti-overlap smoke/sweep on InstrDialog.
# Stage 2: full main + ablation runs on both benchmarks.
# Stage 3: optional multi-seed matrix after Stage 1/2 look healthy.
#
# Examples:
#   bash scripts/run_acl_antioverlap_experiments.sh smoke
#   WANDB_MODE=online bash scripts/run_acl_antioverlap_experiments.sh main
#   bash scripts/run_acl_antioverlap_experiments.sh multiseed

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE="${1:-smoke}"
shift || true

export WANDB_PROJECT="${WANDB_PROJECT:-lora-citb-acl}"
export WANDB_GROUP="${WANDB_GROUP:-acl_antioverlap_${STAGE}}"
export WANDB_MODE="${WANDB_MODE:-online}"

COMMON_WANDB_ARGS=(
  --set output.tracking.use_wandb=true
  --set output.tracking.wandb_project="${WANDB_PROJECT}"
  --set output.tracking.wandb_group="${WANDB_GROUP}"
  --set output.tracking.wandb_mode="${WANDB_MODE}"
)

case "$STAGE" in
  smoke)
    python3 scripts/run_paper_matrix.py \
      --benchmarks instrdialog \
      --variant-ids ours_full,ours_no_overlap,ours_beta001,ours_beta003,ours_beta006 \
      --max-segments 4 \
      --max-train-examples 16 \
      --max-eval-examples 8 \
      --epochs-per-segment 1 \
      --run-name-suffix acl_smoke \
      "${COMMON_WANDB_ARGS[@]}" \
      "$@"
    ;;
  main)
    python3 scripts/run_paper_matrix.py \
      --benchmarks instrdialog,instrdialog++ \
      --categories main,ablation \
      --skip-existing \
      "${COMMON_WANDB_ARGS[@]}" \
      "$@"
    ;;
  sweep)
    python3 scripts/run_paper_matrix.py \
      --benchmarks instrdialog \
      --categories sweep \
      --variant-ids ours_beta001,ours_beta003,ours_beta006,ours_no_meta_threshold,ours_reverse_curriculum \
      --skip-existing \
      "${COMMON_WANDB_ARGS[@]}" \
      "$@"
    ;;
  multiseed)
    python3 scripts/build_paper_run_matrix.py --include-sweeps --seeds 123,456,789
    python3 scripts/run_paper_matrix.py \
      --benchmarks instrdialog,instrdialog++ \
      --categories main,ablation \
      --skip-existing \
      "${COMMON_WANDB_ARGS[@]}" \
      "$@"
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    echo "Expected one of: smoke | main | sweep | multiseed" >&2
    exit 2
    ;;
esac
