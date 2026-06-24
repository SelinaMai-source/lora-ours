#!/usr/bin/env bash
# Overnight LoRA paper experiment queue.
# Uses the autoresearch discipline: run one auditable experiment queue, skip completed
# runs, keep logs, and refresh paper artifacts from whatever completes.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

LOG_DIR="${ROOT}/results/overnight_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/lora_submission_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOG_FILE" 2>&1

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-lora-citb-acl}"
export WANDB_GROUP="${WANDB_GROUP:-lora_submission_$(date +%Y%m%d)}"
export PYTHONUNBUFFERED=1

echo "== LoRA overnight submission queue started: $(date) =="
echo "root=${ROOT}"
echo "log=${LOG_FILE}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_GROUP=${WANDB_GROUP}"
echo

run_step() {
  local name="$1"
  shift
  echo
  echo "========================================================================"
  echo "== ${name}: start $(date)"
  echo "== command: $*"
  echo "========================================================================"
  "$@"
  local rc=$?
  echo "== ${name}: exit ${rc} at $(date)"
  return "${rc}"
}

# Record provenance for paper auditability.
run_step "provenance" bash -lc 'git status --short --branch || true; git log --oneline -5 || true'

# Ensure matrix contains the publication-oriented seeds/sweeps.
run_step "build paper matrix" python3 scripts/build_paper_run_matrix.py --include-sweeps --seeds 123,456,789 || true

# Priority 1: full current main + ablation package on both benchmarks.
run_step "main and ablation matrix" \
  bash scripts/run_acl_antioverlap_experiments.sh main || true

# Priority 2: mechanism sweeps that support reviewer-facing claims.
run_step "mechanism sweep" \
  bash scripts/run_acl_antioverlap_experiments.sh sweep || true

# Priority 3: multi-seed coverage. skip-existing protects completed seed-123 runs.
run_step "multi-seed main and ablation matrix" \
  bash scripts/run_acl_antioverlap_experiments.sh multiseed || true

# Always refresh paper artifacts after completed runs.
run_step "sync paper executions" python3 scripts/sync_paper_executions.py || true
run_step "build paper artifacts" python3 scripts/build_paper_artifacts.py || true
run_step "build RP v3" python3 scripts/build_rp_lora_v3.py || true
run_step "package paper results" python3 scripts/package_paper_results.py || true
run_step "select paper winner" python3 scripts/select_paper_winner.py || true

echo
echo "== LoRA overnight submission queue finished: $(date) =="
