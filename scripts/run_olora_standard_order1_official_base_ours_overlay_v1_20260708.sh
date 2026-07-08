#!/usr/bin/env bash
# Standard Ours v1: v69 anchor + class-coverage SSRG + assess retention gate (threshold 0.25).
# Single mechanism delta from v85: SC_CLASS_COVERAGE_ORDER=1, ASSESS_RETENTION_THRESHOLD=0.25.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORMAL="${FORMAL:-0}"

if [[ "${FORMAL}" == "1" ]]; then
  RUN_NAME="${RUN_NAME:-olora_official_base_ours_overlay_v1_20260708_formal_order1_seed1}"
  WANDB_GROUP="${WANDB_GROUP:-published-base-standard-olora-plus-ours-overlay-v1-20260708-formal}"
  RUN_LABEL="${RUN_LABEL:-published_base_olora_v69_plus_classcov_ssrg_assess_gate_overlay_v1_20260708_formal}"
else
  RUN_NAME="${RUN_NAME:-olora_official_base_ours_overlay_v1_20260708_smoke_order1_seed1}"
  WANDB_GROUP="${WANDB_GROUP:-published-base-standard-olora-plus-ours-overlay-v1-20260708-smoke}"
  RUN_LABEL="${RUN_LABEL:-published_base_olora_v69_plus_classcov_ssrg_assess_gate_overlay_v1_20260708_smoke}"
fi

export RUN_NAME WANDB_GROUP RUN_LABEL
export WANDB_PROJECT="${WANDB_PROJECT:-lora-ours-v1}"
export REPLAY_MODE="${REPLAY_MODE:-ssrg}"
export REPLAY_PER_TASK="${REPLAY_PER_TASK:-64}"
export ASSESS_RETENTION_GATE="${ASSESS_RETENTION_GATE:-1}"
export ASSESS_RETENTION_THRESHOLD="${ASSESS_RETENTION_THRESHOLD:-0.25}"
export SC_CLASS_COVERAGE_ORDER="${SC_CLASS_COVERAGE_ORDER:-1}"
export SSRG_SPECTRAL_TOP_K="${SSRG_SPECTRAL_TOP_K:-8}"
export SSRG_ENERGY_THRESHOLD="${SSRG_ENERGY_THRESHOLD:-0.85}"
export EARLY_GATE_DBPEDIA_EM="${EARLY_GATE_DBPEDIA_EM:-90}"
export EARLY_GATE_AMAZON_EM="${EARLY_GATE_AMAZON_EM:-50}"
export TRAIN_HELDOUT_GATE="${TRAIN_HELDOUT_GATE:-0}"
export FORMAL

exec "${REPO_ROOT}/scripts/run_olora_standard_order1_official_base_ours_overlay_v58.sh" "$@"
