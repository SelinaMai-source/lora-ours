#!/usr/bin/env bash
# v85: v69 anchor + SSRG replay + assess-update retention gate.
# Delegates to v58 launcher; official O-LoRA entry/protocol unchanged.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORMAL="${FORMAL:-0}"

if [[ "${FORMAL}" == "1" ]]; then
  RUN_NAME="${RUN_NAME:-olora_official_base_ours_overlay_ssrg_v85_formal_order1_seed1}"
  WANDB_GROUP="${WANDB_GROUP:-published-base-standard-olora-plus-ours-overlay-v85-formal}"
  RUN_LABEL="${RUN_LABEL:-published_base_olora_official_runtime_plus_ours_ssrg_assess_gate_overlay_v85_formal}"
else
  RUN_NAME="${RUN_NAME:-olora_official_base_ours_overlay_ssrg_v85_smoke_order1_seed1}"
  WANDB_GROUP="${WANDB_GROUP:-published-base-standard-olora-plus-ours-overlay-v85-smoke}"
  RUN_LABEL="${RUN_LABEL:-published_base_olora_official_runtime_plus_ours_ssrg_assess_gate_overlay_v85_smoke}"
fi

export RUN_NAME WANDB_GROUP RUN_LABEL
export WANDB_PROJECT="${WANDB_PROJECT:-lora-ours}"
export REPLAY_MODE="${REPLAY_MODE:-ssrg}"
export REPLAY_PER_TASK="${REPLAY_PER_TASK:-64}"
export ASSESS_RETENTION_GATE="${ASSESS_RETENTION_GATE:-1}"
export ASSESS_RETENTION_THRESHOLD="${ASSESS_RETENTION_THRESHOLD:-0.3}"
export SSRG_SPECTRAL_TOP_K="${SSRG_SPECTRAL_TOP_K:-8}"
export SSRG_ENERGY_THRESHOLD="${SSRG_ENERGY_THRESHOLD:-0.85}"
export EARLY_GATE_DBPEDIA_EM="${EARLY_GATE_DBPEDIA_EM:-90}"
export EARLY_GATE_AMAZON_EM="${EARLY_GATE_AMAZON_EM:-45}"
export TRAIN_HELDOUT_GATE="${TRAIN_HELDOUT_GATE:-0}"
export FORMAL

exec "${REPO_ROOT}/scripts/run_olora_standard_order1_official_base_ours_overlay_v58.sh" "$@"
