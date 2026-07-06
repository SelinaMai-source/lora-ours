#!/usr/bin/env bash
# Master dispatcher for CITB InstrDialog order1 official-script baselines (500/50/50).
# Usage: METHOD=replay50 DRY_RUN=0 bash scripts/run_citb_instrdialog_all_baselines_repro.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

METHOD="${METHOD:-}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
SEED="${SEED:-1}"
ORDER="${ORDER:-1}"
SPLIT_POLICY="${SPLIT_POLICY:-official_script_500_50_50}"

if [[ -z "${METHOD}" ]]; then
  cat <<'EOF'
CITB InstrDialog order1 baseline dispatcher.

METHOD values:
  ft_init   — FT_INSTR (official run_cit_ft_instr.sh)
  l2        — L2 reg=0.01
  ewc       — EWC reg=0.01
  agem10    — AGEM memory 10
  agem50    — AGEM memory 50
  replay10  — Replay memory 10
  replay50  — Replay memory 50 (paper best non-Multi)
  multi     — MULTI_TASK upper bound (separate entrypoint; not yet wired)

Env: DRY_RUN=1 | SMOKE=1 | RUN_NAME=... | SEED=1 | ORDER=1
EOF
  exit 0
fi

_common() {
  export DRY_RUN SMOKE SEED ORDER SPLIT_POLICY
  export WANDB_PROJECT="${WANDB_PROJECT:-lora-ours}"
  export FIX_STAGE1_TIE_WORD_EMBEDDINGS="${FIX_STAGE1_TIE_WORD_EMBEDDINGS:-1}"
}

_suffix() {
  if [[ "${DRY_RUN}" == "1" ]]; then echo "_dryrun"
  elif [[ "${SMOKE}" == "1" ]]; then echo "_smoke"
  else echo "_formal"
  fi
}

case "${METHOD}" in
  ft_init)
    _common
    export RUN_NAME="${RUN_NAME:-citb_instrdialog_order1_seed${SEED}_official_script_500_50_50_tie_fixed_ft_instr_stage1_v54$(_suffix)}"
    exec bash scripts/run_citb_instrdialog_order1_official_base_repro.sh
    ;;
  l2)
    _common
    export CL_METHOD=L2 CL_REG=0.01
    export RUN_NAME="${RUN_NAME:-citb_instrdialog_order1_seed${SEED}_official_script_500_50_50_tie_fixed_l2_v56$(_suffix)}"
    exec bash scripts/run_citb_instrdialog_replay50_official_base_repro.sh
    ;;
  ewc)
    _common
    export CL_METHOD=EWC CL_REG=0.01
    export RUN_NAME="${RUN_NAME:-citb_instrdialog_order1_seed${SEED}_official_script_500_50_50_tie_fixed_ewc_v56$(_suffix)}"
    exec bash scripts/run_citb_instrdialog_replay50_official_base_repro.sh
    ;;
  agem10)
    _common
    export CL_METHOD=AGEM CL_REG=1 REPLAY_NUM_INSTANCE_PER_TASK=10
    export RUN_NAME="${RUN_NAME:-citb_instrdialog_order1_seed${SEED}_official_script_500_50_50_tie_fixed_agem10_v56$(_suffix)}"
    exec bash scripts/run_citb_instrdialog_replay50_official_base_repro.sh
    ;;
  agem50)
    _common
    export CL_METHOD=AGEM CL_REG=1 REPLAY_NUM_INSTANCE_PER_TASK=50
    export RUN_NAME="${RUN_NAME:-citb_instrdialog_order1_seed${SEED}_official_script_500_50_50_tie_fixed_agem50_v56$(_suffix)}"
    exec bash scripts/run_citb_instrdialog_replay50_official_base_repro.sh
    ;;
  replay10)
    _common
    export CL_METHOD=REPLAY REPLAY_NUM_INSTANCE_PER_TASK=10
    export RUN_NAME="${RUN_NAME:-citb_instrdialog_order1_seed${SEED}_official_script_500_50_50_tie_fixed_replay10_v56$(_suffix)}"
    exec bash scripts/run_citb_instrdialog_replay50_official_base_repro.sh
    ;;
  replay50)
    _common
    export CL_METHOD=REPLAY REPLAY_NUM_INSTANCE_PER_TASK=50
    export RUN_NAME="${RUN_NAME:-citb_instrdialog_order1_seed${SEED}_official_script_500_50_50_tie_fixed_replay50_formal_v56$(_suffix)}"
    exec bash scripts/run_citb_instrdialog_replay50_official_base_repro.sh
    ;;
  multi)
    echo "MULTI_TASK upper bound uses run_initial_multitask_tuning_with_CL.sh (different entrypoint)." >&2
    echo "Paper AR ~42.1; local launcher pending — use CITB_ROOT scripts/run_initial_multitask_tuning_with_CL.sh manually." >&2
    exit 9
    ;;
  *)
    echo "Unknown METHOD=${METHOD}" >&2
    exit 2
    ;;
esac
