#!/usr/bin/env bash
# CITB Replay(50) paper-alignment v2 — official Stage-1 path, native tokenizer, minimal shims.
# Wraps run_citb_instrdialog_replay50_official_base_repro.sh with v2 defaults from audit 20260707.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CITB_ROOT="${CITB_ROOT:-/root/autodl-tmp/Lora-code/external_baselines/citb_official}"
OFFICIAL_STAGE1="${CITB_ROOT}/output/initial_multitask_model/base_epoch15_lr1e-05_seed50/checkpoint-14000"
FALLBACK_STAGE1="${FALLBACK_STAGE1:-/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
ALLOW_FALLBACK_STAGE1="${ALLOW_FALLBACK_STAGE1:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-paper_aligned_v2}"

PREFLIGHT_OUT="${REPO_ROOT}/results/logs/citb_official_split_counts_500_50_100_replay50.json"

echo "=== CITB Replay(50) paper-aligned v2 preflight ==="

python3 scripts/preflight_citb_official_split_counts.py \
  --citb-root "${CITB_ROOT}" \
  --order 1 \
  --train 500 --dev 50 --test 100 \
  --output "${PREFLIGHT_OUT}" >/dev/null || true
short_count="$(python3 -c "import json; print(json.load(open('${PREFLIGHT_OUT}'))['short_task_count'])")"
echo "Split 500/50/100: ${short_count}/19 short tasks (paper-text not fully supported; using official_script_500_50_50)"

MODEL_PATH=""
TOKENIZER_NAME=""
USE_TIE_FIX=1

if [[ -d "${OFFICIAL_STAGE1}" && -f "${OFFICIAL_STAGE1}/config.json" ]]; then
  MODEL_PATH="${OFFICIAL_STAGE1}"
  TOKENIZER_NAME="${OFFICIAL_STAGE1}"
  echo "Stage-1: official checkpoint-14000 (seed50)"
elif [[ "${ALLOW_FALLBACK_STAGE1}" == "1" && -d "${FALLBACK_STAGE1}" ]]; then
  MODEL_PATH="${FALLBACK_STAGE1}"
  # seed469 ships tokenizer.json incompatible with CITB transformers 4.25 T5 slow path
  TOKENIZER_NAME="${TOKENIZER_NAME:-/root/autodl-tmp/model_cache/hf_snapshots/google__t5-small-lm-adapt}"
  echo "WARNING: official checkpoint-14000 missing; using fallback seed469 + lm-adapt tokenizer shim." >&2
else
  cat >&2 <<EOF
BLOCKED: official Stage-1 not found at:
  ${OFFICIAL_STAGE1}

Required for paper-aligned v2. Options:
  1. Train or download CITB Stage-1 FT_INSTR → checkpoint-14000 (seed50)
  2. Re-run with ALLOW_FALLBACK_STAGE1=1 (diagnostic only; not paper-comparable)

Current fallback available: ${FALLBACK_STAGE1}
EOF
  exit 4
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  suffix="_dryrun"
elif [[ "${SMOKE}" == "1" ]]; then
  suffix="_smoke"
else
  suffix="_formal"
fi

export MODEL_PATH TOKENIZER_NAME
export CITB_ROOT
export SPLIT_POLICY=official_script_500_50_50
export SEED="${SEED:-50}"
export FIX_STAGE1_TIE_WORD_EMBEDDINGS="${USE_TIE_FIX}"
export CL_METHOD=REPLAY
export REPLAY_NUM_INSTANCE_PER_TASK=50
export RUN_NAME="citb_instrdialog_order1_seed${SEED}_official_script_500_50_50_paper_aligned_replay50_${RUN_SUFFIX}${suffix}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-citb_instrdialog_order1_paper_aligned_replay50_v2}"
export WANDB_NAME="${RUN_NAME}"

echo "RUN_NAME=${RUN_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOKENIZER_NAME=${TOKENIZER_NAME}"
echo "SEED=${SEED} SPLIT_POLICY=${SPLIT_POLICY} DRY_RUN=${DRY_RUN} SMOKE=${SMOKE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  export DRY_RUN=1
fi
if [[ "${SMOKE}" == "1" ]]; then
  export SMOKE=1
fi

exec bash scripts/run_citb_instrdialog_replay50_official_base_repro.sh
