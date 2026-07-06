#!/usr/bin/env bash
# Paper-aligned CITB Stage-1: official scripts/run_initial_multitask_tuning.sh entry,
# fixed seed=50, output under citb_official/output/.../checkpoint-14000 (Stage-2 init).
# Do not run on GPU occupied by ARPER v87 unless you accept contention; prefer serial queue.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CITB_ROOT="${CITB_ROOT:-/root/autodl-tmp/Lora-code/external_baselines/citb_official}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/lora_v10_citb/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/model_cache/hf_snapshots/google__t5-small-lm-adapt}"
CITB_GPT2_TOKENIZER_NAME="${CITB_GPT2_TOKENIZER_NAME:-/root/autodl-tmp/Lora-code/external_baselines/o_lora/data/gpt2tokenizer}"

SEED="${SEED:-50}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-15}"
LEARNING_RATE="${LEARNING_RATE:-1e-05}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-250}"
USE_BF16="${USE_BF16:-1}"
DRY_RUN="${DRY_RUN:-0}"
FIX_STAGE1_TIE_WORD_EMBEDDINGS="${FIX_STAGE1_TIE_WORD_EMBEDDINGS:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-${CITB_ROOT}/output/initial_multitask_model/base_epoch${NUM_TRAIN_EPOCHS}_lr${LEARNING_RATE}_seed${SEED}}"
TARGET_CHECKPOINT="${TARGET_CHECKPOINT:-${OUTPUT_DIR}/checkpoint-14000}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/citb_stage1_repro/logs}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CITB_GPT2_TOKENIZER_NAME
export PYTHONPATH="${REPO_ROOT}/scripts/citb_runtime_shims:${CITB_ROOT}/Tk-Instruct/src:${CITB_ROOT}/continual_learning:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-lora-ours}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-citb_stage1_seed50_official}"
export WANDB_NAME="${WANDB_NAME:-base_epoch${NUM_TRAIN_EPOCHS}_lr${LEARNING_RATE}_seed${SEED}}"
export WANDB_DIR="${WANDB_DIR:-/root/autodl-tmp/citb_stage1_repro/wandb}"

"${PYTHON_BIN}" - <<'PY' "${BASE_MODEL}" "${CITB_ROOT}"
from pathlib import Path
import sys

base_model, citb_root = map(Path, sys.argv[1:3])
missing = []
for path in [
    base_model / "config.json",
    citb_root / "continual_learning/run_initial_multitask_tuning.py",
    citb_root / "scripts/run_initial_multitask_tuning.sh",
    citb_root / "data/CIT_data/initial_multitask_learning/defintion_pos_2/train",
    citb_root / "data/CIT_data/official_test_data",
]:
    if not path.exists():
        missing.append(str(path))
if missing:
    raise SystemExit("Stage-1 seed50 preflight missing:\n" + "\n".join(missing))
print("stage1 seed50 preflight ok")
PY

if [[ -d "${TARGET_CHECKPOINT}" && -f "${TARGET_CHECKPOINT}/config.json" ]]; then
  echo "Stage-1 target already present: ${TARGET_CHECKPOINT}"
  exit 0
fi

cd "${CITB_ROOT}"

EXTRA_ARGS=()
if [[ "${USE_BF16}" == "1" ]]; then
  EXTRA_ARGS+=(--bf16)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA_ARGS+=(
    --max_steps 2
    --max_eval_samples 2
    --save_strategy no
    --report_to none
  )
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
else
  EXTRA_ARGS+=(
    --save_strategy steps
    --save_steps "${SAVE_STEPS}"
    --save_total_limit 1
    --load_best_model_at_end
    --metric_for_best_model rougeL
    --evaluation_strategy steps
    --eval_steps "${EVAL_STEPS}"
    --report_to wandb
  )
fi

# Mirrors scripts/run_initial_multitask_tuning.sh (official); local BASE_MODEL path only.
"${PYTHON_BIN}" continual_learning/run_initial_multitask_tuning.py \
  --do_train \
  --do_eval \
  --do_predict \
  --predict_with_generate \
  --model_name_or_path "${BASE_MODEL}" \
  --tokenizer_name "${BASE_MODEL}" \
  --use_fast_tokenizer False \
  --data_dir data/CIT_data/initial_multitask_learning/defintion_pos_2 \
  --data_dir_for_official_test data/CIT_data/official_test_data \
  --task_dir data/tasks/ \
  --max_source_length 1024 \
  --max_target_length 128 \
  --generation_max_length 128 \
  --add_task_name False \
  --add_task_definition True \
  --num_pos_examples 2 \
  --num_neg_examples 0 \
  --add_explanation False \
  --tk_instruct False \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --cache_dir ./cache/ \
  --overwrite_cache \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --lr_scheduler_type constant \
  --warmup_steps 0 \
  --logging_strategy steps \
  --logging_steps "${LOGGING_STEPS}" \
  --run_name initial_multitask_model \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN complete (no checkpoint-14000 expected)."
  exit 0
fi

if [[ ! -d "${TARGET_CHECKPOINT}" ]] || { [[ ! -f "${TARGET_CHECKPOINT}/pytorch_model.bin" ]] && [[ ! -f "${TARGET_CHECKPOINT}/model.safetensors" ]]; }; then
  echo "WARNING: ${TARGET_CHECKPOINT} not found after training; check ${OUTPUT_DIR} for latest checkpoint-*" >&2
  exit 2
fi

if [[ "${FIX_STAGE1_TIE_WORD_EMBEDDINGS}" == "1" ]]; then
  "${PYTHON_BIN}" - <<'PY' "${TARGET_CHECKPOINT}"
import json
import sys
from pathlib import Path

ckpt = Path(sys.argv[1])
config_path = ckpt / "config.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
original = config.get("tie_word_embeddings")
if original is True:
    config["tie_word_embeddings"] = False
    config["_citb_runtime_patch"] = {
        "reason": "Stage-1 checkpoint: disable tie for transformers 4.25 Stage-2 load (lm_head vs shared weights).",
        "original_tie_word_embeddings": original,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"patched {config_path} tie_word_embeddings {original!r} -> False")
else:
    print(f"no tie patch needed ({config_path}, tie_word_embeddings={original!r})")
PY
fi

echo "Stage-1 ready for CITB v2: ${TARGET_CHECKPOINT}"
