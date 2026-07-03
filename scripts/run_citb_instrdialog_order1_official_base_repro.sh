#!/usr/bin/env bash
set -euo pipefail

# Fixed, auditable launcher for the CITB official Stage-2 FT_INSTR base on
# InstrDialog order1. Outputs are kept outside this repo by default.

# The hyintell/CITB checkout under external_sources has an untracked
# Tk-Instruct copy whose collator lacks add_task_id. This tracked local
# official-baseline tree matches the CL entrypoint's collator contract.
CITB_ROOT="${CITB_ROOT:-/root/autodl-tmp/Lora-code/external_baselines/citb_official}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469}"
TOKENIZER_NAME="${TOKENIZER_NAME:-/root/autodl-tmp/model_cache/hf_snapshots/google__t5-small-lm-adapt}"
OUTPUT_BASE="${OUTPUT_BASE:-/root/autodl-tmp/citb_official_base_repro}"
RUN_NAME="${RUN_NAME:-citb_instrdialog_order1_seed1_official_ft_instr_stage1_v53}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}}"
ORDER="${ORDER:-1}"
SEED="${SEED:-1}"
MAX_TRAIN_INSTANCES="${MAX_TRAIN_INSTANCES:-500}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/lora_v10_citb/bin/python}"

# The unmodified CITB Stage-2 script exposes one value for both dev and per-task
# test split size. Keeping it at 50 is script-strict; 500/50/100 needs a local
# split-policy patch and should be reported as a comparability RED FLAG.
MAX_EVAL_INSTANCES="${MAX_EVAL_INSTANCES:-50}"

DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" == "1" ]]; then
  RUN_NAME="${RUN_NAME}_dryrun"
  OUTPUT_DIR="${OUTPUT_DIR}_dryrun"
  NUM_EPOCHS="${NUM_EPOCHS:-1}"
  MAX_STEPS_ARGS=(--max_steps 1 --max_eval_samples 1)
  PREDICT_ARGS=()
  BF16_ARGS=()
  REPORT_ARGS=(--report_to none)
  WANDB_MODE_DEFAULT="offline"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  ORDER_DIR="${OUTPUT_DIR}/dryrun_task_order/stream=cl_dialogue_tasks"
  mkdir -p "${ORDER_DIR}"
  "${PYTHON_BIN}" - <<'PY' "${CITB_ROOT}" "${ORDER_DIR}" "${ORDER}"
from pathlib import Path
import sys

citb_root = Path(sys.argv[1])
order_dir = Path(sys.argv[2])
order = sys.argv[3]
src = citb_root / "data/CIT_data/task_orders/stream=cl_dialogue_tasks" / f"order{order}.txt"
first_task = src.read_text(encoding="utf-8").splitlines()[0]
(order_dir / f"order{order}.txt").write_text(first_task + "\n", encoding="utf-8")
print(f"dryrun task order: {first_task}")
PY
  DATA_DIR_FOR_TASK_ORDER="${ORDER_DIR}"
else
  NUM_EPOCHS="${NUM_EPOCHS:-15}"
  MAX_STEPS_ARGS=()
  PREDICT_ARGS=(--do_predict)
  BF16_ARGS=(--bf16)
  REPORT_ARGS=(--report_to wandb)
  WANDB_MODE_DEFAULT="online"
  DATA_DIR_FOR_TASK_ORDER="data/CIT_data/task_orders/stream=cl_dialogue_tasks"
fi

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_BASE}/wandb"

export WANDB_MODE="${WANDB_MODE:-${WANDB_MODE_DEFAULT}}"
export WANDB_PROJECT="${WANDB_PROJECT:-lora-ours-v53-citb-official-base}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-citb_instrdialog_order1_official_base_repro}"
export WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_BASE}/wandb}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${CITB_ROOT}/Tk-Instruct/src:${CITB_ROOT}/continual_learning:${PYTHONPATH:-}"

cd "${CITB_ROOT}"

"${PYTHON_BIN}" - <<'PY'
from dataclasses import fields
import sys

sys.path.insert(1, "Tk-Instruct/src")
from ni_collator import DataCollatorForNI  # noqa: E402

field_names = {field.name for field in fields(DataCollatorForNI)}
if "add_task_id" not in field_names:
    raise SystemExit(
        "CITB/Tk-Instruct collator mismatch: DataCollatorForNI lacks add_task_id. "
        "Point CITB_ROOT at a checkout whose Tk-Instruct/src/ni_collator.py matches "
        "continual_learning/run_continual_instruct_tuning.py."
    )
print(f"collator preflight ok: {DataCollatorForNI.__module__} supports add_task_id")
PY

CMD=(
  "${PYTHON_BIN}" continual_learning/run_continual_instruct_tuning.py
  --do_train
  --do_eval
  --predict_with_generate
  --model_name_or_path "${MODEL_PATH}"
  --tokenizer_name "${TOKENIZER_NAME}"
  --use_fast_tokenizer False
  --cl_method FT_INSTR
  --order "${ORDER}"
  --data_dir_for_task_order "${DATA_DIR_FOR_TASK_ORDER}"
  --task_split_file_name cl_dialogue_tasks
  --data_dir_for_official_test data/CIT_data/official_test_data
  --data_dir_for_initial_training_dir data/CIT_data/initial_multitask_learning/defintion_pos_2
  --max_source_length 1024
  --max_target_length 128
  --generation_max_length 128
  --max_num_instances_per_task "${MAX_TRAIN_INSTANCES}"
  --max_num_instances_per_eval_task "${MAX_EVAL_INSTANCES}"
  --add_task_name False
  --add_task_definition True
  --num_pos_examples 2
  --num_neg_examples 0
  --add_explanation False
  --tk_instruct False
  --data_dir data/splits/CIT_splits/
  --task_dir data/tasks/
  --output_dir "${OUTPUT_DIR}"
  --overwrite_output_dir
  --cache_dir ./cache/
  --overwrite_cache
  --per_device_train_batch_size 8
  --per_device_eval_batch_size 32
  --gradient_accumulation_steps 1
  --learning_rate 1e-05
  --num_train_epochs "${NUM_EPOCHS}"
  --lr_scheduler_type constant
  --warmup_steps 0
  --logging_strategy steps
  --logging_steps 50
  --evaluation_strategy epoch
  --save_strategy epoch
  --save_total_limit 1
  --load_best_model_at_end
  --metric_for_best_model rougeL
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
)

CMD+=("${PREDICT_ARGS[@]}" "${BF16_ARGS[@]}" "${REPORT_ARGS[@]}" "${MAX_STEPS_ARGS[@]}")
"${CMD[@]}"
