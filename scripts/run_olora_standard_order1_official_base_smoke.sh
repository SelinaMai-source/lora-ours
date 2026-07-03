#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFICIAL_ROOT="${OLORA_OFFICIAL_ROOT:-/root/autodl-tmp/lora-baselines-run_v1/external_sources/o_lora}"
ENV_PREFIX="${OLORA_ENV_PREFIX:-/root/autodl-tmp/conda_envs/lora_v10_o_lora}"
BASE_MODEL="${OLORA_T5_LARGE:-/root/autodl-tmp/model_cache/hf_snapshots/t5-large}"

RUN_NAME="${RUN_NAME:-olora_t5large_standard_order1_seed1_official_base_smoke_v55}"
WANDB_PROJECT="${WANDB_PROJECT:-lora-ours-published-base-standard}"
WANDB_GROUP="${WANDB_GROUP:-published-base-standard-olora-order1}"
MAX_STEPS="${MAX_STEPS:-1}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-8}"
MAX_PREDICT_SAMPLES="${MAX_PREDICT_SAMPLES:-16}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-16}"
LORA_DIM="${LORA_DIM:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"

RUN_DIR="${REPO_ROOT}/results/runs/${RUN_NAME}"
LOG_DIR="${REPO_ROOT}/results/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
STATUS_FILE="${LOG_DIR}/${RUN_NAME}_status.md"
MANIFEST_FILE="${RUN_DIR}/run_manifest.json"
RUNTIME_ROOT="${RUN_DIR}/official_runtime"
OUTPUT_ROOT="${RUN_DIR}/official_outputs"

TASKS=(dbpedia amazon yahoo agnews)

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${OUTPUT_ROOT}"

write_status() {
  local state="$1"
  local reason="${2:-}"
  python - "$STATUS_FILE" "$RUN_NAME" "$state" "$reason" "$WANDB_PROJECT" "$WANDB_GROUP" "$LOG_FILE" "$MANIFEST_FILE" <<'PY'
import sys
from datetime import datetime
from pathlib import Path

status_file, run_name, state, reason, project, group, log_file, manifest_file = sys.argv[1:]
lines = [
    f"# {run_name}",
    "",
    f"- updated: {datetime.now().isoformat(timespec='seconds')}",
    f"- state: {state}",
    f"- reason: {reason}",
    "- selected base: O-LoRA official T5-large Standard CL order1 seed1",
    "- label: published-base official-runtime smoke, not ours improvement",
    f"- W&B project: {project}",
    f"- W&B group: {group}",
    f"- log: {log_file}",
    f"- manifest: {manifest_file}",
    "- setting: dbpedia -> amazon -> yahoo -> agnews; max_steps=1; max_train_samples=8; max_predict_samples=16",
    "- engineering notes: single-GPU runtime; eval batch reduced for smoke; runtime copy only unblocks W&B env handling",
    "",
]
Path(status_file).write_text("\n".join(lines), encoding="utf-8")
PY
}

write_manifest() {
  local state="$1"
  local reason="${2:-}"
  python - "$MANIFEST_FILE" "$RUN_NAME" "$state" "$reason" "$OFFICIAL_ROOT" "$ENV_PREFIX" "$BASE_MODEL" "$RUNTIME_ROOT" "$OUTPUT_ROOT" "$WANDB_PROJECT" "$WANDB_GROUP" "$MAX_STEPS" "$MAX_TRAIN_SAMPLES" "$MAX_PREDICT_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

(
    manifest_file,
    run_name,
    state,
    reason,
    official_root,
    env_prefix,
    base_model,
    runtime_root,
    output_root,
    wandb_project,
    wandb_group,
    max_steps,
    max_train_samples,
    max_predict_samples,
) = sys.argv[1:]

manifest = {
    "run_name": run_name,
    "state": state,
    "reason": reason,
    "label": "published-base official-runtime smoke, not ours improvement",
    "base": "O-LoRA official T5-large Standard CL order1 seed1",
    "official_root": official_root,
    "env_prefix": env_prefix,
    "base_model": base_model,
    "runtime_root": runtime_root,
    "output_root": output_root,
    "task_order": ["dbpedia", "amazon", "yahoo", "agnews"],
    "official_hyperparameters_preserved": {
        "learning_rate": "1e-3",
        "epochs_source": "official order_1.sh; smoke uses max_steps=1",
        "train_batch_size": "8",
        "max_source_length": "512",
        "max_target_length": "50",
        "generation_max_length": "50",
        "lamda_1": "0.5",
        "lamda_2": "0",
    },
    "smoke_limits": {
        "max_steps": int(max_steps),
        "max_train_samples": int(max_train_samples),
        "max_predict_samples": int(max_predict_samples),
    },
    "engineering_changes": [
        "run official entry from a runtime copy that respects WANDB_DISABLED instead of hard-disabling W&B",
        "single-GPU process instead of official 8-GPU deepspeed launcher",
        "reduced eval batch and sample caps for smoke only",
    ],
    "wandb": {"project": wandb_project, "group": wandb_group, "mode": "online"},
}
Path(manifest_file).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY
}

preflight() {
  local blockers=()
  [[ -d "$OFFICIAL_ROOT" ]] || blockers+=("missing official root: $OFFICIAL_ROOT")
  [[ -f "$OFFICIAL_ROOT/src/run_uie_lora.py" ]] || blockers+=("missing official entry: $OFFICIAL_ROOT/src/run_uie_lora.py")
  [[ -f "$OFFICIAL_ROOT/scripts/order_1.sh" ]] || blockers+=("missing official order_1.sh")
  [[ -f "$OFFICIAL_ROOT/configs/instruction_config.json" ]] || blockers+=("missing instruction config")
  [[ -d "$OFFICIAL_ROOT/CL_Benchmark" ]] || blockers+=("missing CL_Benchmark")
  [[ -d "$ENV_PREFIX" ]] || blockers+=("missing conda env: $ENV_PREFIX")
  [[ -d "$BASE_MODEL" ]] || blockers+=("missing T5-large snapshot: $BASE_MODEL")
  for task in "${TASKS[@]}"; do
    [[ -f "$OFFICIAL_ROOT/configs/order1_configs/${task}/train_tasks.json" ]] || blockers+=("missing ${task} train config")
    [[ -f "$OFFICIAL_ROOT/configs/order1_configs/${task}/test_tasks.json" ]] || blockers+=("missing ${task} test config")
  done
  if (( ${#blockers[@]} > 0 )); then
    printf '%s\n' "${blockers[@]}" >&2
    write_manifest "blocked" "$(IFS='; '; echo "${blockers[*]}")"
    write_status "blocked" "$(IFS='; '; echo "${blockers[*]}")"
    exit 2
  fi

  conda run -p "$ENV_PREFIX" --no-capture-output python "$OFFICIAL_ROOT/src/run_uie_lora.py" --help >/dev/null
}

prepare_runtime() {
  rm -rf "$RUNTIME_ROOT"
  cp -a "$OFFICIAL_ROOT" "$RUNTIME_ROOT"
  python - "$RUNTIME_ROOT/src/run_uie_lora.py" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = "os.environ['WANDB_DISABLED'] = \"True\""
new = "os.environ['WANDB_DISABLED'] = os.environ.get('WANDB_DISABLED', 'False')"
if old not in text:
    raise SystemExit(f"expected W&B disable line not found in {path}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY
}

run_round() {
  local index="$1"
  local task="$2"
  local model_path="$3"
  local output_dir="${OUTPUT_ROOT}/${index}-${task}"
  local round_name="${RUN_NAME}_round${index}_${task}"

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  WANDB_DISABLED=False \
  WANDB_MODE=online \
  WANDB_PROJECT="$WANDB_PROJECT" \
  WANDB_GROUP="$WANDB_GROUP" \
  WANDB_NAME="$round_name" \
  conda run -p "$ENV_PREFIX" --no-capture-output python "$RUNTIME_ROOT/src/run_uie_lora.py" \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path "$model_path" \
    --data_dir "$RUNTIME_ROOT/CL_Benchmark" \
    --task_config_dir "$RUNTIME_ROOT/configs/order1_configs/${task}" \
    --instruction_file "$RUNTIME_ROOT/configs/instruction_config.json" \
    --instruction_strategy single \
    --output_dir "$output_dir" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-03 \
    --num_train_epochs 1 \
    --max_steps "$MAX_STEPS" \
    --run_name "$round_name" \
    --max_source_length 512 \
    --max_target_length 50 \
    --generation_max_length 50 \
    --max_train_samples "$MAX_TRAIN_SAMPLES" \
    --max_predict_samples "$MAX_PREDICT_SAMPLES" \
    --add_task_name True \
    --add_dataset_name True \
    --overwrite_output_dir \
    --overwrite_cache \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 1 \
    --evaluation_strategy no \
    --save_strategy no \
    --lamda_1 0.5 \
    --lamda_2 0 \
    --lora_dim "$LORA_DIM" \
    --seed 1 \
    --report_to wandb
}

main() {
  write_manifest "preflight" ""
  write_status "preflight" ""
  preflight
  prepare_runtime
  write_manifest "ready" ""
  write_status "ready" ""

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] ${RUN_NAME} is ready"
    return 0
  fi

  write_manifest "running" ""
  write_status "running" ""
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo "[run] ${RUN_NAME}"
  echo "[wandb] project=${WANDB_PROJECT} group=${WANDB_GROUP}"
  local model_path="$BASE_MODEL"
  local idx=1
  for task in "${TASKS[@]}"; do
    echo "[round] ${idx} ${task} model=${model_path}"
    run_round "$idx" "$task" "$model_path"
    model_path="${OUTPUT_ROOT}/${idx}-${task}/adapter"
    idx=$((idx + 1))
  done
  write_manifest "completed" ""
  write_status "completed" ""
}

trap 'write_manifest failed "launcher exited unexpectedly"; write_status failed "launcher exited unexpectedly"' ERR
main "$@"
