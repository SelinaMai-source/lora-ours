#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFICIAL_ROOT="${OLORA_OFFICIAL_ROOT:-/root/autodl-tmp/lora-baselines-run_v1/external_sources/o_lora}"
ENV_PREFIX="${OLORA_ENV_PREFIX:-/root/autodl-tmp/conda_envs/lora_v10_o_lora}"
BASE_MODEL="${OLORA_T5_LARGE:-/root/autodl-tmp/model_cache/hf_snapshots/t5-large}"

RUN_NAME="${RUN_NAME:-olora_t5large_standard_order1_seed1_official_base_smoke_v55}"
WANDB_PROJECT="${WANDB_PROJECT:-lora-ours}"
WANDB_GROUP="${WANDB_GROUP:-published-base-standard-olora-order1}"
RUN_LABEL="${RUN_LABEL:-published-base official-runtime smoke, not ours improvement}"
FORMAL="${FORMAL:-0}"
if [[ "${FORMAL}" == "1" ]]; then
  MAX_STEPS="${MAX_STEPS:--1}"
  MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
  MAX_PREDICT_SAMPLES="${MAX_PREDICT_SAMPLES:-}"
else
  MAX_STEPS="${MAX_STEPS:-1}"
  MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-8}"
  MAX_PREDICT_SAMPLES="${MAX_PREDICT_SAMPLES:-16}"
fi
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
  python - "$STATUS_FILE" "$RUN_NAME" "$state" "$reason" "$WANDB_PROJECT" "$WANDB_GROUP" "$LOG_FILE" "$MANIFEST_FILE" "$RUN_LABEL" "$MAX_STEPS" "$MAX_TRAIN_SAMPLES" "$MAX_PREDICT_SAMPLES" <<'PY'
import sys
from datetime import datetime
from pathlib import Path

(
    status_file,
    run_name,
    state,
    reason,
    project,
    group,
    log_file,
    manifest_file,
    run_label,
    max_steps,
    max_train_samples,
    max_predict_samples,
) = sys.argv[1:]

limits = []
if max_steps:
    limits.append(f"max_steps={max_steps}")
if max_train_samples:
    limits.append(f"max_train_samples={max_train_samples}")
if max_predict_samples:
    limits.append(f"max_predict_samples={max_predict_samples}")
setting = "dbpedia -> amazon -> yahoo -> agnews"
if limits:
    setting = f"{setting}; " + "; ".join(limits)

lines = [
    f"# {run_name}",
    "",
    f"- updated: {datetime.now().isoformat(timespec='seconds')}",
    f"- state: {state}",
    f"- reason: {reason}",
    "- selected base: O-LoRA official T5-large Standard CL order1 seed1",
    f"- label: {run_label}",
    f"- W&B project: {project}",
    f"- W&B group: {group}",
    f"- log: {log_file}",
    f"- manifest: {manifest_file}",
    f"- setting: {setting}",
    "- engineering notes: single-GPU runtime; runtime copy only unblocks W&B env handling; sample/step caps are non-paper-comparable when present",
    "",
]
Path(status_file).write_text("\n".join(lines), encoding="utf-8")
PY
}

write_manifest() {
  local state="$1"
  local reason="${2:-}"
  python - "$MANIFEST_FILE" "$RUN_NAME" "$state" "$reason" "$OFFICIAL_ROOT" "$ENV_PREFIX" "$BASE_MODEL" "$RUNTIME_ROOT" "$OUTPUT_ROOT" "$WANDB_PROJECT" "$WANDB_GROUP" "$MAX_STEPS" "$MAX_TRAIN_SAMPLES" "$MAX_PREDICT_SAMPLES" "$RUN_LABEL" <<'PY'
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
    run_label,
) = sys.argv[1:]

def maybe_int(value):
    return int(value) if value not in {"", "None"} else None

manifest = {
    "run_name": run_name,
    "state": state,
    "reason": reason,
    "label": run_label,
    "base": "O-LoRA official T5-large Standard CL order1 seed1",
    "official_root": official_root,
    "env_prefix": env_prefix,
    "base_model": base_model,
    "runtime_root": runtime_root,
    "output_root": output_root,
    "task_order": ["dbpedia", "amazon", "yahoo", "agnews"],
    "official_hyperparameters_preserved": {
        "learning_rate": "1e-3",
        "epochs_source": "official order_1.sh; formal uses one epoch, diagnostic runs may use max_steps/sample caps",
        "train_batch_size": "8",
        "max_source_length": "512",
        "max_target_length": "50",
        "generation_max_length": "50",
        "lamda_1": "0.5",
        "lamda_2": "0",
    },
    "smoke_limits": {
        "max_steps": maybe_int(max_steps),
        "max_train_samples": maybe_int(max_train_samples),
        "max_predict_samples": maybe_int(max_predict_samples),
    },
    "engineering_changes": [
        "run official entry from a runtime copy that respects WANDB_DISABLED instead of hard-disabling W&B",
        "single-GPU process instead of official 8-GPU deepspeed launcher",
        "reduced eval batch and optional sample/step caps for non-paper-comparable diagnostics",
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
  local limit_args=()
  if [[ -n "${MAX_STEPS}" && "${MAX_STEPS}" != "-1" ]]; then
    limit_args+=(--max_steps "${MAX_STEPS}")
  fi
  if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
    limit_args+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
  fi
  if [[ -n "${MAX_PREDICT_SAMPLES}" ]]; then
    limit_args+=(--max_predict_samples "${MAX_PREDICT_SAMPLES}")
  fi

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
    --run_name "$round_name" \
    --max_source_length 512 \
    --max_target_length 50 \
    --generation_max_length 50 \
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
    --report_to wandb \
    "${limit_args[@]}"
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
