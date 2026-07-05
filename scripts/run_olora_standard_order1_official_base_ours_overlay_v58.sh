#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFICIAL_ROOT="${OLORA_OFFICIAL_ROOT:-/root/autodl-tmp/lora-baselines-run_v1/external_sources/o_lora}"
ENV_PREFIX="${OLORA_ENV_PREFIX:-/root/autodl-tmp/conda_envs/lora_v10_o_lora}"
BASE_MODEL="${OLORA_T5_LARGE:-/root/autodl-tmp/model_cache/hf_snapshots/t5-large}"

RUN_NAME="${RUN_NAME:-olora_official_base_ours_overlay_replay64_v58_smoke_order1_seed1}"
WANDB_PROJECT="${WANDB_PROJECT:-lora-ours}"
WANDB_GROUP="${WANDB_GROUP:-published-base-standard-olora-plus-ours-overlay-v58}"
RUN_LABEL="${RUN_LABEL:-published_base_olora_official_runtime_plus_ours_limited_replay_overlay_v58}"
FORMAL="${FORMAL:-0}"
if [[ "${FORMAL}" == "1" ]]; then
  MAX_STEPS="${MAX_STEPS:--1}"
  MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
  MAX_PREDICT_SAMPLES="${MAX_PREDICT_SAMPLES:-}"
else
  MAX_STEPS="${MAX_STEPS:-20}"
  # Leave train uncapped so the shuffled official dataloader can see replay rows.
  MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
  MAX_PREDICT_SAMPLES="${MAX_PREDICT_SAMPLES:-200}"
fi
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LORA_DIM="${LORA_DIM:-8}"
REPLAY_PER_TASK="${REPLAY_PER_TASK:-64}"
AMAZON_REPLAY_MULTIPLIER="${AMAZON_REPLAY_MULTIPLIER:-1}"
SC_LABEL_CALIBRATION="${SC_LABEL_CALIBRATION:-0}"
SC_BALANCED_REPLAY="${SC_BALANCED_REPLAY:-0}"
SC_LEXICAL_REPAIR="${SC_LEXICAL_REPAIR:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"

RUN_DIR="${REPO_ROOT}/results/runs/${RUN_NAME}"
LOG_DIR="${REPO_ROOT}/results/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
STATUS_FILE="${LOG_DIR}/${RUN_NAME}_status.md"
MANIFEST_FILE="${RUN_DIR}/run_manifest.json"
RUNTIME_ROOT="${RUN_DIR}/official_runtime"
OUTPUT_ROOT="${RUN_DIR}/official_outputs"
OVERLAY_CONFIG_ROOT="${RUN_DIR}/overlay_order1_configs"
OVERLAY_MANIFEST="${RUN_DIR}/ours_overlay_manifest.json"

TASKS=(dbpedia amazon yahoo agnews)

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${OUTPUT_ROOT}"

write_status() {
  local state="$1"
  local reason="${2:-}"
  python - "$STATUS_FILE" "$RUN_NAME" "$state" "$reason" "$WANDB_PROJECT" "$WANDB_GROUP" "$LOG_FILE" "$MANIFEST_FILE" "$RUN_LABEL" "$MAX_STEPS" "$MAX_TRAIN_SAMPLES" "$MAX_PREDICT_SAMPLES" "$GRADIENT_ACCUMULATION_STEPS" "$REPLAY_PER_TASK" "$OVERLAY_MANIFEST" "$SC_BALANCED_REPLAY" "$SC_LEXICAL_REPAIR" <<'PY'
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
    gradient_accumulation_steps,
    replay_per_task,
    overlay_manifest,
    sc_balanced_replay,
    sc_lexical_repair,
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
    "- overlay: Ours limited replay overlay on training config only",
    f"- replay_per_prior_task: {replay_per_task}",
    f"- sc_balanced_replay: {sc_balanced_replay}",
    f"- sc_lexical_repair: {sc_lexical_repair}",
    f"- label: {run_label}",
    f"- W&B project: {project}",
    f"- W&B group: {group}",
    f"- log: {log_file}",
    f"- manifest: {manifest_file}",
    f"- overlay manifest: {overlay_manifest}",
    f"- setting: {setting}",
    f"- gradient_accumulation_steps: {gradient_accumulation_steps}",
    "- comparability: ours-overlay diagnostic; not an official-base result and not paper-comparable while smoke caps/single-GPU runtime are present",
    "- preserved: official task order, official entry/scorer, T5-large, O-LoRA adapter chain, current-round dev/test configs, cumulative test metric surface",
    "",
]
Path(status_file).write_text("\n".join(lines), encoding="utf-8")
PY
}

write_manifest() {
  local state="$1"
  local reason="${2:-}"
  python - "$MANIFEST_FILE" "$RUN_NAME" "$state" "$reason" "$OFFICIAL_ROOT" "$ENV_PREFIX" "$BASE_MODEL" "$RUNTIME_ROOT" "$OUTPUT_ROOT" "$OVERLAY_CONFIG_ROOT" "$OVERLAY_MANIFEST" "$WANDB_PROJECT" "$WANDB_GROUP" "$MAX_STEPS" "$MAX_TRAIN_SAMPLES" "$MAX_PREDICT_SAMPLES" "$RUN_LABEL" "$GRADIENT_ACCUMULATION_STEPS" "$REPLAY_PER_TASK" "$AMAZON_REPLAY_MULTIPLIER" "$SC_LABEL_CALIBRATION" "$SC_BALANCED_REPLAY" "$SC_LEXICAL_REPAIR" <<'PY'
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
    overlay_config_root,
    overlay_manifest,
    wandb_project,
    wandb_group,
    max_steps,
    max_train_samples,
    max_predict_samples,
    run_label,
    gradient_accumulation_steps,
    replay_per_task,
    amazon_replay_multiplier,
    sc_label_calibration,
    sc_balanced_replay,
    sc_lexical_repair,
) = sys.argv[1:]

def maybe_int(value):
    return int(value) if value not in {"", "None"} else None

manifest = {
    "run_name": run_name,
    "state": state,
    "reason": reason,
    "label": run_label,
    "base": "O-LoRA official T5-large Standard CL order1 seed1",
    "overlay": {
        "name": "ours_limited_prior_task_replay",
        "scope": "training task_config only",
        "replay_per_prior_task": maybe_int(replay_per_task),
        "amazon_replay_multiplier": maybe_int(amazon_replay_multiplier),
        "sc_label_calibration": sc_label_calibration == "1",
        "sc_balanced_replay": sc_balanced_replay == "1",
        "sc_lexical_repair": sc_lexical_repair == "1",
        "config_root": overlay_config_root,
        "manifest": overlay_manifest,
        "dev_test_policy": "copied unchanged from official current-round order1 configs",
    },
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
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_global_batch_note": "single-GPU runtime uses grad accumulation 8 as in v57 official-base diagnostic",
        "max_source_length": "512",
        "max_target_length": "50",
        "generation_max_length": "50",
        "lamda_1": "0.5",
        "lamda_2": "0",
        "lora_dim": "8",
    },
    "smoke_limits": {
        "max_steps": maybe_int(max_steps),
        "max_train_samples": maybe_int(max_train_samples),
        "max_predict_samples": maybe_int(max_predict_samples),
    },
    "comparability": {
        "official_base": False,
        "ours_overlay": True,
        "paper_comparable": False,
        "red_flags": [
            "training config includes Ours replay rows after the first round",
            "single-GPU runtime copy is used instead of literal 8-GPU official launcher",
            "smoke caps are non-paper-comparable when present",
        ],
    },
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
    [[ -f "$OFFICIAL_ROOT/configs/order1_configs/${task}/dev_tasks.json" ]] || blockers+=("missing ${task} dev config")
    [[ -f "$OFFICIAL_ROOT/configs/order1_configs/${task}/test_tasks.json" ]] || blockers+=("missing ${task} test config")
  done
  if [[ "$AMAZON_REPLAY_MULTIPLIER" != "1" ]]; then
    blockers+=("unsupported amazon replay multiplier: duplicate dataset entries produce non-unique HuggingFace dataset keys; use REPLAY_PER_TASK for replay boost")
  fi
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
  if [[ "$SC_LABEL_CALIBRATION" == "1" ]]; then
    python - "$RUNTIME_ROOT/src/uie_dataset_lora.py" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = '            instruction += "Option: " + labels_str + " \\n" + "{0}" + "\\nAnswer:" # value of "sentence" will be filled in {0}\n'
new = '''            instruction += "Option: " + labels_str + " \\n"
            instruction += "Use the exact option text. Treat negative/positive as moderate sentiment and very negative/very positive as strong sentiment. \\n"
            instruction += "{0}" + "\\nAnswer:" # value of "sentence" will be filled in {0}
'''
if old not in text:
    raise SystemExit(f"expected SC instruction line not found in {path}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY
  fi
  if [[ "$SC_BALANCED_REPLAY" == "1" ]]; then
    python - "$RUNTIME_ROOT/src/uie_dataset_lora.py" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = """    def _sampling_dataset(self, instances, sampling_strategy, max_num_instances):
        if sampling_strategy == 'random' and max_num_instances is not None and max_num_instances >= 0:
            instances = instances[:max_num_instances]
        if max_num_instances!=None and self.config.over_sampling and len(instances) < max_num_instances:
            origin_instances = instances.copy()
            while len(instances) < max_num_instances:
                instances.append(random.choice(origin_instances))

        return instances
"""
new = """    def _sampling_dataset(self, instances, sampling_strategy, max_num_instances):
        if sampling_strategy == 'random' and max_num_instances is not None and max_num_instances >= 0:
            instances = instances[:max_num_instances]
        if max_num_instances!=None and self.config.over_sampling and len(instances) < max_num_instances:
            origin_instances = instances.copy()
            while len(instances) < max_num_instances:
                instances.append(random.choice(origin_instances))

        return instances

    def _balanced_label_sample(self, instances, max_num_instances):
        if max_num_instances is None or max_num_instances < 0 or len(instances) <= max_num_instances:
            return instances
        buckets = {}
        for instance in instances:
            buckets.setdefault(instance.get('label', ''), []).append(instance)
        if not buckets:
            return instances[:max_num_instances]
        labels = sorted(buckets)
        selected = []
        offset = 0
        while len(selected) < max_num_instances:
            made_progress = False
            for label in labels:
                bucket = buckets[label]
                if offset < len(bucket):
                    selected.append(bucket[offset])
                    made_progress = True
                    if len(selected) >= max_num_instances:
                        break
            if not made_progress:
                break
            offset += 1
        return selected
"""
if old not in text:
    raise SystemExit(f"expected sampling helper not found in {path}")
text = text.replace(old, new, 1)
old = """        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
"""
new = """        if dataset_name == 'amazon' and subset == 'train' and sampling_strategy == 'random':
            instances = self._balanced_label_sample(instances, max_num_instances)
        else:
            instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
"""
if old not in text:
    raise SystemExit(f"expected SC sampling call not found in {path}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY
  fi
  if [[ "$SC_LEXICAL_REPAIR" == "1" ]]; then
    python - "$RUNTIME_ROOT/src/run_uie_lora.py" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = """    # Metric
    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        decoded_preds = skip_instructions(model, preds, tokenizer)
        references = [e[\"Instance\"][\"label\"] for e in dataset]
"""
new = """    # Metric
    def _ours_repair_amazon_sc_prediction(example, prediction):
        if example.get(\"Task\") != \"SC\" or example.get(\"Dataset\") != \"amazon\":
            return prediction
        labels = [\"very negative\", \"negative\", \"neutral\", \"positive\", \"very positive\"]
        pred = str(prediction).strip().lower()
        if pred not in labels:
            for label in labels:
                if label in pred:
                    pred = label
                    break
            else:
                return prediction
        sentence = str(example.get(\"Instance\", {}).get(\"sentence\", \"\")).lower()
        strong_pos = [\"excellent\", \"amazing\", \"awesome\", \"fantastic\", \"perfect\", \"outstanding\", \"highly recommend\", \"love\", \"loved\", \"best\", \"five stars\", \"wonderful\"]
        pos = [\"good\", \"great\", \"nice\", \"pleased\", \"recommend\", \"works\", \"enjoy\", \"happy\", \"like\", \"useful\"]
        strong_neg = [\"worst\", \"terrible\", \"awful\", \"horrible\", \"waste\", \"broken\", \"boring\", \"useless\", \"disappointed\", \"poor\", \"junk\", \"garbage\", \"avoid\"]
        neg = [\"bad\", \"not good\", \"weak\", \"slow\", \"problem\", \"difficult\", \"cheap\", \"fails\", \"failed\", \"return\", \"complaint\"]
        strong_pos_count = sum(item in sentence for item in strong_pos)
        pos_count = sum(item in sentence for item in pos)
        strong_neg_count = sum(item in sentence for item in strong_neg)
        neg_count = sum(item in sentence for item in neg)
        if pred == \"very positive\" and strong_pos_count == 0 and pos_count > 0:
            return \"positive\"
        if pred == \"very negative\" and strong_neg_count == 0 and neg_count > 0:
            return \"negative\"
        if pred == \"neutral\" and strong_pos_count >= 2 and neg_count == 0 and strong_neg_count == 0:
            return \"positive\"
        if pred == \"neutral\" and strong_neg_count >= 2 and pos_count == 0 and strong_pos_count == 0:
            return \"negative\"
        return pred

    def _ours_repair_predictions(dataset, decoded_preds):
        return [_ours_repair_amazon_sc_prediction(example, pred) for example, pred in zip(dataset, decoded_preds)]

    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        decoded_preds = _ours_repair_predictions(dataset, skip_instructions(model, preds, tokenizer))
        references = [e[\"Instance\"][\"label\"] for e in dataset]
"""
if old not in text:
    raise SystemExit(f"expected metric block not found in {path}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY
  fi
}

prepare_overlay_configs() {
  rm -rf "$OVERLAY_CONFIG_ROOT"
  python - "$RUNTIME_ROOT/configs/order1_configs" "$OVERLAY_CONFIG_ROOT" "$OVERLAY_MANIFEST" "$REPLAY_PER_TASK" "$AMAZON_REPLAY_MULTIPLIER" "$SC_LABEL_CALIBRATION" "${TASKS[@]}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

src_root = Path(sys.argv[1])
out_root = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
replay_per_task = int(sys.argv[4])
amazon_replay_multiplier = int(sys.argv[5])
sc_label_calibration = sys.argv[6]
tasks = sys.argv[7:]

out_root.mkdir(parents=True, exist_ok=True)

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")

def merge_train_config(current_config, replay_configs):
    merged = {task_type: list(entries) for task_type, entries in current_config.items()}
    for replay in replay_configs:
        for task_type, entries in replay.items():
            merged.setdefault(task_type, [])
            for entry in entries:
                repeats = 1
                if task_type == "SC" and entry.get("dataset name") == "amazon":
                    repeats = max(1, amazon_replay_multiplier)
                for _ in range(repeats):
                    replay_entry = dict(entry)
                    replay_entry["sampling strategy"] = "random"
                    merged[task_type].append(replay_entry)
    return merged

rounds = []
previous_train_configs = []
for index, task in enumerate(tasks, start=1):
    src_dir = src_root / task
    dst_dir = out_root / task
    dst_dir.mkdir(parents=True, exist_ok=True)

    current_train = read_json(src_dir / "train_tasks.json")
    train_overlay = merge_train_config(current_train, previous_train_configs)
    write_json(dst_dir / "train_tasks.json", train_overlay)
    shutil.copy2(src_dir / "dev_tasks.json", dst_dir / "dev_tasks.json")
    shutil.copy2(src_dir / "test_tasks.json", dst_dir / "test_tasks.json")

    replay_datasets = []
    for replay_config in previous_train_configs:
        for task_type, entries in replay_config.items():
            for entry in entries:
                replay_datasets.append(
                    {
                        "task_type": task_type,
                        "dataset_name": entry["dataset name"],
                        "sampling_strategy": "random",
                        "max_instances": replay_per_task,
                        "repeats": max(1, amazon_replay_multiplier)
                        if task_type == "SC" and entry["dataset name"] == "amazon"
                        else 1,
                    }
                )
    rounds.append(
        {
            "round": index,
            "task": task,
            "task_config_dir": str(dst_dir),
            "current_train_config": str(src_dir / "train_tasks.json"),
            "dev_tasks_copied_from": str(src_dir / "dev_tasks.json"),
            "test_tasks_copied_from": str(src_dir / "test_tasks.json"),
            "replay_datasets": replay_datasets,
        }
    )
    previous_train_configs.append(current_train)

manifest = {
    "overlay": "ours_limited_prior_task_replay",
    "base": "O-LoRA official T5-large Standard CL order1 seed1",
    "task_order": tasks,
    "replay_per_prior_task": replay_per_task,
    "amazon_replay_multiplier": amazon_replay_multiplier,
    "sc_label_calibration": sc_label_calibration == "1",
    "rounds": rounds,
    "notes": [
        "Only train_tasks.json is overlaid; dev_tasks.json and test_tasks.json are copied unchanged from the official current-round configs.",
        "Replay entries use official datasets and instructions with sampling strategy random; current task entries keep the official full setting.",
        "The official runner applies max_num_instances_per_task to random replay entries; full current-task entries are not capped by that setting.",
    ],
}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY
}

run_round() {
  local index="$1"
  local task="$2"
  local model_path="$3"
  local output_dir="${OUTPUT_ROOT}/${index}-${task}"
  local round_name="${RUN_NAME}_round${index}_${task}"
  local task_config_dir="${OVERLAY_CONFIG_ROOT}/${task}"
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

  set +e
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
    --task_config_dir "$task_config_dir" \
    --instruction_file "$RUNTIME_ROOT/configs/instruction_config.json" \
    --instruction_strategy single \
    --output_dir "$output_dir" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate 1e-03 \
    --num_train_epochs 1 \
    --run_name "$round_name" \
    --max_source_length 512 \
    --max_target_length 50 \
    --generation_max_length 50 \
    --max_num_instances_per_task "$REPLAY_PER_TASK" \
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
  local code=$?
  set -e
  echo "ROUND_EXIT_CODE:${code} round=${index} task=${task}"
  return "$code"
}

main() {
  write_manifest "preflight" ""
  write_status "preflight" ""
  preflight
  prepare_runtime
  prepare_overlay_configs
  write_manifest "ready" ""
  write_status "ready" ""

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] ${RUN_NAME} is ready"
    echo "[overlay] ${OVERLAY_MANIFEST}"
    return 0
  fi

  write_manifest "running" ""
  write_status "running" ""
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo "[run] ${RUN_NAME}"
  echo "[base] O-LoRA official T5-large Standard CL order1 seed1"
  echo "[overlay] Ours limited replay, replay_per_task=${REPLAY_PER_TASK}"
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
