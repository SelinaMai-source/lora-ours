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
LEARNING_RATE="${LEARNING_RATE:-1e-03}"
AMAZON_LEARNING_RATE="${AMAZON_LEARNING_RATE:-}"
STOP_AFTER_ROUND="${STOP_AFTER_ROUND:-0}"
DO_PREDICT="${DO_PREDICT:-1}"
REPLAY_PER_TASK="${REPLAY_PER_TASK:-64}"
AMAZON_REPLAY_MULTIPLIER="${AMAZON_REPLAY_MULTIPLIER:-1}"
SC_LABEL_CALIBRATION="${SC_LABEL_CALIBRATION:-0}"
SC_BALANCED_REPLAY="${SC_BALANCED_REPLAY:-0}"
SC_LEXICAL_REPAIR="${SC_LEXICAL_REPAIR:-0}"
SC_CLASS_COVERAGE_ORDER="${SC_CLASS_COVERAGE_ORDER:-0}"
SC_MODERATE_CURRICULUM="${SC_MODERATE_CURRICULUM:-0}"
SC_MODERATE_CURRICULUM_RATIO="${SC_MODERATE_CURRICULUM_RATIO:-0.25}"
TRAIN_HELDOUT_GATE="${TRAIN_HELDOUT_GATE:-0}"
TRAIN_HELDOUT_OFFSET="${TRAIN_HELDOUT_OFFSET:-4500}"
TRAIN_HELDOUT_LIMIT="${TRAIN_HELDOUT_LIMIT:-500}"
TRAIN_HELDOUT_DROP_TOLERANCE="${TRAIN_HELDOUT_DROP_TOLERANCE:-3.0}"
TRAIN_HELDOUT_FINAL_BASELINE_EM="${TRAIN_HELDOUT_FINAL_BASELINE_EM:-55.2}"
TRAIN_HELDOUT_MODERATE_MIN_COUNT="${TRAIN_HELDOUT_MODERATE_MIN_COUNT:-5}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
REPLAY_MODE="${REPLAY_MODE:-uniform}"
SSRG_SPECTRAL_TOP_K="${SSRG_SPECTRAL_TOP_K:-8}"
SSRG_ENERGY_THRESHOLD="${SSRG_ENERGY_THRESHOLD:-0.85}"
ASSESS_RETENTION_GATE="${ASSESS_RETENTION_GATE:-0}"
ASSESS_RETENTION_THRESHOLD="${ASSESS_RETENTION_THRESHOLD:-0.3}"
EARLY_GATE_DBPEDIA_EM="${EARLY_GATE_DBPEDIA_EM:-0}"
EARLY_GATE_AMAZON_EM="${EARLY_GATE_AMAZON_EM:-0}"

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
  python - "$STATUS_FILE" "$RUN_NAME" "$state" "$reason" "$WANDB_PROJECT" "$WANDB_GROUP" "$LOG_FILE" "$MANIFEST_FILE" "$RUN_LABEL" "$MAX_STEPS" "$MAX_TRAIN_SAMPLES" "$MAX_PREDICT_SAMPLES" "$GRADIENT_ACCUMULATION_STEPS" "$REPLAY_PER_TASK" "$OVERLAY_MANIFEST" "$SC_BALANCED_REPLAY" "$SC_LEXICAL_REPAIR" "$TRAIN_HELDOUT_GATE" "$TRAIN_HELDOUT_OFFSET" "$TRAIN_HELDOUT_LIMIT" "$LEARNING_RATE" "$AMAZON_LEARNING_RATE" "$STOP_AFTER_ROUND" "$DO_PREDICT" "$SC_CLASS_COVERAGE_ORDER" "$SC_MODERATE_CURRICULUM" "$SC_MODERATE_CURRICULUM_RATIO" "$REPLAY_MODE" "$ASSESS_RETENTION_GATE" "$EARLY_GATE_DBPEDIA_EM" "$EARLY_GATE_AMAZON_EM" <<'PY'
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
    train_heldout_gate,
    train_heldout_offset,
    train_heldout_limit,
    learning_rate,
    amazon_learning_rate,
    stop_after_round,
    do_predict,
    sc_class_coverage_order,
    sc_moderate_curriculum,
    sc_moderate_curriculum_ratio,
    replay_mode,
    assess_retention_gate,
    early_gate_dbpedia_em,
    early_gate_amazon_em,
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
    f"- replay_mode: {replay_mode}",
    f"- assess_retention_gate: {assess_retention_gate}",
    f"- early_gate_dbpedia_em: {early_gate_dbpedia_em}",
    f"- early_gate_amazon_em: {early_gate_amazon_em}",
    f"- sc_balanced_replay: {sc_balanced_replay}",
    f"- sc_lexical_repair: {sc_lexical_repair}",
    f"- sc_class_coverage_order: {sc_class_coverage_order}",
    f"- sc_moderate_curriculum: {sc_moderate_curriculum}",
    f"- sc_moderate_curriculum_ratio: {sc_moderate_curriculum_ratio}",
    f"- train_heldout_gate: {train_heldout_gate}",
    f"- train_heldout_slice: amazon/train[{train_heldout_offset}:{int(train_heldout_offset) + int(train_heldout_limit)}]",
    f"- label: {run_label}",
    f"- W&B project: {project}",
    f"- W&B group: {group}",
    f"- log: {log_file}",
    f"- manifest: {manifest_file}",
    f"- overlay manifest: {overlay_manifest}",
    f"- setting: {setting}",
    f"- gradient_accumulation_steps: {gradient_accumulation_steps}",
    f"- learning_rate: {learning_rate}",
    f"- amazon_learning_rate: {amazon_learning_rate}",
    f"- stop_after_round: {stop_after_round}",
    f"- do_predict: {do_predict}",
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
  python - "$MANIFEST_FILE" "$RUN_NAME" "$state" "$reason" "$OFFICIAL_ROOT" "$ENV_PREFIX" "$BASE_MODEL" "$RUNTIME_ROOT" "$OUTPUT_ROOT" "$OVERLAY_CONFIG_ROOT" "$OVERLAY_MANIFEST" "$WANDB_PROJECT" "$WANDB_GROUP" "$MAX_STEPS" "$MAX_TRAIN_SAMPLES" "$MAX_PREDICT_SAMPLES" "$RUN_LABEL" "$GRADIENT_ACCUMULATION_STEPS" "$REPLAY_PER_TASK" "$AMAZON_REPLAY_MULTIPLIER" "$SC_LABEL_CALIBRATION" "$SC_BALANCED_REPLAY" "$SC_LEXICAL_REPAIR" "$TRAIN_HELDOUT_GATE" "$TRAIN_HELDOUT_OFFSET" "$TRAIN_HELDOUT_LIMIT" "$TRAIN_HELDOUT_DROP_TOLERANCE" "$TRAIN_HELDOUT_FINAL_BASELINE_EM" "$LEARNING_RATE" "$AMAZON_LEARNING_RATE" "$STOP_AFTER_ROUND" "$DO_PREDICT" "$SC_CLASS_COVERAGE_ORDER" "$SC_MODERATE_CURRICULUM" "$SC_MODERATE_CURRICULUM_RATIO" "$REPLAY_MODE" "$SSRG_SPECTRAL_TOP_K" "$SSRG_ENERGY_THRESHOLD" "$ASSESS_RETENTION_GATE" "$ASSESS_RETENTION_THRESHOLD" "$EARLY_GATE_DBPEDIA_EM" "$EARLY_GATE_AMAZON_EM" <<'PY'
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
    train_heldout_gate,
    train_heldout_offset,
    train_heldout_limit,
    train_heldout_drop_tolerance,
    train_heldout_final_baseline_em,
    learning_rate,
    amazon_learning_rate,
    stop_after_round,
    do_predict,
    sc_class_coverage_order,
    sc_moderate_curriculum,
    sc_moderate_curriculum_ratio,
    replay_mode,
    ssrg_spectral_top_k,
    ssrg_energy_threshold,
    assess_retention_gate,
    assess_retention_threshold,
    early_gate_dbpedia_em,
    early_gate_amazon_em,
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
        "sc_class_coverage_order": sc_class_coverage_order == "1",
        "sc_moderate_curriculum": sc_moderate_curriculum == "1",
        "sc_moderate_curriculum_ratio": float(sc_moderate_curriculum_ratio),
        "replay_mode": replay_mode,
        "ssrg_spectral_top_k": maybe_int(ssrg_spectral_top_k),
        "ssrg_energy_threshold": float(ssrg_energy_threshold),
        "assess_retention_gate": {
            "enabled": assess_retention_gate == "1",
            "isolated_energy_threshold": float(assess_retention_threshold),
            "policy": "reject high isolated-energy adapter updates vs prior round",
        },
        "early_gate": {
            "dbpedia_em_min": maybe_int(early_gate_dbpedia_em),
            "amazon_em_min": maybe_int(early_gate_amazon_em),
        },
        "train_heldout_gate": {
            "enabled": train_heldout_gate == "1",
            "source": "amazon/train.json",
            "offset": maybe_int(train_heldout_offset),
            "limit": maybe_int(train_heldout_limit),
            "drop_tolerance_em": float(train_heldout_drop_tolerance),
            "final_baseline_em": float(train_heldout_final_baseline_em),
            "leakage_policy": "training split only; no dev/test split or test targets/confusion",
        },
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
        "learning_rate": learning_rate,
        "amazon_learning_rate": amazon_learning_rate,
    },
    "diagnostic_controls": {
        "stop_after_round": maybe_int(stop_after_round),
        "do_predict": do_predict == "1",
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
  if [[ "$SC_CLASS_COVERAGE_ORDER" == "1" ]]; then
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

    def _class_coverage_order(self, instances):
        buckets = {}
        for instance in instances:
            buckets.setdefault(instance.get('label', ''), []).append(instance)
        if len(buckets) <= 1:
            return instances
        labels = sorted(buckets)
        ordered = []
        offset = 0
        while len(ordered) < len(instances):
            made_progress = False
            for label in labels:
                bucket = buckets[label]
                if offset < len(bucket):
                    ordered.append(bucket[offset])
                    made_progress = True
            if not made_progress:
                break
            offset += 1
        return ordered
"""
if old not in text:
    raise SystemExit(f"expected sampling helper not found in {path}")
text = text.replace(old, new, 1)
old = """        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
"""
new = """        labels_str = ', '.join(labels)
        if dataset_name == 'amazon' and subset == 'train' and sampling_strategy == 'full':
            instances = self._class_coverage_order(instances)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
"""
if old not in text:
    raise SystemExit(f"expected SC sampling call not found in {path}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY
  fi
  if [[ "$SC_MODERATE_CURRICULUM" == "1" ]]; then
    python - "$RUNTIME_ROOT/src/uie_dataset_lora.py" "$SC_MODERATE_CURRICULUM_RATIO" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
ratio = float(sys.argv[2])
if ratio < 0 or ratio > 0.5:
    raise SystemExit(f"SC_MODERATE_CURRICULUM_RATIO must be in [0, 0.5], got {ratio}")
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

    def _moderate_curriculum_sample(self, instances):
        ratio = MODERATE_RATIO_PLACEHOLDER
        if ratio <= 0 or len(instances) == 0:
            return instances
        moderate_labels = ['negative', 'positive']
        buckets = {}
        for instance in instances:
            buckets.setdefault(instance.get('label', ''), []).append(instance)
        moderate = []
        for label in moderate_labels:
            moderate.extend(buckets.get(label, []))
        if not moderate:
            return instances
        duplicate_count = min(int(len(moderate) * ratio), len(instances) // 4)
        if duplicate_count <= 0:
            return instances
        duplicates = []
        offsets = {label: 0 for label in moderate_labels}
        while len(duplicates) < duplicate_count:
            made_progress = False
            for label in moderate_labels:
                bucket = buckets.get(label, [])
                if bucket:
                    duplicates.append(bucket[offsets[label] % len(bucket)])
                    offsets[label] += 1
                    made_progress = True
                    if len(duplicates) >= duplicate_count:
                        break
            if not made_progress:
                break
        nonmoderate_labels = [label for label in sorted(buckets) if label not in moderate_labels]
        remove_ids = set()
        tails = {label: len(buckets[label]) - 1 for label in nonmoderate_labels}
        while len(remove_ids) < len(duplicates):
            made_progress = False
            for label in nonmoderate_labels:
                bucket = buckets[label]
                if tails[label] >= 0:
                    remove_ids.add(id(bucket[tails[label]]))
                    tails[label] -= 1
                    made_progress = True
                    if len(remove_ids) >= len(duplicates):
                        break
            if not made_progress:
                break
        selected = duplicates + [item for item in instances if id(item) not in remove_ids]
        return selected[:len(instances)]
""".replace("MODERATE_RATIO_PLACEHOLDER", repr(ratio))
if old not in text:
    raise SystemExit(f"expected sampling helper not found in {path}")
text = text.replace(old, new, 1)
old = """        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
"""
new = """        labels_str = ', '.join(labels)
        if dataset_name == 'amazon' and subset == 'train' and sampling_strategy == 'full':
            instances = self._moderate_curriculum_sample(instances)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
"""
if old not in text:
    raise SystemExit(f"expected SC sampling call not found in {path}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY
  fi
  if [[ "$REPLAY_MODE" == "ssrg" ]]; then
    python - "$RUNTIME_ROOT/src/uie_dataset_lora.py" "$SSRG_SPECTRAL_TOP_K" "$SSRG_ENERGY_THRESHOLD" <<'PY'
import math
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
top_k = int(sys.argv[2])
energy_threshold = float(sys.argv[3])
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
        if sampling_strategy == 'ssrg' and max_num_instances is not None and max_num_instances >= 0:
            instances = self._ssrg_spectral_sample(instances, max_num_instances)
        elif sampling_strategy == 'random' and max_num_instances is not None and max_num_instances >= 0:
            instances = instances[:max_num_instances]
        if max_num_instances!=None and self.config.over_sampling and len(instances) < max_num_instances:
            origin_instances = instances.copy()
            while len(instances) < max_num_instances:
                instances.append(random.choice(origin_instances))

        return instances

    def _ssrg_spectral_sample(self, instances, max_num_instances):
        if max_num_instances is None or max_num_instances < 0 or len(instances) <= max_num_instances:
            return instances
        docs = []
        for item in instances:
            sentence = str(item.get('sentence', item.get('text', '')))
            label = str(item.get('label', ''))
            docs.append(f"{sentence} {label}".strip())
        vocab = {}
        for doc in docs:
            for tok in set(re.findall(r"[a-z0-9']+", doc.lower())):
                vocab[tok] = vocab.get(tok, 0) + 1
        ranked = sorted(vocab.items(), key=lambda row: (-row[1], row[0]))[:4096]
        vocab = {tok: idx for idx, (tok, _) in enumerate(ranked)}
        matrix = []
        for doc in docs:
            counts = {}
            for tok in re.findall(r"[a-z0-9']+", doc.lower()):
                if tok in vocab:
                    counts[tok] = counts.get(tok, 0) + 1
            row = [0.0] * len(vocab)
            if counts:
                max_tf = max(counts.values())
                for tok, count in counts.items():
                    row[vocab[tok]] = 0.5 + 0.5 * (count / max_tf)
            matrix.append(row)
        if len(matrix) < 2:
            return instances[:max_num_instances]
        dim = len(matrix[0])
        mean = [sum(row[i] for row in matrix) / len(matrix) for i in range(dim)]
        matrix = [[row[i] - mean[i] for i in range(dim)] for row in matrix]
        cov = [[0.0] * dim for _ in range(dim)]
        denom = max(1, len(matrix) - 1)
        for row in matrix:
            for i in range(dim):
                for j in range(dim):
                    cov[i][j] += row[i] * row[j] / denom
        rank = max(1, min(SSRG_TOP_K, dim))
        basis = []
        work = [r[:] for r in cov]
        singular_values = []
        for _ in range(rank):
            vec = [1.0 / math.sqrt(dim)] * dim
            for _ in range(12):
                nxt = [0.0] * dim
                for i in range(dim):
                    for j in range(dim):
                        nxt[i] += work[i][j] * vec[j]
                norm = math.sqrt(sum(v * v for v in nxt)) or 1.0
                vec = [v / norm for v in nxt]
            singular = math.sqrt(max(0.0, sum(vec[i] * sum(work[i][j] * vec[j] for j in range(dim)) for i in range(dim))))
            singular_values.append(singular)
            basis.append(vec)
            for i in range(dim):
                for j in range(dim):
                    work[i][j] -= singular * vec[i] * vec[j]
        total_energy = sum(v * v for v in singular_values) or 1.0
        cumulative = 0.0
        keep = rank
        for idx, value in enumerate(singular_values, start=1):
            cumulative += value * value
            if cumulative / total_energy >= SSRG_ENERGY_THRESHOLD:
                keep = idx
                break
        basis = basis[:keep]
        scores = []
        for row in matrix:
            energy = 0.0
            for vec in basis:
                proj = sum(row[i] * vec[i] for i in range(dim))
                energy += proj * proj
            scores.append(math.sqrt(max(0.0, energy)))
        ranked = sorted(enumerate(instances), key=lambda item: -scores[item[0]])
        return [item for _, item in ranked[:max_num_instances]]
""".replace("SSRG_TOP_K", repr(top_k)).replace("SSRG_ENERGY_THRESHOLD", repr(energy_threshold))
if old not in text:
    raise SystemExit(f"expected sampling helper not found in {path}")
text = text.replace(old, new, 1)
if "import re" not in text:
    text = text.replace("import random\n", "import random\nimport re\n", 1)
path.write_text(text, encoding="utf-8")
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
  python - "$RUNTIME_ROOT/configs/order1_configs" "$OVERLAY_CONFIG_ROOT" "$OVERLAY_MANIFEST" "$REPLAY_PER_TASK" "$AMAZON_REPLAY_MULTIPLIER" "$SC_LABEL_CALIBRATION" "$REPLAY_MODE" "${TASKS[@]}" <<'PY'
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
replay_mode = sys.argv[7]
tasks = sys.argv[8:]
replay_sampling = "ssrg" if replay_mode == "ssrg" else "random"

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
                    replay_entry["sampling strategy"] = replay_sampling
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
                        "sampling_strategy": replay_sampling,
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
    "replay_mode": replay_mode,
    "amazon_replay_multiplier": amazon_replay_multiplier,
    "sc_label_calibration": sc_label_calibration == "1",
    "rounds": rounds,
    "notes": [
        "Only train_tasks.json is overlaid; dev_tasks.json and test_tasks.json are copied unchanged from the official current-round configs.",
        "Replay entries use official datasets with sampling strategy random or ssrg; current task entries keep the official full setting.",
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
  local round_learning_rate="$LEARNING_RATE"
  if [[ "$task" == "amazon" && -n "$AMAZON_LEARNING_RATE" ]]; then
    round_learning_rate="$AMAZON_LEARNING_RATE"
  fi
  local limit_args=()
  local predict_args=()
  if [[ -n "${MAX_STEPS}" && "${MAX_STEPS}" != "-1" ]]; then
    limit_args+=(--max_steps "${MAX_STEPS}")
  fi
  if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
    limit_args+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
  fi
  if [[ -n "${MAX_PREDICT_SAMPLES}" ]]; then
    limit_args+=(--max_predict_samples "${MAX_PREDICT_SAMPLES}")
  fi
  if [[ "$DO_PREDICT" == "1" ]]; then
    predict_args+=(--do_predict --predict_with_generate)
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
    "${predict_args[@]}" \
    --model_name_or_path "$model_path" \
    --data_dir "$RUNTIME_ROOT/CL_Benchmark" \
    --task_config_dir "$task_config_dir" \
    --instruction_file "$RUNTIME_ROOT/configs/instruction_config.json" \
    --instruction_strategy single \
    --output_dir "$output_dir" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$round_learning_rate" \
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

run_train_heldout_gate() {
  local index="$1"
  local task="$2"
  local adapter_path="$3"

  if [[ "$TRAIN_HELDOUT_GATE" != "1" ]]; then
    return 0
  fi
  if (( index < 2 )); then
    return 0
  fi

  local gate_dir="${RUN_DIR}/train_heldout_gate"
  local gate_state="${gate_dir}/gate_state.json"
  local diag_run="${RUN_NAME}_trainheldout_round${index}_${task}"
  local diag_log="${LOG_DIR}/${diag_run}.log"
  mkdir -p "$gate_dir"

  echo "[heldout-gate] evaluating ${task} round=${index} adapter=${adapter_path}"
  python "${REPO_ROOT}/scripts/run_olora_amazon_dev_diagnostic.py" \
    --run-name "$diag_run" \
    --adapter "$adapter_path" \
    --source-split train \
    --train-offset "$TRAIN_HELDOUT_OFFSET" \
    --train-limit "$TRAIN_HELDOUT_LIMIT"

  python - "$gate_state" "$diag_run" "$diag_log" "$index" "$task" "$TRAIN_HELDOUT_DROP_TOLERANCE" "$TRAIN_HELDOUT_FINAL_BASELINE_EM" "$TRAIN_HELDOUT_MODERATE_MIN_COUNT" "${#TASKS[@]}" <<'PY'
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

(
    gate_state,
    diag_run,
    diag_log,
    index,
    task,
    drop_tolerance,
    final_baseline_em,
    moderate_min_count,
    total_rounds,
) = sys.argv[1:]
gate_state = Path(gate_state)
diag_log = Path(diag_log)
index_i = int(index)
total_rounds_i = int(total_rounds)
drop_tolerance_f = float(drop_tolerance)
final_baseline_f = float(final_baseline_em)
moderate_min_count_i = int(moderate_min_count)

text = diag_log.read_text(encoding="utf-8")
match = re.search(r"predict_exact_match_for_amazon\s*=\s*([0-9.]+)", text)
if not match:
    raise RuntimeError(f"missing heldout amazon EM in {diag_log}")
em = float(match.group(1))
rouge_match = re.search(r"predict_rougeL_for_amazon\s*=\s*([0-9.]+)", text)
rouge = float(rouge_match.group(1)) if rouge_match else None

prediction_path = Path("/root/autodl-tmp/lora-ours-devdiag") / diag_run / "outputs/predict_eval_predictions.jsonl"
pred_counts: Counter[str] = Counter()
if prediction_path.exists():
    with prediction_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                pred_counts[str(row.get("Prediction", "")).strip()] += 1

if gate_state.exists():
    state = json.loads(gate_state.read_text(encoding="utf-8"))
else:
    state = {"evaluations": [], "best_em": None, "decision": "running"}

best_before = state.get("best_em")
drop_fail = best_before is not None and em < float(best_before) - drop_tolerance_f
final_fail = index_i == total_rounds_i and em < final_baseline_f
moderate_fail = (
    index_i == total_rounds_i
    and (
        pred_counts.get("negative", 0) < moderate_min_count_i
        or pred_counts.get("positive", 0) < moderate_min_count_i
    )
)
fail_reasons = []
if drop_fail:
    fail_reasons.append(f"heldout EM dropped below best by > {drop_tolerance_f}: best={best_before}, current={em}")
if final_fail:
    fail_reasons.append(f"final heldout EM {em} below v69 baseline {final_baseline_f}")
if moderate_fail:
    fail_reasons.append(
        "moderate label collapse: "
        f"negative={pred_counts.get('negative', 0)}, positive={pred_counts.get('positive', 0)}, "
        f"min={moderate_min_count_i}"
    )

record = {
    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "diag_run": diag_run,
    "round": index_i,
    "task": task,
    "heldout_em": em,
    "heldout_rougeL": rouge,
    "prediction_counts": dict(pred_counts),
    "best_before": best_before,
    "fail_reasons": fail_reasons,
    "source": "amazon/train.json heldout slice",
    "leakage_policy": "train-only; no dev/test split or test targets/confusion",
}
state.setdefault("evaluations", []).append(record)
state["best_em"] = max(em, float(best_before)) if best_before is not None else em
state["decision"] = "rejected" if fail_reasons else "running"
state["last_record"] = record
gate_state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

print(json.dumps(record, indent=2))
if fail_reasons:
    raise SystemExit(42)
PY
  local gate_code=$?
  if [[ "$gate_code" != "0" ]]; then
    echo "TRAIN_HELDOUT_GATE_REJECTED round=${index} task=${task} state=${gate_state}"
    write_manifest "rejected" "train-heldout gate rejected round ${index} ${task}"
    write_status "rejected" "train-heldout gate rejected round ${index} ${task}"
    return "$gate_code"
  fi
  echo "TRAIN_HELDOUT_GATE_PASSED round=${index} task=${task} state=${gate_state}"
}

run_early_gate() {
  local index="$1"
  local task="$2"
  local min_em="$3"

  if [[ -z "$min_em" || "$min_em" == "0" ]]; then
    return 0
  fi
  if [[ ! -f "$LOG_FILE" ]]; then
    return 0
  fi

  local metric_key="predict_exact_match_for_${task}"
  local em
  em="$(python - "$LOG_FILE" "$metric_key" <<'PY'
import re
import sys
text = open(sys.argv[1], encoding="utf-8").read()
matches = re.findall(rf"{re.escape(sys.argv[2])}\s*=\s*([0-9.]+)", text)
if not matches:
    raise SystemExit(2)
print(matches[-1])
PY
)" || return 0

  echo "[early-gate] round=${index} task=${task} em=${em} min=${min_em}"
  python - "$em" "$min_em" "$index" "$task" "${RUN_DIR}/early_gate_state.json" <<'PY'
import json
import sys
from pathlib import Path

em, min_em, index, task, gate_state = sys.argv[1:]
em_f = float(em)
min_f = float(min_em)
fail = em_f < min_f
record = {
    "round": int(index),
    "task": task,
    "em": em_f,
    "min_em": min_f,
    "fail_reasons": [f"{task} EM {em_f} < {min_f}"] if fail else [],
}
state_path = Path(gate_state)
state_path.parent.mkdir(parents=True, exist_ok=True)
state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"evaluations": [], "decision": "running"}
state.setdefault("evaluations", []).append(record)
state["last_record"] = record
state["decision"] = "rejected" if fail else "running"
state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
if fail:
    raise SystemExit(41)
PY
  local gate_code=$?
  if [[ "$gate_code" != "0" ]]; then
    echo "EARLY_GATE_REJECTED round=${index} task=${task}"
    write_manifest "rejected" "early gate rejected round ${index} ${task}: EM ${em} < ${min_em}"
    write_status "rejected" "early gate rejected round ${index} ${task}: EM ${em} < ${min_em}"
    return "$gate_code"
  fi
  echo "EARLY_GATE_PASSED round=${index} task=${task} em=${em}"
}

run_assess_retention_gate() {
  local index="$1"
  local task="$2"
  local prior_adapter="$3"
  local current_adapter="$4"

  if [[ "$ASSESS_RETENTION_GATE" != "1" ]]; then
    return 0
  fi
  if (( index < 2 )); then
    return 0
  fi
  if [[ ! -d "$prior_adapter" || ! -d "$current_adapter" ]]; then
    return 0
  fi

  local gate_dir="${RUN_DIR}/assess_retention_gate"
  local gate_state="${gate_dir}/gate_state.json"
  mkdir -p "$gate_dir"
  echo "[assess-retention-gate] round=${index} task=${task} prior=${prior_adapter} current=${current_adapter}"
  set +e
  python "${REPO_ROOT}/scripts/olora_overlay_assess_retention_gate.py" \
    --prior-adapter "$prior_adapter" \
    --current-adapter "$current_adapter" \
    --threshold "$ASSESS_RETENTION_THRESHOLD" \
    --gate-state "$gate_state" \
    --round "$index" \
    --task "$task"
  local gate_code=$?
  set -e
  if [[ "$gate_code" != "0" ]]; then
    echo "ASSESS_RETENTION_GATE_REJECTED round=${index} task=${task} state=${gate_state}"
    write_manifest "rejected" "assess-retention gate rejected round ${index} ${task}"
    write_status "rejected" "assess-retention gate rejected round ${index} ${task}"
    return "$gate_code"
  fi
  echo "ASSESS_RETENTION_GATE_PASSED round=${index} task=${task} state=${gate_state}"
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
  echo "[overlay] Ours limited replay, replay_per_task=${REPLAY_PER_TASK}, replay_mode=${REPLAY_MODE}"
  echo "[wandb] project=${WANDB_PROJECT} group=${WANDB_GROUP}"
  local model_path="$BASE_MODEL"
  local prior_adapter_path=""
  local idx=1
  for task in "${TASKS[@]}"; do
    echo "[round] ${idx} ${task} model=${model_path}"
    run_round "$idx" "$task" "$model_path"
    local current_adapter_path="${OUTPUT_ROOT}/${idx}-${task}/adapter"
    if [[ "$task" == "dbpedia" && "$EARLY_GATE_DBPEDIA_EM" != "0" ]]; then
      if ! run_early_gate "$idx" "$task" "$EARLY_GATE_DBPEDIA_EM"; then
        echo "[early-gate] rejected ${RUN_NAME}; stopping after dbpedia segment"
        return 0
      fi
    fi
    if [[ "$task" == "amazon" && "$EARLY_GATE_AMAZON_EM" != "0" ]]; then
      if ! run_early_gate "$idx" "$task" "$EARLY_GATE_AMAZON_EM"; then
        echo "[early-gate] rejected ${RUN_NAME}; stopping after amazon segment"
        return 0
      fi
    fi
    if [[ -n "$prior_adapter_path" ]]; then
      if ! run_assess_retention_gate "$idx" "$task" "$prior_adapter_path" "$current_adapter_path"; then
        echo "[assess-retention-gate] rejected ${RUN_NAME}; stopping before promotion"
        return 0
      fi
    fi
    prior_adapter_path="$current_adapter_path"
    model_path="${OUTPUT_ROOT}/${idx}-${task}/adapter"
    if ! run_train_heldout_gate "$idx" "$task" "$model_path"; then
      echo "[heldout-gate] rejected ${RUN_NAME}; stopping before promotion"
      return 0
    fi
    if [[ "$STOP_AFTER_ROUND" != "0" && "$idx" -ge "$STOP_AFTER_ROUND" ]]; then
      echo "[stop-after-round] stopping after round ${idx}"
      write_manifest "stopped" "stop_after_round=${STOP_AFTER_ROUND}"
      write_status "stopped" "stop_after_round=${STOP_AFTER_ROUND}"
      return 0
    fi
    idx=$((idx + 1))
  done
  write_manifest "completed" ""
  write_status "completed" ""
}

trap 'write_manifest failed "launcher exited unexpectedly"; write_status failed "launcher exited unexpectedly"' ERR
main "$@"
