#!/usr/bin/env bash
# ToDCL official ADAPTER NLG 37-domain anchor reproduction.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TODCL_ROOT="${TODCL_ROOT:-/root/autodl-tmp/lora-baselines-run_v1/external_sources/todcl}"
GPT2_LOCAL="${GPT2_LOCAL:-${TODCL_ROOT}/gpt2}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/todcl_legacy_py37/bin/python}"
RUN_ID="${RUN_ID:-todcl_adapter_nlg_official_anchor_20260706}"
SESSION="${TMUX_SESSION:-lora-ours-todcl-adapter-anchor}"
OUTPUT_BASE="${OUTPUT_BASE:-/root/autodl-tmp/todcl_official_runs}"
LOG_PATH="${LOG_PATH:-/root/autodl-tmp/lora-ours-logs/${RUN_ID}.log}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
FORCE="${FORCE:-0}"
BOUNDED_SMOKE="${BOUNDED_SMOKE:-0}"

mkdir -p /root/autodl-tmp/lora-ours-logs "${OUTPUT_BASE}"

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 2
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 2
  fi
}

require_dir "${TODCL_ROOT}"
require_file "${TODCL_ROOT}/train.py"
require_file "${PYTHON_BIN}"
require_file "${GPT2_LOCAL}/pytorch_model.bin"
require_file "${GPT2_LOCAL}/config.json"

PREFLIGHT_JSON="/root/autodl-tmp/lora-ours-logs/${RUN_ID}_preflight.json"
ln -sf "${PREFLIGHT_JSON}" "${REPO_ROOT}/results/logs/${RUN_ID}_preflight.json" 2>/dev/null || true

run_preflight() {
  "${PYTHON_BIN}" - <<'PY' "${TODCL_ROOT}" "${PREFLIGHT_JSON}"
import json, os, sys
from pathlib import Path
todcl_root, out_path = sys.argv[1:3]
os.chdir(todcl_root)
sys.path.insert(0, todcl_root)
from utils.preprocess import get_datasets
out = get_datasets(dataset_list=['TM19','TM20','MWOZ','SGD'], setting='single', verbose=False, develop=False)
train = out['TOTAL']['train']
dev = out['TOTAL']['dev']
test = out['TOTAL']['test']
bydomain = out['BYDOMAIN']
payload = {
    "status": "passed",
    "cwd": os.getcwd(),
    "train_total": len(train),
    "dev_total": len(dev),
    "test_total": len(test),
    "bydomain_train": len(bydomain['train']),
    "bydomain_dev": len(bydomain['dev']),
    "bydomain_test": len(bydomain['test']),
}
Path(out_path).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload))
PY
}

echo "Running ToDCL dataloader preflight..."
run_preflight | tee "/root/autodl-tmp/lora-ours-logs/${RUN_ID}_preflight.stdout"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "PREFLIGHT_ONLY=1; skipping launch"
  exit 0
fi

gpu_compute="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
if [[ -n "${gpu_compute// /}" && "${FORCE}" != "1" ]]; then
  cat > "/root/autodl-tmp/lora-ours-logs/${RUN_ID}_launch.json" <<EOF
{
  "updated_at": "$(date -Iseconds)",
  "run_id": "${RUN_ID}",
  "state": "queued_gpu_busy",
  "session": "${SESSION}",
  "preflight_json": "${PREFLIGHT_JSON}"
}
EOF
  echo "GPU busy; anchor queued." >&2
  exit 75
fi

if tmux has-session -t "${SESSION}" 2>/dev/null && [[ "${FORCE}" != "1" ]]; then
  echo "tmux session ${SESSION} already exists" >&2
  exit 76
fi

TRAIN_ARGS=(
  --task_type NLG
  --CL ADAPTER
  --bottleneck_size 50
  --lr 6.25e-3
  --n_epochs 10
  --train_batch_size 10
  --gradient_accumulation_steps 8
  --dataset_list SGD,TM19,TM20,MWOZ
  --setting single
  --seed 1
  --model_checkpoint "${GPT2_LOCAL}"
)

if [[ "${BOUNDED_SMOKE}" == "1" ]]; then
  TRAIN_ARGS+=(--debug)
  RUN_ID="${RUN_ID}_bounded_smoke"
  LOG_PATH="/root/autodl-tmp/lora-ours-logs/${RUN_ID}.log"
fi

TRAIN_CMD="cd ${TODCL_ROOT} && CUDA_VISIBLE_DEVICES=${GPU_ID} PYTHONUNBUFFERED=1 ${PYTHON_BIN} train.py ${TRAIN_ARGS[*]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 SESSION=${SESSION} TRAIN_CMD=${TRAIN_CMD}"
  exit 0
fi

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -n train \
  "cd ${TODCL_ROOT} && ${TRAIN_CMD} > ${LOG_PATH} 2>&1; echo EXIT=\$? >> ${LOG_PATH}"

cat > "/root/autodl-tmp/lora-ours-logs/${RUN_ID}_launch.json" <<EOF
{
  "updated_at": "$(date -Iseconds)",
  "run_id": "${RUN_ID}",
  "session": "${SESSION}",
  "state": "launched",
  "method": "ADAPTER NLG official",
  "domains": 37,
  "log_path": "${LOG_PATH}",
  "paper_reference": {"bleu": 21.7719, "eer": 0.163975}
}
EOF

echo "Launched ${SESSION}; log=${LOG_PATH}"
