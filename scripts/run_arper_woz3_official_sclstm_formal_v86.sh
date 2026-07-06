#!/usr/bin/env bash
# Official ARPER WOZ3 SCLSTM Path B formal anchor (paper/default epochs).
# Uses run_woz3.py with WOZ3 unique-DA split and task order 1,7,0,6,4,2,8.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-arper_woz3_official_sclstm_formal_v86}"
SESSION="${TMUX_SESSION:-lora-ours-arper-v86-formal}"
ARPER_ROOT="${ARPER_ROOT:-${REPO_ROOT}/baselines/advanced_baselines/arper_dialog_nlg/external}"
CONFIG="${CONFIG:-${REPO_ROOT}/results/logs/${RUN_ID}.cfg}"
LOG_PATH="${LOG_PATH:-/root/autodl-tmp/lora-ours-logs/${RUN_ID}.log}"
STATUS_BASENAME="${STATUS_BASENAME:-${RUN_ID}_status}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

mkdir -p /root/autodl-tmp/lora-ours-logs "${REPO_ROOT}/results/logs"
ln -sf "${LOG_PATH}" "${REPO_ROOT}/results/logs/${RUN_ID}.log" 2>/dev/null || true

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 2
  fi
}

require_file "${CONFIG}"
require_file "${ARPER_ROOT}/run_woz3.py"

gpu_compute="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
if [[ -n "${gpu_compute// /}" && "${FORCE}" != "1" ]]; then
  cat > "/root/autodl-tmp/lora-ours-logs/${RUN_ID}_launch.json" <<EOF
{
  "updated_at": "$(date -Iseconds)",
  "run_id": "${RUN_ID}",
  "session": "${SESSION}",
  "state": "queued_gpu_busy",
  "gpu_compute": "${gpu_compute//$'\n'/; }",
  "config": "${CONFIG}"
}
EOF
  ln -sf "/root/autodl-tmp/lora-ours-logs/${RUN_ID}_launch.json" "${REPO_ROOT}/results/logs/${RUN_ID}_launch.json" 2>/dev/null || true
  echo "GPU busy; queued. Active compute:" >&2
  echo "${gpu_compute}" >&2
  exit 75
fi

if tmux has-session -t "${SESSION}" 2>/dev/null && [[ "${FORCE}" != "1" ]]; then
  echo "tmux session ${SESSION} already exists; attach or set FORCE=1" >&2
  exit 76
fi

TRAIN_CMD="cd ${ARPER_ROOT} && CUDA_VISIBLE_DEVICES=${GPU_ID} PYTHONUNBUFFERED=1 ${PYTHON_BIN} -W ignore run_woz3.py \
  --mode train \
  --random_seed 1111 \
  --sv_len_weight 0.5 \
  --adaptive True \
  --ewc_importance 300000 \
  --lr 0.005 \
  --dropout 0 \
  --_lambda 2.0 \
  --config_file ${CONFIG}"

MONITOR_CMD="cd ${REPO_ROOT} && while true; do \
  ${PYTHON_BIN} scripts/monitor_arper_woz3_formal.py \
    --run-id ${RUN_ID} \
    --log-path ${LOG_PATH} \
    --status-basename ${STATUS_BASENAME}; \
  st=\$(python3 -c \"import json; print(json.load(open('results/logs/${STATUS_BASENAME}.json')).get('state',''))\" 2>/dev/null || echo running); \
  [[ \"\$st\" == \"completed_or_stopped\" ]] && break; \
  sleep 60; \
done"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1"
  echo "SESSION=${SESSION}"
  echo "ARPER_ROOT=${ARPER_ROOT}"
  echo "CONFIG=${CONFIG}"
  echo "TRAIN_CMD=${TRAIN_CMD}"
  echo "MONITOR_CMD=${MONITOR_CMD}"
  exit 0
fi

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -n train \
  "cd ${REPO_ROOT} && ${TRAIN_CMD} > ${LOG_PATH} 2>&1; echo EXIT=\$? >> ${LOG_PATH}"
tmux new-window -t "${SESSION}" -n monitor "${MONITOR_CMD}"

cat > "/root/autodl-tmp/lora-ours-logs/${RUN_ID}_launch.json" <<EOF
{
  "updated_at": "$(date -Iseconds)",
  "run_id": "${RUN_ID}",
  "session": "${SESSION}",
  "state": "launched",
  "path": "official ARPER SCLSTM Path B",
  "config": "${CONFIG}",
  "log_path": "${LOG_PATH}",
  "arper_root": "${ARPER_ROOT}",
  "paper_defaults": {
    "n_epochs": 100,
    "batch_size": 64,
    "exemplar_size": 250,
    "task_seq": "1,7,0,6,4,2,8",
    "random_seed": 1111,
    "sv_len_weight": 0.5,
    "ewc_importance": 300000
  },
  "v66_reference": {"bleu4": 0.63231, "ser": 4.817},
  "paper_reference": {"bleu4": 0.701, "ser": 3.63}
}
EOF

echo "Launched ${SESSION}; log=${LOG_PATH}"
