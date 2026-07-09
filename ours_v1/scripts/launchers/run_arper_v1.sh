#!/usr/bin/env bash
# ARPER WOZ3 v1: v89 SCLSTM anchor + SSRG exemplar selection overlay.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID="${RUN_ID:-arper_woz3_sclstm_v89_ssrg_exemplar_overlay_v1_20260708}"
SESSION="${TMUX_SESSION:-lora-ours-arper-v1-ssrg-overlay}"
BASE_CFG="${BASE_CFG:-${REPO_ROOT}/results/logs/arper_woz3_paper_aligned_exemplar500_batch128_formal_v89.cfg}"
CONFIG="${CONFIG:-${REPO_ROOT}/results/logs/${RUN_ID}.cfg}"
LOG_DIR="${LORA_OURS_LOG_DIR:-/root/autodl-tmp/lora-ours-logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${RUN_ID}.log}"
STATUS_BASENAME="${STATUS_BASENAME:-${RUN_ID}_status}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
BOUNDED_SMOKE="${BOUNDED_SMOKE:-0}"

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/results/logs"

python3 - "$BASE_CFG" "$CONFIG" "$RUN_ID" "$BOUNDED_SMOKE" <<'PY'
import configparser, sys
base, out, run_id, bounded = sys.argv[1:5]
cfg = configparser.ConfigParser()
cfg.read(base)
prefix = cfg["EXPERIMENT"]["experiment_prefix"].rstrip("/")
cfg["EXPERIMENT"]["experiment_prefix"] = prefix.replace(
    "arper_woz3_paper_aligned_exemplar500_batch128_formal_v89", run_id
) + "/"
exp = cfg["EXPERIMENT"]["experiment"]
if "ssrg" not in exp:
    cfg["EXPERIMENT"]["experiment"] = exp.replace("v89", "v89_ssrg")
sel = cfg["EXPERIMENT"]["exemplar_selection"]
if "ssrg" not in sel:
    cfg["EXPERIMENT"]["exemplar_selection"] = "ssrg," + sel
if bounded == "1":
    cfg["TRAINING"]["n_epochs"] = "10"
with open(out, "w") as f:
    cfg.write(f)
print(f"Wrote {out}")
PY

ln -sf "${LOG_PATH}" "${REPO_ROOT}/results/logs/${RUN_ID}.log" 2>/dev/null || true

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 RUN_ID=${RUN_ID}"
  test -f "${CONFIG}"
  exit 0
fi

gpu_compute="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
if [[ -n "${gpu_compute// /}" && "${FORCE}" != "1" ]]; then
  echo "GPU busy; queued ${RUN_ID}" >&2
  exit 75
fi

if tmux has-session -t "${SESSION}" 2>/dev/null && [[ "${FORCE}" != "1" ]]; then
  echo "tmux session ${SESSION} exists" >&2
  exit 76
fi

export ARPER_OVERLAY_CONFIG="${CONFIG}"
TRAIN_CMD="cd ${REPO_ROOT} && CUDA_VISIBLE_DEVICES=${GPU_ID} PYTHONUNBUFFERED=1 ${PYTHON_BIN} ours_v1/suites/arper/arper_ssrg_overlay_train_wrapper.py 2>&1 | tee -a ${LOG_PATH}"
MONITOR_CMD="cd ${REPO_ROOT} && while true; do ${PYTHON_BIN} scripts/monitor_arper_woz3_formal.py --run-id ${RUN_ID} --log-path ${LOG_PATH} --status-basename ${STATUS_BASENAME}; st=\$(${PYTHON_BIN} -c \"import json; print(json.load(open('results/logs/${STATUS_BASENAME}.json')).get('state',''))\" 2>/dev/null || echo running); [[ \"\$st\" == \"completed_or_stopped\" ]] && break; sleep 60; done"

tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${TRAIN_CMD}")"
tmux new-window -t "${SESSION}" -n monitor "bash -lc $(printf '%q' "${MONITOR_CMD}")"
echo "Launched ${SESSION}; log=${LOG_PATH}"
