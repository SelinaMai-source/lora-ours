#!/usr/bin/env bash
# ToDCL v1: ADAPTER anchor + Assess-then-Update orthogonal penalty overlay.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TODCL_ROOT="${TODCL_ROOT:-/root/autodl-tmp/lora-baselines-run_v1/external_sources/todcl}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/todcl_legacy_py37/bin/python}"
RUN_ID="${RUN_ID:-todcl_adapter_nlg_ours_assess_overlay_v1_20260708}"
SESSION="${TMUX_SESSION:-lora-ours-todcl-v1-assess-overlay}"
LOG_PATH="${LOG_PATH:-/root/autodl-tmp/lora-ours-logs/${RUN_ID}.log}"
ANCHOR_RUN_ID="${ANCHOR_RUN_ID:-todcl_adapter_nlg_official_anchor_20260706}"
PRIOR_CKPT="${PRIOR_CKPT:-}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
BOUNDED_SMOKE="${BOUNDED_SMOKE:-0}"

ASSESS_THRESHOLD="${ASSESS_THRESHOLD:-0.25}"
ORTHOGONAL_PENALTY="${ORTHOGONAL_PENALTY:-0.1}"
N_EPOCHS="${N_EPOCHS:-10}"
if [[ "${BOUNDED_SMOKE}" == "1" ]]; then
  N_EPOCHS=3
fi

mkdir -p /root/autodl-tmp/lora-ours-logs "${REPO_ROOT}/results/logs"
ln -sf "${LOG_PATH}" "${REPO_ROOT}/results/logs/${RUN_ID}.log" 2>/dev/null || true

# Resolve prior checkpoint from anchor run output
if [[ -z "${PRIOR_CKPT}" ]]; then
  PRIOR_CKPT="$(find /root/autodl-tmp/todcl_official_runs -name '*.ckpt' -o -name 'pytorch_model.bin' 2>/dev/null | head -1 || true)"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 RUN_ID=${RUN_ID} PRIOR_CKPT=${PRIOR_CKPT:-pending_anchor}"
  test -f "${TODCL_ROOT}/train.py"
  exit 0
fi

if [[ -z "${PRIOR_CKPT}" || ! -f "${PRIOR_CKPT}" ]]; then
  echo "Anchor checkpoint not ready; overlay blocked until ${ANCHOR_RUN_ID} completes" >&2
  exit 77
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

CMD="cd ${REPO_ROOT} && CUDA_VISIBLE_DEVICES=${GPU_ID} PYTHONUNBUFFERED=1 \
  OURS_PRIOR_CKPT=${PRIOR_CKPT} OURS_ASSESS_THRESHOLD=${ASSESS_THRESHOLD} \
  OURS_ORTHOGONAL_PENALTY=${ORTHOGONAL_PENALTY} N_EPOCHS=${N_EPOCHS} TODCL_ROOT=${TODCL_ROOT} \
  ${PYTHON_BIN} scripts/todcl_assess_orthogonal_overlay.py 2>&1 | tee -a ${LOG_PATH}"

tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${CMD}")"
echo "Launched ${SESSION}; log=${LOG_PATH}"
