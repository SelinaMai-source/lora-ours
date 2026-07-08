#!/usr/bin/env bash
# CITB Ours v1 smoke/formal — Replay50 base + replay_ratio 0.5 SSRG alignment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v1_20260708_smoke_strict.yaml}"
FORMAL="${FORMAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
SESSION="${TMUX_SESSION:-lora-ours-citb-ours-v1-smoke}"
LOG_PATH="${LOG_PATH:-/root/autodl-tmp/lora-ours-logs/citb_ours_v1_20260708_smoke.log}"

if [[ "${FORMAL}" == "1" ]]; then
  CONFIG="${FORMAL_CONFIG:-configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v1_20260708_formal_strict.yaml}"
  SESSION="${TMUX_SESSION:-lora-ours-citb-ours-v1-formal}"
  LOG_PATH="${LOG_PATH:-/root/autodl-tmp/lora-ours-logs/citb_ours_v1_20260708_formal.log}"
fi

mkdir -p /root/autodl-tmp/lora-ours-logs "${REPO_ROOT}/results/logs"
ln -sf "${LOG_PATH}" "${REPO_ROOT}/results/logs/$(basename "${LOG_PATH}")" 2>/dev/null || true

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 CONFIG=${CONFIG} FORMAL=${FORMAL}"
  python -m core.train --config "${REPO_ROOT}/${CONFIG}" --dry-run 2>/dev/null || \
    python -c "import yaml; yaml.safe_load(open('${REPO_ROOT}/${CONFIG}')); print('config OK')"
  exit 0
fi

CMD="cd ${REPO_ROOT} && python -m core.train --config ${CONFIG} 2>&1 | tee -a ${LOG_PATH}"

if [[ -n "${TMUX:-}" ]] && tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists"
  exit 76
fi

if command -v tmux >/dev/null 2>&1 && [[ -z "${NO_TMUX:-}" ]]; then
  tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${CMD}")"
  echo "Launched ${SESSION}; log=${LOG_PATH}"
else
  eval "${CMD}"
fi
