#!/usr/bin/env bash
# CITB InstrDialog++ Ours v1 smoke/formal (public_script_100_25_25 — NOT paper-exact).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FORMAL="${FORMAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
SESSION="${TMUX_SESSION:-lora-ours-citb-pp-ours-v1-smoke}"
LOG_DIR="${LORA_OURS_LOG_DIR:-/root/autodl-tmp/lora-ours-logs}"
GATE_JSON="${REPO_ROOT}/results/manifests/citb_instrdialogpp_split_gate.json"

if [[ "${FORMAL}" == "1" ]]; then
  CONFIG="${CONFIG:-configs/ccfa_three_suite/citb_instrdialogpp_order1_seed1_ours_v1_20260708_formal_strict.yaml}"
  SESSION="${TMUX_SESSION:-lora-ours-citb-pp-ours-v1-formal}"
  LOG_PATH="${LOG_PATH:-${LOG_DIR}/citb_instrdialogpp_ours_v1_20260708_formal.log}"
else
  CONFIG="${CONFIG:-configs/ccfa_three_suite/citb_instrdialogpp_order1_seed1_ours_v1_20260708_smoke_strict.yaml}"
  LOG_PATH="${LOG_PATH:-${LOG_DIR}/citb_instrdialogpp_ours_v1_20260708_smoke.log}"
fi

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/results/logs" "${REPO_ROOT}/results/manifests"
ln -sf "${LOG_PATH}" "${REPO_ROOT}/results/logs/$(basename "${LOG_PATH}")" 2>/dev/null || true

if [[ ! -f "${GATE_JSON}" ]]; then
  python3 "${REPO_ROOT}/scripts/preflight_citb_official_split_counts.py" \
    --citb-root /root/autodl-tmp/lora-baselines-run_v1/external_sources/citb \
    --task-order-dir /root/autodl-tmp/lora-baselines-run_v1/external_sources/citb/data/CIT_data/task_orders/stream=cl_dialogue_long_tasks \
    --order 1 --train 100 --dev 25 --test 25 \
    --output "${GATE_JSON}" || true
fi

if [[ "${FORMAL}" == "1" && "${SKIP_SPLIT_GATE:-0}" != "1" && -f "${GATE_JSON}" ]]; then
  python3 - <<PY
import json, sys
g = json.load(open("${GATE_JSON}"))
if not g.get("all_tasks_meet_target", False):
    print("BLOCKER: InstrDialog++ formal blocked —", g.get("short_task_count"), "tasks under 100/25/25;", file=sys.stderr)
    print("Disclosure: public_script_100_25_25_not_paper_exact (NOT paper 100/50/100)", file=sys.stderr)
    sys.exit(78)
PY
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 CONFIG=${CONFIG} FORMAL=${FORMAL} split=public_script_100_25_25_not_paper_exact"
  python -c "import yaml; yaml.safe_load(open('${REPO_ROOT}/${CONFIG}')); print('config OK')"
  exit 0
fi

CMD="cd ${REPO_ROOT} && python -m core.train --config ${CONFIG} 2>&1 | tee -a ${LOG_PATH}"
if command -v tmux >/dev/null 2>&1 && [[ -z "${NO_TMUX:-}" ]]; then
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session ${SESSION} already exists" >&2
    exit 76
  fi
  tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${CMD}")"
  echo "Launched ${SESSION}; log=${LOG_PATH}"
else
  eval "${CMD}"
fi
