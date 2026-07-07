#!/usr/bin/env bash
# ARPER WOZ3 v89: domain-wise + exemplar 500 + batch 128 (hybrid after v88 paper-exact fail).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_ID="${RUN_ID:-arper_woz3_paper_aligned_exemplar500_batch128_formal_v89}"
export TMUX_SESSION="${TMUX_SESSION:-lora-ours-arper-v89-formal}"
export CONFIG="${CONFIG:-${REPO_ROOT}/results/logs/${RUN_ID}.cfg}"
export STATUS_BASENAME="${STATUS_BASENAME:-${RUN_ID}_status}"

exec bash "${REPO_ROOT}/scripts/run_arper_woz3_official_sclstm_formal_v86.sh"
