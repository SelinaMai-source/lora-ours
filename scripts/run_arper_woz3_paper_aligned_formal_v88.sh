#!/usr/bin/env bash
# ARPER WOZ3 paper-exact repro v88: config.cfg defaults (domain-wise, exemplar 250, batch 128).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_ID="${RUN_ID:-arper_woz3_paper_aligned_exemplar250_formal_v88}"
export TMUX_SESSION="${TMUX_SESSION:-lora-ours-arper-v88-formal}"
export CONFIG="${CONFIG:-${REPO_ROOT}/results/logs/${RUN_ID}.cfg}"
export STATUS_BASENAME="${STATUS_BASENAME:-${RUN_ID}_status}"

exec bash "${REPO_ROOT}/scripts/run_arper_woz3_official_sclstm_formal_v86.sh"
