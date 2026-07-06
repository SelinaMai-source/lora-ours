#!/usr/bin/env bash
# ARPER WOZ3 paper-aligned repro: domain-wise (granularity=0) + exemplar 500.
# Thin wrapper around v86 launcher with v87 config defaults.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_ID="${RUN_ID:-arper_woz3_paper_aligned_exemplar500_formal_v87}"
export TMUX_SESSION="${TMUX_SESSION:-lora-ours-arper-v87-formal}"
export CONFIG="${CONFIG:-${REPO_ROOT}/results/logs/${RUN_ID}.cfg}"
export STATUS_BASENAME="${STATUS_BASENAME:-${RUN_ID}_status}"

exec bash "${REPO_ROOT}/scripts/run_arper_woz3_official_sclstm_formal_v86.sh"
