#!/usr/bin/env bash
# ARPER WOZ3 v90: GPU-tuned next-run launcher (batch 1024 + optional cudnn.benchmark).
# Use ONLY after v89 completes or if v89 FAILs gate — do not kill a healthy v89 mid-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_ID="${RUN_ID:-arper_woz3_paper_aligned_exemplar500_batch1024_gpu_tuned_formal_v90}"
export TMUX_SESSION="${TMUX_SESSION:-lora-ours-arper-v90-formal}"
export CONFIG="${CONFIG:-${REPO_ROOT}/results/logs/${RUN_ID}.cfg}"
export STATUS_BASENAME="${STATUS_BASENAME:-${RUN_ID}_status}"

# Enable cuDNN autotune for throughput (disables strict determinism; v90 only).
export ARPER_CUDNN_BENCHMARK="${ARPER_CUDNN_BENCHMARK:-1}"

# Soft cap allocator fragmentation; leave ~20% VRAM headroom vs 80% batch probe cap.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec bash "${REPO_ROOT}/scripts/run_arper_woz3_official_sclstm_formal_v86.sh"
