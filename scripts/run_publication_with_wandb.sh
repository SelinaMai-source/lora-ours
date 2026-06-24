#!/usr/bin/env bash
# 投稿向实验：开启 W&B 并跑 paper matrix 子集。
#
# 用法示例：
#   export WANDB_API_KEY=...          # 或 wandb login
#   export WANDB_MODE=online          # 离线: offline
#   bash scripts/run_publication_with_wandb.sh --benchmarks instrdialog --categories main
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export WANDB_PROJECT="${WANDB_PROJECT:-lora-citb-acl}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_GROUP="${WANDB_GROUP:-acl_antioverlap}"

python3 scripts/run_paper_matrix.py \
  --set output.tracking.use_wandb=true \
  --set output.tracking.wandb_project="${WANDB_PROJECT}" \
  --set output.tracking.wandb_mode="${WANDB_MODE}" \
  --set output.tracking.wandb_group="${WANDB_GROUP}" \
  "$@"
