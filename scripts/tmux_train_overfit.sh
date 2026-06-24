#!/usr/bin/env bash
# 在 tmux 会话里跑 overfit-8 训练，可断开 SSH；断点续训见下方说明。
#
# 用法:
#   bash scripts/tmux_train_overfit.sh [SESSION_NAME] [CONFIG_PATH]
# 默认:
#   SESSION_NAME=lora-overfit
#   CONFIG_PATH=configs/behavior_gate_manual_qkvo.yaml
#
# 连接/断开:
#   tmux attach -t lora-overfit    # 回到会话
#   Ctrl+B 然后 D                  # detach（训练继续）
#
# 断点续训:
#   1) 保持与上次相同的 output.run_name（同一 results/runs/<run_name>/ 目录）
#   2) 在 YAML 的 debug_tools 里设 overfit_resume: true
#   3) 再执行本脚本（或 attach 后在同目录手动 python core/train.py --config ...）
#   会从 overfit_state.json 的 last_completed_step+1 继续，并加载 lora_ckpt/latest/
#
# 全新重跑同一 run 目录:
#   debug_tools.overfit_fresh_start: true 且 overfit_resume: false（会删该次 run 下 overfit8 的 csv/state/ckpt）
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SESSION="${1:-lora-overfit}"
CONFIG="${2:-configs/behavior_gate_manual_qkvo.yaml}"
LOG_DIR="$ROOT/results/logs"
mkdir -p "$LOG_DIR"
CONSOLE_LOG="$LOG_DIR/${SESSION}_tmux_console.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux 会话已存在: $SESSION"
  echo "  连接: tmux attach -t $SESSION"
  echo "  或删了重建: tmux kill-session -t $SESSION && bash $0 $@"
  exit 0
fi

CMD="cd '$ROOT' && python3 core/train.py --config '$CONFIG' 2>&1 | tee -a '$CONSOLE_LOG'"
tmux new-session -d -s "$SESSION" bash -lc "$CMD"

echo "已在后台启动 tmux 会话: $SESSION"
echo "  配置: $CONFIG"
echo "  终端日志: $CONSOLE_LOG"
echo "  连接: tmux attach -t $SESSION"
