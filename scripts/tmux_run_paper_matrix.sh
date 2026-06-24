#!/usr/bin/env bash
# 在 tmux 会话里运行 paper matrix，可断开 SSH 持续训练。
#
# 用法:
#   bash scripts/tmux_run_paper_matrix.sh [SESSION_NAME] --benchmarks "instrdialog" --categories "main"
# 例如:
#   bash scripts/tmux_run_paper_matrix.sh paper-instrdialog --benchmarks "instrdialog" --categories "main"
#
# 连接/断开:
#   tmux attach -t paper-instrdialog
#   Ctrl+B 然后 D
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SESSION="${1:-paper-matrix}"
shift || true

if [ "$#" -eq 0 ]; then
  echo "用法: bash scripts/tmux_run_paper_matrix.sh [SESSION_NAME] <run_paper_matrix.py args...>"
  echo "示例: bash scripts/tmux_run_paper_matrix.sh paper-instrdialog --benchmarks \"instrdialog\" --categories \"main\""
  exit 1
fi

LOG_DIR="$ROOT/results/logs"
mkdir -p "$LOG_DIR"
CONSOLE_LOG="$LOG_DIR/${SESSION}_tmux_console.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux 会话已存在: $SESSION"
  echo "  连接: tmux attach -t $SESSION"
  echo "  或删了重建: tmux kill-session -t $SESSION && bash $0 $SESSION $*"
  exit 0
fi

printf -v RUN_ARGS '%q ' "$@"
CMD="cd '$ROOT' && python3 scripts/run_paper_matrix.py ${RUN_ARGS} 2>&1 | tee -a '$CONSOLE_LOG'"
tmux new-session -d -s "$SESSION" bash -lc "$CMD"

echo "已在后台启动 tmux 会话: $SESSION"
echo "  命令: python3 scripts/run_paper_matrix.py $*"
echo "  终端日志: $CONSOLE_LOG"
echo "  连接: tmux attach -t $SESSION"
