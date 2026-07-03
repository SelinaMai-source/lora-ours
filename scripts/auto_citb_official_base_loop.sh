#!/usr/bin/env bash
set -euo pipefail

# Autonomous CITB official-base loop:
# 1. monitor the script-strict 500/50/50 smoke;
# 2. launch the paper-target 500/50/100 run only when explicitly opted in;
# 3. emit AGENT_LOOP_WAKE_LORA_OURS only on blockers/failures/stalls.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-/root/autodl-tmp/citb_official_base_repro}"
LOG_DIR="${OUTPUT_BASE}/logs"
mkdir -p "${LOG_DIR}"

SMOKE_RUN="${SMOKE_RUN:-citb_instrdialog_order1_seed1_official_script_500_50_50_tie_fixed_ft_instr_stage1_v54}"
SMOKE_STATUS_BASENAME="${SMOKE_STATUS_BASENAME:-citb_official_script_500_50_50_tie_fixed_status}"
FORMAL_RUN="${FORMAL_RUN:-citb_instrdialog_order1_seed1_paper_target_500_50_100_ft_instr_stage1_v54}"
FORMAL_SESSION="${FORMAL_SESSION:-citb-paper-target-500-50-100-v54}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
AUTO_LAUNCH_FORMAL="${AUTO_LAUNCH_FORMAL:-0}"

cd "${REPO_ROOT}"

json_get() {
  python - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
if not path.is_file():
    print("")
    raise SystemExit(0)
data = json.loads(path.read_text(encoding="utf-8"))
value = data
for part in key.split("."):
    value = value.get(part, "") if isinstance(value, dict) else ""
print(value)
PY
}

has_active_training() {
  pgrep -af "run_continual_instruct_tuning.py" >/dev/null 2>&1
}

launch_formal() {
  if tmux has-session -t "${FORMAL_SESSION}" 2>/dev/null; then
    return 0
  fi
  if has_active_training; then
    return 1
  fi
  tmux new-session -d -s "${FORMAL_SESSION}" -n train \
    "cd '${REPO_ROOT}' && SPLIT_POLICY=paper_target_500_50_100 RUN_NAME='${FORMAL_RUN}' WANDB_PROJECT='lora-ours' WANDB_RUN_GROUP='citb_instrdialog_order1_paper_target_500_50_100' bash scripts/run_citb_instrdialog_order1_official_base_repro.sh 2>&1 | tee '${LOG_DIR}/${FORMAL_RUN}.log'"
  tmux new-window -t "${FORMAL_SESSION}" -n monitor \
    "cd '${REPO_ROOT}' && while true; do python scripts/monitor_citb_official_base_repro.py --run-name '${FORMAL_RUN}' --basename citb_paper_target_500_50_100_status --stale-minutes 30 --expected-tasks 19 --emit-sentinel; sleep 60; done"
  echo "AGENT_LOOP_WAKE_LORA_OURS {\"scope\":\"citb_official_base_repro\",\"state\":\"formal_started\",\"run_name\":\"${FORMAL_RUN}\",\"tmux\":\"${FORMAL_SESSION}\"}"
}

while true; do
  python scripts/monitor_citb_official_base_repro.py \
    --run-name "${SMOKE_RUN}" \
    --basename "${SMOKE_STATUS_BASENAME}" \
    --stale-minutes 30 \
    --expected-tasks 19 \
    --emit-sentinel > "${LOG_DIR}/${SMOKE_RUN}.auto_status.jsonl" 2>&1 || true

  STATUS_JSON="${REPO_ROOT}/results/logs/${SMOKE_STATUS_BASENAME}.json"
  state="$(json_get "${STATUS_JSON}" state)"
  result_dirs="$(json_get "${STATUS_JSON}" num_result_dirs)"

  case "${state}" in
    completed)
      if [[ "${AUTO_LAUNCH_FORMAL}" != "1" ]]; then
        echo "AGENT_LOOP_WAKE_LORA_OURS {\"scope\":\"citb_official_base_repro\",\"state\":\"smoke_completed_formal_blocked\",\"run_name\":\"${SMOKE_RUN}\",\"reason\":\"AUTO_LAUNCH_FORMAL is not enabled\"}"
        exit 0
      fi
      if launch_formal; then
        exit 0
      fi
      ;;
    failed|stalled|stopped_incomplete)
      echo "AGENT_LOOP_WAKE_LORA_OURS {\"scope\":\"citb_official_base_repro\",\"state\":\"${state}\",\"run_name\":\"${SMOKE_RUN}\",\"result_dirs\":\"${result_dirs}\",\"status_file\":\"results/logs/${SMOKE_STATUS_BASENAME}.md\"}"
      exit 0
      ;;
  esac

  sleep "${CHECK_INTERVAL_SECONDS}"
done
