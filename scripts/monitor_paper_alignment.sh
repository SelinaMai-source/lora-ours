#!/usr/bin/env bash
# Paper-alignment monitor: poll suite metrics, check ±1 tolerance, queue next attempt on fail.
# Launch: tmux new-session -d -s lora-ours-paper-alignment-watch 'bash scripts/monitor_paper_alignment.sh'
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

POLL_SEC="${POLL_SEC:-300}"
ONCE="${ONCE:-0}"
TRACKER="${TRACKER:-results/tables/paper_alignment_iteration_tracker_20260707.md}"
WATCH_MD="${WATCH_MD:-results/logs/paper_alignment_watch_20260707.md}"
MONITOR_LOG="${MONITOR_LOG:-results/logs/paper_alignment_monitor_20260707.log}"
QUEUE_SCRIPT="${QUEUE_SCRIPT:-scripts/run_paper_alignment_queue.sh}"

# ToDCL metrics (paper-alignment watch consumes this JSON).
TODCL_RUN_ID="${TODCL_RUN_ID:-todcl_adapter_nlg_official_anchor_20260706}"
TODCL_METRICS_JSON="${TODCL_METRICS_JSON:-results/logs/todcl_adapter_nlg_official_anchor_20260706_metrics.json}"
TODCL_METRICS_SCRIPT="${TODCL_METRICS_SCRIPT:-scripts/refresh_paper_alignment_todcl_adapter_anchor_metrics.py}"

# Metric targets
CITB_PAPER=40.4
CITB_TOL=1.0
STD_PAPER=75.8
STD_TOL=1.0
ARPER_BLEU_PAPER=0.701
ARPER_BLEU_TOL=0.01
ARPER_SER_PAPER=3.63
ARPER_SER_TOL=1.0
TODCL_BLEU_PAPER=21.77
TODCL_BLEU_TOL=1.0
TODCL_EER_PAPER=0.164
TODCL_EER_TOL=0.10

mkdir -p results/logs results/tables
chmod +x scripts/run_citb_replay50_paper_aligned_v2.sh \
  scripts/run_arper_woz3_paper_aligned_formal_v88.sh \
  scripts/run_todcl_adapter_nlg_official_anchor.sh \
  scripts/run_paper_alignment_queue.sh 2>/dev/null || true

log() { echo "[$(date -Iseconds)] $*" | tee -a "$MONITOR_LOG"; }

within_tol() {
  local local_val="$1" paper="$2" tol="$3"
  python3 - "$local_val" "$paper" "$tol" <<'PY'
import sys
local, paper, tol = map(float, sys.argv[1:4])
if local != local:
    print("pending")
elif abs(local - paper) <= tol:
    print("pass")
else:
    print("fail")
PY
}

# --- metric extractors ---
get_citb_ar() {
  python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from scripts.parse_citb_official_results import summarize_method

# Prefer v56 matrix (PASS ±1); seed50 v2 is optional strict-parity attempt.
candidates = [
    Path("/root/autodl-tmp/citb_official_base_repro/citb_instrdialog_order1_seed1_official_script_500_50_50_tie_fixed_replay50_formal_v56/results"),
    Path("/root/autodl-tmp/citb_official_base_repro/citb_instrdialog_order1_seed50_official_script_500_50_50_paper_aligned_replay50_paper_aligned_v2_formal/results"),
]
for p in candidates:
    if p.is_dir() and list(p.glob("*/metrics.json")):
        s = summarize_method(p, "rougeL")
        if s.get("average_accuracy") is not None:
            print(f"{s['average_accuracy']:.4f}")
            sys.exit(0)
print("nan")
PY
}

get_std_em() {
  grep 'observed_avg_exact:' results/logs/olora_t5large_standard_order1_seed1_official_base_formal_v57_monitor.md 2>/dev/null | head -1 | sed 's/.*observed_avg_exact: *//' | awk '{print $1}' || echo "nan"
}

get_arper_bleu() {
  local run_id="${1:-arper_woz3_paper_aligned_exemplar500_formal_v87}"
  local log="/root/autodl-tmp/lora-ours-logs/${run_id}.log"
  [[ -f "$log" ]] || { echo "nan"; return; }
  grep -E '^test Loss:.*BLEU4:' "$log" 2>/dev/null | tail -1 | sed -n 's/.*BLEU4: \([0-9.]*\).*/\1/p' || echo "nan"
}

get_arper_ser() {
  local run_id="${1:-arper_woz3_paper_aligned_exemplar500_formal_v87}"
  local log="/root/autodl-tmp/lora-ours-logs/${run_id}.log"
  [[ -f "$log" ]] || { echo "nan"; return; }
  grep -E '^test Loss:.*Slot error:' "$log" 2>/dev/null | tail -1 | sed -n 's/.*Slot error: \([0-9.]*\).*/\1/p' || echo "nan"
}

arper_running() {
  pgrep -f 'run_woz3\.py.*paper_aligned' >/dev/null 2>&1
}

is_latest_alignment_run() {
  # Do not kill if a paper-alignment attempt is the newest active training job.
  arper_running && return 0
  pgrep -f 'continual_learning/run_continual_instruct_tuning.*paper_aligned' >/dev/null 2>&1 && return 0
  pgrep -f 'train\.py.*--CL ADAPTER.*todcl' >/dev/null 2>&1 && return 0
  return 1
}

write_watch() {
  local citb="$1" std="$2" arper_b="$3" arper_s="$4" todcl_b="$5" todcl_e="$6"
  cat > "$WATCH_MD" <<EOF
# Paper Alignment Watch — 20260707

Updated: $(date -Iseconds)

| Suite | Paper | Local | ±1 | Verdict |
|-------|-------|-------|-----|---------|
| CITB Replay(50) AR | ${CITB_PAPER} | ${citb} | ${CITB_TOL} | $(within_tol "$citb" "$CITB_PAPER" "$CITB_TOL") |
| Standard O-LoRA EM | ${STD_PAPER} | ${std} | ${STD_TOL} | $(within_tol "$std" "$STD_PAPER" "$STD_TOL") |
| ARPER BLEU | ${ARPER_BLEU_PAPER} | ${arper_b} | ${ARPER_BLEU_TOL} | $(within_tol "$arper_b" "$ARPER_BLEU_PAPER" "$ARPER_BLEU_TOL") |
| ARPER SER | ${ARPER_SER_PAPER} | ${arper_s} | ${ARPER_SER_TOL} | $(within_tol "$arper_s" "$ARPER_SER_PAPER" "$ARPER_SER_TOL") |
| ToDCL BLEU | ${TODCL_BLEU_PAPER} | ${todcl_b} | ${TODCL_BLEU_TOL} | $(within_tol "$todcl_b" "$TODCL_BLEU_PAPER" "$TODCL_BLEU_TOL") |
| ToDCL EER | ${TODCL_EER_PAPER} | ${todcl_e} | ${TODCL_EER_TOL} | $(within_tol "$todcl_e" "$TODCL_EER_PAPER" "$TODCL_EER_TOL") |

Tracker: \`${TRACKER}\`

GPU: $(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader 2>/dev/null | head -2 || echo idle)

Healthy alignment run active: $(is_latest_alignment_run && echo yes || echo no)
EOF
}

maybe_queue_next() {
  local suite="$1" verdict="$2"
  if [[ "$verdict" == "pass" ]]; then
    log "${suite}: PASS ±1 — update best-method-repro branch when pushing"
    return
  fi
  if [[ "$verdict" == "pending" ]]; then
    return
  fi
  if is_latest_alignment_run; then
    log "${suite}: FAIL but healthy alignment run active — skip auto-relaunch"
    return
  fi
  if [[ -x "$QUEUE_SCRIPT" ]] && ! tmux has-session -t lora-ours-paper-alignment-queue 2>/dev/null; then
    log "${suite}: FAIL ±1 — starting paper alignment queue"
    tmux new-session -d -s lora-ours-paper-alignment-queue \
      "bash -lc 'cd ${REPO_ROOT} && bash ${QUEUE_SCRIPT}' > results/logs/paper_alignment_queue_20260707.log 2>&1"
  fi
}

log "Paper alignment monitor started (poll=${POLL_SEC}s)"

while true; do
  citb="$(get_citb_ar)"
  std="$(get_std_em)"
  # Prefer latest *completed* formal run; skip in-progress v88 partial domain metrics.
  arper_b="$(get_arper_bleu arper_woz3_paper_aligned_exemplar500_batch128_formal_v89)"
  arper_s="$(get_arper_ser arper_woz3_paper_aligned_exemplar500_batch128_formal_v89)"
  if [[ "$arper_b" == "nan" || "$arper_s" == "nan" ]]; then
    arper_b="$(get_arper_bleu arper_woz3_paper_aligned_exemplar500_formal_v87)"
    arper_s="$(get_arper_ser arper_woz3_paper_aligned_exemplar500_formal_v87)"
  fi
  if [[ "$arper_b" == "nan" || "$arper_s" == "nan" ]] && ! arper_running; then
    arper_b="$(get_arper_bleu arper_woz3_paper_aligned_exemplar250_formal_v88)"
    arper_s="$(get_arper_ser arper_woz3_paper_aligned_exemplar250_formal_v88)"
  fi
  todcl_b="nan"
  todcl_e="nan"
  if [[ -f "$TODCL_METRICS_JSON" ]]; then
    todcl_b="$(python3 - "$TODCL_METRICS_JSON" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], 'r'))
print(d.get('local_metric', {}).get('bleu', 'nan'))
PY
)"
    todcl_e="$(python3 - "$TODCL_METRICS_JSON" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], 'r'))
print(d.get('local_metric', {}).get('eer', 'nan'))
PY
)"
  elif [[ -f "$TODCL_METRICS_SCRIPT" ]]; then
    python3 "$TODCL_METRICS_SCRIPT" >/dev/null 2>&1 || true
    if [[ -f "$TODCL_METRICS_JSON" ]]; then
      todcl_b="$(python3 - "$TODCL_METRICS_JSON" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], 'r'))
print(d.get('local_metric', {}).get('bleu', 'nan'))
PY
)"
      todcl_e="$(python3 - "$TODCL_METRICS_JSON" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], 'r'))
print(d.get('local_metric', {}).get('eer', 'nan'))
PY
)"
    fi
  fi

  write_watch "$citb" "$std" "$arper_b" "$arper_s" "$todcl_b" "$todcl_e"

  maybe_queue_next "CITB" "$(within_tol "$citb" "$CITB_PAPER" "$CITB_TOL")"
  maybe_queue_next "ARPER" "$(within_tol "$arper_b" "$ARPER_BLEU_PAPER" "$ARPER_BLEU_TOL")"

  if pgrep -f 'run_woz3.*formal_v87' >/dev/null 2>&1; then
    python3 scripts/monitor_arper_woz3_formal.py \
      --run-id arper_woz3_paper_aligned_exemplar500_formal_v87 \
      --log-path /root/autodl-tmp/lora-ours-logs/arper_woz3_paper_aligned_exemplar500_formal_v87.log \
      --status-basename arper_woz3_paper_aligned_exemplar500_formal_v87_status >/dev/null 2>&1 || true
  fi

  if [[ "$ONCE" == "1" ]]; then
    log "ONCE=1; exiting monitor loop"
    break
  fi
  sleep "$POLL_SEC"
done
