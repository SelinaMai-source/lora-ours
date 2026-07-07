#!/usr/bin/env bash
# Strict paper reproduction iteration loop with ±1 gate per suite.
# GPU serial; does not kill healthy training jobs.
#
# Usage:
#   tmux new-session -d -s lora-ours-strict-paper-repro \
#     'bash scripts/run_strict_paper_repro_iteration.sh'
#
# Env:
#   ITERATION=1          — label for logs
#   POLL_SEC=120
#   SKIP_TODCL=0         — set 1 if ToDCL already running/done externally
#   FORCE_CITB_STAGE1=0  — set 1 to train Stage-1 even if AR already ±1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ITERATION="${ITERATION:-1}"
POLL_SEC="${POLL_SEC:-120}"
SKIP_TODCL="${SKIP_TODCL:-0}"
FORCE_CITB_STAGE1="${FORCE_CITB_STAGE1:-0}"
LOG="${LOG:-results/logs/strict_paper_repro_iteration_${ITERATION}_$(date +%Y%m%d).log}"
GATE_JSON="${GATE_JSON:-results/logs/strict_paper_repro_gate_${ITERATION}_$(date +%Y%m%d).json}"
TRACKER="${TRACKER:-results/tables/baseline_reproduction_tracker_20260706.md}"

mkdir -p results/logs results/tables
chmod +x scripts/run_todcl_adapter_nlg_official_anchor.sh \
  scripts/run_citb_stage1_seed50_train.sh \
  scripts/run_citb_replay50_paper_aligned_v2.sh \
  scripts/run_arper_woz3_paper_aligned_formal_v88.sh 2>/dev/null || true

log() { echo "[$(date -Iseconds)] [iter=${ITERATION}] $*" | tee -a "$LOG"; }

gpu_idle() {
  [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]
}

wait_gpu() {
  local reason="$1"
  while ! gpu_idle; do
    log "waiting GPU (${reason})"
    sleep "$POLL_SEC"
  done
}

wait_tmux() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep "$POLL_SEC"
  done
}

wait_process_pattern() {
  local pattern="$1"
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    sleep "$POLL_SEC"
  done
}

# --- metric extractors (paper-primary only) ---
get_citb_ar() {
  python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from scripts.parse_citb_official_results import summarize_method

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
  grep 'observed_avg_exact:' results/logs/olora_t5large_standard_order1_seed1_official_base_formal_v57_monitor.md 2>/dev/null \
    | head -1 | sed 's/.*observed_avg_exact: *//' | awk '{print $1}' || echo "nan"
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

get_todcl_bleu() {
  local log="/root/autodl-tmp/lora-ours-logs/todcl_adapter_nlg_official_anchor_20260706.log"
  [[ -f "$log" ]] || { echo "nan"; return; }
  grep -E 'BLEU|bleu' "$log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1 || echo "nan"
}

within_tol() {
  python3 - "$1" "$2" "$3" <<'PY'
import math, sys
local, paper, tol = map(float, sys.argv[1:4])
if math.isnan(local):
    print("pending")
elif abs(local - paper) <= tol:
    print("pass")
else:
    print("fail")
PY
}

write_gate() {
  local citb_ar std_em arper_bleu arper_ser todcl_bleu
  citb_ar="$(get_citb_ar)"
  std_em="$(get_std_em)"
  arper_bleu="$(get_arper_bleu arper_woz3_paper_aligned_exemplar250_formal_v88)"
  [[ "$arper_bleu" == "nan" ]] && arper_bleu="$(get_arper_bleu arper_woz3_paper_aligned_exemplar500_formal_v87)"
  arper_ser="$(get_arper_ser arper_woz3_paper_aligned_exemplar250_formal_v88)"
  [[ "$arper_ser" == "nan" ]] && arper_ser="$(get_arper_ser arper_woz3_paper_aligned_exemplar500_formal_v87)"
  todcl_bleu="$(get_todcl_bleu)"

  python3 - "$GATE_JSON" "$ITERATION" "$citb_ar" "$std_em" "$arper_bleu" "$arper_ser" "$todcl_bleu" <<'PY'
import json, math, sys
from datetime import datetime
from pathlib import Path

out, iteration, citb_ar, std_em, arper_bleu, arper_ser, todcl_bleu = sys.argv[1:8]

def f(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None

def gate(local, paper, tol):
    if local is None:
        return {"local": None, "paper": paper, "gap": None, "within_pm1": "pending"}
    gap = local - paper
    return {"local": local, "paper": paper, "gap": round(gap, 4), "within_pm1": "pass" if abs(gap) <= tol else "fail"}

doc = {
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "iteration": int(iteration),
    "tolerance": {
        "EM_AR_percent_points": 1.0,
        "BLEU_absolute": 0.01,
        "SER_absolute": 1.0,
        "ToDCL_BLEU_percent_points": 1.0,
    },
    "suites": {
        "CITB_Replay50_AR": gate(f(citb_ar), 40.4, 1.0),
        "Standard_OLoRA_EM": gate(f(std_em), 75.8, 1.0),
        "ARPER_BLEU4": gate(f(arper_bleu), 0.701, 0.01),
        "ARPER_SER": gate(f(arper_ser), 3.63, 1.0),
        "ToDCL_ADAPTER_BLEU": gate(f(todcl_bleu), 21.77, 1.0),
    },
}
Path(out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
for name, row in doc["suites"].items():
    print(f"{name}: {row['within_pm1']} local={row['local']} paper={row['paper']} gap={row['gap']}")
PY
}

launch_tmux() {
  local session="$1"
  local cmd="$2"
  if tmux has-session -t "$session" 2>/dev/null; then
    log "tmux ${session} already exists — skip launch"
    return 0
  fi
  tmux new-session -d -s "$session" "bash -lc 'cd ${REPO_ROOT} && ${cmd}'"
  log "launched tmux ${session}"
}

log "strict paper repro iteration ${ITERATION} started"
write_gate | tee -a "$LOG"

# --- Phase: respect in-flight healthy jobs ---
if pgrep -f 'todcl.*train.py.*ADAPTER' >/dev/null 2>&1; then
  log "ToDCL ADAPTER healthy — waiting (do not kill)"
  wait_process_pattern 'todcl.*train.py.*ADAPTER'
fi

citb_ar="$(get_citb_ar)"
citb_gate="$(within_tol "$citb_ar" 40.4 1.0)"
arper_bleu="$(get_arper_bleu arper_woz3_paper_aligned_exemplar500_formal_v87)"
arper_gate="$(within_tol "$arper_bleu" 0.701 0.01)"

# --- Fix queue (single fix per iteration, serial GPU) ---
if [[ "$SKIP_TODCL" != "1" ]] && [[ "$(get_todcl_bleu)" == "nan" ]] && ! pgrep -f 'todcl.*train.py.*ADAPTER' >/dev/null 2>&1; then
  log "Fix: launch ToDCL ADAPTER anchor"
  wait_gpu "todcl"
  bash scripts/run_todcl_adapter_nlg_official_anchor.sh || true
  wait_tmux lora-ours-todcl-adapter-anchor
  write_gate | tee -a "$LOG"
fi

if [[ "$arper_gate" == "fail" ]]; then
  log "Fix: ARPER v88 (exemplar 250, batch 128, domain-wise)"
  wait_gpu "arper_v88"
  bash scripts/run_arper_woz3_paper_aligned_formal_v88.sh || true
  wait_tmux lora-ours-arper-v88-formal
  write_gate | tee -a "$LOG"
fi

if [[ "$FORCE_CITB_STAGE1" == "1" ]] || [[ "$citb_gate" == "fail" ]]; then
  OFFICIAL_CKPT="/root/autodl-tmp/Lora-code/external_baselines/citb_official/output/initial_multitask_model/base_epoch15_lr1e-05_seed50/checkpoint-14000"
  if [[ ! -d "$OFFICIAL_CKPT" ]]; then
    log "Fix: CITB Stage-1 seed50 train (checkpoint-14000 missing)"
    wait_gpu "citb_stage1"
    launch_tmux lora-ours-citb-stage1-seed50 \
      "bash scripts/run_citb_stage1_seed50_train.sh > results/logs/citb_stage1_seed50_train_formal.log 2>&1; echo EXIT=\$? >> results/logs/citb_stage1_seed50_train_formal.log"
    wait_tmux lora-ours-citb-stage1-seed50
  fi
  if [[ -d "$OFFICIAL_CKPT" ]]; then
    log "Fix: CITB Replay(50) v2 formal (official Stage-1)"
    wait_gpu "citb_v2"
    launch_tmux lora-ours-citb-replay50-v2 \
      "ALLOW_FALLBACK_STAGE1=0 DRY_RUN=0 bash scripts/run_citb_replay50_paper_aligned_v2.sh > results/logs/citb_replay50_paper_aligned_v2_formal.log 2>&1; echo EXIT=\$? >> results/logs/citb_replay50_paper_aligned_v2_formal.log"
    wait_tmux lora-ours-citb-replay50-v2
  fi
  write_gate | tee -a "$LOG"
else
  log "CITB AR gate pass (${citb_ar}) — skip Stage-1/v2 unless FORCE_CITB_STAGE1=1"
fi

write_gate | tee -a "$LOG"
log "iteration ${ITERATION} complete — gate: ${GATE_JSON}"
