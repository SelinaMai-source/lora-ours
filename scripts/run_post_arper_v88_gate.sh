#!/usr/bin/env bash
# Wait for ARPER v88, evaluate ±1 gate, optionally launch v89, then ToDCL (GPU serial).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

POLL_SEC="${POLL_SEC:-60}"
LOG="${LOG:-results/logs/post_arper_v88_gate_20260707.log}"
TRACKER="${TRACKER:-results/tables/paper_repro_iteration_tracker_20260707.md}"
GAP_MD="${GAP_MD:-docs/experiments/paper_repro_gap_analysis_20260707.md}"

V88_RUN_ID="arper_woz3_paper_aligned_exemplar250_formal_v88"
V88_SESSION="lora-ours-arper-v88-formal"
V88_LOG="/root/autodl-tmp/lora-ours-logs/${V88_RUN_ID}.log"

ARPER_BLEU_PAPER=0.701
ARPER_BLEU_TOL=0.01
ARPER_SER_PAPER=3.63
ARPER_SER_TOL=1.0

mkdir -p results/logs results/tables docs/experiments
chmod +x scripts/run_arper_woz3_paper_aligned_formal_v89.sh \
  scripts/run_todcl_adapter_nlg_official_anchor.sh 2>/dev/null || true

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

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

get_arper_bleu() {
  local run_id="$1"
  local logf="/root/autodl-tmp/lora-ours-logs/${run_id}.log"
  [[ -f "$logf" ]] || { echo "nan"; return; }
  grep -E '^test Loss:.*BLEU4:' "$logf" 2>/dev/null | tail -1 | sed -n 's/.*BLEU4: \([0-9.]*\).*/\1/p' || echo "nan"
}

get_arper_ser() {
  local run_id="$1"
  local logf="/root/autodl-tmp/lora-ours-logs/${run_id}.log"
  [[ -f "$logf" ]] || { echo "nan"; return; }
  grep -E '^test Loss:.*Slot error:' "$logf" 2>/dev/null | tail -1 | sed -n 's/.*Slot error: \([0-9.]*\).*/\1/p' || echo "nan"
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

update_tracker_row() {
  local version="$1" status="$2" local_bleu="$3" local_ser="$4" verdict="$5" next="$6"
  python3 - "$TRACKER" "$version" "$status" "$local_bleu" "$local_ser" "$verdict" "$next" <<'PY'
import re, sys
from datetime import datetime
from pathlib import Path

path, version, status, bleu, ser, verdict, nxt = sys.argv[1:8]
text = Path(path).read_text(encoding="utf-8")
now = datetime.now().strftime("%Y-%m-%dT%H:%M+08:00")
row = f"| 2 | {version} | ARPER | {status} | 0.701/3.63 | — | {verdict} | {nxt} |"
text = re.sub(
    r"\| 2 \| v88 \| ARPER \|[^\n]+\n",
    row + "\n",
    text,
    count=1,
)
text = re.sub(r"\*Updated: [^\*]+\*", f"*Updated: {now}*", text, count=1)
Path(path).write_text(text, encoding="utf-8")
PY
}

append_v89_gap() {
  local bleu="$1" ser="$2" bleu_gap="$3" ser_gap="$4"
  cat >> "$GAP_MD" <<EOF

---

## v88 result → v89 hypothesis ($(date -Iseconds))

| Metric | Paper | v88 local | Gap | ±1? |
|--------|-------|-----------|-----|-----|
| BLEU | 0.701 | ${bleu} | ${bleu_gap} | $(within_tol "$bleu" "$ARPER_BLEU_PAPER" "$ARPER_BLEU_TOL") |
| SER | 3.63 | ${ser} | ${ser_gap} | $(within_tol "$ser" "$ARPER_SER_PAPER" "$ARPER_SER_TOL") |

**v89 hypothesis:** domain-wise + exemplar **500** + batch **128** (v87 exemplar count, v88 batch size).
EOF
}

log "post-v88 gate watcher started"

# Wait for v88 tmux + process
while tmux has-session -t "$V88_SESSION" 2>/dev/null; do
  sleep "$POLL_SEC"
done
while pgrep -f "run_woz3\.py.*${V88_RUN_ID}" >/dev/null 2>&1; do
  sleep "$POLL_SEC"
done
log "v88 session ended — parsing metrics"

bleu="$(get_arper_bleu "$V88_RUN_ID")"
ser="$(get_arper_ser "$V88_RUN_ID")"
bleu_gate="$(within_tol "$bleu" "$ARPER_BLEU_PAPER" "$ARPER_BLEU_TOL")"
ser_gate="$(within_tol "$ser" "$ARPER_SER_PAPER" "$ARPER_SER_TOL")"

if [[ "$bleu" == "nan" || "$ser" == "nan" ]]; then
  log "v88 metrics not found in log — check ${V88_LOG}"
  update_tracker_row "v88" "parse_error" "$bleu" "$ser" "TBD" "manual"
  exit 1
fi

bleu_gap="$(python3 -c "print(round(float('$bleu') - $ARPER_BLEU_PAPER, 4))")"
ser_gap="$(python3 -c "print(round(float('$ser') - $ARPER_SER_PAPER, 4))")"
log "v88 BLEU=${bleu} (gap ${bleu_gap}) SER=${ser} (gap ${ser_gap}) bleu_gate=${bleu_gate} ser_gate=${ser_gate}"

if [[ "$bleu_gate" == "pass" && "$ser_gate" == "pass" ]]; then
  update_tracker_row "v88" "BLEU ${bleu} SER ${ser}" "$bleu" "$ser" "**PASS**" "ToDCL"
  log "v88 PASS ±1 — skip v89"
else
  update_tracker_row "v88" "BLEU ${bleu} SER ${ser}" "$bleu" "$ser" "FAIL" "v89"
  append_v89_gap "$bleu" "$ser" "$bleu_gap" "$ser_gap"
  log "v88 FAIL — launching v89 after GPU free"
  wait_gpu "before ARPER v89"
  bash scripts/run_arper_woz3_paper_aligned_formal_v89.sh || {
    ec=$?; [[ "$ec" -eq 75 ]] && log "v89 queued (GPU busy)" || exit "$ec"
  }
  while tmux has-session -t lora-ours-arper-v89-formal 2>/dev/null; do
    sleep "$POLL_SEC"
  done
  bleu="$(get_arper_bleu arper_woz3_paper_aligned_exemplar500_batch128_formal_v89)"
  ser="$(get_arper_ser arper_woz3_paper_aligned_exemplar500_batch128_formal_v89)"
  log "v89 final BLEU=${bleu} SER=${ser}"
fi

# ToDCL after ARPER line completes
if pgrep -f 'todcl.*train.py.*ADAPTER' >/dev/null 2>&1 || tmux has-session -t lora-ours-todcl-adapter-anchor 2>/dev/null; then
  log "ToDCL already running — waiting"
  while pgrep -f 'todcl.*train.py.*ADAPTER' >/dev/null 2>&1 || tmux has-session -t lora-ours-todcl-adapter-anchor 2>/dev/null; do
    sleep "$POLL_SEC"
  done
else
  log "launching ToDCL ADAPTER anchor"
  wait_gpu "before ToDCL"
  bash scripts/run_todcl_adapter_nlg_official_anchor.sh || true
  while tmux has-session -t lora-ours-todcl-adapter-anchor 2>/dev/null; do
    sleep "$POLL_SEC"
  done
fi

log "post-v88 gate complete"
