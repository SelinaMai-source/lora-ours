#!/usr/bin/env bash
# Monitor best-method-only reproduction until all suites complete or blocker.
# Launch: tmux new-session -d -s lora-ours-best-method-watch 'bash scripts/monitor_best_method_repro_complete.sh'
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

POLL_SEC="${POLL_SEC:-300}"
WATCH_MD="${WATCH_MD:-results/logs/best_method_repro_watch_20260706.md}"
FLAG_FILE="${FLAG_FILE:-BEST_METHOD_REPRO_COMPLETE.flag}"
MONITOR_LOG="${MONITOR_LOG:-results/logs/best_method_repro_monitor_20260706.log}"

# --- paths ---
CITB_RUN="citb_instrdialog_order1_seed1_official_script_500_50_50_tie_fixed_replay50_formal_v56"
CITB_OUT="/root/autodl-tmp/citb_official_base_repro/${CITB_RUN}"
CITB_LOG="results/logs/citb_replay50_formal_20260706.log"
CITB_STATUS="results/logs/citb_replay50_formal_20260706"

V57_STATUS="results/logs/olora_t5large_standard_order1_seed1_official_base_formal_v57_status.md"
V57_MANIFEST="results/runs/olora_t5large_standard_order1_seed1_official_base_formal_v57/run_manifest.json"
V57_MONITOR="results/logs/olora_t5large_standard_order1_seed1_official_base_formal_v57_monitor.md"

TODCL_LOG="/root/autodl-tmp/lora-ours-logs/todcl_adapter_nlg_official_anchor_20260706.log"
TODCL_STATUS="results/logs/todcl_adapter_nlg_official_anchor_20260706_status.md"
TODCL_SESSION="lora-ours-todcl-adapter-anchor"

ARPER_RUN="arper_woz3_paper_aligned_exemplar500_formal_v87"
ARPER_LOG="/root/autodl-tmp/lora-ours-logs/${ARPER_RUN}.log"
ARPER_STATUS_JSON="results/logs/${ARPER_RUN}_status.json"
ARPER_STATUS_MD="results/logs/${ARPER_RUN}_status.md"
ARPER_SESSION="lora-ours-arper-v87-formal"
ARPER_TASKS=6

QUEUE_SESSION="lora-ours-baseline-repro-queue"

mkdir -p results/logs
chmod +x scripts/push_best_method_repro_results_20260706.sh 2>/dev/null || true

log() { echo "[$(date -Iseconds)] $*" | tee -a "$MONITOR_LOG"; }

gpu_summary() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -3 || echo "(no GPU apps)"
}

tmux_summary() {
  local items=()
  for s in lora-ours-citb-replay50-formal "$TODCL_SESSION" "$ARPER_SESSION" "$QUEUE_SESSION" lora-ours-best-method-watch; do
    if tmux has-session -t "$s" 2>/dev/null; then items+=("$s"); fi
  done
  if ((${#items[@]})); then printf '%s ' "${items[@]}"; else echo "(none)"; fi
}

refresh_citb_status() {
  python3 scripts/monitor_citb_official_base_repro.py \
    --run-name "$CITB_RUN" \
    --output-dir "$CITB_OUT" \
    --expected-tasks 19 \
    --basename "${CITB_STATUS##*/}" \
    --log-path "$REPO_ROOT/$CITB_LOG" >/dev/null 2>&1 || true
  ln -sf "${CITB_STATUS##*/}.md" "${CITB_STATUS}_status.md" 2>/dev/null || true
}

refresh_arper_status() {
  if [[ -f "$ARPER_LOG" ]]; then
    python3 scripts/monitor_arper_woz3_formal.py \
      --run-id "$ARPER_RUN" \
      --log-path "$ARPER_LOG" \
      --status-basename "${ARPER_RUN}_status" >/dev/null 2>&1 || true
  fi
}

write_todcl_status() {
  local state="$1" bleu="$2" eer="$3" notes="$4"
  cat > "$TODCL_STATUS" <<EOF
# ToDCL ADAPTER NLG Official Anchor — 20260706

- Updated: $(date -Iseconds)
- State: \`${state}\`
- Session: \`${TODCL_SESSION}\`
- Log: \`${TODCL_LOG}\`
- Launcher: \`scripts/run_todcl_adapter_nlg_official_anchor.sh\`
- Paper reference: BLEU **21.7719**, EER **0.164**
- Local BLEU: ${bleu:-*pending*}
- Local EER: ${eer:-*pending*}
- Notes: ${notes}

## Exit / error signals
$(tail -20 "$TODCL_LOG" 2>/dev/null | sed 's/^/- /' || echo "- (no log)")
EOF
}

# Returns via stdout: status|metric|pct|eta|detail
check_citb() {
  refresh_citb_status
  local n=0 rouge="—"
  if [[ -d "$CITB_OUT/results" ]]; then
    n=$(find "$CITB_OUT/results" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  fi
  if [[ -f "${CITB_STATUS}.json" ]]; then
    rouge=$(python3 -c "import json; d=json.load(open('${CITB_STATUS}.json')); print(d.get('all_results',{}).get('predict_official_rougeL', d.get('all_results',{}).get('predict_official_rougeL_for_dialogue_generation','—')))" 2>/dev/null || echo "—")
  fi
  local pct=$(( n * 100 / 19 ))
  if [[ -f "$CITB_LOG" ]] && grep -q 'EXIT_CODE=0' "$CITB_LOG" && [[ "$n" -ge 19 ]]; then
    echo "done|ROUGE-L AR ${rouge}|100|0|19/19 tasks"
  elif tmux has-session -t lora-ours-citb-replay50-formal 2>/dev/null || pgrep -f '[p]ython.*run_continual_instruct_tuning' >/dev/null 2>&1; then
    echo "running|ROUGE-L partial|${pct}|$(( (19-n) * 25 ))|${n}/19 tasks"
  elif [[ "$n" -ge 19 ]]; then
    echo "done|ROUGE-L AR ${rouge}|100|0|19/19 (log may lack EXIT marker)"
  else
    echo "incomplete|—|${pct}|?|${n}/19 tasks"
  fi
}

check_v57() {
  local em="—" state="pending"
  if [[ -f "$V57_MONITOR" ]]; then
    em=$(grep 'observed_avg_exact:' "$V57_MONITOR" 2>/dev/null | head -1 | sed 's/.*observed_avg_exact: *//' | awk '{printf "%.4f", $1}' || echo "—")
  fi
  if [[ -f "$V57_MANIFEST" ]]; then
    state=$(python3 -c "import json; print(json.load(open('$V57_MANIFEST')).get('state',''))" 2>/dev/null || echo "")
  fi
  if [[ "$state" == "completed" ]] || [[ -f "$V57_STATUS" ]] && grep -q 'state: completed' "$V57_STATUS" 2>/dev/null; then
    echo "done|EM ${em}|100|0|v57 formal manifest completed"
  else
    echo "pending|EM ${em}|0|?|manifest=${state}"
  fi
}

check_todcl() {
  local bleu="" eer="" exit_code=""
  if [[ -f "$TODCL_LOG" ]]; then
    exit_code=$(grep -oE 'EXIT=[0-9]+' "$TODCL_LOG" 2>/dev/null | tail -1 | cut -d= -f2 || true)
    bleu=$(grep -oE 'BLEU[^0-9]*[0-9]+\.[0-9]+' "$TODCL_LOG" 2>/dev/null | tail -1 || true)
    eer=$(grep -oE 'EER[^0-9]*[0-9]+\.[0-9]+' "$TODCL_LOG" 2>/dev/null | tail -1 || true)
    if grep -qE 'BLEU|Slot error|test Loss' "$TODCL_LOG" 2>/dev/null; then
      bleu=$(grep 'BLEU' "$TODCL_LOG" 2>/dev/null | tail -1 || true)
      eer=$(grep -iE 'EER|Slot error' "$TODCL_LOG" 2>/dev/null | tail -1 || true)
    fi
  fi

  if tmux has-session -t "$TODCL_SESSION" 2>/dev/null || pgrep -f 'train\.py.*--CL ADAPTER' >/dev/null 2>&1; then
    write_todcl_status "running" "—" "—" "37-domain NLG anchor in progress"
    echo "running|—|50|~hours|tmux active"
    return
  fi

  if [[ -f "$TODCL_LOG" ]] && grep -qE 'test.*BLEU|Final BLEU|avg.*BLEU' "$TODCL_LOG" 2>/dev/null; then
    write_todcl_status "completed" "${bleu:-extracted}" "${eer:-extracted}" "Formal anchor finished"
    echo "done|${bleu:-BLEU logged}|100|0|completed with metrics"
    return
  fi

  if [[ "$exit_code" == "1" ]] || grep -q 'Unable to load weights.*gpt2' "$TODCL_LOG" 2>/dev/null; then
    write_todcl_status "failed_blocker" "—" "—" "GPT-2 pytorch_model.bin corrupt/missing; HF download unreachable — unrecoverable without model fix"
    echo "blocker|—|0|—|gpt2 load failure EXIT=${exit_code:-1}"
    return
  fi

  if [[ -f "$TODCL_LOG" ]] && grep -q 'EXIT=0' "$TODCL_LOG" 2>/dev/null; then
    write_todcl_status "completed_exit0" "${bleu:-—}" "${eer:-—}" "Process exited 0; verify BLEU/EER in log"
    echo "done|${bleu:-see log}|100|0|EXIT=0"
    return
  fi

  write_todcl_status "queued" "—" "—" "Not started or awaiting GPU after ARPER"
  echo "queued|—|0|after ARPER|not running"
}

check_arper() {
  refresh_arper_status
  local state="unknown" tasks_done=0 bleu="" ser=""
  if [[ -f "$ARPER_STATUS_JSON" ]]; then
    state=$(python3 -c "import json; print(json.load(open('$ARPER_STATUS_JSON')).get('state',''))" 2>/dev/null || echo "")
  fi
  if [[ -f "$ARPER_LOG" ]]; then
    tasks_done=$(grep -c 'Current task:' "$ARPER_LOG" 2>/dev/null || echo 0)
    bleu=$(grep 'test Loss:' "$ARPER_LOG" 2>/dev/null | tail -1 | grep -oE 'BLEU4: [0-9.]+' | awk '{print $2}' || true)
    ser=$(grep 'test Loss:' "$ARPER_LOG" 2>/dev/null | tail -1 | grep -oE 'Slot error: [0-9.]+' | awk '{print $2}' || true)
  fi
  local pct=$(( tasks_done * 100 / ARPER_TASKS ))
  [[ "$pct" -gt 99 ]] && pct=99

  if [[ "$state" == "completed_or_stopped" ]] && [[ -n "$bleu" ]]; then
    echo "done|BLEU ${bleu} SER ${ser}|100|0|formal finished"
    return
  fi
  if [[ "$state" == "completed_or_stopped" ]] && ! tmux has-session -t "$ARPER_SESSION" 2>/dev/null; then
    if [[ -n "$bleu" ]]; then
      echo "done|BLEU ${bleu} SER ${ser}|100|0|process stopped with test metrics"
    else
      echo "incomplete|—|${pct}|?|stopped without final BLEU"
    fi
    return
  fi
  if tmux has-session -t "$ARPER_SESSION" 2>/dev/null || pgrep -f 'run_woz3.py' >/dev/null 2>&1; then
    # ETA: ~45-90 min per task early-stop, rough
    local eta_min=$(( (ARPER_TASKS - tasks_done) * 60 + 30 ))
    echo "running|BLEU partial ${bleu:-0}|${pct}|~${eta_min}m|task ~${tasks_done}/${ARPER_TASKS}"
    return
  fi
  echo "pending|—|${pct}|?|state=${state}"
}

write_watch_md() {
  local citb="$1" v57="$2" todcl="$3" arper="$4"
  IFS='|' read -r c_st c_met c_pct c_eta c_det <<< "$citb"
  IFS='|' read -r v_st v_met v_pct v_eta v_det <<< "$v57"
  IFS='|' read -r t_st t_met t_pct t_eta t_det <<< "$todcl"
  IFS='|' read -r a_st a_met a_pct a_eta a_det <<< "$arper"

  cat > "$WATCH_MD" <<EOF
# Best-Method Repro Watch — 2026-07-06

**Updated:** $(date -Iseconds)  
**Poll interval:** ${POLL_SEC}s  
**Monitor tmux:** \`lora-ours-best-method-watch\`  
**Queue tmux:** \`$QUEUE_SESSION\`

## Suite progress

| Suite | Best method | Status | Metric | Progress | ETA |
|-------|-------------|--------|--------|----------|-----|
| CITB | Replay(50) formal | **${c_st}** | ${c_met} | ${c_pct}% | ${c_eta} |
| Standard | O-LoRA v57 | **${v_st}** | ${v_met} | ${v_pct}% | ${v_eta} |
| Dialogue ToDCL | ADAPTER 37-domain | **${t_st}** | ${t_met} | ${t_pct}% | ${t_eta} |
| Dialogue ARPER | v87 paper-aligned | **${a_st}** | ${a_met} | ${a_pct}% | ${a_eta} |

### Details
- CITB: ${c_det}
- Standard: ${v_det}
- ToDCL: ${t_det}
- ARPER: ${a_det}

## GPU
\`\`\`
$(gpu_summary)
\`\`\`

## Active tmux
$(tmux_summary)

## Queue
See \`results/logs/baseline_reproduction_queue_status_20260706.md\`

EOF
}

write_flag_and_push() {
  local overall="$1"
  local citb="$2" v57="$3" todcl="$4" arper="$5"
  python3 - <<'PY' "$FLAG_FILE" "$overall" "$citb" "$v57" "$todcl" "$arper"
import json, sys
from datetime import datetime
from pathlib import Path

flag_path, overall, citb, v57, todcl, arper = sys.argv[1:7]

def parse(line):
    p = line.split("|")
    return {"status": p[0], "metric": p[1], "pct": p[2], "eta": p[3], "detail": p[4] if len(p)>4 else ""}

c, v, t, a = map(parse, [citb, v57, todcl, arper])
blockers = []
if t["status"] == "blocker":
    blockers.append("ToDCL ADAPTER: corrupt/missing GPT-2 weights; network unreachable for re-download")

payload = {
    "completed_at": datetime.now().isoformat(timespec="seconds"),
    "overall_status": overall,
    "branch": "best-method-repro-results-20260706",
    "watch_md": "results/logs/best_method_repro_watch_20260706.md",
    "blockers": blockers,
    "suites": {
        "citb_replay50": {
            "method": "Replay(50) formal 19-task",
            "status": c["status"],
            "paper_metric": "ROUGE-L AR 40.4",
            "local_metric": c["metric"] if c["status"]=="done" else "32.442 (prior)",
            "match": "No" if c["status"]=="done" else "—",
            "notes": "official_script_500_50_50; gap −7.96 vs paper",
            "tmux": "lora-ours-citb-replay50-formal",
            "wandb": "4r1vg0x9 / citb_instrdialog_order1_official_replay50_base_repro",
        },
        "standard_olora_v57": {
            "method": "O-LoRA v57 formal",
            "status": v["status"],
            "paper_metric": "EM avg 75.8",
            "local_metric": v["metric"] if v["status"]=="done" else "76.8059",
            "match": "Yes / Close",
            "notes": "single-GPU grad_accum=8 official-equivalent",
            "tmux": "(completed prior)",
            "wandb": "published-base-standard-olora-order1-formal-v57",
        },
        "todcl_adapter": {
            "method": "ADAPTER 37-domain NLG anchor",
            "status": t["status"],
            "paper_metric": "BLEU 21.77 / EER 0.164",
            "local_metric": t["metric"] if t["status"]=="done" else "—",
            "match": "—" if t["status"] != "done" else "TBD",
            "notes": t["detail"],
            "tmux": "lora-ours-todcl-adapter-anchor",
            "wandb": "—",
        },
        "arper_v87": {
            "method": "v87 paper-aligned domain/exemplar500",
            "status": a["status"],
            "paper_metric": "BLEU 0.701 / SER 3.63",
            "local_metric": a["metric"] if a["status"]=="done" else "—",
            "match": "TBD",
            "notes": "granularity=0 exemplar 500",
            "tmux": "lora-ours-arper-v87-formal",
            "wandb": "—",
        },
    },
}
Path(flag_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"overall": overall, "blockers": blockers}))
PY

  log "Flag written: $FLAG_FILE (overall=$overall)"
  if bash scripts/push_best_method_repro_results_20260706.sh; then
    log "Push succeeded"
  else
    log "Push failed (will retry next tick if flag exists)"
  fi
}

ensure_queue_active() {
  if ! tmux has-session -t "$QUEUE_SESSION" 2>/dev/null; then
    log "Restarting baseline repro queue session"
    tmux new-session -d -s "$QUEUE_SESSION" \
      "cd $REPO_ROOT && bash scripts/run_baseline_reproduction_queue.sh" 2>/dev/null || true
  fi
}

log "Best-method repro monitor started (poll=${POLL_SEC}s)"

while true; do
  if [[ -f "$FLAG_FILE" ]]; then
    log "Flag already set; exiting monitor loop"
    break
  fi

  ensure_queue_active

  citb=$(check_citb)
  v57=$(check_v57)
  todcl=$(check_todcl)
  arper=$(check_arper)

  write_watch_md "$citb" "$v57" "$todcl" "$arper"
  log "tick citb=${citb%%|*} v57=${v57%%|*} todcl=${todcl%%|*} arper=${arper%%|*}"

  c_st="${citb%%|*}"
  v_st="${v57%%|*}"
  t_st="${todcl%%|*}"
  a_st="${arper%%|*}"

  all_done=false
  if [[ "$c_st" == "done" && "$v_st" == "done" && "$t_st" == "done" && "$a_st" == "done" ]]; then
    all_done=true
    write_flag_and_push "all_complete" "$citb" "$v57" "$todcl" "$arper"
    break
  fi

  # Blocker path: ToDCL unrecoverable but other three finished
  if [[ "$t_st" == "blocker" && "$c_st" == "done" && "$v_st" == "done" && "$a_st" == "done" ]]; then
    write_flag_and_push "complete_with_blocker" "$citb" "$v57" "$todcl" "$arper"
    break
  fi

  sleep "$POLL_SEC"
done

log "Monitor loop ended"
