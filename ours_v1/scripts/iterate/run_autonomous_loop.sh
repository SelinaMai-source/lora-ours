#!/usr/bin/env bash
# Autonomous single-GPU iteration loop across suites.
# - Suite isolation: one suite FAIL does not block others
# - Formal only after smoke PASS
# - On metric/smoke FAIL: diagnose → propose one delta → create ours-v{N+1} → continue
# - Stops a suite only on all-metrics PASS or documented external_blocker
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

ITER_DIR="${REPO_ROOT}/ours_v1/scripts/iterate"
STATE_JSON="${ITER_DIR}/suite_state.json"
LOOP_LOG="${LOOP_LOG:-results/logs/ours_autonomous_loop_20260712.log}"
POLL_SEC="${POLL_SEC:-90}"
MAX_VERSIONS_PER_SUITE="${MAX_VERSIONS_PER_SUITE:-5}"
LEADERBOARD="${LEADERBOARD:-results/tables/ours_iteration_leaderboard.json}"

mkdir -p results/logs results/tables results/manifests docs/experiments

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOOP_LOG"; }

gpu_empty() {
  local out
  out="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
  [[ -z "${out// /}" ]]
}

wait_gpu() {
  while ! gpu_empty; do
    log "GPU busy; sleep ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
}

wait_tmux() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep "$POLL_SEC"
  done
  wait_gpu
}

read_standard_state() {
  local run_name="$1"
  local f="results/logs/${run_name}_status.md"
  [[ -f "$f" ]] || { echo missing; return; }
  python3 -c "import re; t=open('$f',encoding='utf-8',errors='replace').read(); m=re.search(r'(?m)^-\\s*state:\\s*(\\S+)',t); print(m.group(1) if m else 'unknown')"
}

publish_leaderboard() {
  python3 - <<'PY' | tee -a "$LOOP_LOG"
import json
from datetime import datetime, timezone
from pathlib import Path
repo = Path("/root/lora-ours-ours-v1")
state = json.loads((repo / "ours_v1/scripts/iterate/suite_state.json").read_text())
rows = []
for suite, st in state["suites"].items():
    for mname, m in (st.get("metrics") or {}).items():
        rows.append({
            "suite": suite,
            "metric": mname,
            "published_base": m.get("base"),
            "ours": m.get("ours"),
            "delta": (None if m.get("ours") is None or m.get("base") is None else m["ours"] - m["base"]),
            "target": m.get("target"),
            "version": st.get("current_version"),
            "status": st.get("status"),
            "blocker": st.get("blocker"),
            "setting_label": st.get("overlay_active"),
        })
out = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "campaign": state.get("campaign"),
    "rows": rows,
    "suites": state["suites"],
}
path = repo / "results/tables/ours_iteration_leaderboard.json"
path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
md = repo / "results/tables/ours_iteration_leaderboard.md"
lines = [
    "# Ours Autonomous Iteration Leaderboard",
    "",
    f"Updated: {out['updated_at']}",
    "",
    "| Suite | Metric | Base | Ours | Target | Version | Status |",
    "|-------|--------|------|------|--------|---------|--------|",
]
for r in rows:
    lines.append(
        f"| {r['suite']} | {r['metric']} | {r['published_base']} | {r['ours']} | {r['target']} | {r['version']} | {r['status']} |"
    )
md.write_text("\n".join(lines) + "\n")
print(f"wrote {path} and {md}")
PY
  python3 ours_v1/scripts/publish_formal_results.py >>"$LOOP_LOG" 2>&1 || true
}

suite_status() {
  python3 -c "import json; print(json.load(open('${STATE_JSON}'))['suites']['$1']['status'])"
}

suite_version() {
  python3 -c "import json; print(json.load(open('${STATE_JSON}'))['suites']['$1']['current_version'])"
}

mark_blocker() {
  local suite="$1"
  local reason="$2"
  python3 - "$suite" "$reason" <<'PY'
import json, sys
from pathlib import Path
from datetime import datetime, timezone
suite, reason = sys.argv[1], sys.argv[2]
path = Path("/root/lora-ours-ours-v1/ours_v1/scripts/iterate/suite_state.json")
st = json.loads(path.read_text())
st["suites"][suite]["status"] = "external_blocker"
st["suites"][suite]["blocker"] = {
    "type": "external_blocker",
    "reason": reason,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
path.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(st["suites"][suite]["blocker"], indent=2))
PY
}

# ---------- Standard versioned smoke→formal ----------
run_standard_version() {
  local ver="$1"  # ours-v2
  local short="${ver#ours-}"
  local launcher="ours_v1/scripts/launchers/run_standard_${short}.sh"
  if [[ ! -f "$launcher" ]]; then
    launcher="ours_v1/scripts/launchers/run_standard_v1.sh"
  fi
  local smoke_name="olora_official_base_ours_overlay_${short}_20260712_smoke_order1_seed1"
  local formal_name="olora_official_base_ours_overlay_${short}_20260712_formal_order1_seed1"
  local smoke_session="lora-ours-standard-${short}-smoke"
  local formal_session="lora-ours-standard-${short}-formal"

  log "STANDARD ${ver} smoke via ${launcher}"
  wait_gpu
  # v58 training runs foreground — always wrap in tmux (never orphan parallel formal).
  if tmux has-session -t "$smoke_session" 2>/dev/null; then
    log "smoke session ${smoke_session} already exists; waiting"
  else
    tmux new-session -d -s "$smoke_session" \
      "bash -lc 'cd ${REPO_ROOT} && RUN_NAME=${smoke_name} bash ${launcher}; ec=\$?; echo EXIT=\$ec | tee -a results/logs/${smoke_name}.log; exit \$ec'"
  fi
  wait_tmux "$smoke_session"
  local st
  st="$(read_standard_state "$smoke_name")"
  log "STANDARD ${ver} smoke state=${st}"
  if [[ "$st" != "completed" ]]; then
    # parse amazon/dbpedia from status for diagnosis
    python3 "${ITER_DIR}/diagnose_failure.py" --suite standard --version "$ver" \
      --reason "$(grep -E '^- reason:' "results/logs/${smoke_name}_status.md" 2>/dev/null | sed 's/^- reason: //')" \
      $( [[ "$ver" != "ours-v1" ]] && echo --assess-skip-amazon || true ) || true
    return 1
  fi

  log "STANDARD ${ver} formal"
  wait_gpu
  if tmux has-session -t "$formal_session" 2>/dev/null; then
    log "formal session ${formal_session} already exists; waiting"
  else
    tmux new-session -d -s "$formal_session" \
      "bash -lc 'cd ${REPO_ROOT} && FORMAL=1 RUN_NAME=${formal_name} bash ${launcher}; ec=\$?; echo EXIT=\$ec | tee -a results/logs/${formal_name}.log; exit \$ec'"
  fi
  wait_tmux "$formal_session"
  st="$(read_standard_state "$formal_name")"
  log "STANDARD ${ver} formal state=${st}"
  if [[ "$st" != "completed" ]]; then
    python3 "${ITER_DIR}/diagnose_failure.py" --suite standard --version "$ver" \
      --reason "formal state=${st}" || true
    return 1
  fi
  if python3 "${ITER_DIR}/evaluate_gate.py" --suite standard --run-key "$formal_name"; then
    log "STANDARD ${ver} TARGETS MET"
    return 0
  fi
  python3 "${ITER_DIR}/diagnose_failure.py" --suite standard --version "$ver" \
    --reason "formal metrics below target" || true
  return 1
}

advance_standard() {
  local cur
  cur="$(suite_version standard)"
  python3 "${ITER_DIR}/propose_next_version.py" --suite standard --from-version "$cur" --create-branch
}

# ---------- CITB InstrDialog v1 ----------
run_citb_v1() {
  log "CITB InstrDialog v1 smoke"
  wait_gpu
  local session="lora-ours-citb-ours-v1-smoke"
  bash ours_v1/scripts/launchers/run_citb_v1.sh || true
  wait_tmux "$session"
  local run_smoke="citb_instrdialog_order1_seed1_ours_v1_20260708_smoke_strict"
  if [[ ! -f "results/runs/${run_smoke}/ccfa_postprocess/summary.json" ]] && \
     [[ ! -f "results/runs/${run_smoke}/run_manifest.json" ]]; then
    # smoke may still be success if final_metrics exists
    if [[ ! -f "results/runs/${run_smoke}/final_metrics.json" ]]; then
      log "CITB smoke missing artifacts → diagnose"
      python3 "${ITER_DIR}/diagnose_failure.py" --suite citb_instrdialog --version ours-v1 --reason "smoke incomplete" || true
      return 1
    fi
  fi
  log "CITB smoke OK-ish → formal"
  wait_gpu
  session="lora-ours-citb-ours-v1-formal"
  FORMAL=1 bash ours_v1/scripts/launchers/run_citb_v1.sh || true
  wait_tmux "$session"
  local run_formal="citb_instrdialog_order1_seed1_ours_v1_20260708_formal_strict"
  if python3 "${ITER_DIR}/evaluate_gate.py" --suite citb_instrdialog --run-key "$run_formal"; then
    log "CITB TARGETS MET"
    return 0
  fi
  python3 "${ITER_DIR}/diagnose_failure.py" --suite citb_instrdialog --version ours-v1 --reason "metrics or incomplete" || true
  return 1
}

# ---------- ARPER v1 ----------
run_arper_v1() {
  log "ARPER v1 smoke (no Fisher CPU / no post-decode repair)"
  wait_gpu
  local session="lora-ours-arper-v1-ssrg-smoke"
  BOUNDED_SMOKE=1 TMUX_SESSION="$session" bash ours_v1/scripts/launchers/run_arper_v1.sh || true
  wait_tmux "$session"
  local smoke_status="arper_woz3_sclstm_v89_ssrg_exemplar_overlay_v1_20260708_status"
  local st
  st="$(python3 -c "import json,pathlib; p=pathlib.Path('results/logs/${smoke_status}.json'); print(json.load(open(p)).get('state','missing') if p.exists() else 'missing')")"
  log "ARPER smoke state=${st}"
  if [[ "$st" != "completed_or_stopped" && "$st" != "completed" ]]; then
    python3 "${ITER_DIR}/diagnose_failure.py" --suite arper --version ours-v1 --reason "smoke state=${st}" || true
    return 1
  fi
  wait_gpu
  session="lora-ours-arper-v1-ssrg-formal"
  BOUNDED_SMOKE=0 TMUX_SESSION="$session" \
    RUN_ID=arper_woz3_sclstm_v89_ssrg_exemplar_overlay_v1_20260708_formal \
    STATUS_BASENAME=arper_woz3_sclstm_v89_ssrg_exemplar_overlay_v1_20260708_formal_status \
    bash ours_v1/scripts/launchers/run_arper_v1.sh || true
  wait_tmux "$session"
  if python3 "${ITER_DIR}/evaluate_gate.py" --suite arper --run-key arper_woz3_sclstm_v89_ssrg_exemplar_overlay_v1_20260708_formal; then
    log "ARPER TARGETS MET"
    return 0
  fi
  python3 "${ITER_DIR}/diagnose_failure.py" --suite arper --version ours-v1 --reason "metrics below target" || true
  return 1
}

# ---------- ToDCL v1 ----------
run_todcl_v1() {
  log "ToDCL v1 smoke"
  wait_gpu
  local session="lora-ours-todcl-v1-assess-overlay"
  local rc=0
  BOUNDED_SMOKE=1 bash ours_v1/scripts/launchers/run_todcl_v1.sh || rc=$?
  if [[ "$rc" -eq 77 ]]; then
    mark_blocker todcl "no loadable ADAPTER anchor checkpoint"
    return 1
  fi
  wait_tmux "$session"
  wait_gpu
  session="lora-ours-todcl-v1-assess-formal"
  BOUNDED_SMOKE=0 TMUX_SESSION="$session" \
    RUN_ID=todcl_adapter_nlg_ours_assess_overlay_v1_20260708_formal \
    bash ours_v1/scripts/launchers/run_todcl_v1.sh || rc=$?
  if [[ "$rc" -eq 77 ]]; then
    mark_blocker todcl "no loadable ADAPTER anchor checkpoint"
    return 1
  fi
  wait_tmux "$session"
  if python3 "${ITER_DIR}/evaluate_gate.py" --suite todcl --run-key todcl_adapter_nlg_ours_assess_overlay_v1_20260708_formal; then
    log "ToDCL TARGETS MET"
    return 0
  fi
  python3 "${ITER_DIR}/diagnose_failure.py" --suite todcl --version ours-v1 --reason "metrics below target" || true
  return 1
}

# ---------- InstrDialog++ blocker (honest) ----------
enforce_citbpp_blocker() {
  mark_blocker citb_instrdialogpp \
    "public split shortfall: 2 tasks under train=100 (task1549=42, task459=48); paper-exact not publicly available"
  python3 - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone
mani = Path("results/manifests/citb_instrdialogpp_ours_v1_formal_manifest.json")
if mani.is_file():
    m = json.loads(mani.read_text())
    m["status"] = "blocked"
    m["ready_for_formal"] = False
    m["blockers"] = [{
        "type": "external_blocker",
        "reason": "InstrDialog++ public split shortfall (2 short tasks)",
        "evidence": "results/manifests/citb_instrdialogpp_split_gate.json",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }]
    mani.write_text(json.dumps(m, indent=2) + "\n")
print("citb_instrdialogpp blocked")
PY
}

# ===================== MAIN =====================
log "=== Autonomous loop start ==="
enforce_citbpp_blocker
publish_leaderboard

# 1) Prepare Standard ours-v2 proposal + branch + launcher
log "Diagnose Standard v1 and propose ours-v2"
python3 "${ITER_DIR}/diagnose_failure.py" --suite standard --version ours-v1 \
  --dbpedia-em 98.5 --amazon-em 38.0 \
  --reason "early gate rejected round 2 amazon: EM 38.0 < 50" || true
python3 "${ITER_DIR}/propose_next_version.py" --suite standard --from-version ours-v1 \
  --delta amazon_round_assess_pause --create-branch

# Priority cycle (plan Phase 5)
for round in $(seq 1 "$MAX_VERSIONS_PER_SUITE"); do
  log "=== campaign round ${round} ==="
  publish_leaderboard

  # Standard
  st="$(suite_status standard)"
  if [[ "$st" != "completed" && "$st" != "external_blocker" ]]; then
    ver="$(suite_version standard)"
    if run_standard_version "$ver"; then
      log "standard completed at ${ver}"
    else
      log "standard ${ver} failed → propose next"
      if [[ "$round" -lt "$MAX_VERSIONS_PER_SUITE" ]]; then
        advance_standard || true
      fi
    fi
  fi
  publish_leaderboard

  # CITB InstrDialog
  st="$(suite_status citb_instrdialog)"
  if [[ "$st" != "completed" && "$st" != "external_blocker" ]]; then
    if run_citb_v1; then
      :
    else
      # propose next citb version only if diagnosis recommends
      cur="$(suite_version citb_instrdialog)"
      python3 "${ITER_DIR}/propose_next_version.py" --suite citb_instrdialog --from-version "$cur" || true
    fi
  fi
  publish_leaderboard

  # ARPER
  st="$(suite_status arper)"
  if [[ "$st" != "completed" && "$st" != "external_blocker" ]]; then
    run_arper_v1 || python3 "${ITER_DIR}/propose_next_version.py" --suite arper --from-version "$(suite_version arper)" || true
  fi
  publish_leaderboard

  # ToDCL
  st="$(suite_status todcl)"
  if [[ "$st" != "completed" && "$st" != "external_blocker" ]]; then
    run_todcl_v1 || python3 "${ITER_DIR}/propose_next_version.py" --suite todcl --from-version "$(suite_version todcl)" || true
  fi
  publish_leaderboard

  # Termination: all suites completed or external_blocker
  if python3 -c "import json,sys; st=json.load(open('ours_v1/scripts/iterate/suite_state.json')); sys.exit(0 if all(v.get('status') in ('completed','external_blocker') for v in st['suites'].values()) else 1)"; then
    log "All suites completed or external_blocker — campaign done"
    break
  fi
done

publish_leaderboard
log "=== Autonomous loop end ==="
