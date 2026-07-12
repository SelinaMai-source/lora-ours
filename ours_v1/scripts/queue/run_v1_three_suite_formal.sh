#!/usr/bin/env bash
# Serial smoke→formal GPU queue with suite isolation.
# - After launcher returns, ALWAYS read status/gate files (do not only wait for tmux).
# - Suite failure marks that suite FAIL and continues others (STOP_ON_FAIL=0 default).
# - Formal only launches when corresponding smoke PASS.
# - No parallel orphan formals.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

QUEUE_LOG="${QUEUE_LOG:-results/logs/ours_v1_three_suite_formal_queue_20260712.log}"
STATE_JSON="${STATE_JSON:-results/logs/ours_v1_three_suite_formal_queue_20260712.json}"
POLL_SEC="${POLL_SEC:-60}"
# Suite isolation: do not halt the whole campaign on one suite failure.
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

mkdir -p results/logs results/manifests results/tables

log() { echo "[$(date -Iseconds)] $*" | tee -a "$QUEUE_LOG"; }

gpu_empty() {
  local out
  out="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
  [[ -z "${out// /}" ]]
}

wait_gpu() {
  while ! gpu_empty; do
    log "GPU busy; waiting ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
}

wait_session_done() {
  local session="$1"
  # Wait until tmux session is gone AND GPU is idle.
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep "$POLL_SEC"
  done
  wait_gpu
}

# Read Standard overlay status.md → state token (completed|rejected|failed|...).
read_standard_state() {
  local run_name="$1"
  local status_file="results/logs/${run_name}_status.md"
  if [[ ! -f "$status_file" ]]; then
    echo "missing"
    return
  fi
  python3 - "$status_file" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
m = re.search(r"(?m)^-\s*state:\s*(\S+)", text)
print(m.group(1) if m else "unknown")
PY
}

# CITB / generic: look for completed markers in log or run summary.
read_citb_pass() {
  local run_name="$1"
  local summary="results/runs/${run_name}/ccfa_postprocess/summary.json"
  local manifest="results/runs/${run_name}/run_manifest.json"
  if [[ -f "$summary" ]]; then
    echo "completed"
    return
  fi
  if [[ -f "$manifest" ]]; then
    python3 -c "import json; d=json.load(open('$manifest')); print(d.get('state','unknown'))" 2>/dev/null || echo "unknown"
    return
  fi
  echo "missing"
}

read_arper_state() {
  local status_basename="$1"
  local jf="results/logs/${status_basename}.json"
  if [[ ! -f "$jf" ]]; then
    echo "missing"
    return
  fi
  python3 -c "import json; print(json.load(open('$jf')).get('state','unknown'))" 2>/dev/null || echo "unknown"
}

read_todcl_pass() {
  local run_id="$1"
  local logf="/root/autodl-tmp/lora-ours-logs/${run_id}.log"
  if [[ -f "$logf" ]] && grep -Eqi 'BLEU|EER|completed|finished' "$logf"; then
    # Prefer explicit exit marker if present
    if grep -Eqi 'TODCL_OVERLAY_COMPLETE|formal complete|Training completed' "$logf"; then
      echo "completed"
      return
    fi
  fi
  if [[ -f "results/logs/${run_id}.exit.json" ]]; then
    python3 -c "import json; print('completed' if json.load(open('results/logs/${run_id}.exit.json')).get('ok') else 'failed')" 2>/dev/null || echo "unknown"
    return
  fi
  echo "unknown"
}

state_is_pass() {
  local st="$1"
  case "$st" in
    completed|completed_or_stopped|pass|PASS|success) return 0 ;;
    *) return 1 ;;
  esac
}

state_is_fail() {
  local st="$1"
  case "$st" in
    rejected|failed|fail|FAIL|stopped_invalid|blocked|error) return 0 ;;
    *) return 1 ;;
  esac
}

update_job() {
  local job_id="$1"
  local status="$2"
  local detail="${3:-}"
  python3 - "$STATE_JSON" "$job_id" "$status" "$detail" <<'PY'
import json, sys
from pathlib import Path
path, job_id, status, detail = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
data = json.loads(path.read_text()) if path.is_file() else {"jobs": []}
jobs = data.setdefault("jobs", [])
found = False
for j in jobs:
    if j.get("id") == job_id:
        j["status"] = status
        j["detail"] = detail
        found = True
        break
if not found:
    jobs.append({"id": job_id, "status": status, "detail": detail})
path.write_text(json.dumps(data, indent=2) + "\n")
PY
}

# Launch a job, wait for its tmux session (if any), then evaluate gate/status.
# Returns 0 on PASS, 1 on FAIL. Never aborts the whole queue unless STOP_ON_FAIL=1.
run_job() {
  local job_id="$1"
  local session="$2"
  local gate_kind="$3"   # standard|citb|arper|todcl|none
  local gate_key="$4"    # run_name / status basename / run_id
  shift 4
  log "=== ${job_id} ==="
  wait_gpu

  local rc=0
  "$@" || rc=$?
  if [[ "$rc" -ne 0 && "$rc" -ne 75 && "$rc" -ne 76 && "$rc" -ne 77 ]]; then
    log "LAUNCH_FAILED ${job_id} exit=${rc}"
    update_job "$job_id" "FAIL" "launcher_exit=${rc}"
    echo "${job_id}:failed:${rc}" >> "${STATE_JSON}.failures"
    if [[ "${STOP_ON_FAIL}" == "1" ]]; then
      return "$rc"
    fi
    return 1
  fi
  if [[ "$rc" -eq 77 ]]; then
    log "BLOCKED ${job_id} exit=77"
    update_job "$job_id" "BLOCKED" "launcher_exit=77"
    echo "${job_id}:blocked:77" >> "${STATE_JSON}.failures"
    return 1
  fi
  if [[ "$rc" -eq 75 || "$rc" -eq 76 ]]; then
    log "SKIP ${job_id} exit=${rc} (gpu busy or session exists)"
    update_job "$job_id" "SKIP" "launcher_exit=${rc}"
    return 1
  fi

  if [[ -n "$session" ]]; then
    log "waiting session ${session} (then gate read)"
    wait_session_done "$session"
  else
    wait_gpu
  fi

  local st="unknown"
  case "$gate_kind" in
    standard) st="$(read_standard_state "$gate_key")" ;;
    citb) st="$(read_citb_pass "$gate_key")" ;;
    arper) st="$(read_arper_state "$gate_key")" ;;
    todcl) st="$(read_todcl_pass "$gate_key")" ;;
    none) st="completed" ;;
  esac

  if state_is_pass "$st"; then
    log "PASS ${job_id} state=${st}"
    update_job "$job_id" "PASS" "state=${st}"
    return 0
  fi
  if state_is_fail "$st"; then
    log "FAIL ${job_id} state=${st}"
    update_job "$job_id" "FAIL" "state=${st}"
    echo "${job_id}:failed:${st}" >> "${STATE_JSON}.failures"
    return 1
  fi
  # unknown: treat as fail for formal gating honesty
  log "UNKNOWN_GATE ${job_id} state=${st} → treat as FAIL for formal trigger"
  update_job "$job_id" "FAIL" "unknown_gate state=${st}"
  echo "${job_id}:failed:unknown:${st}" >> "${STATE_JSON}.failures"
  return 1
}

run_smoke_then_formal() {
  local suite="$1"
  local smoke_id="$2"
  local smoke_session="$3"
  local smoke_gate_kind="$4"
  local smoke_gate_key="$5"
  local formal_id="$6"
  local formal_session="$7"
  local formal_gate_kind="$8"
  local formal_gate_key="$9"
  shift 9
  # remaining: smoke_cmd... --- formal_cmd...
  local smoke_cmd=()
  local formal_cmd=()
  local mode=smoke
  for arg in "$@"; do
    if [[ "$arg" == "---" ]]; then
      mode=formal
      continue
    fi
    if [[ "$mode" == "smoke" ]]; then
      smoke_cmd+=("$arg")
    else
      formal_cmd+=("$arg")
    fi
  done

  log "suite=${suite} smoke→formal"
  if run_job "$smoke_id" "$smoke_session" "$smoke_gate_kind" "$smoke_gate_key" "${smoke_cmd[@]}"; then
    log "smoke PASS → launching formal for ${suite}"
    if run_job "$formal_id" "$formal_session" "$formal_gate_kind" "$formal_gate_key" "${formal_cmd[@]}"; then
      update_job "${suite}:suite" "PASS" "smoke+formal"
      return 0
    else
      update_job "${suite}:suite" "FAIL" "formal_failed"
      return 1
    fi
  else
    log "smoke FAIL/BLOCKED → skipping formal for ${suite}"
    update_job "$formal_id" "SKIPPED" "smoke_did_not_pass"
    update_job "${suite}:suite" "FAIL" "smoke_failed"
    return 1
  fi
}

log "Phase 1: freeze matrix + configs + preflights"
python3 ours_v1/scripts/freeze_v1_matrix.py | tee -a "$QUEUE_LOG" || true
python3 ours_v1/scripts/generate_formal_configs.py | tee -a "$QUEUE_LOG" || true

if [[ ! -f results/logs/arper_woz3_paper_aligned_exemplar500_batch128_formal_v89_status.json ]]; then
  for f in arper_woz3_paper_aligned_exemplar500_batch128_formal_v89.cfg \
           arper_woz3_paper_aligned_exemplar500_batch128_formal_v89_status.json \
           arper_woz3_paper_aligned_exemplar500_batch128_formal_v89_status.md; do
    if [[ -f "/root/lora-ours/results/logs/${f}" ]]; then
      cp "/root/lora-ours/results/logs/${f}" "results/logs/${f}"
      log "synced ${f}"
    fi
  done
fi

python3 ours_v1/scripts/run_strict_preflights.py | tee -a "$QUEUE_LOG" || {
  log "preflight reported blockers; continuing only for ready suites"
}

# Force InstrDialog++ blocked when split gate says formal_allowed=false
python3 - <<'PY' | tee -a "$QUEUE_LOG"
import json
from pathlib import Path
from datetime import datetime, timezone
repo = Path(".")
split = repo / "results/manifests/citb_instrdialogpp_split_gate.json"
mani = repo / "results/manifests/citb_instrdialogpp_ours_v1_formal_manifest.json"
if split.is_file() and mani.is_file():
    sg = json.loads(split.read_text())
    m = json.loads(mani.read_text())
    if not sg.get("formal_allowed", False) or sg.get("all_tasks_meet_target") is False:
        m["status"] = "blocked"
        m["ready_for_formal"] = False
        m["blockers"] = [{
            "type": "external_blocker",
            "reason": sg.get("blocker") or "InstrDialog++ public split shortfall",
            "short_tasks": sg.get("short_tasks"),
            "disclosure": "paper-exact 100/50/100 not available on public assets; do not skip/resample to fake paper-exact",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }]
        mani.write_text(json.dumps(m, indent=2) + "\n")
        print(f"FORCE_BLOCKED citb_instrdialogpp: {m['blockers'][0]['reason']}")
    else:
        print("citb_instrdialogpp split OK")
else:
    print("citb_instrdialogpp gate/manifest missing")
PY

: > "${STATE_JSON}.failures" 2>/dev/null || true
echo '{"started_at":"'"$(date -Iseconds)"'","status":"RUNNING","jobs":[],"policy":"suite_isolated_smoke_then_formal"}' > "$STATE_JSON"

log "Phase 2: suite-isolated serial smoke→formal"

# --- Standard: by default SKIP in this legacy queue if already FAILED in v1;
# autonomous loop / ours-v2 handles Standard separately. ---
if [[ "${RUN_STANDARD:-0}" == "1" ]]; then
  run_smoke_then_formal "standard" \
    "standard-smoke" "lora-ours-standard-v1-smoke" "standard" \
    "olora_official_base_ours_overlay_v1_20260708_smoke_order1_seed1" \
    "standard-formal" "lora-ours-standard-v1-formal" "standard" \
    "olora_official_base_ours_overlay_v1_20260708_formal_order1_seed1" \
    bash ours_v1/scripts/launchers/run_standard_v1.sh \
    --- \
    env FORMAL=1 TMUX_SESSION=lora-ours-standard-v1-formal bash ours_v1/scripts/launchers/run_standard_v1.sh \
    || true
else
  log "SKIP standard in legacy queue (use ours-v2 autonomous loop); set RUN_STANDARD=1 to force"
  update_job "standard:suite" "DEFERRED" "handled_by_ours_v2_or_iterate"
fi

# --- CITB InstrDialog ---
run_smoke_then_formal "citb_instrdialog" \
  "citb-smoke" "lora-ours-citb-ours-v1-smoke" "citb" \
  "citb_instrdialog_order1_seed1_ours_v1_20260708_smoke_strict" \
  "citb-formal" "lora-ours-citb-ours-v1-formal" "citb" \
  "citb_instrdialog_order1_seed1_ours_v1_20260708_formal_strict" \
  bash ours_v1/scripts/launchers/run_citb_v1.sh \
  --- \
  env FORMAL=1 bash ours_v1/scripts/launchers/run_citb_v1.sh \
  || true

# --- CITB InstrDialog++ (blocked) ---
if python3 -c "import json; m=json.load(open('results/manifests/citb_instrdialogpp_ours_v1_formal_manifest.json')); exit(0 if m.get('ready_for_formal') else 1)" 2>/dev/null; then
  run_smoke_then_formal "citb_instrdialogpp" \
    "citbpp-smoke" "lora-ours-citb-pp-ours-v1-smoke" "citb" \
    "citb_instrdialogpp_order1_seed1_ours_v1_20260708_smoke_strict" \
    "citbpp-formal" "lora-ours-citb-pp-ours-v1-formal" "citb" \
    "citb_instrdialogpp_order1_seed1_ours_v1_20260708_formal_strict" \
    bash ours_v1/scripts/launchers/run_citb_pp_v1.sh \
    --- \
    env FORMAL=1 bash ours_v1/scripts/launchers/run_citb_pp_v1.sh \
    || true
else
  log "BLOCKER citb_instrdialogpp — keeping blocked with disclosure"
  update_job "citbpp:suite" "BLOCKED" "public_split_shortfall"
  echo "citb_instrdialogpp:blocked" >> "${STATE_JSON}.failures"
fi

# --- ARPER ---
run_smoke_then_formal "arper" \
  "arper-smoke" "lora-ours-arper-v1-ssrg-smoke" "arper" \
  "arper_woz3_sclstm_v89_ssrg_exemplar_overlay_v1_20260708_status" \
  "arper-formal" "lora-ours-arper-v1-ssrg-formal" "arper" \
  "arper_woz3_sclstm_v89_ssrg_exemplar_overlay_v1_20260708_formal_status" \
  env BOUNDED_SMOKE=1 TMUX_SESSION=lora-ours-arper-v1-ssrg-smoke bash ours_v1/scripts/launchers/run_arper_v1.sh \
  --- \
  env BOUNDED_SMOKE=0 TMUX_SESSION=lora-ours-arper-v1-ssrg-formal \
      RUN_ID=arper_woz3_sclstm_v89_ssrg_exemplar_overlay_v1_20260708_formal \
      STATUS_BASENAME=arper_woz3_sclstm_v89_ssrg_exemplar_overlay_v1_20260708_formal_status \
      bash ours_v1/scripts/launchers/run_arper_v1.sh \
  || true

# --- ToDCL ---
run_smoke_then_formal "todcl" \
  "todcl-smoke" "lora-ours-todcl-v1-assess-overlay" "todcl" \
  "todcl_adapter_nlg_ours_assess_overlay_v1_20260708" \
  "todcl-formal" "lora-ours-todcl-v1-assess-formal" "todcl" \
  "todcl_adapter_nlg_ours_assess_overlay_v1_20260708_formal" \
  env BOUNDED_SMOKE=1 bash ours_v1/scripts/launchers/run_todcl_v1.sh \
  --- \
  env BOUNDED_SMOKE=0 TMUX_SESSION=lora-ours-todcl-v1-assess-formal \
      RUN_ID=todcl_adapter_nlg_ours_assess_overlay_v1_20260708_formal \
      bash ours_v1/scripts/launchers/run_todcl_v1.sh \
  || true

log "Phase 3: publish results"
python3 ours_v1/scripts/publish_formal_results.py | tee -a "$QUEUE_LOG" || true

python3 - "$STATE_JSON" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text())
data["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
data["status"] = "COMPLETED_WITH_ISOLATION"
p.write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PY

log "Queue complete; state=${STATE_JSON}"
