#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/results/logs"
mkdir -p "$LOG_DIR"

PRIMARY_BENCHMARK="instrdialog"
SECONDARY_BENCHMARK="instrdialog++"
BASELINE_VARIANT_IDS="bank_no_router,periodic_latest,replay_b10,replay_b50,router_only,seq"
SECONDARY_MAX_SEEN_GAP="0.02"
SECONDARY_MAX_TOKEN_F1_GAP="0.02"
SECONDARY_MAX_FORGETTING_GAP="0.05"

select_primary_winner_json() {
  python3 scripts/select_paper_winner.py --benchmark "$PRIMARY_BENCHMARK"
}

select_cross_benchmark_json() {
  python3 scripts/select_paper_winner.py \
    --benchmark "$PRIMARY_BENCHMARK" \
    --secondary-benchmark "$SECONDARY_BENCHMARK" \
    --secondary-max-seen-gap "$SECONDARY_MAX_SEEN_GAP" \
    --secondary-max-token-f1-gap "$SECONDARY_MAX_TOKEN_F1_GAP" \
    --secondary-max-forgetting-gap "$SECONDARY_MAX_FORGETTING_GAP"
}

sync_execution_manifest() {
  python3 scripts/sync_paper_executions.py
}

json_field() {
  local payload="$1"
  local field="$2"
  python3 - "$payload" "$field" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
field = sys.argv[2]
value = payload
for part in field.split("."):
    if isinstance(value, dict):
        value = value.get(part)
    else:
        value = None
        break
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
else:
    print(value)
PY
}

build_repair_variant_ids() {
  local payload="$1"
  python3 - "$payload" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
items = []
for variant in ("ours_no_overlap", "ours_full"):
    if variant not in items:
        items.append(variant)
winner = str(payload.get("winner_variant_id", "")).strip()
if winner and winner not in items:
    items.append(winner)
for variant in payload.get("top_ours_variant_ids", []) or []:
    variant = str(variant).strip()
    if variant and variant not in items:
        items.append(variant)
print(",".join(items))
PY
}

phase_wait_for_session_end() {
  local session_name="$1"
  while tmux has-session -t "$session_name" 2>/dev/null; do
    sync_execution_manifest
    echo "[watch] waiting for tmux session: $session_name"
    sleep 60
  done
}

run_until_success() {
  local phase_name="$1"
  shift
  local attempt=1
  while true; do
    echo "[watch] phase=$phase_name attempt=$attempt"
    if "$@"; then
      echo "[watch] phase=$phase_name status=success"
      break
    fi
    echo "[watch] phase=$phase_name status=failed attempt=$attempt"
    attempt=$((attempt + 1))
    sleep 60
  done
}

refresh_primary_winner_state() {
  WINNER_JSON="$(select_primary_winner_json)"
  echo "[watch] primary_winner_state=$WINNER_JSON"
  WINNER_VARIANT="$(json_field "$WINNER_JSON" "winner_variant_id")"
  WINNER_RUN_NAME="$(json_field "$WINNER_JSON" "winner_run_name")"
  WINNER_BEATS_BASELINE="$(json_field "$WINNER_JSON" "winner_beats_best_baseline")"
  REPAIR_VARIANT_IDS="$(build_repair_variant_ids "$WINNER_JSON")"
}

refresh_cross_benchmark_state() {
  CROSS_BENCHMARK_JSON="$(select_cross_benchmark_json)"
  echo "[watch] cross_benchmark_state=$CROSS_BENCHMARK_JSON"
  WINNER_CROSS_BENCHMARK_PASS="$(json_field "$CROSS_BENCHMARK_JSON" "winner_cross_benchmark_pass")"
  WINNER_BEATS_SECONDARY_BASELINE="$(json_field "$CROSS_BENCHMARK_JSON" "winner_beats_secondary_baseline")"
  SECONDARY_MATCH_RUN_NAME="$(json_field "$CROSS_BENCHMARK_JSON" "secondary_winner_match.resolved_run_name")"
}

append_override_args_from_json() {
  local overrides_json="$1"
  if [ -z "$overrides_json" ]; then
    return 0
  fi
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    WINNER_OVERRIDE_ARGS+=(--set "$line")
  done < <(python3 - "$overrides_json" <<'PY'
import json
import sys

payload = sys.argv[1].strip()
if payload:
    data = json.loads(payload)
    for key in sorted(data):
        print(f"{key}={json.dumps(data[key], ensure_ascii=False)}")
PY
)
}

configure_winner_overrides() {
  local winner_payload="$1"
  WINNER_RUN_SUFFIX="$(json_field "$winner_payload" "best_ours.run_name_suffix")"
  WINNER_OVERRIDE_ARGS=()
  if [ -n "$WINNER_RUN_SUFFIX" ]; then
    WINNER_OVERRIDE_ARGS+=(--run-name-suffix "$WINNER_RUN_SUFFIX")
  fi
  append_override_args_from_json "$(json_field "$winner_payload" "best_ours.generic_overrides_json")"
}

run_repair_preset() {
  local suffix="$1"
  shift
  run_until_success "repair_${suffix}" \
    python3 scripts/run_paper_matrix.py \
      --benchmarks "$PRIMARY_BENCHMARK" \
      --variant-ids "$REPAIR_VARIANT_IDS" \
      --run-name-suffix "$suffix" \
      --skip-existing \
      "$@"
  run_until_success "artifacts_${suffix}" \
    python3 scripts/build_paper_artifacts.py
  run_until_success "package_${suffix}" \
    python3 scripts/package_paper_results.py
}

run_secondary_winner_check() {
  if [ -z "${WINNER_VARIANT:-}" ]; then
    echo "[watch] secondary winner check skipped: empty winner variant"
    return 1
  fi
  configure_winner_overrides "$WINNER_JSON"
  run_until_success "secondary_${WINNER_VARIANT}" \
    python3 scripts/run_paper_matrix.py \
      --benchmarks "$SECONDARY_BENCHMARK" \
      --variant-ids "$WINNER_VARIANT" \
      --skip-existing \
      "${WINNER_OVERRIDE_ARGS[@]}"
}

winner_passes_all_guards() {
  refresh_primary_winner_state
  if [ -z "$WINNER_VARIANT" ]; then
    echo "[watch] no completed ours winner found yet"
    return 1
  fi
  if [ "$WINNER_BEATS_BASELINE" != "true" ]; then
    echo "[watch] primary guard failed: winner=$WINNER_VARIANT run=$WINNER_RUN_NAME repair_variants=$REPAIR_VARIANT_IDS"
    return 1
  fi
  run_secondary_winner_check
  refresh_cross_benchmark_state
  if [ "$WINNER_CROSS_BENCHMARK_PASS" = "true" ]; then
    echo "[watch] winner passed both benchmarks: primary=$WINNER_RUN_NAME secondary=${SECONDARY_MATCH_RUN_NAME:-}"
    return 0
  fi
  echo "[watch] secondary guard failed: winner=$WINNER_VARIANT run=$WINNER_RUN_NAME secondary_match=${SECONDARY_MATCH_RUN_NAME:-none}"
  return 1
}

repair_until_sota() {
  if winner_passes_all_guards; then
    return 0
  fi

  echo "[watch] launching adaptive repair search from winner=${WINNER_VARIANT:-none} variants=${REPAIR_VARIANT_IDS:-none}"

  run_repair_preset "rw0_gap0" \
    --set router.router_warmup_segments=0 \
    --set router.margin_filter_min_gap=0.0
  winner_passes_all_guards && return 0

  run_repair_preset "rw0_gap0_hits1" \
    --set router.router_warmup_segments=0 \
    --set router.margin_filter_min_gap=0.0 \
    --set drift.min_consecutive_probe_hits=1
  winner_passes_all_guards && return 0

  run_repair_preset "rw0_gap0_thr008" \
    --set router.router_warmup_segments=0 \
    --set router.margin_filter_min_gap=0.0 \
    --set drift.threshold=0.08
  winner_passes_all_guards && return 0

  run_repair_preset "rw0_gap0_hits1_thr008" \
    --set router.router_warmup_segments=0 \
    --set router.margin_filter_min_gap=0.0 \
    --set drift.min_consecutive_probe_hits=1 \
    --set drift.threshold=0.08
  winner_passes_all_guards && return 0

  run_repair_preset "rw0_gap002_hits1_thr008" \
    --set router.router_warmup_segments=0 \
    --set router.margin_filter_min_gap=0.02 \
    --set drift.min_consecutive_probe_hits=1 \
    --set drift.threshold=0.08
  winner_passes_all_guards && return 0

  run_repair_preset "rw0_gap0_hits1_thr006" \
    --set router.router_warmup_segments=0 \
    --set router.margin_filter_min_gap=0.0 \
    --set drift.min_consecutive_probe_hits=1 \
    --set drift.threshold=0.06
  winner_passes_all_guards && return 0

  run_repair_preset "rw0_gap0_hits1_thr008_lr2e3" \
    --set router.router_warmup_segments=0 \
    --set router.margin_filter_min_gap=0.0 \
    --set drift.min_consecutive_probe_hits=1 \
    --set drift.threshold=0.08 \
    --set router.learning_rate=0.002
  winner_passes_all_guards && return 0

  run_repair_preset "rw0_gap0_hits1_thr008_lr5e4" \
    --set router.router_warmup_segments=0 \
    --set router.margin_filter_min_gap=0.0 \
    --set drift.min_consecutive_probe_hits=1 \
    --set drift.threshold=0.08 \
    --set train.lr=5.0e-5
  winner_passes_all_guards && return 0

  run_repair_preset "rw0_gap0_hits1_thr008_br4" \
    --set router.router_warmup_segments=0 \
    --set router.margin_filter_min_gap=0.0 \
    --set drift.min_consecutive_probe_hits=1 \
    --set drift.threshold=0.08 \
    --set bank.max_branches=4
  winner_passes_all_guards && return 0

  run_repair_preset "rw0_gap0_hits1_thr008_anchor128" \
    --set router.router_warmup_segments=0 \
    --set router.margin_filter_min_gap=0.0 \
    --set drift.min_consecutive_probe_hits=1 \
    --set drift.threshold=0.08 \
    --set drift.anchor_size=128
  winner_passes_all_guards && return 0

  run_repair_preset "rw0_gap0_hits1_thr008_k100" \
    --set router.router_warmup_segments=0 \
    --set router.margin_filter_min_gap=0.0 \
    --set drift.min_consecutive_probe_hits=1 \
    --set drift.threshold=0.08 \
    --set drift.monitor_interval=100
  winner_passes_all_guards && return 0

  run_repair_preset "rw0_gap0_hits1_thr008_r32" \
    --set router.router_warmup_segments=0 \
    --set router.margin_filter_min_gap=0.0 \
    --set drift.min_consecutive_probe_hits=1 \
    --set drift.threshold=0.08 \
    --set lora.r=32 \
    --set lora.alpha=64
  winner_passes_all_guards && return 0

  echo "[watch] adaptive repair presets exhausted without clearing both guards"
  return 1
}

echo "[watch] full run watchdog started"

# Phase 1: current single-seed InstrDialog main batch is already running elsewhere.
phase_wait_for_session_end "paper-instrdialog-main-s123"

# Recover/complete any missing single-seed main runs.
run_until_success "instrdialog_main_recover" \
  python3 scripts/run_paper_matrix.py --benchmarks "$PRIMARY_BENCHMARK" --categories "main" --skip-existing

# Single-seed ablations and transfer benchmark.
run_until_success "instrdialog_ablation" \
  python3 scripts/run_paper_matrix.py --benchmarks "$PRIMARY_BENCHMARK" --categories "ablation" --skip-existing

run_until_success "instrdialogpp_baselines" \
  python3 scripts/run_paper_matrix.py --benchmarks "$SECONDARY_BENCHMARK" --variant-ids "$BASELINE_VARIANT_IDS" --skip-existing

run_until_success "artifacts_single_seed" \
  python3 scripts/build_paper_artifacts.py

run_until_success "package_single_seed" \
  python3 scripts/package_paper_results.py

run_until_success "sync_single_seed_manifest" sync_execution_manifest

if ! repair_until_sota; then
  echo "[watch] failed to find a winner that clears both SOTA guards"
  exit 1
fi

configure_winner_overrides "$WINNER_JSON"
echo "[watch] selected_multiseed_winner variant=$WINNER_VARIANT run_name=$WINNER_RUN_NAME suffix=${WINNER_RUN_SUFFIX:-none}"

run_until_success "artifacts_single_seed_validated" \
  python3 scripts/build_paper_artifacts.py

run_until_success "package_single_seed_validated" \
  python3 scripts/package_paper_results.py

# Multi-seed expansion for main tables using the best validated single-seed winner.
run_until_success "build_multiseed_matrix" \
  python3 scripts/build_paper_run_matrix.py --seeds "123,231,340" --ours-main-variant "$WINNER_VARIANT"

run_until_success "multiseed_baselines" \
  python3 scripts/run_paper_matrix.py \
    --benchmarks "$PRIMARY_BENCHMARK,$SECONDARY_BENCHMARK" \
    --variant-ids "$BASELINE_VARIANT_IDS" \
    --skip-existing

run_until_success "multiseed_ours_winner" \
  python3 scripts/run_paper_matrix.py \
    --benchmarks "$PRIMARY_BENCHMARK,$SECONDARY_BENCHMARK" \
    --variant-ids "$WINNER_VARIANT" \
    --skip-existing \
    "${WINNER_OVERRIDE_ARGS[@]}"

run_until_success "artifacts_multiseed" \
  python3 scripts/build_paper_artifacts.py

run_until_success "package_multiseed" \
  python3 scripts/package_paper_results.py

echo "[watch] full run watchdog finished"
