#!/usr/bin/env bash
# Master dispatcher for Standard T5-large PEFT CL baselines (order1 seed1).
# Usage: METHOD=olora DRY_RUN=0 bash scripts/run_standard_all_baselines_repro.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

METHOD="${METHOD:-}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"

O_LORA_ROOT="${O_LORA_ROOT:-/root/autodl-tmp/lora-baselines-run_v1/external_sources/o_lora}"
LFPT5_ROOT="${LFPT5_ROOT:-/root/autodl-tmp/lora-baselines-run_v1/external_sources/lfpt5}"
PROG_ROOT="${PROG_ROOT:-/root/autodl-tmp/lora-baselines-run_v1/external_sources/progressive_prompts}"

if [[ -z "${METHOD}" ]]; then
  cat <<'EOF'
Standard T5-large CL baseline dispatcher (order1 seed1).

METHOD values:
  olora            — O-LoRA official-equivalent (v57 path; verify only if RUN_NAME set)
  olora_smoke      — capped smoke via run_olora_standard_order1_official_base_smoke.sh
  lfpt5            — LFPT5 (repo downloaded; T5-large CL script audit pending)
  progressive      — Progressive Prompts (BERT codebase in repo; T5 CL path TBD)
  seqlora          — SeqLoRA (paper baseline; no isolated official script in O-LoRA repo)
  inclora          — IncLoRA (same)
  replay           — Replay baseline (no isolated official script found)
  lb_cl            — paper_only — no author code
  mtl              — paper_only upper bound ~80.0 EM

Env: DRY_RUN=1 | SMOKE=1
EOF
  exit 0
fi

case "${METHOD}" in
  olora)
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "DRY_RUN: would run O-LoRA official-equivalent formal (see v57 manifest)"
      echo "Manifest: results/runs/olora_t5large_standard_order1_seed1_official_base_formal_v57/run_manifest.json"
      exit 0
    fi
    echo "O-LoRA v57 formal already completed. Re-run requires explicit RUN_NAME and launcher audit." >&2
    echo "See results/logs/olora_t5large_standard_order1_seed1_official_base_formal_v57_status.md" >&2
    exit 0
    ;;
  olora_smoke)
    export DRY_RUN SMOKE
    exec bash scripts/run_olora_standard_order1_official_base_smoke.sh
    ;;
  lfpt5)
    echo "LFPT5: repo at ${LFPT5_ROOT}" >&2
    echo "Blocker: T5-large standard CL order script not verified in LFPT5 repo root." >&2
    echo "Paper target EM ~72.7 (O-LoRA table) / ~71.3 (LB-CL table)." >&2
    exit 8
    ;;
  progressive)
    echo "Progressive Prompts: repo at ${PROG_ROOT}" >&2
    echo "Blocker: run_prog_prompts.sh is BERT codebase; T5-large CL protocol differs from O-LoRA benchmark." >&2
    echo "Paper target EM ~75.1–76.1 depending table." >&2
    exit 8
    ;;
  seqlora|inclora|replay)
    echo "${METHOD}: baseline cited in O-LoRA/LB-CL papers but no isolated runnable script in ${O_LORA_ROOT}." >&2
    echo "Audit O-LoRA README/supplement or implement from paper — not exposed as order_1.sh variant." >&2
    exit 8
    ;;
  lb_cl)
    echo "LB-CL: paper_only_baseline — NeurIPS checklist says code not attached; no author repo found." >&2
    echo "Paper target avg EM 76.7 (order1/2/3: 76.9/76.5/76.8)." >&2
    exit 7
    ;;
  mtl)
    echo "MTL upper bound ~80.0 EM — paper reference only, not a CL deployable method." >&2
    exit 7
    ;;
  *)
    echo "Unknown METHOD=${METHOD}" >&2
    exit 2
    ;;
esac
