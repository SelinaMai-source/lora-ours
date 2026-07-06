#!/usr/bin/env bash
# Push best-method reproduction artifacts to GitHub (no weights).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BRANCH="best-method-repro-results-20260706"
FLAG_FILE="${REPO_ROOT}/BEST_METHOD_REPRO_COMPLETE.flag"
LOG="${REPO_ROOT}/results/logs/best_method_repro_push_20260706.log"

mkdir -p results/logs results/tables docs/experiments

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

if [[ ! -f "$FLAG_FILE" ]]; then
  log "ABORT: missing $FLAG_FILE"
  exit 1
fi

# Ensure status symlinks / aliases expected by package README
ln -sf citb_replay50_formal_20260706.md results/logs/citb_replay50_formal_20260706_status.md 2>/dev/null || true

# Generate final summary + package README from flag JSON if not already present
python3 - <<'PY' "$FLAG_FILE" "$REPO_ROOT"
import json, sys
from pathlib import Path
from datetime import datetime

flag = json.loads(Path(sys.argv[1]).read_text())
repo = Path(sys.argv[2])
now = datetime.now().strftime("%Y-%m-%d %H:%M UTC+8")

suites = flag.get("suites", {})
blockers = flag.get("blockers", [])

def row(suite, method, paper, local, match, notes):
    return f"| {suite} | {method} | {paper} | {local} | {match} | {notes} |"

lines = [
    "# Best-Method Reproduction — Final Summary (2026-07-06)",
    "",
    f"**Generated:** {now}  ",
    f"**Branch:** `{flag.get('branch', 'best-method-repro-results-20260706')}`  ",
    f"**Overall:** `{flag.get('overall_status', 'unknown')}`",
    "",
    "## Per-suite best methods",
    "",
    "| Suite | Best method | Paper | Local | Match? | Notes |",
    "|-------|-------------|-------|-------|--------|-------|",
]

for key, label in [
    ("citb_replay50", "CITB"),
    ("standard_olora_v57", "Standard"),
    ("todcl_adapter", "Dialogue ToDCL"),
    ("arper_v87", "Dialogue ARPER"),
]:
    s = suites.get(key, {})
    lines.append(row(
        label,
        s.get("method", "—"),
        s.get("paper_metric", "—"),
        s.get("local_metric", "—"),
        s.get("match", "—"),
        s.get("notes", ""),
    ))

lines += [
    "",
    "## W&B / tmux",
    "",
    "| Suite | tmux session | W&B run / group |",
    "|-------|--------------|-----------------|",
]
for key in ("citb_replay50", "standard_olora_v57", "todcl_adapter", "arper_v87"):
    s = suites.get(key, {})
    lines.append(f"| {key} | `{s.get('tmux', '—')}` | `{s.get('wandb', '—')}` |")

lines += [
    "",
    "## Launchers",
    "",
    "- CITB Replay(50): `scripts/run_citb_instrdialog_replay50_official_base_repro.sh`",
    "- Standard O-LoRA v57: `scripts/run_standard_all_baselines_repro.sh` (METHOD=olora)",
    "- ToDCL ADAPTER: `scripts/run_todcl_adapter_nlg_official_anchor.sh`",
    "- ARPER v87: `scripts/run_arper_woz3_paper_aligned_formal_v87.sh`",
    "",
    "## Setting disclosures",
    "",
    "1. CITB: official-script `500/50/50` (paper text `500/50/100` not in released scripts).",
    "2. Standard v57: single-GPU + `grad_accum=8` official-equivalent.",
    "3. LB-CL: `paper_only` (no code); O-LoRA v57 is best-available anchor.",
    "4. ARPER v87: domain-wise `granularity=0`, exemplar 500, `task_seq=0,5,2,1,3,4`.",
    "5. ToDCL: legacy py37 env, GPT-2 base required for paper anchor.",
    "",
]
if blockers:
    lines += ["## Blockers", ""]
    for b in blockers:
        lines.append(f"- {b}")
    lines.append("")

summary_path = repo / "results/logs/best_method_repro_final_summary_20260706.md"
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

pkg = [
    "# Best-Method Reproduction Package — 2026-07-06",
    "",
    "Review branch: `best-method-repro-results-20260706` on GitHub.",
    "",
    "## Contents",
    "",
    "| Path | Description |",
    "|------|-------------|",
    "| `results/tables/baseline_reproduction_tracker_20260706.md` | Live tracker |",
    "| `results/tables/published_vs_local_comparison_20260706.md` | Paper vs local |",
    "| `results/tables/sota_main_table_20260706.md` | Main table snapshot |",
    "| `results/logs/best_method_repro_final_summary_20260706.md` | Executive summary |",
    "| `results/logs/*_status.md` | Per-run status (citb, o-lora, todcl, arper) |",
    "| `scripts/run_*` launchers | Repro entrypoints |",
    "",
    f"**Overall status:** `{flag.get('overall_status')}`",
    "",
    "No model weights or `wandb/` dirs are committed.",
    "",
]
(repo / "docs/experiments/best_method_repro_package_20260706.md").write_text("\n".join(pkg) + "\n", encoding="utf-8")

# User notification
notify = [
    "# Best-Method Reproduction Complete — User Notification",
    "",
    f"**Time:** {now}",
    "",
    f"**Branch URL:** https://github.com/SelinaMai-source/lora-ours/tree/{flag.get('branch', 'best-method-repro-results-20260706')}",
    "",
    "## Summary (paper vs local)",
    "",
    "| Suite | Method | Paper | Local | Verdict |",
    "|-------|--------|-------|-------|---------|",
]
for key, label in [
    ("citb_replay50", "CITB Replay(50)"),
    ("standard_olora_v57", "Standard O-LoRA"),
    ("todcl_adapter", "ToDCL ADAPTER"),
    ("arper_v87", "ARPER v87"),
]:
    s = suites.get(key, {})
    notify.append(f"| {label} | {s.get('method','—')} | {s.get('paper_metric','—')} | {s.get('local_metric','—')} | {s.get('match','—')} |")

notify += [
    "",
    "## What matched",
    "",
]
matched = [k for k, s in suites.items() if s.get("match") in ("Yes", "Close", "Yes / Close")]
if matched:
    for k in matched:
        s = suites[k]
        notify.append(f"- **{k}**: {s.get('local_metric')} ({s.get('notes','')})")
else:
    notify.append("- See per-suite rows above.")

notify += ["", "## What did not match", ""]
for k, s in suites.items():
    if s.get("match") not in ("Yes", "Close", "Yes / Close"):
        notify.append(f"- **{k}**: {s.get('notes', s.get('status', 'failed or pending'))}")

if blockers:
    notify += ["", "## Blockers", ""]
    for b in blockers:
        notify.append(f"- {b}")

(repo / "results/logs/USER_NOTIFY_BEST_METHOD_REPRO_COMPLETE.md").write_text("\n".join(notify) + "\n", encoding="utf-8")
print("wrote summary + notify")
PY

log "Creating branch $BRANCH from HEAD"
git checkout -B "$BRANCH"

FILES=(
  results/tables/baseline_reproduction_tracker_20260706.md
  results/tables/published_vs_local_comparison_20260706.md
  results/tables/sota_main_table_20260706.md
  results/logs/citb_replay50_formal_20260706_status.md
  results/logs/citb_replay50_formal_20260706.md
  results/logs/citb_replay50_formal_20260706.json
  results/logs/olora_t5large_standard_order1_seed1_official_base_formal_v57_status.md
  results/logs/olora_t5large_standard_order1_seed1_official_base_formal_v57_monitor.md
  results/logs/todcl_adapter_nlg_official_anchor_20260706_status.md
  results/logs/arper_woz3_paper_aligned_exemplar500_formal_v87_status.md
  results/logs/arper_woz3_paper_aligned_exemplar500_formal_v87_status.json
  results/logs/best_method_repro_final_summary_20260706.md
  results/logs/best_method_repro_watch_20260706.md
  results/logs/USER_NOTIFY_BEST_METHOD_REPRO_COMPLETE.md
  docs/experiments/best_method_repro_package_20260706.md
  scripts/run_citb_instrdialog_replay50_official_base_repro.sh
  scripts/run_citb_instrdialog_all_baselines_repro.sh
  scripts/run_standard_all_baselines_repro.sh
  scripts/run_todcl_adapter_nlg_official_anchor.sh
  scripts/run_arper_woz3_paper_aligned_formal_v87.sh
  scripts/run_arper_woz3_official_sclstm_formal_v86.sh
  scripts/monitor_best_method_repro_complete.sh
  scripts/push_best_method_repro_results_20260706.sh
  BEST_METHOD_REPRO_COMPLETE.flag
)

for f in "${FILES[@]}"; do
  [[ -e "$f" ]] && git add "$f" || log "skip missing: $f"
done

if git diff --cached --quiet; then
  log "Nothing to commit"
else
  git commit -m "$(cat <<'EOF'
Add best-method reproduction results package (2026-07-06).

Artifacts-only branch for CITB Replay(50), Standard O-LoRA v57,
ToDCL ADAPTER, and ARPER v87 paper-aligned repro — tables, status
logs, launchers, and honest paper-vs-local disclosure (no weights).
EOF
)"
fi

log "Pushing $BRANCH"
git push -u origin "$BRANCH"
log "Push complete: https://github.com/SelinaMai-source/lora-ours/tree/$BRANCH"
