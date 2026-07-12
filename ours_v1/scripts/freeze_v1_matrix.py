#!/usr/bin/env python3
"""Freeze ours-v1 SHA, official base SHAs, and three-suite formal experiment matrix."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "manifests"
AUTODL = Path("/root/autodl-tmp/lora-baselines-run_v1/external_sources")

OFFICIAL_REPOS = {
    "citb": AUTODL / "citb",
    "o_lora": AUTODL / "o_lora",
    "arper": AUTODL / "arper",
    "todcl": AUTODL / "todcl",
}

MATRIX = [
    {
        "suite": "citb_instrdialog",
        "benchmark": "InstrDialog-order1",
        "official_base": "Replay(50)",
        "ours_v1_delta": "SSRG replay_ratio=0.5, min_replay_per_segment=64",
        "metric": "ROUGE-L AR / BWT",
        "split_policy": "official_script_500_50_50",
        "split_disclosure": "Paper 500/50/100; public script 500/50/50",
        "smoke_config": "configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v1_20260708_smoke_strict.yaml",
        "formal_config": "configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v1_20260708_formal_strict.yaml",
        "launcher": "ours_v1/scripts/launchers/run_citb_v1.sh",
        "locked_base_metric": {"AR": 39.98},
        "target_plus_one_third": {"AR": 53.87},
    },
    {
        "suite": "citb_instrdialogpp",
        "benchmark": "InstrDialog++-order1",
        "official_base": "Replay(50) public long stream",
        "ours_v1_delta": "Same SSRG mechanism; skip_empty_segments; no v49-v51 patches",
        "metric": "ROUGE-L AR / BWT",
        "split_policy": "public_actual_100_50_100_with_shortfall",
        "split_disclosure": "Paper 100/50/100; public data ~100/25/25 on short tasks",
        "smoke_config": "configs/ccfa_three_suite/citb_instrdialogpp_order1_seed1_ours_v1_20260708_smoke_strict.yaml",
        "formal_config": "configs/ccfa_three_suite/citb_instrdialogpp_order1_seed1_ours_v1_20260708_formal_strict.yaml",
        "launcher": "ours_v1/scripts/launchers/run_citb_pp_v1.sh",
        "locked_base_metric": {"AR": None},
        "target_plus_one_third": {"AR": None},
    },
    {
        "suite": "standard",
        "benchmark": "O-LoRA T5-large order1",
        "official_base": "O-LoRA v57",
        "ours_v1_delta": "class-coverage SSRG + assess-retention gate 0.25",
        "metric": "final average EM",
        "split_policy": "official-equivalent",
        "split_disclosure": "single-GPU + grad_accum=8; not literal multi-GPU official",
        "smoke_config": "env:ours_v1/scripts/launchers/run_standard_v1.sh",
        "formal_config": "env:FORMAL=1 run_standard_v1.sh",
        "launcher": "ours_v1/scripts/launchers/run_standard_v1.sh",
        "locked_base_metric": {"EM": 76.81},
        "target_plus_one_third": {"EM": 84.5},
    },
    {
        "suite": "arper",
        "benchmark": "ARPER MultiWOZ-2.0 Path B SCLSTM",
        "official_base": "v89 SCLSTM exemplar500 batch128",
        "ours_v1_delta": "SSRG exemplar selection overlay",
        "metric": "BLEU4 / SER",
        "split_policy": "official Path B",
        "split_disclosure": "task_seq 0,5,2,1,3,4; local BLEU below paper",
        "smoke_config": "BOUNDED_SMOKE=1 run_arper_v1.sh",
        "formal_config": "run_arper_v1.sh",
        "launcher": "ours_v1/scripts/launchers/run_arper_v1.sh",
        "locked_base_metric": {"BLEU4": 0.59890, "SER": 5.938},
        "target_plus_one_third": {"BLEU4": 0.935, "SER": 2.72},
    },
    {
        "suite": "todcl",
        "benchmark": "ToDCL ADAPTER 37-domain NLG",
        "official_base": "ADAPTER anchor 20260706",
        "ours_v1_delta": "Assess-then-Update orthogonal penalty",
        "metric": "BLEU / EER",
        "split_policy": "official 37-domain single setting",
        "split_disclosure": "legacy py37 env; paper BLEU best on ADAPTER, EER best on REPLAY",
        "smoke_config": "BOUNDED_SMOKE=1 run_todcl_v1.sh",
        "formal_config": "run_todcl_v1.sh",
        "launcher": "ours_v1/scripts/launchers/run_todcl_v1.sh",
        "locked_base_metric": {"BLEU": 21.7719, "EER": 0.163975},
        "target_plus_one_third": {"BLEU": 29.0, "EER": 0.123},
    },
]


def git_sha(path: Path) -> str:
    if not (path / ".git").exists():
        return "missing"
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def ours_v1_sha() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "branch": "ours-v1",
        "ours_v1_sha": ours_v1_sha(),
        "worktree": str(REPO),
        "official_repo_shas": {k: git_sha(v) for k, v in OFFICIAL_REPOS.items()},
        "matrix": MATRIX,
        "phase0_seal": "/root/lora-ours/results/logs/phase0_stop_seal_20260712.md",
        "policy": "smoke early-gate then formal; blocker stops suite; no pseudo-comparable results",
    }
    json_path = OUT_DIR / "ours_v1_frozen_matrix_20260712.json"
    md_path = OUT_DIR / "ours_v1_frozen_matrix_20260712.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Ours v1 Frozen Matrix — 2026-07-12",
        "",
        f"- **ours-v1 SHA:** `{payload['ours_v1_sha']}`",
        f"- **Worktree:** `{REPO}`",
        "",
        "## Official repo SHAs",
        "",
        "| Repo | SHA |",
        "|------|-----|",
    ]
    for name, sha in payload["official_repo_shas"].items():
        lines.append(f"| {name} | `{sha}` |")
    lines.extend(["", "## Formal matrix", ""])
    for row in MATRIX:
        lines.append(f"### {row['suite']}")
        lines.append(f"- Benchmark: {row['benchmark']}")
        lines.append(f"- Base: {row['official_base']}")
        lines.append(f"- Ours v1: {row['ours_v1_delta']}")
        lines.append(f"- Metric: {row['metric']}")
        lines.append(f"- Split: {row['split_policy']} ({row['split_disclosure']})")
        lines.append(f"- Smoke: `{row['smoke_config']}`")
        lines.append(f"- Formal: `{row['formal_config']}`")
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
