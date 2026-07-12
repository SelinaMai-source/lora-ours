#!/usr/bin/env python3
"""Freeze ours-v1 SHA, official base SHAs, and three-suite formal matrix."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "manifests" / "ours_v1_formal_matrix_frozen.json"


def git_sha(path: Path) -> str:
    if not (path / ".git").exists():
        return "no-git"
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


MATRIX = {
    "frozen_at": datetime.now().isoformat(timespec="seconds"),
    "workspace": str(REPO),
    "branch": "ours-v1",
    "ours_v1_sha": git_sha(REPO),
    "official_base_shas": {
        "citb": git_sha(Path("/root/autodl-tmp/lora-baselines-run_v1/external_sources/citb")),
        "o_lora": git_sha(Path("/root/autodl-tmp/lora-baselines-run_v1/external_sources/o_lora")),
        "arper": git_sha(Path("/root/autodl-tmp/lora-baselines-run_v1/external_sources/arper")),
        "todcl": git_sha(Path("/root/autodl-tmp/lora-baselines-run_v1/external_sources/todcl")),
    },
    "suites": [
        {
            "suite": "CITB InstrDialog",
            "official_base": "Replay(50) official_script_500_50_50",
            "ours_delta": "SSRG replay_ratio=0.5",
            "metric": "ROUGE-L AR/BWT",
            "smoke_config": "configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v1_20260708_smoke_strict.yaml",
            "formal_config": "configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v1_20260708_formal_strict.yaml",
            "launcher": "ours_v1/scripts/launchers/run_citb_v1.sh",
        },
        {
            "suite": "CITB InstrDialog++",
            "official_base": "Replay(50) public long stream",
            "split_label": "public_script_100_25_25_not_paper_exact",
            "ours_delta": "SSRG replay_ratio=0.5 (same mechanism)",
            "metric": "ROUGE-L AR/BWT",
            "smoke_config": "configs/ccfa_three_suite/citb_instrdialogpp_order1_seed1_ours_v1_20260708_smoke_strict.yaml",
            "formal_config": "configs/ccfa_three_suite/citb_instrdialogpp_order1_seed1_ours_v1_20260708_formal_strict.yaml",
            "launcher": "ours_v1/scripts/launchers/run_citb_pp_v1.sh",
        },
        {
            "suite": "Standard",
            "official_base": "O-LoRA T5-large",
            "hardware_label": "official-equivalent single-GPU grad_accum=8",
            "ours_delta": "class-coverage SSRG + assess-retention gate",
            "metric": "final average EM / ROUGE-L",
            "launcher": "ours_v1/scripts/launchers/run_standard_v1.sh",
        },
        {
            "suite": "ARPER",
            "official_base": "SCLSTM exemplar Path B v89",
            "ours_delta": "SSRG exemplar selection overlay",
            "metric": "BLEU4 / SER",
            "anchor_cfg": "results/logs/arper_woz3_paper_aligned_exemplar500_batch128_formal_v89.cfg",
            "launcher": "ours_v1/scripts/launchers/run_arper_v1.sh",
            "forbidden": ["ARPER_EWC_FISHER_CPU", "post-decode repair"],
        },
        {
            "suite": "ToDCL",
            "official_base": "ADAPTER NLG anchor",
            "ours_delta": "Assess-then-Update orthogonal penalty",
            "metric": "BLEU / EER",
            "launcher": "ours_v1/scripts/launchers/run_todcl_v1.sh",
            "checkpoint_gate": "requires loadable .ckpt",
        },
    ],
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(MATRIX, indent=2) + "\n")
    md = REPO / "results" / "manifests" / "ours_v1_formal_matrix_frozen.md"
    lines = ["# Frozen Ours v1 Formal Matrix", "", f"SHA: `{MATRIX['ours_v1_sha']}`", ""]
    for row in MATRIX["suites"]:
        lines.append(f"## {row['suite']}")
        lines.append(f"- base: {row['official_base']}")
        lines.append(f"- ours: {row['ours_delta']}")
        lines.append(f"- metric: {row['metric']}")
        if "split_label" in row:
            lines.append(f"- split: **{row['split_label']}**")
        lines.append("")
    md.write_text("\n".join(lines))
    print(OUT)


if __name__ == "__main__":
    main()
