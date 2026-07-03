#!/usr/bin/env python3
"""Write a compact status report for the CITB official-base reproduction."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN_NAME = "citb_instrdialog_order1_seed1_official_ft_instr_stage1_v53"
DEFAULT_OUTPUT_BASE = Path("/root/autodl-tmp/citb_official_base_repro")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": repr(exc)}


def _result_dirs(output_dir: Path) -> List[Path]:
    results = output_dir / "results"
    if not results.is_dir():
        return []
    return sorted([p for p in results.iterdir() if p.is_dir()], key=lambda p: p.name)


def build_status(output_dir: Path, run_name: str) -> Dict[str, Any]:
    result_dirs = _result_dirs(output_dir)
    latest_dir = result_dirs[-1] if result_dirs else None
    latest_metrics = _load_json(latest_dir / "metrics.json") if latest_dir else {}
    train_state = _load_json(output_dir / "trainer_state.json")
    all_results = _load_json(output_dir / "all_results.json")
    return {
        "updated": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "output_dir": str(output_dir),
        "state": "has_results" if result_dirs else "pending_or_starting",
        "num_result_dirs": len(result_dirs),
        "latest_result_dir": str(latest_dir) if latest_dir else "",
        "latest_metric_keys": sorted(latest_metrics.keys())[:40],
        "latest_metrics": {
            key: latest_metrics.get(key)
            for key in sorted(latest_metrics)
            if key.endswith("_rougeL")
            or key.endswith("_exact_match")
            or key.endswith("_samples")
            or key in {"train_runtime", "train_samples", "epoch"}
        },
        "trainer_state_exists": bool(train_state),
        "all_results": all_results,
        "red_flags": [
            "CITB paper states InstrDialog uses 500/50/100 train/dev/test instances, but the official short-stream FT_INSTR script sets max_num_instances_per_eval_task=50.",
            "Official CITB Stage-2 code uses one max_num_instances_per_eval_task for both dev and per-task test split size, yielding script-strict 500/50/50 unless split code is patched.",
            "Default launcher keeps the script-strict 500/50/50 policy; requested 500/50/100 remains blocked unless split code is patched and audited.",
            "Official dry-run currently requires the lora_v10_citb Python 3.9 environment; base Python lacks datasets.load_metric.",
            "Stage-1 tokenizer files are incompatible with the old official stack, so the launcher uses the local google__t5-small-lm-adapt tokenizer as a compatibility override.",
            "The hyintell/CITB checkout has an untracked Tk-Instruct copy whose collator lacks add_task_id; launcher now defaults to the tracked local citb_official tree whose collator matches the CL entrypoint.",
        ],
    }


def write_status(status: Dict[str, Any], basename: str) -> None:
    logs_dir = REPO / "results/logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_path = logs_dir / f"{basename}.json"
    md_path = logs_dir / f"{basename}.md"
    json_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CITB Official Base Repro Status",
        "",
        f"- Updated: `{status['updated']}`",
        f"- Run: `{status['run_name']}`",
        f"- State: `{status['state']}`",
        f"- Output dir: `{status['output_dir']}`",
        f"- Result dirs: `{status['num_result_dirs']}`",
        f"- Latest result dir: `{status['latest_result_dir']}`",
        "",
        "## Latest Metrics",
    ]
    latest_metrics = status.get("latest_metrics") or {}
    if latest_metrics:
        for key, value in latest_metrics.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No per-task metrics yet.")
    lines.extend(["", "## RED FLAG"])
    for flag in status["red_flags"]:
        lines.append(f"- {flag}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=os.environ.get("CITB_OFFICIAL_OUTPUT_DIR", ""))
    parser.add_argument("--run-name", default=os.environ.get("CITB_OFFICIAL_RUN_NAME", DEFAULT_RUN_NAME))
    parser.add_argument("--basename", default="citb_official_base_repro_v53_status")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_BASE / args.run_name
    status = build_status(output_dir=output_dir, run_name=args.run_name)
    write_status(status, args.basename)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
