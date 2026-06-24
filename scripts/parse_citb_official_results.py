#!/usr/bin/env python3
"""Parse CITB official per-task metrics and compare with paper targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.citb_t5.citb_metrics import (
    compute_citb_metric_summary,
    matrix_from_official_metric_jsons,
)


PAPER_TARGETS: Dict[str, Dict[str, Any]] = {
    "FT_INSTR": {
        "paper_method": "FT-init",
        "source": "CITB paper Table 1 / official scores/continual_instruction_tuning/stream=cl_dialogue_tasks/CL=FT_INSTR/scores.json",
        "rougeL": {
            "average_accuracy": {"mean": 35.7, "std": 0.2},
            "average_FWT": {"mean": 18.5, "std": 0.7},
            "average_BWT": {"mean": -4.6, "std": 0.2},
            "final_initial_multi_test_score": {"mean": 38.6, "std": 0.3},
            "final_official_test_score": {"mean": 32.3, "std": 0.6},
        },
    },
    "REPLAY": {
        "paper_method": "Replay(50)",
        "source": "CITB paper Table 1 / official scores/continual_instruction_tuning/stream=cl_dialogue_tasks/CL=REPLAY/scores.json",
        "rougeL": {
            "average_accuracy": {"mean": 40.4, "std": 0.0},
            "average_FWT": {"mean": 22.9, "std": 0.1},
            "average_BWT": {"mean": 1.6, "std": 1.2},
            "final_initial_multi_test_score": {"mean": 47.1, "std": 0.5},
            "final_official_test_score": {"mean": 31.8, "std": 1.0},
        },
    },
}


def _task_index(metrics_path: Path) -> int:
    return int(metrics_path.parent.name.split("_", 1)[0])


def load_metric_jsons(results_dir: Path) -> List[Mapping[str, Any]]:
    files = sorted(results_dir.glob("*/metrics.json"), key=_task_index)
    return [json.loads(path.read_text()) for path in files]


def summarize_method(results_dir: Path, metric: str) -> Dict[str, Any]:
    metric_jsons = load_metric_jsons(results_dir)
    if not metric_jsons:
        return {"status": "missing", "task_count": 0}

    matrix = matrix_from_official_metric_jsons(metric_jsons, metric=metric)
    summary = compute_citb_metric_summary(matrix).as_dict()
    final = metric_jsons[-1]
    summary.update(
        {
            "status": "complete" if len(metric_jsons) == 19 else "partial",
            "task_count": len(metric_jsons),
            "final_initial_multi_test_score": final.get(f"predict_initial_multi_{metric}"),
            "final_official_test_score": final.get(f"predict_official_{metric}"),
            "matrix": matrix,
        }
    )
    return summary


def compare_to_target(local: Mapping[str, Any], target: Mapping[str, Any]) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {}
    key_map = {
        "average_accuracy": "average_accuracy",
        "average_fwt": "average_FWT",
        "average_bwt": "average_BWT",
        "final_initial_multi_test_score": "final_initial_multi_test_score",
        "final_official_test_score": "final_official_test_score",
    }
    for local_key, target_key in key_map.items():
        if local.get(local_key) is None or target_key not in target:
            continue
        paper = target[target_key]
        delta = float(local[local_key]) - float(paper["mean"])
        std = float(paper.get("std") or 0.0)
        comparisons[target_key] = {
            "local": float(local[local_key]),
            "paper_mean": float(paper["mean"]),
            "paper_std": std,
            "delta": delta,
            "within_2std_or_0_5": abs(delta) <= max(2 * std, 0.5),
        }
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--citb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    base = args.citb_root / "output/continual_instruction_tuning/stream=cl_dialogue_tasks"
    seed = args.seed
    methods = {
        "FT_INSTR": base / f"CL=FT_INSTR/lora_run_v10_instrdialog_citb_t5_sequential_s{seed}/results",
        "REPLAY": base / f"CL=REPLAY/lora_run_v10_instrdialog_citb_t5_replay50_s{seed}/results",
    }

    report: Dict[str, Any] = {
        "benchmark": "CITB InstrDialog",
        "seed": args.seed,
        "metric": "rougeL",
        "strict_allowed": False,
        "paper_targets": PAPER_TARGETS,
        "methods": {},
        "notes": [
            "AR/FWT/BWT are computed from the official per-task score matrix.",
            "CITB paper FR columns are Final ROUGE-L on T_init and T_unseen; they correspond here to final_initial_multi_test_score and final_official_test_score.",
            "The local forgetting_rate field is an auxiliary CL forgetting calculation and is not the CITB paper FR column.",
        ],
    }

    all_complete = True
    all_match = True
    for method, results_dir in methods.items():
        local = summarize_method(results_dir, metric="rougeL")
        target = PAPER_TARGETS[method]["rougeL"]
        comparison = compare_to_target(local, target)
        method_complete = local.get("status") == "complete"
        method_match = method_complete and all(v["within_2std_or_0_5"] for v in comparison.values())
        all_complete = all_complete and method_complete
        all_match = all_match and method_match
        report["methods"][method] = {
            "results_dir": str(results_dir),
            "paper_method": PAPER_TARGETS[method]["paper_method"],
            "local": local,
            "comparison": comparison,
            "matches_paper_target": method_match,
        }

    report["strict_allowed"] = bool(all_complete and all_match)
    report["remaining_blocker"] = None if report["strict_allowed"] else "full run and paper comparison are not closed"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
