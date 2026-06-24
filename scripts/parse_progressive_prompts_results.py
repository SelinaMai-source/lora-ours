#!/usr/bin/env python3
"""Parse official Progressive Prompts ``results_dict.npy`` outputs.

The official T5 runner stores Python dictionaries with validation/test
accuracies in a NumPy object file. This parser converts that artifact into
JSON/CSV evidence that lora-run_v10 can track without modifying the official
baseline source.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]

PP_TO_SEQGLUE = {
    "sst2": "glue_sst2",
    "mrpc": "glue_mrpc",
    "rte": "glue_rte",
    "cola": "glue_cola",
    "boolq": "super_glue_boolq",
    "wic": "super_glue_wic",
    "cb": "super_glue_cb",
    "copa": "super_glue_copa",
}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _load_npy(path: Path) -> Dict[str, Any]:
    obj = np.load(path, allow_pickle=True)
    if hasattr(obj, "item"):
        loaded = obj.item()
    else:
        loaded = obj
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict-like results_dict.npy, got {type(loaded).__name__}")
    return loaded


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _final_test_scores(test_payload: Any) -> Dict[str, float]:
    """Return the latest per-task test scores from official result variants."""
    if not isinstance(test_payload, dict):
        return {}
    if all(_as_float(value) is not None for value in test_payload.values()):
        return {str(task): float(value) for task, value in test_payload.items()}
    numeric_steps = []
    for key, value in test_payload.items():
        try:
            numeric_steps.append((int(key), value))
        except (TypeError, ValueError):
            continue
    if not numeric_steps:
        return {}
    _, latest = sorted(numeric_steps, key=lambda item: item[0])[-1]
    if not isinstance(latest, dict):
        return {}
    return {str(task): float(value) for task, value in latest.items() if _as_float(value) is not None}


def _iter_val_rows(results: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for task, value in results.items():
        if task == "test":
            continue
        score = _as_float(value)
        if score is None:
            continue
        yield {
            "official_task": task,
            "seqglue_segment": PP_TO_SEQGLUE.get(str(task), ""),
            "split": "validation",
            "score": score,
        }


def parse_results(results_path: Path, config_path: Path | None) -> Dict[str, Any]:
    results = _load_npy(results_path)
    cfg: Dict[str, Any] = {}
    if config_path and config_path.is_file():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    val_rows = list(_iter_val_rows(results))
    test_scores = _final_test_scores(results.get("test"))
    test_rows = [
        {
            "official_task": task,
            "seqglue_segment": PP_TO_SEQGLUE.get(str(task), ""),
            "split": "test",
            "score": score,
        }
        for task, score in sorted(test_scores.items())
    ]
    rows = val_rows + test_rows

    expected_tasks = list(((cfg.get("official_preflight") or {}).get("task_list")) or [])
    if not expected_tasks:
        expected_tasks = list(PP_TO_SEQGLUE)
    observed_tasks = [row["official_task"] for row in val_rows]
    missing_expected = [task for task in expected_tasks if task not in observed_tasks and task not in test_scores]

    validation_scores = [row["score"] for row in val_rows]
    test_score_values = [row["score"] for row in test_rows]
    metric_basis = "test_final" if test_score_values else "validation_after_task"
    avg_scores = test_score_values or validation_scores

    paper_protocol_notes = [
        "Official Progressive Prompts paper uses T5-large, frozen backbone, prefix_len=10, lr=0.3, 10 epochs, early stopping, and select_k_per_class=1000.",
        "Paper Table 6 long orders contain 15 tasks; the local lora-run_v10 Seq-GLUE bridge currently has 8 train50/eval10 segments.",
        "This parser is an environment/provenance adapter only; it does not make local train50/eval10 outputs strict paper reproduction results.",
    ]

    return {
        "cell": "Progressive Prompts / Seq-GLUE",
        "status": "parsed-results-needs-paper-protocol-match",
        "strict_allowed": False,
        "results_path": _rel(results_path),
        "config": _rel(config_path) if config_path else None,
        "metric_basis": metric_basis,
        "num_rows": len(rows),
        "average_score": sum(avg_scores) / len(avg_scores) if avg_scores else None,
        "validation_tasks": observed_tasks,
        "test_tasks": sorted(test_scores),
        "missing_expected_tasks": missing_expected,
        "paper_protocol_notes": paper_protocol_notes,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse official Progressive Prompts results_dict.npy")
    parser.add_argument("--results", type=Path, required=True, help="Path to official results_dict.npy")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper/lora_run_v10/seqglue__progressive_prompts_official__s123.yaml"),
    )
    parser.add_argument("--out-json", type=Path, default=Path("results/tables/progressive_prompts_seqglue_results.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("results/tables/progressive_prompts_seqglue_results.csv"))
    args = parser.parse_args()

    results_path = args.results if args.results.is_absolute() else REPO / args.results
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    out_json = args.out_json if args.out_json.is_absolute() else REPO / args.out_json
    out_csv = args.out_csv if args.out_csv.is_absolute() else REPO / args.out_csv

    parsed = parse_results(results_path, config_path if config_path.is_file() else None)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["official_task", "seqglue_segment", "split", "score"])
        writer.writeheader()
        writer.writerows(parsed["rows"])

    print(json.dumps({"out_json": _rel(out_json), "out_csv": _rel(out_csv), "average_score": parsed["average_score"]}, indent=2))
    return 0 if parsed["num_rows"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
