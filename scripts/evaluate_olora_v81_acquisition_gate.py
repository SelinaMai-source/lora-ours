#!/usr/bin/env python3
"""Evaluate amazon round2 acquisition quality from train-heldout diagnostics.

The gate uses only amazon/train heldout confusion summaries. It rejects weak
amazon anchors before later retention candidates can be promoted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]


def load_runs(confusion_paths: list[Path]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in confusion_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("runs", []):
            item = dict(item)
            item["_source_confusion"] = str(path)
            item["_name"] = infer_name(str(item.get("path", "")))
            runs.append(item)
    return runs


def infer_name(path: str) -> str:
    explicit_names = {
        "olora_v82_low_lr_round2_trainheldout_diag": "v82_r2",
        "olora_v83_classcov_round2_trainheldout_diag": "v83_r2",
    }
    for marker, name in explicit_names.items():
        if marker in path:
            return name
    markers = [
        "olora_v75_amazon_sc_trainheldout500_",
        "olora_v81_acquisition_trainheldout500_",
    ]
    for marker in markers:
        if marker in path:
            return path.split(marker, 1)[1].split("/", 1)[0]
    return Path(path).parts[-3] if path else "unknown"


def moderate_counts(run: dict[str, Any]) -> tuple[int, int]:
    counts = run.get("prediction_counts", {})
    return int(counts.get("negative", 0)), int(counts.get("positive", 0))


def evaluate_run(
    run: dict[str, Any],
    *,
    baseline_accuracy: float,
    baseline_negative: int,
    baseline_positive: int,
    min_negative_accuracy: float,
    min_positive_accuracy: float,
    strict: bool,
) -> dict[str, Any]:
    neg_count, pos_count = moderate_counts(run)
    per_label = run.get("per_label", {})
    neg_acc = per_label.get("negative", {}).get("accuracy")
    pos_acc = per_label.get("positive", {}).get("accuracy")
    accuracy = float(run.get("accuracy", 0.0))
    reasons: list[str] = []
    if strict:
        if accuracy <= baseline_accuracy:
            reasons.append(f"heldout EM {accuracy} does not exceed baseline {baseline_accuracy}")
    elif accuracy < baseline_accuracy:
        reasons.append(f"heldout EM {accuracy} below baseline {baseline_accuracy}")
    if neg_count < baseline_negative:
        reasons.append(f"negative prediction count {neg_count} < baseline {baseline_negative}")
    if pos_count < baseline_positive:
        reasons.append(f"positive prediction count {pos_count} < baseline {baseline_positive}")
    if neg_acc is None or float(neg_acc) < min_negative_accuracy:
        reasons.append(f"negative per-label accuracy {neg_acc} < {min_negative_accuracy}")
    if pos_acc is None or float(pos_acc) < min_positive_accuracy:
        reasons.append(f"positive per-label accuracy {pos_acc} < {min_positive_accuracy}")
    return {
        "name": run["_name"],
        "path": run.get("path"),
        "accuracy": accuracy,
        "prediction_counts": run.get("prediction_counts", {}),
        "negative_prediction_count": neg_count,
        "positive_prediction_count": pos_count,
        "negative_accuracy": neg_acc,
        "positive_accuracy": pos_acc,
        "decision": "pass" if not reasons else "reject",
        "reasons": reasons,
        "source_confusion": run.get("_source_confusion"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confusion-json", action="append", required=True)
    parser.add_argument("--baseline-name", default="v69_r2")
    parser.add_argument("--min-negative-accuracy", type=float, default=20.0)
    parser.add_argument("--min-positive-accuracy", type=float, default=5.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    runs = load_runs([Path(item) for item in args.confusion_json])
    baseline = next((run for run in runs if run["_name"] == args.baseline_name), None)
    if baseline is None:
        raise RuntimeError(f"baseline {args.baseline_name} not found")
    baseline_accuracy = float(baseline.get("accuracy", 0.0))
    baseline_negative, baseline_positive = moderate_counts(baseline)

    evaluations = [
        evaluate_run(
            run,
            baseline_accuracy=baseline_accuracy,
            baseline_negative=baseline_negative,
            baseline_positive=baseline_positive,
            min_negative_accuracy=args.min_negative_accuracy,
            min_positive_accuracy=args.min_positive_accuracy,
            strict=(run["_name"] != args.baseline_name),
        )
        for run in runs
        if run["_name"].endswith("_r2") or run["_name"] == args.baseline_name
    ]

    payload = {
        "policy": {
            "gate": "amazon_round2_acquisition_quality",
            "uses_dev_or_test": False,
            "source": "amazon/train heldout confusion summaries",
            "baseline_name": args.baseline_name,
            "baseline_accuracy": baseline_accuracy,
            "baseline_negative_prediction_count": baseline_negative,
            "baseline_positive_prediction_count": baseline_positive,
            "min_negative_accuracy": args.min_negative_accuracy,
            "min_positive_accuracy": args.min_positive_accuracy,
            "promotion_rule": (
                "new acquisition candidates must exceed v69 round2 heldout EM and preserve at least "
                "the v69 moderate-label prediction coverage before retention/smoke/formal promotion"
            ),
        },
        "evaluations": evaluations,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = ["# v81 Amazon Round2 Acquisition Quality Gate", ""]
    lines.append("- Leakage policy: amazon train-heldout only; no dev/test predictions or targets.")
    lines.append(
        f"- Baseline `{args.baseline_name}`: EM `{baseline_accuracy}`, negative count `{baseline_negative}`, positive count `{baseline_positive}`."
    )
    lines.append("")
    for item in evaluations:
        lines.extend(
            [
                f"## {item['name']}",
                "",
                f"- Decision: `{item['decision']}`",
                f"- Heldout EM: `{item['accuracy']}`",
                f"- Negative/positive counts: `{item['negative_prediction_count']}` / `{item['positive_prediction_count']}`",
                f"- Negative/positive accuracy: `{item['negative_accuracy']}` / `{item['positive_accuracy']}`",
            ]
        )
        if item["reasons"]:
            lines.append(f"- Reject reasons: `{' | '.join(item['reasons'])}`")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(out_json), "md": str(out_md), "evaluations": evaluations}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
