#!/usr/bin/env python3
"""Analyze amazon/SC prediction confusion for O-LoRA overlay runs."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                obj = json.loads(line)
                if obj.get("Task") == "SC" and obj.get("Dataset") == "amazon":
                    rows.append(obj)
    return rows


def _summarize(path: Path) -> dict[str, Any]:
    rows = _read_rows(path)
    pairs = [
        (str(row["Instance"]["label"]).strip(), str(row.get("Prediction", "")).strip())
        for row in rows
    ]
    total = len(pairs)
    correct = sum(gold == pred for gold, pred in pairs)
    per_label: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        label_pairs = [(gold, pred) for gold, pred in pairs if gold == label]
        label_total = len(label_pairs)
        label_correct = sum(gold == pred for gold, pred in label_pairs)
        per_label[label] = {
            "total": label_total,
            "correct": label_correct,
            "accuracy": round(100 * label_correct / label_total, 4) if label_total else None,
        }
    confusions = collections.Counter((gold, pred) for gold, pred in pairs if gold != pred)
    pred_counts = collections.Counter(pred for _, pred in pairs)
    gold_counts = collections.Counter(gold for gold, _ in pairs)
    return {
        "path": str(path),
        "total": total,
        "accuracy": round(100 * correct / total, 4) if total else None,
        "gold_counts": dict(gold_counts),
        "prediction_counts": dict(pred_counts),
        "per_label": per_label,
        "top_confusions": [
            {"gold": gold, "prediction": pred, "count": count}
            for (gold, pred), count in confusions.most_common(20)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("prediction_files", nargs="+")
    args = parser.parse_args()

    summaries = [_summarize(Path(item)) for item in args.prediction_files]
    payload = {"runs": summaries}

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = ["# Amazon/SC Prediction Confusion Diagnostics", ""]
    for summary in summaries:
        lines.extend(
            [
                f"## {Path(summary['path']).parts[-4]} / {Path(summary['path']).parts[-2]}",
                "",
                f"- Path: `{summary['path']}`",
                f"- Accuracy: `{summary['accuracy']}` over `{summary['total']}` amazon/SC examples.",
                f"- Prediction counts: `{summary['prediction_counts']}`",
                "",
                "### Per-Label Accuracy",
                "",
            ]
        )
        for label in LABELS:
            item = summary["per_label"][label]
            lines.append(
                f"- `{label}`: `{item['accuracy']}` ({item['correct']}/{item['total']})"
            )
        lines.extend(["", "### Top Confusions", ""])
        for item in summary["top_confusions"][:10]:
            lines.append(f"- `{item['gold']}` -> `{item['prediction']}`: `{item['count']}`")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(out_json), "md": str(out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
