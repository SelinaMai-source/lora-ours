#!/usr/bin/env python3
"""Evaluate v78 train-only behavior-retention regularizer candidates.

The candidate is a target-free regularizer design: preserve the amazon adapter's
train-heldout behavior (teacher predictions and label distribution) while later
tasks train. Promotion still requires the v76 train-heldout gate, which uses
only train-heldout labels, never dev/test targets.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path
from typing import Any


LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("Task") == "SC" and row.get("Dataset") == "amazon":
                rows.append(row)
    return rows


def exact_match(rows: list[dict[str, Any]]) -> float:
    correct = 0
    for row in rows:
        gold = str(row["Instance"]["label"]).strip()
        pred = str(row.get("Prediction", "")).strip()
        correct += int(gold == pred)
    return round(100 * correct / len(rows), 4) if rows else 0.0


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = collections.Counter(str(row.get("Prediction", "")).strip() for row in rows)
    return {label: counts.get(label, 0) for label in LABELS}


def agreement(teacher: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> float:
    if len(teacher) != len(candidate):
        raise ValueError(f"prediction length mismatch: teacher={len(teacher)}, candidate={len(candidate)}")
    same = 0
    for a, b in zip(teacher, candidate):
        if str(a.get("Prediction", "")).strip() == str(b.get("Prediction", "")).strip():
            same += 1
    return round(100 * same / len(teacher), 4) if teacher else 0.0


def l1_distribution_distance(a_counts: dict[str, int], b_counts: dict[str, int]) -> float:
    total_a = max(1, sum(a_counts.values()))
    total_b = max(1, sum(b_counts.values()))
    return round(sum(abs(a_counts[label] / total_a - b_counts[label] / total_b) for label in LABELS), 6)


def parse_manifest_metrics(manifest_path: Path) -> tuple[float | None, float | None]:
    if not manifest_path.exists():
        return None, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    log_path = Path(manifest["log_file"])
    if not log_path.exists():
        return None, None
    text = log_path.read_text(encoding="utf-8")
    em_match = re.search(r"predict_exact_match_for_amazon\s*=\s*([0-9.]+)", text)
    rouge_match = re.search(r"predict_rougeL_for_amazon\s*=\s*([0-9.]+)", text)
    em = float(em_match.group(1)) if em_match else None
    rouge = float(rouge_match.group(1)) if rouge_match else None
    return em, rouge


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--candidate", action="append", required=True, help="name:path[:manifest]")
    parser.add_argument("--baseline-em", type=float, default=55.2)
    parser.add_argument("--min-agreement", type=float, default=60.0)
    parser.add_argument("--max-label-l1", type=float, default=0.35)
    parser.add_argument("--moderate-min-count", type=int, default=5)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    teacher_rows = read_predictions(Path(args.teacher))
    teacher_counts = label_counts(teacher_rows)
    teacher_em = exact_match(teacher_rows)

    candidates = []
    for item in args.candidate:
        parts = item.split(":", 2)
        if len(parts) < 2:
            raise ValueError("--candidate must be name:path[:manifest]")
        name, pred_path = parts[0], Path(parts[1])
        manifest_path = Path(parts[2]) if len(parts) == 3 and parts[2] else None
        rows = read_predictions(pred_path)
        counts = label_counts(rows)
        em = exact_match(rows)
        heldout_em, heldout_rouge = parse_manifest_metrics(manifest_path) if manifest_path else (em, None)
        pseudo_agreement = agreement(teacher_rows, rows)
        dist_l1 = l1_distribution_distance(teacher_counts, counts)
        moderate_ok = counts.get("negative", 0) >= args.moderate_min_count and counts.get("positive", 0) >= args.moderate_min_count
        reasons: list[str] = []
        if pseudo_agreement < args.min_agreement:
            reasons.append(f"teacher agreement {pseudo_agreement} < {args.min_agreement}")
        if dist_l1 > args.max_label_l1:
            reasons.append(f"label distribution L1 {dist_l1} > {args.max_label_l1}")
        if heldout_em is None or heldout_em <= args.baseline_em:
            reasons.append(f"v76 heldout EM {heldout_em} does not exceed baseline {args.baseline_em}")
        if not moderate_ok:
            reasons.append(
                f"moderate label collapse: negative={counts.get('negative', 0)}, positive={counts.get('positive', 0)}"
            )
        candidates.append(
            {
                "name": name,
                "prediction_path": str(pred_path),
                "manifest_path": str(manifest_path) if manifest_path else None,
                "heldout_em_from_predictions": em,
                "heldout_em_from_manifest": heldout_em,
                "heldout_rougeL_from_manifest": heldout_rouge,
                "teacher_agreement": pseudo_agreement,
                "label_distribution_l1": dist_l1,
                "prediction_counts": counts,
                "decision": "pass" if not reasons else "reject",
                "reasons": reasons,
            }
        )

    payload = {
        "policy": {
            "candidate": "behavior_retention_regularizer",
            "teacher_source": args.teacher,
            "teacher_anchor": {
                "heldout_em": teacher_em,
                "prediction_counts": teacher_counts,
            },
            "uses_dev_or_test": False,
            "regularizer_design": (
                "During later-task training, penalize divergence from the amazon-adapter "
                "teacher predictions/distribution on amazon train-heldout prompts; promote "
                "only if v76 heldout EM exceeds the v69 baseline."
            ),
            "thresholds": {
                "baseline_em": args.baseline_em,
                "min_teacher_agreement": args.min_agreement,
                "max_label_distribution_l1": args.max_label_l1,
                "moderate_min_count": args.moderate_min_count,
            },
        },
        "candidates": candidates,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = ["# v78 Behavior-Retention Regularizer Diagnostic", ""]
    lines.append("- Leakage policy: train-heldout predictions only; no dev/test split or targets.")
    lines.append(f"- Teacher anchor heldout EM: `{teacher_em}`")
    lines.append(f"- Teacher prediction counts: `{teacher_counts}`")
    lines.append("")
    for cand in candidates:
        lines.extend(
            [
                f"## {cand['name']}",
                "",
                f"- Decision: `{cand['decision']}`",
                f"- Heldout EM: `{cand['heldout_em_from_manifest']}`",
                f"- Teacher agreement: `{cand['teacher_agreement']}`",
                f"- Label distribution L1: `{cand['label_distribution_l1']}`",
                f"- Prediction counts: `{cand['prediction_counts']}`",
            ]
        )
        if cand["reasons"]:
            lines.append(f"- Reject reasons: `{' | '.join(cand['reasons'])}`")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(out_json), "md": str(out_md), "candidates": candidates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
