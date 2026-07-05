#!/usr/bin/env python3
"""Evaluate v77 train-heldout adapter-selection candidates.

Inputs are the v75 train-heldout diagnostics only. This script does not read
dev/test predictions or targets.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path("/root/lora-ours")


@dataclass
class HeldoutEval:
    family: str
    round_name: str
    run_name: str
    adapter: str
    heldout_em: float
    heldout_rouge_l: float | None
    prediction_counts: dict[str, int]


def _parse_metrics(log_path: Path) -> tuple[float, float | None]:
    text = log_path.read_text(encoding="utf-8")
    em_match = re.search(r"predict_exact_match_for_amazon\s*=\s*([0-9.]+)", text)
    if not em_match:
        raise RuntimeError(f"missing amazon EM in {log_path}")
    rouge_match = re.search(r"predict_rougeL_for_amazon\s*=\s*([0-9.]+)", text)
    return float(em_match.group(1)), float(rouge_match.group(1)) if rouge_match else None


def _load_prediction_counts(confusion_path: Path) -> dict[str, dict[str, int]]:
    payload = json.loads(confusion_path.read_text(encoding="utf-8"))
    counts_by_run: dict[str, dict[str, int]] = {}
    for item in payload["runs"]:
        path = str(item["path"])
        marker = "olora_v75_amazon_sc_trainheldout500_"
        if marker not in path:
            continue
        suffix = path.split(marker, 1)[1].split("/", 1)[0]
        counts_by_run[f"olora_v75_amazon_sc_trainheldout500_{suffix}"] = {
            str(k): int(v) for k, v in item.get("prediction_counts", {}).items()
        }
    return counts_by_run


def load_evals(log_dir: Path, confusion_path: Path) -> list[HeldoutEval]:
    counts_by_run = _load_prediction_counts(confusion_path)
    evals: list[HeldoutEval] = []
    for manifest_path in sorted(log_dir.glob("olora_v75_amazon_sc_trainheldout500_*.manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_name = str(manifest["run_name"])
        suffix = run_name.removeprefix("olora_v75_amazon_sc_trainheldout500_")
        family, round_name = suffix.rsplit("_", 1)
        em, rouge_l = _parse_metrics(Path(manifest["log_file"]))
        evals.append(
            HeldoutEval(
                family=family,
                round_name=round_name,
                run_name=run_name,
                adapter=str(manifest["adapter"]),
                heldout_em=em,
                heldout_rouge_l=rouge_l,
                prediction_counts=counts_by_run.get(run_name, {}),
            )
        )
    return evals


def summarize_family(evals: list[HeldoutEval], baseline_em: float, moderate_min_count: int) -> dict[str, Any]:
    best = max(evals, key=lambda item: item.heldout_em)
    final = next((item for item in evals if item.round_name == "r4"), evals[-1])
    best_exceeds_baseline = best.heldout_em > baseline_em
    final_stable = final.heldout_em >= baseline_em
    moderate_ok = (
        final.prediction_counts.get("negative", 0) >= moderate_min_count
        and final.prediction_counts.get("positive", 0) >= moderate_min_count
    )
    reasons: list[str] = []
    if not best_exceeds_baseline:
        reasons.append(f"best heldout EM {best.heldout_em} does not exceed v69 baseline {baseline_em}")
    if not final_stable:
        reasons.append(f"final heldout EM {final.heldout_em} below v69 baseline {baseline_em}")
    if not moderate_ok:
        reasons.append(
            "final moderate-label collapse: "
            f"negative={final.prediction_counts.get('negative', 0)}, "
            f"positive={final.prediction_counts.get('positive', 0)}"
        )
    return {
        "family": evals[0].family,
        "adapter_selection_policy": {
            "selected_run": best.run_name,
            "selected_round": best.round_name,
            "selected_adapter": best.adapter,
            "selected_heldout_em": best.heldout_em,
            "selected_heldout_rougeL": best.heldout_rouge_l,
        },
        "final_adapter": {
            "run": final.run_name,
            "heldout_em": final.heldout_em,
            "heldout_rougeL": final.heldout_rouge_l,
            "prediction_counts": final.prediction_counts,
        },
        "trajectory": [
            {
                "round": item.round_name,
                "heldout_em": item.heldout_em,
                "heldout_rougeL": item.heldout_rouge_l,
                "prediction_counts": item.prediction_counts,
            }
            for item in evals
        ],
        "decision": "pass" if not reasons else "reject",
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=str(REPO / "results/logs"))
    parser.add_argument(
        "--confusion-json",
        default=str(REPO / "results/logs/olora_v75_amazon_sc_trainheldout500_confusion.json"),
    )
    parser.add_argument("--baseline-em", type=float, default=55.2)
    parser.add_argument("--moderate-min-count", type=int, default=5)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    evals = load_evals(Path(args.log_dir), Path(args.confusion_json))
    by_family: dict[str, list[HeldoutEval]] = {}
    for item in evals:
        by_family.setdefault(item.family, []).append(item)
    summaries = [
        summarize_family(sorted(items, key=lambda item: item.round_name), args.baseline_em, args.moderate_min_count)
        for _, items in sorted(by_family.items())
    ]

    payload = {
        "policy": {
            "source": "v75 train-heldout diagnostics",
            "uses_test_json": False,
            "uses_test_targets": False,
            "baseline_em": args.baseline_em,
            "moderate_min_count": args.moderate_min_count,
            "rp_mapping": {
                "lora_bank": "round adapters are treated as a published-base adapter bank",
                "prototype_router": "train-heldout EM selects the amazon/SC adapter without test labels",
                "drift_assess_update": "heldout drop and moderate-label collapse reject harmful updates",
                "overlap_or_spectral_replay": "not enabled as a new training change unless gate evidence exceeds baseline",
            },
        },
        "candidates": summaries,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = ["# v77 Train-Heldout Adapter-Selection Candidates", ""]
    lines.append("- Leakage policy: uses only v75 train-heldout diagnostics; no dev/test predictions or targets.")
    lines.append("- Pass rule: selected heldout EM must exceed v69 heldout baseline and final adapter must not collapse moderate labels.")
    lines.append("")
    for item in summaries:
        selected = item["adapter_selection_policy"]
        final = item["final_adapter"]
        lines.extend(
            [
                f"## {item['family']}",
                "",
                f"- Decision: `{item['decision']}`",
                f"- Selected adapter: `{selected['selected_round']}` with heldout EM `{selected['selected_heldout_em']}`",
                f"- Final adapter heldout EM: `{final['heldout_em']}`",
                f"- Final prediction counts: `{final['prediction_counts']}`",
            ]
        )
        if item["reasons"]:
            lines.append(f"- Reject reasons: `{' | '.join(item['reasons'])}`")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(out_json), "md": str(out_md), "candidates": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
