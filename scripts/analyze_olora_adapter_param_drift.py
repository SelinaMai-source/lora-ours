#!/usr/bin/env python3
"""Analyze O-LoRA adapter parameter drift from an amazon anchor.

This diagnostic reads only adapter weights and train-heldout diagnostic logs.
It does not read dev/test predictions or targets.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch


def load_state(adapter_dir: Path) -> dict[str, torch.Tensor]:
    path = adapter_dir / "adapter_model.bin"
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu")
    return {key: value.float() for key, value in state.items() if torch.is_tensor(value)}


def parse_heldout(log_path: Path | None) -> tuple[float | None, float | None]:
    if log_path is None or not log_path.exists():
        return None, None
    text = log_path.read_text(encoding="utf-8")
    em = re.search(r"predict_exact_match_for_amazon\s*=\s*([0-9.]+)", text)
    rouge = re.search(r"predict_rougeL_for_amazon\s*=\s*([0-9.]+)", text)
    return (float(em.group(1)) if em else None, float(rouge.group(1)) if rouge else None)


def group_name(key: str) -> str:
    if "lora_A" in key:
        return "lora_A"
    if "lora_B" in key:
        return "lora_B"
    return "other"


def compare(anchor: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]) -> dict[str, Any]:
    common = sorted(set(anchor) & set(candidate))
    if not common:
        raise ValueError("no common tensor keys")

    groups: dict[str, dict[str, Any]] = {}
    total_diff_sq = 0.0
    total_anchor_sq = 0.0
    total_dot = 0.0
    total_cand_sq = 0.0
    total_new_sq = 0.0
    total_params = 0
    total_new_params = 0

    for key in common:
        a_full = anchor[key]
        b_full = candidate[key]
        if a_full.ndim != b_full.ndim:
            continue
        slices = tuple(slice(0, min(a_dim, b_dim)) for a_dim, b_dim in zip(a_full.shape, b_full.shape))
        a = a_full[slices].reshape(-1)
        b = b_full[slices].reshape(-1)
        if a.numel() == 0:
            continue
        diff = b - a
        diff_sq = float(torch.dot(diff, diff).item())
        anchor_sq = float(torch.dot(a, a).item())
        cand_sq = float(torch.dot(b, b).item())
        dot = float(torch.dot(a, b).item())
        new_sq = max(float(torch.dot(b_full.reshape(-1), b_full.reshape(-1)).item()) - cand_sq, 0.0)
        new_params = max(b_full.numel() - b.numel(), 0)
        total_diff_sq += diff_sq
        total_anchor_sq += anchor_sq
        total_cand_sq += cand_sq
        total_dot += dot
        total_new_sq += new_sq
        total_params += a.numel()
        total_new_params += new_params

        group = groups.setdefault(
            group_name(key),
            {
                "diff_sq": 0.0,
                "anchor_sq": 0.0,
                "cand_sq": 0.0,
                "dot": 0.0,
                "new_sq": 0.0,
                "params": 0,
                "new_params": 0,
                "keys": 0,
            },
        )
        group["diff_sq"] += diff_sq
        group["anchor_sq"] += anchor_sq
        group["cand_sq"] += cand_sq
        group["dot"] += dot
        group["new_sq"] += new_sq
        group["params"] += a.numel()
        group["new_params"] += new_params
        group["keys"] += 1

    def finalize(item: dict[str, Any]) -> dict[str, Any]:
        denom = math.sqrt(max(item["anchor_sq"], 1e-12) * max(item["cand_sq"], 1e-12))
        return {
            "l2": math.sqrt(item["diff_sq"]),
            "relative_l2": math.sqrt(item["diff_sq"]) / max(math.sqrt(item["anchor_sq"]), 1e-12),
            "cosine": item["dot"] / denom,
            "new_rank_l2": math.sqrt(item.get("new_sq", 0.0)),
            "new_rank_energy_ratio": item.get("new_sq", 0.0) / max(item["cand_sq"] + item.get("new_sq", 0.0), 1e-12),
            "params": item["params"],
            "new_params": item.get("new_params", 0),
            "keys": item["keys"],
        }

    overall = finalize(
        {
            "diff_sq": total_diff_sq,
            "anchor_sq": total_anchor_sq,
            "cand_sq": total_cand_sq,
            "dot": total_dot,
            "new_sq": total_new_sq,
            "params": total_params,
            "new_params": total_new_params,
            "keys": len(common),
        }
    )
    return {"overall": overall, "groups": {name: finalize(item) for name, item in groups.items()}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--candidate", action="append", required=True, help="name:adapter_dir[:heldout_log]")
    parser.add_argument("--baseline-heldout-em", type=float, default=55.2)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    anchor_state = load_state(Path(args.anchor))
    candidates = []
    for spec in args.candidate:
        parts = spec.split(":", 2)
        if len(parts) < 2:
            raise ValueError("--candidate must be name:adapter_dir[:heldout_log]")
        name = parts[0]
        adapter_dir = Path(parts[1])
        log_path = Path(parts[2]) if len(parts) == 3 and parts[2] else None
        metrics = compare(anchor_state, load_state(adapter_dir))
        heldout_em, heldout_rouge = parse_heldout(log_path)
        candidates.append(
            {
                "name": name,
                "adapter_dir": str(adapter_dir),
                "heldout_em": heldout_em,
                "heldout_rougeL": heldout_rouge,
                "param_drift": metrics,
                "gate_signal": {
                    "exceeds_v69_heldout_baseline": heldout_em is not None and heldout_em > args.baseline_heldout_em,
                    "baseline_heldout_em": args.baseline_heldout_em,
                },
            }
        )

    payload = {
        "policy": {
            "anchor": args.anchor,
            "anchor_role": "amazon round2 adapter parameter anchor",
            "uses_dev_or_test": False,
            "diagnostic": "adapter_model.bin parameter-space drift vs train-heldout outcomes",
            "candidate_regularizer": (
                "For later rounds, penalize drift from the amazon anchor in sensitive LoRA matrices; "
                "promote only if v76 train-heldout gate exceeds the v69 baseline."
            ),
        },
        "candidates": candidates,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = ["# v79 Adapter Parameter Drift Diagnostic", ""]
    lines.append("- Leakage policy: adapter weights and train-heldout logs only; no dev/test predictions or targets.")
    lines.append(f"- Anchor: `{args.anchor}`")
    lines.append("")
    for item in candidates:
        overall = item["param_drift"]["overall"]
        groups = item["param_drift"]["groups"]
        lines.extend(
            [
                f"## {item['name']}",
                "",
                f"- Heldout EM: `{item['heldout_em']}`",
                f"- Overall relative L2: `{overall['relative_l2']:.6f}`",
                f"- Overall cosine: `{overall['cosine']:.6f}`",
            ]
        )
        for group_name_item in sorted(groups):
            group = groups[group_name_item]
            lines.append(
                f"- `{group_name_item}` relative L2: `{group['relative_l2']:.6f}`, cosine: `{group['cosine']:.6f}`"
            )
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(out_json), "md": str(out_md), "candidates": candidates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
