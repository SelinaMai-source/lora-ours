#!/usr/bin/env python3
"""Analyze new-rank LoRA interference against an amazon adapter anchor.

This diagnostic estimates the output-space contribution of ranks added after
the amazon round: for each LoRA module it compares `B_new @ A_new` with the
amazon anchor update `B_anchor @ A_anchor`. It reads only adapter weights and
train-heldout logs, never dev/test predictions or targets.
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
    em_match = re.search(r"predict_exact_match_for_amazon\s*=\s*([0-9.]+)", text)
    rouge_match = re.search(r"predict_rougeL_for_amazon\s*=\s*([0-9.]+)", text)
    return (
        float(em_match.group(1)) if em_match else None,
        float(rouge_match.group(1)) if rouge_match else None,
    )


def module_kind(key: str) -> str:
    if ".q." in key:
        return "q"
    if ".v." in key:
        return "v"
    return "other"


def stack_region(key: str) -> str:
    if ".encoder." in key:
        return "encoder"
    if ".decoder." in key:
        return "decoder"
    return "other"


def iter_lora_a_keys(state: dict[str, torch.Tensor]) -> list[str]:
    return sorted(key for key in state if "lora_A.weight" in key)


def b_key_for(a_key: str) -> str:
    return a_key.replace("lora_A.weight", "lora_B.weight")


def flatten_update(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # A: [rank, in], B: [out, rank] -> [out, in]
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros(0)
    return torch.matmul(b, a).reshape(-1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    denom = math.sqrt(float(torch.dot(a, a).item()) * float(torch.dot(b, b).item()))
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(a, b).item()) / denom


def analyze_candidate(anchor: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]) -> dict[str, Any]:
    modules: list[dict[str, Any]] = []
    totals = {
        "anchor_sq": 0.0,
        "new_sq": 0.0,
        "new_on_anchor_sq": 0.0,
        "abs_dot": 0.0,
        "module_count": 0,
    }
    by_kind: dict[str, dict[str, float]] = {}
    by_stack: dict[str, dict[str, float]] = {}

    for a_key in iter_lora_a_keys(anchor):
        b_key = b_key_for(a_key)
        if a_key not in candidate or b_key not in anchor or b_key not in candidate:
            continue
        a_anchor = anchor[a_key]
        b_anchor = anchor[b_key]
        a_candidate = candidate[a_key]
        b_candidate = candidate[b_key]
        if a_anchor.ndim != 2 or b_anchor.ndim != 2 or a_candidate.ndim != 2 or b_candidate.ndim != 2:
            continue
        rank_anchor = min(a_anchor.shape[0], b_anchor.shape[1], a_candidate.shape[0], b_candidate.shape[1])
        rank_candidate = min(a_candidate.shape[0], b_candidate.shape[1])
        if rank_candidate <= rank_anchor:
            continue

        anchor_update = flatten_update(a_anchor[:rank_anchor, :], b_anchor[:, :rank_anchor])
        new_update = flatten_update(a_candidate[rank_anchor:rank_candidate, :], b_candidate[:, rank_anchor:rank_candidate])
        anchor_sq = float(torch.dot(anchor_update, anchor_update).item())
        new_sq = float(torch.dot(new_update, new_update).item())
        dot = float(torch.dot(anchor_update, new_update).item()) if anchor_update.numel() == new_update.numel() else 0.0
        cos_value = cosine(anchor_update, new_update)
        projection_sq = (dot * dot / max(anchor_sq, 1e-12)) if anchor_sq > 0 else 0.0
        projection_ratio = projection_sq / max(new_sq, 1e-12)
        interference_score = abs(cos_value) * math.sqrt(new_sq / max(anchor_sq, 1e-12))

        kind = module_kind(a_key)
        stack = stack_region(a_key)
        record = {
            "module": a_key.replace(".lora_A.weight", ""),
            "kind": kind,
            "stack": stack,
            "rank_anchor": rank_anchor,
            "rank_candidate": rank_candidate,
            "new_rank": rank_candidate - rank_anchor,
            "anchor_l2": math.sqrt(anchor_sq),
            "new_l2": math.sqrt(new_sq),
            "new_to_anchor_l2_ratio": math.sqrt(new_sq) / max(math.sqrt(anchor_sq), 1e-12),
            "cosine_new_vs_anchor": cos_value,
            "projection_ratio": projection_ratio,
            "interference_score": interference_score,
        }
        modules.append(record)

        totals["anchor_sq"] += anchor_sq
        totals["new_sq"] += new_sq
        totals["new_on_anchor_sq"] += projection_sq
        totals["abs_dot"] += abs(dot)
        totals["module_count"] += 1
        for bucket, name in ((by_kind, kind), (by_stack, stack)):
            item = bucket.setdefault(name, {"anchor_sq": 0.0, "new_sq": 0.0, "projection_sq": 0.0, "modules": 0.0})
            item["anchor_sq"] += anchor_sq
            item["new_sq"] += new_sq
            item["projection_sq"] += projection_sq
            item["modules"] += 1

    def finalize(bucket: dict[str, float]) -> dict[str, float]:
        return {
            "new_to_anchor_l2_ratio": math.sqrt(bucket["new_sq"]) / max(math.sqrt(bucket["anchor_sq"]), 1e-12),
            "projection_ratio": bucket["projection_sq"] / max(bucket["new_sq"], 1e-12),
            "modules": bucket["modules"],
        }

    modules.sort(key=lambda item: item["interference_score"], reverse=True)
    overall = {
        "new_to_anchor_l2_ratio": math.sqrt(totals["new_sq"]) / max(math.sqrt(totals["anchor_sq"]), 1e-12),
        "projection_ratio": totals["new_on_anchor_sq"] / max(totals["new_sq"], 1e-12),
        "module_count": int(totals["module_count"]),
    }
    return {
        "overall": overall,
        "by_kind": {key: finalize(value) for key, value in sorted(by_kind.items())},
        "by_stack": {key: finalize(value) for key, value in sorted(by_stack.items())},
        "top_modules": modules[:12],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--candidate", action="append", required=True, help="name:adapter_dir[:heldout_log]")
    parser.add_argument("--baseline-heldout-em", type=float, default=55.2)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    anchor = load_state(Path(args.anchor))
    candidates = []
    for spec in args.candidate:
        parts = spec.split(":", 2)
        if len(parts) < 2:
            raise ValueError("--candidate must be name:adapter_dir[:heldout_log]")
        name = parts[0]
        adapter_dir = Path(parts[1])
        log_path = Path(parts[2]) if len(parts) == 3 and parts[2] else None
        heldout_em, heldout_rouge = parse_heldout(log_path)
        diagnostics = analyze_candidate(anchor, load_state(adapter_dir))
        candidates.append(
            {
                "name": name,
                "adapter_dir": str(adapter_dir),
                "heldout_em": heldout_em,
                "heldout_rougeL": heldout_rouge,
                "new_rank_interference": diagnostics,
                "gate_signal": {
                    "exceeds_v69_heldout_baseline": heldout_em is not None and heldout_em > args.baseline_heldout_em,
                    "baseline_heldout_em": args.baseline_heldout_em,
                },
            }
        )

    payload = {
        "policy": {
            "anchor": args.anchor,
            "uses_dev_or_test": False,
            "diagnostic": "new-rank LoRA output-update interference vs amazon anchor update",
            "candidate_control": (
                "For later rounds, penalize new-rank output updates that project onto or dominate "
                "the amazon anchor update subspace on q/v modules; promote only via v76 train-heldout gate."
            ),
        },
        "candidates": candidates,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = ["# v80 New-Rank Interference Diagnostic", ""]
    lines.append("- Leakage policy: adapter weights and train-heldout logs only; no dev/test predictions or targets.")
    lines.append(f"- Anchor: `{args.anchor}`")
    lines.append("")
    for item in candidates:
        overall = item["new_rank_interference"]["overall"]
        lines.extend(
            [
                f"## {item['name']}",
                "",
                f"- Heldout EM: `{item['heldout_em']}`",
                f"- New/anchor L2 ratio: `{overall['new_to_anchor_l2_ratio']:.6f}`",
                f"- New-rank projection ratio: `{overall['projection_ratio']:.6f}`",
            ]
        )
        for kind, bucket in item["new_rank_interference"]["by_kind"].items():
            lines.append(
                f"- `{kind}` new/anchor ratio: `{bucket['new_to_anchor_l2_ratio']:.6f}`, projection: `{bucket['projection_ratio']:.6f}`"
            )
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(out_json), "md": str(out_md), "candidates": candidates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
