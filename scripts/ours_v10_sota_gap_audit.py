#!/usr/bin/env python3
"""Ours v10 SOTA gap audit: Ours vs strongest baseline × 1.10 (seed=123)."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results/tables"
OUT_JSON = TABLES / "ours_v10_sota_gap_s123.json"
OUT_MD = TABLES / "ours_v10_sota_gap_s123.md"

SOTA_MARGIN = 1.10
FORGET_MARGIN = 0.90  # lower is better: pass if ours <= baseline_best * 0.9

CORE_METRICS = [
    ("seen_avg_score", "eval.seen_avg_score", "higher"),
    ("seen_avg_task_aware_score", "eval.seen_avg_task_aware_score", "higher"),
    ("token_f1_mean", "eval.token_f1_mean", "higher"),
    ("forgetting", "eval.forgetting", "lower"),
]

FORGET_EXCLUDE_SUFFIXES = {"lfpt5"}  # degenerate forgetting=0 bridge runs

BASELINE_SUFFIXES = [
    ("sequential_lora", "Sequential"),
    ("replay_lora", "Replay"),
    ("o_lora", "O-LoRA"),
    ("lb_cl", "LB-CL"),
    ("progressive_prompts", "PP"),
    ("continual_t0", "Continual-T0"),
    ("lfpt5", "LFPT5"),
]

BENCHMARKS: List[Dict[str, str]] = [
    {"key": "instrdialog", "label": "InstrDialog", "prefix": "published_instrdialog"},
    {"key": "instrdialogpp", "label": "InstrDialog++", "prefix": "published_instrdialogpp"},
    {"key": "trace", "label": "TRACE", "prefix": "trace_full"},
    {"key": "multiwoz", "label": "MultiWOZ NLG", "prefix": "full_multiwoz"},
    {"key": "seqglue", "label": "Seq-GLUE", "prefix": "published_seqglue"},
]


@dataclass
class MetricGap:
    metric: str
    direction: str
    ours: Optional[float]
    best_baseline: Optional[float]
    best_baseline_method: Optional[str]
    target: Optional[float]
    ratio_to_target: Optional[float]
    gap_pct: Optional[float]
    pass_sota: bool
    missing: bool = False


@dataclass
class BenchmarkGap:
    benchmark: str
    label: str
    ours_run_id: Optional[str]
    metrics: List[MetricGap]
    n_fail: int
    n_pass: int
    n_missing: int
    weakest_metric: Optional[str]
    skip_campaign: bool


def _read_last_row(csv_path: Path) -> Optional[Dict[str, str]]:
    if not csv_path.is_file():
        return None
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        last = None
        for row in reader:
            last = row
        return last


def _f(val: Optional[str]) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _metric_pass(
    ours: Optional[float],
    best_bl: Optional[float],
    direction: str,
) -> Tuple[bool, Optional[float], Optional[float], Optional[float]]:
    if ours is None or best_bl is None:
        return False, None, None, None
    if direction == "higher":
        target = best_bl * SOTA_MARGIN
        pass_ok = ours >= target
        ratio = ours / target if target > 0 else (1.0 if pass_ok else 0.0)
        gap_pct = (ours - target) / target * 100.0 if target > 0 else None
        return pass_ok, target, ratio, gap_pct
    target = best_bl * FORGET_MARGIN
    pass_ok = ours <= target
    ratio = target / ours if ours > 0 else (1.0 if pass_ok else 0.0)
    gap_pct = (target - ours) / target * 100.0 if target > 0 else None
    return pass_ok, target, ratio, gap_pct


def audit_benchmark(spec: Dict[str, str]) -> BenchmarkGap:
    prefix = spec["prefix"]
    ours_path = TABLES / f"{prefix}_ours_s123_segment_metrics.csv"
    ours_row = _read_last_row(ours_path)
    ours_run = ours_row.get("run_id") if ours_row else None

    metric_gaps: List[MetricGap] = []
    for metric_name, col, direction in CORE_METRICS:
        ours_val = _f(ours_row.get(col)) if ours_row else None
        best_bl = None
        best_method = None
        for suffix, label in BASELINE_SUFFIXES:
            if metric_name == "forgetting" and suffix in FORGET_EXCLUDE_SUFFIXES:
                continue
            bl_path = TABLES / f"{prefix}_{suffix}_s123_segment_metrics.csv"
            row = _read_last_row(bl_path)
            if row is None:
                continue
            v = _f(row.get(col))
            if v is None:
                continue
            if direction == "higher":
                if best_bl is None or v > best_bl:
                    best_bl = v
                    best_method = label
            else:
                if best_bl is None or v < best_bl:
                    best_bl = v
                    best_method = label

        missing = ours_val is None or best_bl is None
        if missing:
            mg = MetricGap(
                metric=metric_name,
                direction=direction,
                ours=ours_val,
                best_baseline=best_bl,
                best_baseline_method=best_method,
                target=None,
                ratio_to_target=None,
                gap_pct=None,
                pass_sota=False,
                missing=True,
            )
        else:
            ok, target, ratio, gap_pct = _metric_pass(ours_val, best_bl, direction)
            mg = MetricGap(
                metric=metric_name,
                direction=direction,
                ours=ours_val,
                best_baseline=best_bl,
                best_baseline_method=best_method,
                target=target,
                ratio_to_target=ratio,
                gap_pct=gap_pct,
                pass_sota=ok,
                missing=False,
            )
        metric_gaps.append(mg)

    n_fail = sum(1 for m in metric_gaps if not m.pass_sota and not m.missing)
    n_pass = sum(1 for m in metric_gaps if m.pass_sota)
    n_missing = sum(1 for m in metric_gaps if m.missing)
    fail_metrics = [m for m in metric_gaps if not m.pass_sota and not m.missing]
    weakest = None
    if fail_metrics:
        # smallest ratio_to_target = furthest from SOTA
        ranked = sorted(
            fail_metrics,
            key=lambda m: m.ratio_to_target if m.ratio_to_target is not None else 0.0,
        )
        weakest = ranked[0].metric

    return BenchmarkGap(
        benchmark=spec["key"],
        label=spec["label"],
        ours_run_id=ours_run,
        metrics=metric_gaps,
        n_fail=n_fail,
        n_pass=n_pass,
        n_missing=n_missing,
        weakest_metric=weakest,
        skip_campaign=n_fail == 0 and n_missing == 0,
    )


def pick_weakest(benchmarks: List[BenchmarkGap]) -> Optional[BenchmarkGap]:
    active = [b for b in benchmarks if not b.skip_campaign]
    if not active:
        return None

    priority = {"seqglue": 0, "trace": 1}

    def score(b: BenchmarkGap) -> Tuple[int, int, float]:
        fail_ratios = [
            m.ratio_to_target
            for m in b.metrics
            if not m.pass_sota and not m.missing and m.ratio_to_target is not None
        ]
        min_ratio = min(fail_ratios) if fail_ratios else 1.0
        return (-b.n_fail, priority.get(b.benchmark, 9), min_ratio)

    return sorted(active, key=score)[0]


def _md_table(benchmarks: List[BenchmarkGap]) -> str:
    lines = [
        "# Ours v10 SOTA Gap Audit (seed=123)",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "**SOTA rule**: Ours ≥ best baseline × 1.10 (higher-better metrics); "
        "forgetting ≤ best baseline × 0.90.",
        "",
        "## Summary",
        "",
        "| Benchmark | Fail | Pass | Skip? | Weakest metric |",
        "|-----------|------|------|-------|----------------|",
    ]
    for b in benchmarks:
        skip = "✅" if b.skip_campaign else "—"
        lines.append(
            f"| {b.label} | {b.n_fail} | {b.n_pass} | {skip} | {b.weakest_metric or '—'} |"
        )
    weakest = pick_weakest(benchmarks)
    if weakest:
        lines += [
            "",
            f"**Campaign priority**: {weakest.label} (`{weakest.benchmark}`)",
        ]
    else:
        lines += ["", "**Campaign priority**: none — all benchmarks at SOTA ✅"]

    for b in benchmarks:
        lines += [
            "",
            f"## {b.label}",
            "",
            f"- Ours run: `{b.ours_run_id or 'MISSING'}`",
            "",
            "| Metric | Ours | Best BL | BL method | Target | Gap% | Pass |",
            "|--------|------|---------|-----------|--------|------|------|",
        ]
        for m in b.metrics:
            if m.missing:
                lines.append(
                    f"| {m.metric} | {m.ours} | {m.best_baseline} | {m.best_baseline_method} | — | — | ❓ |"
                )
                continue
            status = "✅" if m.pass_sota else "❌"
            gap_s = f"{m.gap_pct:+.1f}%" if m.gap_pct is not None else "—"
            tgt = f"{m.target:.4f}" if m.target is not None else "—"
            lines.append(
                f"| {m.metric} | {m.ours:.4f} | {m.best_baseline:.4f} | {m.best_baseline_method} "
                f"| {tgt} | {gap_s} | {status} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    results = [audit_benchmark(spec) for spec in BENCHMARKS]
    weakest = pick_weakest(results)
    payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "sota_margin": SOTA_MARGIN,
        "forget_margin": FORGET_MARGIN,
        "benchmarks": [],
        "weakest_benchmark": weakest.benchmark if weakest else None,
        "total_fail_cells": sum(b.n_fail for b in results),
        "total_pass_cells": sum(b.n_pass for b in results),
    }
    for b in results:
        payload["benchmarks"].append(
            {
                **{k: v for k, v in asdict(b).items() if k != "metrics"},
                "metrics": [asdict(m) for m in b.metrics],
            }
        )
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(_md_table(results), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    if weakest:
        print(f"Weakest: {weakest.label} ({weakest.n_fail} failing metrics)")
    else:
        print("All benchmarks at SOTA")


if __name__ == "__main__":
    main()
