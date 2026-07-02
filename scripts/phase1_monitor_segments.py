#!/usr/bin/env python3
"""Monitor a phase-1 run directory and emit an auditable status JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def latest_run_dir(run_name: str) -> Path | None:
    root = REPO / "results/runs"
    if not root.is_dir():
        return None
    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name == run_name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _metric(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor phase-1 segment metrics.")
    parser.add_argument("--run-name", default="phase1_ours_v0_debug_smoke")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--min-seen-avg-after-segment", type=float, default=None)
    parser.add_argument("--min-task-aware-after-segment", type=float, default=None)
    parser.add_argument("--min-rouge-l-after-segment", type=float, default=None)
    parser.add_argument("--min-bleu-after-segment", type=float, default=None)
    parser.add_argument("--max-router-fallback-rate", type=float, default=None)
    parser.add_argument("--check-after-segment", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/preflight/phase1_monitor_latest.json"))
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir else latest_run_dir(args.run_name)
    status: dict[str, Any] = {
        "run_name": args.run_name,
        "run_dir": str(run_dir.relative_to(REPO)) if run_dir and run_dir.is_relative_to(REPO) else str(run_dir or ""),
        "found": bool(run_dir and run_dir.is_dir()),
        "recommendation": "no_action",
    }
    if not run_dir or not run_dir.is_dir():
        status["recommendation"] = "wait_or_check_launch"
    else:
        rows = read_jsonl(run_dir / "metrics.jsonl")
        status["segments_observed"] = len(rows)
        status["last_segment"] = rows[-1] if rows else {}
        final_metrics = run_dir / "final_metrics.json"
        status["final_metrics_present"] = final_metrics.is_file()
        if final_metrics.is_file():
            status["final_metrics"] = json.loads(final_metrics.read_text(encoding="utf-8"))
        if args.min_seen_avg_after_segment is not None and rows:
            last = rows[-1]
            seen = last.get("eval.seen_avg_score", last.get("seen_avg_score"))
            sid = int(last.get("segment_id", -1))
            if sid >= args.check_after_segment and seen is not None and float(seen) < args.min_seen_avg_after_segment:
                status["recommendation"] = "stop_and_diagnose"
                status["reason"] = (
                    f"seen_avg_score={float(seen):.6f} below floor "
                    f"{args.min_seen_avg_after_segment:.6f} at segment {sid}"
                )
        if rows and status["recommendation"] != "stop_and_diagnose":
            last = rows[-1]
            sid = int(last.get("segment_id", -1))
            checks = [
                (
                    args.min_task_aware_after_segment,
                    _metric(last, "eval.seen_avg_task_aware_score", "seen_avg_task_aware_score"),
                    "seen_avg_task_aware_score",
                    "below",
                ),
                (
                    args.min_rouge_l_after_segment,
                    _metric(last, "eval.rouge_l_mean", "rouge_l_mean"),
                    "rouge_l_mean",
                    "below",
                ),
                (
                    args.min_bleu_after_segment,
                    _metric(last, "eval.bleu_mean", "bleu_mean"),
                    "bleu_mean",
                    "below",
                ),
            ]
            for threshold, value, name, direction in checks:
                if threshold is None or value is None or sid < args.check_after_segment:
                    continue
                if value < threshold:
                    status["recommendation"] = "stop_and_diagnose"
                    status["reason"] = f"{name}={value:.6f} {direction} floor {threshold:.6f} at segment {sid}"
                    break
            if status["recommendation"] != "stop_and_diagnose" and args.max_router_fallback_rate is not None:
                routed = _metric(last, "train.routed_train_examples", "routed_train_examples") or 0.0
                fallback = _metric(last, "train.routed_train_fallback_to_active", "routed_train_fallback_to_active")
                if sid >= args.check_after_segment and routed > 0 and fallback is not None:
                    fallback_rate = float(fallback / max(1.0, routed))
                    status["router_fallback_rate"] = fallback_rate
                    if fallback_rate > args.max_router_fallback_rate:
                        status["recommendation"] = "stop_and_diagnose"
                        status["reason"] = (
                            f"router_fallback_rate={fallback_rate:.6f} above ceiling "
                            f"{args.max_router_fallback_rate:.6f} at segment {sid}"
                        )

    out = args.out if args.out.is_absolute() else REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 42 if status.get("recommendation") == "stop_and_diagnose" else 0


if __name__ == "__main__":
    raise SystemExit(main())
