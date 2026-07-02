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


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor phase-1 segment metrics.")
    parser.add_argument("--run-name", default="phase1_ours_v0_debug_smoke")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--min-seen-avg-after-segment", type=float, default=None)
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
            if sid >= 1 and seen is not None and float(seen) < args.min_seen_avg_after_segment:
                status["recommendation"] = "stop_and_diagnose"
                status["reason"] = (
                    f"seen_avg_score={float(seen):.6f} below floor "
                    f"{args.min_seen_avg_after_segment:.6f} at segment {sid}"
                )

    out = args.out if args.out.is_absolute() else REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 42 if status.get("recommendation") == "stop_and_diagnose" else 0


if __name__ == "__main__":
    raise SystemExit(main())
