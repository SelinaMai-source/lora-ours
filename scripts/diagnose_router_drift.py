from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _entropy(counts: Dict[str, Any]) -> float:
    total = float(sum(_safe_float(v) for v in counts.values()))
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in counts.values():
        p = _safe_float(v) / total
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    return float(ent)


def _router_collapse_flag(branch_counts: Dict[str, Any]) -> bool:
    total = float(sum(_safe_float(v) for v in branch_counts.values()))
    if total <= 0 or len(branch_counts) <= 1:
        return False
    top = max(_safe_float(v) for v in branch_counts.values())
    return (top / total) >= 0.9


def _summarize_run(run_dir: Path) -> Dict[str, Any]:
    final_path = run_dir / "final_metrics.json"
    final_doc = _read_json(final_path) if final_path.is_file() else {}
    routing = final_doc.get("routing_quality", {}) if isinstance(final_doc.get("routing_quality", {}), dict) else {}
    drift = final_doc.get("drift_quality", {}) if isinstance(final_doc.get("drift_quality", {}), dict) else {}
    branch_counts = routing.get("branch_counts", {}) if isinstance(routing.get("branch_counts", {}), dict) else {}
    oracle_counts = (
        routing.get("oracle_best_branch_counts", {})
        if isinstance(routing.get("oracle_best_branch_counts", {}), dict)
        else {}
    )
    monitor_rows = _read_jsonl(run_dir / "anchor_monitor.jsonl")
    drift_events = _read_jsonl(run_dir / "drift_events.jsonl")
    triggered_monitor = sum(1 for row in monitor_rows if bool(row.get("triggered", False)))
    calibrated_monitor = sum(1 for row in monitor_rows if bool(row.get("calibrated", False)))
    probe_hits = sum(1 for row in monitor_rows if _safe_float(row.get("probe_cusum")) > _safe_float(row.get("threshold")))

    detail_counts: Counter[str] = Counter()
    router_mismatch = 0
    router_total = 0
    for debug_path in sorted((run_dir / "eval_debug").glob("eval_segment_*.json")):
        doc = _read_json(debug_path)
        for ex in doc.get("examples", []) or []:
            if not isinstance(ex, dict):
                continue
            selected = str(ex.get("routing_selected_branch") or "")
            oracle = str(ex.get("routing_oracle_branch") or "")
            if selected:
                detail_counts[selected] += 1
            if selected and oracle:
                router_total += 1
                router_mismatch += int(selected != oracle)

    selected_counts = dict(branch_counts or detail_counts)
    return {
        "run_name": run_dir.name,
        "num_routed": int(_safe_float(routing.get("num_routed"))),
        "router_branch_counts": json.dumps(selected_counts, ensure_ascii=False, sort_keys=True),
        "router_oracle_best_branch_counts": json.dumps(oracle_counts, ensure_ascii=False, sort_keys=True),
        "router_branch_entropy": _entropy(selected_counts),
        "router_oracle_entropy": _entropy(oracle_counts),
        "router_oracle_agreement_rate": _safe_float(routing.get("oracle_agreement_rate")),
        "router_debug_mismatch_rate": router_mismatch / max(1, router_total),
        "router_collapse_flag": _router_collapse_flag(selected_counts),
        "drift_false_alarm_rate": _safe_float(drift.get("false_alarm_rate")),
        "drift_miss_rate": _safe_float(drift.get("miss_rate")),
        "drift_detection_delay_mean": _safe_float(drift.get("detection_delay_mean")),
        "drift_monitor_rows": len(monitor_rows),
        "drift_calibrated_rows": calibrated_monitor,
        "drift_triggered_monitor_rows": triggered_monitor,
        "drift_event_rows": len(drift_events),
        "drift_probe_cusum_over_threshold_rows": probe_hits,
        "recommended_drift_sweep": "threshold,shift_stat=absolute,calibration_window,core_guard_scale"
        if _safe_float(drift.get("miss_rate")) >= 0.5 or (monitor_rows and triggered_monitor == 0)
        else "",
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Router And Drift Diagnostics", ""]
    if not rows:
        lines.append("_No runs found._")
    else:
        lines.extend(
            [
                "| run | routed | branch_entropy | oracle_agreement | collapse | drift_miss | drift_triggers | recommendation |",
                "|---|---:|---:|---:|---|---:|---:|---|",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row['run_name']}` | {int(row['num_routed'])} | "
                f"{_safe_float(row['router_branch_entropy']):.4f} | "
                f"{_safe_float(row['router_oracle_agreement_rate']):.4f} | "
                f"{row['router_collapse_flag']} | "
                f"{_safe_float(row['drift_miss_rate']):.4f} | "
                f"{int(row['drift_triggered_monitor_rows'])} | "
                f"{row['recommended_drift_sweep']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize router collapse and drift miss-rate diagnostics.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = (repo_root / args.results_dir).resolve()
    rows = [_summarize_run(p) for p in sorted((results_dir / "runs").glob("*")) if p.is_dir()]
    rows = [row for row in rows if row["num_routed"] > 0 or row["drift_monitor_rows"] > 0 or row["drift_miss_rate"] > 0]
    _write_csv(results_dir / "tables" / "router_drift_diagnostics.csv", rows)
    _write_markdown(results_dir / "router_drift_diagnostics.md", rows)


if __name__ == "__main__":
    main()
