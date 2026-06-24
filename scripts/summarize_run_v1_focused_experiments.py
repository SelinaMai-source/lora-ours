from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


FOCUSED_PREFIXES = ("smoke_", "mini_")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _collect_rows(repo_root: Path) -> List[Dict[str, Any]]:
    executions = repo_root / "results" / "tables" / "paper_matrix_executions.csv"
    rows: List[Dict[str, Any]] = []
    for row in _read_csv(executions):
        suffix = str(row.get("run_name_suffix", "")).strip()
        if not suffix.startswith(FOCUSED_PREFIXES):
            continue
        run_name = str(row.get("actual_run_name") or row.get("run_name") or "").strip()
        metrics_path = repo_root / "results" / "runs" / run_name / "final_metrics.json"
        if not metrics_path.is_file():
            continue
        doc = _load_json(metrics_path)
        final = doc.get("final", {}) if isinstance(doc.get("final", {}), dict) else {}
        drift = doc.get("drift_quality", {}) if isinstance(doc.get("drift_quality", {}), dict) else {}
        routing = doc.get("routing_quality", {}) if isinstance(doc.get("routing_quality", {}), dict) else {}
        rows.append(
            {
                "run_name": run_name,
                "suffix": suffix,
                "benchmark": row.get("benchmark", ""),
                "variant_id": row.get("variant_id", ""),
                "overrides": row.get("generic_overrides_json", ""),
                "segments": row.get("max_segments_override", ""),
                "train_examples": row.get("max_train_examples_override", ""),
                "eval_examples": row.get("max_eval_examples_override", ""),
                "epochs": row.get("epochs_per_segment_override", ""),
                "seen_avg": _safe_float(final.get("eval.seen_avg_score")),
                "current_score": _safe_float(final.get("eval.current_score")),
                "forgetting": _safe_float(final.get("eval.forgetting")),
                "token_f1": _safe_float(final.get("eval.token_f1_mean")),
                "prefix1": _safe_float(final.get("eval.prefix_1_match_mean")),
                "num_branches": int(_safe_float(final.get("num_branches"))),
                "drift_miss_rate": _safe_float(drift.get("miss_rate")),
                "drift_false_alarm_rate": _safe_float(drift.get("false_alarm_rate")),
                "drift_detection_delay_mean": _safe_float(drift.get("detection_delay_mean")),
                "oracle_agreement": _safe_float(routing.get("oracle_agreement_rate")),
                "decision_entropy": _safe_float(routing.get("decision_entropy_mean")),
            }
        )
    return sorted(rows, key=lambda r: str(r["run_name"]))


def _write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Run V1 Focused Experiment Summary",
        "",
        "This file summarizes smoke/mini override runs. These runs are intentionally excluded from paper tables.",
        "",
        "| suffix | segments | seen_avg | token_f1 | branches | drift_miss | oracle_agreement | note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        note = ""
        suffix = str(row["suffix"])
        if "abs_drift" in suffix:
            note = "absolute drift sweep; triggers branch spawning"
        elif "routed" in suffix:
            note = "routed-training path check"
        elif "drift" in suffix:
            note = "degradation-only aggressive drift check"
        else:
            note = "default compatibility check"
        lines.append(
            f"| {suffix} | {row['segments']} | {row['seen_avg']:.4f} | {row['token_f1']:.4f} | "
            f"{row['num_branches']} | {row['drift_miss_rate']:.4f} | {row['oracle_agreement']:.4f} | {note} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rows = _collect_rows(repo_root)
    _write_csv(repo_root / "results" / "tables" / "run_v1_focused_experiments.csv", rows)
    _write_markdown(repo_root / "results" / "run_v1_focused_experiments_summary.md", rows)
    print(f"[focused_summary] wrote {len(rows)} focused run row(s)")


if __name__ == "__main__":
    main()
