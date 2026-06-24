from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _row_run_name(row: Dict[str, Any]) -> str:
    return str(row.get("actual_run_name") or row.get("run_name") or "")


def _merge_rows(existing: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    index_by_key: Dict[str, int] = {}
    for row in existing:
        key = _row_run_name(row)
        if key and key not in index_by_key:
            index_by_key[key] = len(merged)
            merged.append(dict(row))
    for row in new_rows:
        key = _row_run_name(row)
        if key and key in index_by_key:
            merged[index_by_key[key]] = {**merged[index_by_key[key]], **dict(row)}
        else:
            if key:
                index_by_key[key] = len(merged)
            merged.append(dict(row))
    return merged


def _status_from_artifacts(run_dir: Path) -> str:
    final_metrics_path = run_dir / "final_metrics.json"
    if final_metrics_path.is_file():
        return "completed"
    manifest = _read_json(run_dir / "run_manifest.json")
    manifest_status = str(manifest.get("status", "")).strip().lower()
    if manifest_status in {"running", "completed", "failed"}:
        return manifest_status
    if run_dir.is_dir():
        segment_dirs = list(run_dir.glob("segment_*"))
        if segment_dirs:
            return "running"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync canonical paper run statuses into paper_matrix_executions.csv.")
    parser.add_argument("--matrix-csv", type=Path, default=Path("results/tables/paper_run_matrix.csv"))
    parser.add_argument("--executions-csv", type=Path, default=Path("results/tables/paper_matrix_executions.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    matrix_csv = (repo_root / args.matrix_csv).resolve()
    executions_csv = (repo_root / args.executions_csv).resolve()
    results_dir = (repo_root / args.results_dir).resolve()

    matrix_rows = _read_csv(matrix_csv)
    existing_rows = _read_csv(executions_csv)
    synced_rows: List[Dict[str, Any]] = []

    for row in matrix_rows:
        run_name = str(row.get("run_name", "")).strip()
        if not run_name:
            continue
        run_dir = results_dir / "runs" / run_name
        status = _status_from_artifacts(run_dir)
        if not status:
            continue
        synced_rows.append(
            {
                **row,
                "actual_run_name": run_name,
                "results_dir": str(args.results_dir),
                "skip_existing": True,
                "run_name_suffix": "",
                "generic_overrides_json": "{}",
                "max_segments_override": "",
                "max_train_examples_override": "",
                "max_eval_examples_override": "",
                "epochs_per_segment_override": "",
                "batch_size_override": "",
                "execution_status": status,
            }
        )

    merged = _merge_rows(existing_rows, synced_rows)
    _write_csv(executions_csv, merged)
    print(f"[sync] canonical_rows={len(synced_rows)} total_rows={len(merged)}")


if __name__ == "__main__":
    main()
