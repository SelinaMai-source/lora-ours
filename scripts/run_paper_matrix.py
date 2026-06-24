from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _read_csv(path: Path) -> List[Dict[str, str]]:
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


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_override_pairs(values: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Override must look like key=value, got: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Override key is empty: {raw}")
        out[key] = yaml.safe_load(value)
    return out


def _merge_execution_rows(existing: List[Dict[str, str]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    index_by_key: Dict[str, int] = {}
    for row in existing:
        key = str(row.get("actual_run_name") or row.get("run_name") or "")
        if key and key not in index_by_key:
            index_by_key[key] = len(merged)
            merged.append(dict(row))
    for row in new_rows:
        key = str(row.get("actual_run_name") or row.get("run_name") or "")
        if key and key in index_by_key:
            merged[index_by_key[key]] = dict(row)
        else:
            if key:
                index_by_key[key] = len(merged)
            merged.append(dict(row))
    return merged


def _write_merged_execution_rows(path: Path, new_rows: List[Dict[str, Any]]) -> None:
    existing_rows = _read_csv(path) if path.is_file() else []
    _write_csv(path, _merge_execution_rows(existing_rows, new_rows))


def _matches_filter(value: str, allowed: List[str]) -> bool:
    if not allowed:
        return True
    return value in set(allowed)


def _select_rows(
    rows: List[Dict[str, str]],
    *,
    benchmarks: List[str],
    modes: List[str],
    categories: List[str],
    variant_ids: List[str],
    run_names: List[str],
) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    for row in rows:
        if not _matches_filter(row.get("benchmark", ""), benchmarks):
            continue
        if not _matches_filter(row.get("mode", ""), modes):
            continue
        if not _matches_filter(row.get("category", ""), categories):
            continue
        if not _matches_filter(row.get("variant_id", ""), variant_ids):
            continue
        if run_names and row.get("run_name", "") not in set(run_names):
            continue
        selected.append(row)
    return selected


def _apply_overrides(
    cfg: Dict[str, Any],
    *,
    max_segments: int,
    max_train_examples: int,
    max_eval_examples: int,
    epochs_per_segment: int,
    batch_size: int,
    run_name_suffix: str,
    generic_overrides: Dict[str, Any],
) -> None:
    data_cfg = cfg.setdefault("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("data config must be a mapping")
    if max_segments > 0:
        data_cfg["max_segments"] = int(max_segments)
    if max_train_examples > 0:
        data_cfg["max_train_examples_per_segment"] = int(max_train_examples)
    if max_eval_examples > 0:
        data_cfg["max_eval_examples_per_segment"] = int(max_eval_examples)

    train_cfg = cfg.setdefault("train", {})
    if not isinstance(train_cfg, dict):
        raise ValueError("train config must be a mapping")
    if epochs_per_segment > 0:
        train_cfg["epochs_per_segment"] = int(epochs_per_segment)
    if batch_size > 0:
        train_cfg["batch_size"] = int(batch_size)

    output_cfg = cfg.setdefault("output", {})
    if not isinstance(output_cfg, dict):
        raise ValueError("output config must be a mapping")
    if run_name_suffix:
        base_run_name = str(output_cfg.get("run_name", "")).strip()
        output_cfg["run_name"] = f"{base_run_name}_{run_name_suffix}"
    for dotted_key, value in generic_overrides.items():
        cur: Dict[str, Any] = cfg
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            next_obj = cur.setdefault(part, {})
            if not isinstance(next_obj, dict):
                raise ValueError(f"Override parent must be a mapping: {dotted_key}")
            cur = next_obj
        cur[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute generated paper-matrix configs with optional overrides.")
    parser.add_argument(
        "--matrix-csv",
        type=Path,
        default=Path("results/tables/paper_run_matrix.csv"),
        help="Config manifest generated by build_paper_run_matrix.py",
    )
    parser.add_argument("--benchmarks", type=str, default="", help="Comma-separated benchmark aliases to run.")
    parser.add_argument("--modes", type=str, default="", help="Comma-separated modes to run.")
    parser.add_argument("--categories", type=str, default="", help="Comma-separated categories to run.")
    parser.add_argument("--variant-ids", type=str, default="", help="Comma-separated variant IDs to run.")
    parser.add_argument("--run-names", type=str, default="", help="Comma-separated explicit run_names to run.")
    parser.add_argument("--max-segments", type=int, default=-1, help="Override data.max_segments.")
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=-1,
        help="Override data.max_train_examples_per_segment.",
    )
    parser.add_argument(
        "--max-eval-examples",
        type=int,
        default=-1,
        help="Override data.max_eval_examples_per_segment.",
    )
    parser.add_argument("--epochs-per-segment", type=int, default=-1, help="Override train.epochs_per_segment.")
    parser.add_argument("--batch-size", type=int, default=-1, help="Override train.batch_size.")
    parser.add_argument(
        "--run-name-suffix",
        type=str,
        default="",
        help="Append a suffix to output.run_name so pilot runs do not overwrite full runs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose final_metrics.json already exists.",
    )
    parser.add_argument(
        "--executions-csv",
        type=Path,
        default=Path("results/tables/paper_matrix_executions.csv"),
        help="Where to write the concrete run manifest after overrides/suffixes are applied.",
    )
    parser.add_argument(
        "--set",
        dest="generic_overrides",
        action="append",
        default=[],
        help="Additional dotted config overrides, e.g. --set drift.calibration_window=1",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    rows = _read_csv((repo_root / args.matrix_csv).resolve())
    generic_overrides = _parse_override_pairs(list(args.generic_overrides or []))
    selected = _select_rows(
        rows,
        benchmarks=_parse_csv_list(args.benchmarks),
        modes=_parse_csv_list(args.modes),
        categories=_parse_csv_list(args.categories),
        variant_ids=_parse_csv_list(args.variant_ids),
        run_names=_parse_csv_list(args.run_names),
    )
    if not selected:
        raise ValueError("No configs matched the requested filters.")

    executions_csv_path = (repo_root / args.executions_csv).resolve()
    executed = 0
    skipped = 0
    execution_rows: List[Dict[str, Any]] = []
    for row in selected:
        cfg_path = (repo_root / row["config_path"]).resolve()
        cfg = _load_yaml(cfg_path)
        _apply_overrides(
            cfg,
            max_segments=args.max_segments,
            max_train_examples=args.max_train_examples,
            max_eval_examples=args.max_eval_examples,
            epochs_per_segment=args.epochs_per_segment,
            batch_size=args.batch_size,
            run_name_suffix=args.run_name_suffix.strip(),
            generic_overrides=generic_overrides,
        )

        output_cfg = cfg.get("output", {}) if isinstance(cfg.get("output", {}), dict) else {}
        run_name = str(output_cfg.get("run_name", "")).strip()
        results_dir = str(cfg.get("paths", {}).get("results_dir", "results"))
        final_metrics_path = repo_root / results_dir / "runs" / run_name / "final_metrics.json"
        execution_row: Dict[str, Any] = {
            **row,
            "actual_run_name": run_name,
            "results_dir": results_dir,
            "skip_existing": bool(args.skip_existing),
            "run_name_suffix": args.run_name_suffix.strip(),
            "generic_overrides_json": json.dumps(generic_overrides, ensure_ascii=False, sort_keys=True),
            "max_segments_override": int(args.max_segments) if args.max_segments > 0 else "",
            "max_train_examples_override": int(args.max_train_examples) if args.max_train_examples > 0 else "",
            "max_eval_examples_override": int(args.max_eval_examples) if args.max_eval_examples > 0 else "",
            "epochs_per_segment_override": int(args.epochs_per_segment) if args.epochs_per_segment > 0 else "",
            "batch_size_override": int(args.batch_size) if args.batch_size > 0 else "",
            "execution_status": "queued",
        }
        execution_rows.append(execution_row)
        _write_merged_execution_rows(executions_csv_path, [execution_row])
        if args.skip_existing and final_metrics_path.is_file():
            print(f"[skip-existing] {run_name}")
            skipped += 1
            execution_row["execution_status"] = "skipped_existing"
            _write_merged_execution_rows(executions_csv_path, [execution_row])
            continue

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp_cfg:
            yaml.safe_dump(cfg, tmp_cfg, sort_keys=False, allow_unicode=True)
            tmp_cfg_path = Path(tmp_cfg.name)

        print(f"[run] {run_name} <- {row['config_path']}")
        execution_row["execution_status"] = "running"
        _write_merged_execution_rows(executions_csv_path, [execution_row])
        try:
            subprocess.run([sys.executable, "core/train.py", "--config", str(tmp_cfg_path)], cwd=repo_root, check=True)
            execution_row["execution_status"] = "completed"
        except Exception:
            execution_row["execution_status"] = "failed"
            raise
        finally:
            tmp_cfg_path.unlink(missing_ok=True)
            _write_merged_execution_rows(executions_csv_path, [execution_row])
        executed += 1

    if execution_rows:
        _write_merged_execution_rows(executions_csv_path, execution_rows)
    print(f"[done] executed={executed} skipped={skipped} selected={len(selected)}")


if __name__ == "__main__":
    main()
