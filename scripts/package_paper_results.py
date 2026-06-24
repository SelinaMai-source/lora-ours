from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _default_exec_csv(repo_root: Path) -> Path:
    exec_csv = repo_root / "results" / "tables" / "paper_matrix_executions.csv"
    if exec_csv.is_file():
        return exec_csv
    return repo_root / "results" / "tables" / "paper_run_matrix.csv"


def _truncate(text: Any, limit: int = 240) -> str:
    s = str(text or "").strip().replace("\n", " ")
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _collect_failure_cases(
    *,
    results_dir: Path,
    run_rows: List[Dict[str, str]],
    max_cases: int,
) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for row in run_rows:
        run_name = str(row.get("resolved_run_name") or row.get("actual_run_name") or row.get("run_name") or "").strip()
        if not run_name:
            continue
        eval_debug_dir = results_dir / "runs" / run_name / "eval_debug"
        if not eval_debug_dir.is_dir():
            continue
        for path in sorted(eval_debug_dir.glob("eval_segment_*.json")):
            payload = _read_json(path)
            seg_id = int(payload.get("segment_id", -1))
            for ex in payload.get("examples", []) if isinstance(payload.get("examples", []), list) else []:
                if bool(ex.get("match", False)):
                    continue
                failures.append(
                    {
                        "run_name": run_name,
                        "segment_id": seg_id,
                        "token_f1": _safe_float(ex.get("token_f1")),
                        "lcs_overlap": _safe_float(ex.get("lcs_overlap")),
                        "bad_prefix_mismatch": bool(ex.get("bad_prefix_mismatch", False)),
                        "routing_selected_branch": ex.get("routing_selected_branch", ""),
                        "routing_oracle_branch": ex.get("routing_oracle_branch", ""),
                        "routing_oracle_margin": _safe_float(ex.get("routing_oracle_margin")),
                        "instruction": _truncate(ex.get("instruction"), 180),
                        "input_text": _truncate(ex.get("input_text"), 220),
                        "gold_output": str(ex.get("gold_output", "")),
                        "predicted_output": str(ex.get("raw_generated_output", "")),
                    }
                )
    failures.sort(
        key=lambda x: (
            -int(x.get("bad_prefix_mismatch", False)),
            _safe_float(x.get("token_f1")),
            -_safe_float(x.get("routing_oracle_margin")),
            -int(x.get("segment_id", -1)),
        )
    )
    return failures[:max_cases]


def _write_markdown(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _build_final_report(
    *,
    out_path: Path,
    main_agg_rows: List[Dict[str, str]],
    ablation_agg_rows: List[Dict[str, str]],
) -> None:
    lines = [
        "# Final Paper Package",
        "",
        f"- generated_at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- main_rows: `{len(main_agg_rows)}`",
        f"- ablation_rows: `{len(ablation_agg_rows)}`",
        "",
        "## Main Table Snapshot",
        "",
    ]
    if main_agg_rows:
        lines.append("| benchmark | method | n | seen_avg | forgetting | token_f1 | oracle_agreement |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in main_agg_rows:
            lines.append(
                f"| {row.get('benchmark', '')} | {row.get('method_label', row.get('variant_id', ''))} | "
                f"{row.get('num_runs', '0')} | "
                f"{_safe_float(row.get('eval.seen_avg_score.mean')):.4f} | "
                f"{_safe_float(row.get('eval.forgetting.mean')):.4f} | "
                f"{_safe_float(row.get('eval.token_f1_mean.mean')):.4f} | "
                f"{_safe_float(row.get('routing.oracle_agreement_rate.mean')):.4f} |"
            )
    else:
        lines.append("_No aggregated main rows yet._")

    lines.extend(["", "## Ablation Snapshot", ""])
    if ablation_agg_rows:
        lines.append("| benchmark | method | seen_avg | forgetting | token_f1 |")
        lines.append("|---|---|---:|---:|---:|")
        for row in ablation_agg_rows:
            lines.append(
                f"| {row.get('benchmark', '')} | {row.get('method_label', row.get('variant_id', ''))} | "
                f"{_safe_float(row.get('eval.seen_avg_score.mean')):.4f} | "
                f"{_safe_float(row.get('eval.forgetting.mean')):.4f} | "
                f"{_safe_float(row.get('eval.token_f1_mean.mean')):.4f} |"
            )
    else:
        lines.append("_No ablation rows yet._")
    _write_markdown(out_path, lines)


def _build_failure_cases_report(*, out_path: Path, failures: List[Dict[str, Any]]) -> None:
    lines = [
        "# Failure Cases",
        "",
        f"- num_cases: `{len(failures)}`",
        "",
    ]
    if not failures:
        lines.append("_No failure cases available yet._")
        _write_markdown(out_path, lines)
        return
    for idx, row in enumerate(failures, start=1):
        lines.extend(
            [
                f"## Case {idx}",
                "",
                f"- run: `{row['run_name']}`",
                f"- segment: `{row['segment_id']}`",
                f"- token_f1: `{row['token_f1']:.4f}`",
                f"- lcs_overlap: `{row['lcs_overlap']:.4f}`",
                f"- bad_prefix_mismatch: `{row['bad_prefix_mismatch']}`",
                f"- routing: `{row['routing_selected_branch']}` vs oracle `{row['routing_oracle_branch']}` "
                f"(margin `{row['routing_oracle_margin']:.4f}`)",
                f"- instruction: `{row['instruction']}`",
                f"- input_excerpt: `{row['input_text']}`",
                "",
                "### Gold",
                "",
                row["gold_output"] or "_empty_",
                "",
                "### Prediction",
                "",
                row["predicted_output"] or "_empty_",
                "",
            ]
        )
    _write_markdown(out_path, lines)


def _build_repro_checklist(
    *,
    out_path: Path,
    results_dir: Path,
    exec_rows: List[Dict[str, str]],
) -> None:
    lines = [
        "# Reproducibility Checklist",
        "",
        "| run | config_snapshot | final_metrics | run_manifest | segment_table | log_file |",
        "|---|---|---|---|---|---|",
    ]
    if not exec_rows:
        lines.append("| _none_ | - | - | - | - | - |")
        _write_markdown(out_path, lines)
        return
    for row in exec_rows:
        run_name = str(row.get("actual_run_name") or row.get("run_name") or "").strip()
        if not run_name:
            continue
        run_dir = results_dir / "runs" / run_name
        seg_table = results_dir / "tables" / f"{run_name}_segment_metrics.csv"
        log_file = results_dir / "logs" / f"{run_name}.log"
        status = lambda p: "yes" if p.exists() else "no"
        lines.append(
            f"| `{run_name}` | {status(run_dir / 'config_snapshot.yaml')} | {status(run_dir / 'final_metrics.json')} | "
            f"{status(run_dir / 'run_manifest.json')} | {status(seg_table)} | {status(log_file)} |"
        )
    _write_markdown(out_path, lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package final paper-ready outputs: summary, failure cases, reproducibility.")
    parser.add_argument(
        "--executions-csv",
        type=Path,
        default=None,
        help="Execution manifest to package. Defaults to paper_matrix_executions.csv if available.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/final_package"))
    parser.add_argument("--max-failure-cases", type=int, default=8)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = (repo_root / args.results_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    exec_csv = (repo_root / args.executions_csv).resolve() if args.executions_csv is not None else _default_exec_csv(repo_root)
    exec_rows = _read_csv(exec_csv)
    main_agg_rows = _read_csv(results_dir / "tables" / "paper_main_results_agg.csv")
    ablation_agg_rows = _read_csv(results_dir / "tables" / "paper_ablation_results_agg.csv")
    main_rows = _read_csv(results_dir / "tables" / "paper_main_results.csv")

    failures = _collect_failure_cases(
        results_dir=results_dir,
        run_rows=main_rows or exec_rows,
        max_cases=max(1, int(args.max_failure_cases)),
    )
    _build_final_report(
        out_path=out_dir / "final_report.md",
        main_agg_rows=main_agg_rows,
        ablation_agg_rows=ablation_agg_rows,
    )
    _build_failure_cases_report(out_path=out_dir / "failure_cases.md", failures=failures)
    _build_repro_checklist(out_path=out_dir / "reproducibility_checklist.md", results_dir=results_dir, exec_rows=exec_rows)


if __name__ == "__main__":
    main()
