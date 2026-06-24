from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_OURS_CANDIDATES = ["ours_no_overlap", "ours_full", "ours_no_router", "ours_no_drift", "ours_no_bank"]
DEFAULT_BASELINE_CANDIDATES = ["seq", "replay_b10", "replay_b50", "periodic_latest", "bank_no_router", "router_only"]
EXCLUDED_RUN_TOKENS = ("pilot", "smoke", "sweep")


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


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _row_run_name(row: Dict[str, Any]) -> str:
    return str(row.get("actual_run_name") or row.get("run_name") or "")


def _is_paper_candidate_run(run_name: str) -> bool:
    lower = run_name.lower()
    return not any(token in lower for token in EXCLUDED_RUN_TOKENS)


def _metric_key(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        _safe_float(row.get("seen_avg_score")),
        _safe_float(row.get("token_f1_mean")),
        _safe_float(row.get("oracle_agreement_rate")),
        -_safe_float(row.get("forgetting")),
    )


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
            merged[index_by_key[key]] = dict(row)
        else:
            if key:
                index_by_key[key] = len(merged)
            merged.append(dict(row))
    return merged


def _load_matrix_rows(repo_root: Path, matrix_csv: Path) -> List[Dict[str, Any]]:
    base_csv = (repo_root / "results" / "tables" / "paper_run_matrix.csv").resolve()
    base_rows = _read_csv(base_csv)
    execution_rows = _read_csv(matrix_csv)
    return _merge_rows(base_rows, execution_rows)


def _load_candidate_rows(
    *,
    matrix_rows: List[Dict[str, Any]],
    results_dir: Path,
    benchmark: str,
    variant_ids: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in matrix_rows:
        if str(row.get("benchmark", "")) != benchmark:
            continue
        if str(row.get("variant_id", "")) not in set(variant_ids):
            continue
        run_name = _row_run_name(row)
        if not run_name:
            continue
        if not _is_paper_candidate_run(run_name):
            continue
        final_doc = _read_json(results_dir / "runs" / run_name / "final_metrics.json")
        if not final_doc:
            continue
        final = final_doc.get("final", {}) if isinstance(final_doc.get("final", {}), dict) else {}
        routing = final_doc.get("routing_quality", {}) if isinstance(final_doc.get("routing_quality", {}), dict) else {}
        rows.append(
            {
                **row,
                "resolved_run_name": run_name,
                "seen_avg_score": _safe_float(final.get("eval.seen_avg_score")),
                "token_f1_mean": _safe_float(final.get("eval.token_f1_mean")),
                "forgetting": _safe_float(final.get("eval.forgetting")),
                "anytime_score": _safe_float(final.get("eval.anytime_score")),
                "oracle_agreement_rate": _safe_float(routing.get("oracle_agreement_rate")),
                "run_name_suffix": str(row.get("run_name_suffix", "")),
                "generic_overrides_json": str(row.get("generic_overrides_json", "")),
            }
        )
    return rows


def _beats(lhs: Dict[str, Any], rhs: Dict[str, Any]) -> bool:
    lhs_seen = _safe_float(lhs.get("seen_avg_score"))
    rhs_seen = _safe_float(rhs.get("seen_avg_score"))
    if lhs_seen > rhs_seen + 1e-9:
        return True
    if lhs_seen + 1e-9 < rhs_seen:
        return False
    lhs_f1 = _safe_float(lhs.get("token_f1_mean"))
    rhs_f1 = _safe_float(rhs.get("token_f1_mean"))
    if lhs_f1 > rhs_f1 + 1e-9:
        return True
    if lhs_f1 + 1e-9 < rhs_f1:
        return False
    lhs_forgetting = _safe_float(lhs.get("forgetting"))
    rhs_forgetting = _safe_float(rhs.get("forgetting"))
    if lhs_forgetting + 1e-9 < rhs_forgetting:
        return True
    if lhs_forgetting > rhs_forgetting + 1e-9:
        return False
    return _safe_float(lhs.get("oracle_agreement_rate")) >= _safe_float(rhs.get("oracle_agreement_rate"))


def _top_variant_ids(rows: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    seen = set()
    out: List[str] = []
    for row in sorted(rows, key=_metric_key, reverse=True):
        variant_id = str(row.get("variant_id", ""))
        if not variant_id or variant_id in seen:
            continue
        seen.add(variant_id)
        out.append(variant_id)
        if len(out) >= limit:
            break
    return out


def _secondary_match_rows(rows: List[Dict[str, Any]], winner_row: Dict[str, Any], benchmark: str) -> List[Dict[str, Any]]:
    if not winner_row:
        return []
    variant_id = str(winner_row.get("variant_id", ""))
    run_name_suffix = str(winner_row.get("run_name_suffix", ""))
    overrides_json = str(winner_row.get("generic_overrides_json", ""))
    scoped = [
        row
        for row in rows
        if str(row.get("benchmark", "")) == benchmark and str(row.get("variant_id", "")) == variant_id
    ]
    exact = [
        row
        for row in scoped
        if str(row.get("run_name_suffix", "")) == run_name_suffix
        and str(row.get("generic_overrides_json", "")) == overrides_json
    ]
    if exact:
        return exact
    if not run_name_suffix and overrides_json in ("", "{}"):
        canonical = [
            row
            for row in scoped
            if str(row.get("run_name_suffix", "")) in ("",)
            and str(row.get("generic_overrides_json", "")) in ("", "{}")
        ]
        if canonical:
            return canonical
    return scoped


def _secondary_guard_pass(
    candidate: Dict[str, Any],
    baseline: Dict[str, Any],
    *,
    max_seen_gap: float,
    max_token_f1_gap: float,
    max_forgetting_gap: float,
) -> bool:
    if not candidate or not baseline:
        return False
    seen_gap = _safe_float(baseline.get("seen_avg_score")) - _safe_float(candidate.get("seen_avg_score"))
    token_f1_gap = _safe_float(baseline.get("token_f1_mean")) - _safe_float(candidate.get("token_f1_mean"))
    forgetting_gap = _safe_float(candidate.get("forgetting")) - _safe_float(baseline.get("forgetting"))
    return (
        seen_gap <= max_seen_gap + 1e-9
        and token_f1_gap <= max_token_f1_gap + 1e-9
        and forgetting_gap <= max_forgetting_gap + 1e-9
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the current best ours variant from completed paper runs.")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--matrix-csv", type=Path, default=Path("results/tables/paper_matrix_executions.csv"))
    parser.add_argument("--ours-candidates", type=str, default=",".join(DEFAULT_OURS_CANDIDATES))
    parser.add_argument("--baseline-candidates", type=str, default=",".join(DEFAULT_BASELINE_CANDIDATES))
    parser.add_argument("--secondary-benchmark", type=str, default="")
    parser.add_argument("--secondary-max-seen-gap", type=float, default=0.02)
    parser.add_argument("--secondary-max-token-f1-gap", type=float, default=0.02)
    parser.add_argument("--secondary-max-forgetting-gap", type=float, default=0.05)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = (repo_root / args.results_dir).resolve()
    matrix_csv = (repo_root / args.matrix_csv).resolve()
    matrix_rows = _load_matrix_rows(repo_root, matrix_csv)

    ours_rows = _load_candidate_rows(
        matrix_rows=matrix_rows,
        results_dir=results_dir,
        benchmark=str(args.benchmark),
        variant_ids=_parse_csv_list(args.ours_candidates),
    )
    baseline_rows = _load_candidate_rows(
        matrix_rows=matrix_rows,
        results_dir=results_dir,
        benchmark=str(args.benchmark),
        variant_ids=_parse_csv_list(args.baseline_candidates),
    )

    best_ours = max(ours_rows, key=_metric_key) if ours_rows else {}
    best_baseline = max(baseline_rows, key=_metric_key) if baseline_rows else {}
    primary_pass = bool(best_ours and best_baseline and _beats(best_ours, best_baseline))

    secondary_best_baseline: Dict[str, Any] = {}
    secondary_winner_match: Dict[str, Any] = {}
    secondary_pass = False
    secondary_rows: List[Dict[str, Any]] = []
    if str(args.secondary_benchmark).strip():
        secondary_rows = _load_candidate_rows(
            matrix_rows=matrix_rows,
            results_dir=results_dir,
            benchmark=str(args.secondary_benchmark),
            variant_ids=_parse_csv_list(args.ours_candidates),
        )
        secondary_baseline_rows = _load_candidate_rows(
            matrix_rows=matrix_rows,
            results_dir=results_dir,
            benchmark=str(args.secondary_benchmark),
            variant_ids=_parse_csv_list(args.baseline_candidates),
        )
        secondary_best_baseline = max(secondary_baseline_rows, key=_metric_key) if secondary_baseline_rows else {}
        matching_secondary_rows = _secondary_match_rows(secondary_rows, best_ours, str(args.secondary_benchmark))
        secondary_winner_match = max(matching_secondary_rows, key=_metric_key) if matching_secondary_rows else {}
        secondary_pass = _secondary_guard_pass(
            secondary_winner_match,
            secondary_best_baseline,
            max_seen_gap=float(args.secondary_max_seen_gap),
            max_token_f1_gap=float(args.secondary_max_token_f1_gap),
            max_forgetting_gap=float(args.secondary_max_forgetting_gap),
        )

    payload = {
        "benchmark": str(args.benchmark),
        "num_ours_candidates_completed": len(ours_rows),
        "num_baseline_candidates_completed": len(baseline_rows),
        "best_ours": best_ours,
        "best_baseline": best_baseline,
        "top_ours_variant_ids": _top_variant_ids(ours_rows),
        "winner_variant_id": str(best_ours.get("variant_id", "")),
        "winner_run_name": str(best_ours.get("resolved_run_name", "")),
        "winner_beats_best_baseline": primary_pass,
        "secondary_benchmark": str(args.secondary_benchmark),
        "secondary_best_baseline": secondary_best_baseline,
        "secondary_winner_match": secondary_winner_match,
        "winner_beats_secondary_baseline": bool(
            secondary_winner_match and secondary_best_baseline and _beats(secondary_winner_match, secondary_best_baseline)
        ),
        "winner_cross_benchmark_pass": bool(
            primary_pass and (secondary_pass if str(args.secondary_benchmark).strip() else True)
        ),
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
