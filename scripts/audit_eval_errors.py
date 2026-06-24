from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.evaluate import (  # noqa: E402
    _basic_answer_normalize,
    _extract_after_step,
    _score_task_aware,
    _token_f1,
    _truncate_prediction_for_scoring,
)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _iter_debug_examples(run_dir: Path) -> Iterable[Dict[str, Any]]:
    for debug_path in sorted((run_dir / "eval_debug").glob("eval_segment_*.json")):
        doc = _read_json(debug_path)
        segment_id = doc.get("segment_id")
        for idx, ex in enumerate(doc.get("examples", []) or []):
            if isinstance(ex, dict):
                row = dict(ex)
                row["debug_file"] = str(debug_path)
                row["debug_segment_id"] = segment_id
                row["debug_example_index"] = idx
                row["source_segment_id"] = ex.get("source_segment_id", segment_id)
                row["source_segment_name"] = ex.get("source_segment_name", "")
                row["source_example_idx"] = ex.get("source_example_idx", idx)
                yield row


def _looks_task_polluted(pred: str, instruction: str, input_text: str) -> bool:
    text = _basic_answer_normalize(pred)
    source = _basic_answer_normalize(f"{instruction} {input_text}")
    pollution_terms = [
        "calendar",
        "contact",
        "email",
        "meeting",
        "restaurant",
        "weather",
        "alarm",
        "reminder",
    ]
    hits = sum(1 for term in pollution_terms if term in text and term not in source)
    return hits >= 2


def _classify_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    pred = str(ex.get("raw_generated_output") or ex.get("raw_generated_text") or "")
    gold = str(ex.get("gold_output") or "")
    instruction = str(ex.get("instruction") or "")
    input_text = str(ex.get("input_text") or "")
    norm_pred = str(ex.get("normalized_prediction") or _basic_answer_normalize(pred))
    norm_gold = str(ex.get("normalized_gold") or _basic_answer_normalize(gold))
    strict_match = bool(ex.get("strict_match", ex.get("match", norm_pred == norm_gold)))
    token_f1 = _safe_float(ex.get("token_f1", _token_f1(norm_pred, norm_gold)))

    task_score = _score_task_aware(
        pred=pred,
        gold=gold,
        norm_pred=norm_pred,
        norm_gold=norm_gold,
        instruction=instruction,
        input_text=input_text,
        cfg={},
    )
    pred_scoring = str(task_score.get("prediction_for_scoring") or _basic_answer_normalize(_truncate_prediction_for_scoring(pred, {})))
    gold_basic = _basic_answer_normalize(gold)
    pred_basic = _basic_answer_normalize(pred)

    error_type = "strict_correct"
    if not strict_match:
        if bool(task_score.get("task_aware_match", False)):
            if str(task_score.get("task_score_type")) == "after_step_extracted_em":
                error_type = "extractable_after_step"
            elif str(task_score.get("task_score_type")) == "label_accuracy":
                error_type = "extractable_label"
            else:
                error_type = "format_or_boundary"
        elif gold_basic and pred_basic.startswith(gold_basic) and len(pred_basic) > len(gold_basic):
            error_type = "correct_prefix_overgenerated"
        elif pred_scoring == gold_basic:
            error_type = "first_line_or_sentence_correct"
        elif token_f1 >= 0.5:
            error_type = "partial_content_overlap"
        elif _extract_after_step(gold) is not None and _extract_after_step(pred) is not None:
            error_type = "wrong_extracted_step"
        elif _looks_task_polluted(pred, instruction, input_text):
            error_type = "task_pollution"
        elif not pred_basic:
            error_type = "empty_output"
        else:
            error_type = "unrelated_or_wrong"

    router_selected = str(ex.get("routing_selected_branch") or "")
    router_oracle = str(ex.get("routing_oracle_branch") or "")
    router_mismatch = bool(router_selected and router_oracle and router_selected != router_oracle)
    return {
        "error_type": error_type,
        "strict_match": strict_match,
        "task_aware_match": bool(task_score.get("task_aware_match", strict_match)),
        "task_score_type": str(task_score.get("task_score_type", "")),
        "token_f1": token_f1,
        "lcs_overlap": _safe_float(ex.get("lcs_overlap")),
        "router_mismatch": router_mismatch,
        "extracted_prediction": str(task_score.get("extracted_prediction", "")),
        "extracted_gold": str(task_score.get("extracted_gold", "")),
        "prediction_for_scoring": pred_scoring,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary_rows: List[Dict[str, Any]], sample_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Eval Error Audit", ""]
    if summary_rows:
        lines.extend(
            [
                "| run | debug_examples | strict_em | task_aware | top_error_type | router_mismatch_rate |",
                "|---|---:|---:|---:|---|---:|",
            ]
        )
        for row in summary_rows:
            lines.append(
                f"| `{row['run_name']}` | {int(row['num_debug_examples'])} | "
                f"{_safe_float(row['strict_em']):.4f} | {_safe_float(row['task_aware_score']):.4f} | "
                f"{row['top_error_type']} | {_safe_float(row['router_mismatch_rate']):.4f} |"
            )
    else:
        lines.append("_No eval_debug examples found. Run with `save_debug_examples_dir` enabled or inspect final metrics only._")

    if sample_rows:
        lines.extend(["", "## Representative Strict-EM Failures", ""])
        for row in sample_rows[:20]:
            lines.append(
                f"- `{row['run_name']}` seg={row.get('debug_segment_id', '')} "
                f"type=`{row['error_type']}` f1={_safe_float(row['token_f1']):.3f} "
                f"gold=`{row.get('gold_output', '')}` pred=`{row.get('raw_generated_output', '')[:140]}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit strict-EM failures in saved eval_debug examples.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-samples-per-run", type=int, default=50)
    args = parser.parse_args()

    results_dir = (REPO_ROOT / args.results_dir).resolve()
    runs_root = results_dir / "runs"
    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []

    for run_dir in sorted(p for p in runs_root.glob("*") if p.is_dir()):
        final_path = run_dir / "final_metrics.json"
        final_doc = _read_json(final_path) if final_path.is_file() else {}
        final_metrics = final_doc.get("final", {}) if isinstance(final_doc.get("final", {}), dict) else {}
        counters: Counter[str] = Counter()
        router_mismatch_count = 0
        strict_count = 0
        task_count = 0
        debug_count = 0

        for ex in _iter_debug_examples(run_dir):
            debug_count += 1
            audit = _classify_example(ex)
            counters[audit["error_type"]] += 1
            strict_count += int(bool(audit["strict_match"]))
            task_count += int(bool(audit["task_aware_match"]))
            router_mismatch_count += int(bool(audit["router_mismatch"]))
            row = {
                "run_name": run_dir.name,
                **audit,
                "debug_segment_id": ex.get("debug_segment_id", ""),
                "debug_example_index": ex.get("debug_example_index", ""),
                "source_segment_id": ex.get("source_segment_id", ""),
                "source_segment_name": ex.get("source_segment_name", ""),
                "source_example_idx": ex.get("source_example_idx", ""),
                "gold_output": ex.get("gold_output", ""),
                "raw_generated_output": ex.get("raw_generated_output", ex.get("raw_generated_text", "")),
            }
            detail_rows.append(row)
            if not bool(audit["strict_match"]) and len(sample_rows) < max(1, args.max_samples_per_run) * 20:
                sample_rows.append(row)

        if debug_count > 0 or final_metrics:
            top_error = counters.most_common(1)[0][0] if counters else ""
            summary_rows.append(
                {
                    "run_name": run_dir.name,
                    "num_debug_examples": debug_count,
                    "strict_em": strict_count / max(1, debug_count),
                    "task_aware_score": task_count / max(1, debug_count),
                    "top_error_type": top_error,
                    "router_mismatch_rate": router_mismatch_count / max(1, debug_count),
                    "final_seen_avg_score": _safe_float(final_metrics.get("eval.seen_avg_score")),
                    "final_seen_avg_task_aware_score": _safe_float(final_metrics.get("eval.seen_avg_task_aware_score")),
                    "final_token_f1_mean": _safe_float(final_metrics.get("eval.token_f1_mean")),
                    **{f"count.{k}": v for k, v in sorted(counters.items())},
                }
            )

    _write_csv(results_dir / "tables" / "eval_error_audit_summary.csv", summary_rows)
    if detail_rows:
        _write_csv(results_dir / "tables" / "eval_error_audit_examples.csv", detail_rows)
    _write_markdown(results_dir / "eval_error_audit.md", summary_rows, sample_rows)


if __name__ == "__main__":
    main()
