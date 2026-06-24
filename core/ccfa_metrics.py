from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.data import ContinualStream


def matrix_columns_from_eval(eval_metrics: Dict[str, Any]) -> Dict[str, Any]:
    extra = eval_metrics.get("extra", {}) if isinstance(eval_metrics.get("extra"), dict) else {}
    out: Dict[str, Any] = {}
    for item in extra.get("per_segment_accuracy", []) or []:
        sid = int(item.get("segment_id"))
        out[f"eval.matrix.segment_{sid:03d}.accuracy"] = float(item.get("accuracy", 0.0))
    for item in extra.get("per_segment_task_aware_accuracy", []) or []:
        sid = int(item.get("segment_id"))
        out[f"eval.matrix.segment_{sid:03d}.task_aware_accuracy"] = float(item.get("task_aware_accuracy", 0.0))
    for item in extra.get("forgetting_by_segment", []) or []:
        sid = int(item.get("segment_id"))
        out[f"eval.matrix.segment_{sid:03d}.forgetting"] = float(item.get("forgetting", 0.0))
    for item in extra.get("task_aware_forgetting_by_segment", []) or []:
        sid = int(item.get("segment_id"))
        out[f"eval.matrix.segment_{sid:03d}.task_aware_forgetting"] = float(
            item.get("task_aware_forgetting", 0.0)
        )
    return out


def write_ccfa_postprocess_outputs(
    *,
    run_dir: str,
    cfg: Dict[str, Any],
    stream: ContinualStream,
    segment_metrics_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    out_dir = Path(run_dir) / "ccfa_postprocess"
    out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = [int(seg.segment_id) for seg in stream.stream]
    task_names = {int(seg.segment_id): str(seg.segment_name) for seg in stream.stream}

    score_matrix: List[List[Optional[float]]] = []
    task_aware_matrix: List[List[Optional[float]]] = []
    per_task_time_rows: List[Dict[str, Any]] = []
    for time_idx, row in enumerate(segment_metrics_rows):
        acc_row: List[Optional[float]] = []
        task_row: List[Optional[float]] = []
        trained_segment_id = int(row.get("segment_id", time_idx))
        for sid in task_ids:
            acc = _optional_float(row.get(f"eval.matrix.segment_{sid:03d}.accuracy"))
            task_acc = _optional_float(row.get(f"eval.matrix.segment_{sid:03d}.task_aware_accuracy"))
            acc_row.append(acc)
            task_row.append(task_acc)
            per_task_time_rows.append(
                {
                    "time_index": int(time_idx),
                    "trained_segment_id": int(trained_segment_id),
                    "eval_segment_id": int(sid),
                    "eval_segment_name": task_names.get(sid, ""),
                    "accuracy": "" if acc is None else acc,
                    "task_aware_accuracy": "" if task_acc is None else task_acc,
                    "forgetting": _csv_value(row.get(f"eval.matrix.segment_{sid:03d}.forgetting")),
                    "task_aware_forgetting": _csv_value(
                        row.get(f"eval.matrix.segment_{sid:03d}.task_aware_forgetting")
                    ),
                }
            )
        score_matrix.append(acc_row)
        task_aware_matrix.append(task_row)

    final_scores = [x for x in (score_matrix[-1] if score_matrix else []) if x is not None]
    final_task_scores = [x for x in (task_aware_matrix[-1] if task_aware_matrix else []) if x is not None]
    final_average_accuracy = _mean(final_scores)
    final_average_task_aware_accuracy = _mean(final_task_scores)
    average_forgetting = _optional_float(
        segment_metrics_rows[-1].get("eval.forgetting") if segment_metrics_rows else None
    )
    task_aware_forgetting = _optional_float(
        segment_metrics_rows[-1].get("eval.task_aware_forgetting") if segment_metrics_rows else None
    )
    matrix_fwt = _matrix_fwt(task_aware_matrix if final_task_scores else score_matrix)
    matrix_bwt = _matrix_bwt(task_aware_matrix if final_task_scores else score_matrix)
    arper_bleu4 = _optional_float(
        (
            segment_metrics_rows[-1].get("eval.arper_woz3_corpus_bleu4")
            if segment_metrics_rows[-1].get("eval.arper_woz3_corpus_bleu4") is not None
            else segment_metrics_rows[-1].get("eval.corpus_bleu4")
        )
        if segment_metrics_rows
        else None
    )
    arper_ser = _optional_float(
        (
            segment_metrics_rows[-1].get("eval.arper_woz3_ser_percent")
            if segment_metrics_rows[-1].get("eval.arper_woz3_ser_percent") is not None
            else segment_metrics_rows[-1].get("eval.slot_error_rate")
        )
        if segment_metrics_rows
        else None
    )
    summary = {
        "suite": (cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}).get("suite", ""),
        "benchmark": stream.benchmark,
        "version": stream.version,
        "num_tasks": int(len(task_ids)),
        "task_ids": task_ids,
        "task_names": [task_names[sid] for sid in task_ids],
        "score_matrix": score_matrix,
        "task_aware_score_matrix": task_aware_matrix,
        "final_average_accuracy": final_average_accuracy,
        "final_average_task_aware_accuracy": final_average_task_aware_accuracy,
        "average_forgetting": average_forgetting,
        "bwt": matrix_bwt,
        "official_metrics": {
            "CITB": {
                "ROUGE-L AR": _mean(final_task_scores or final_scores),
                "FWT": matrix_fwt,
                "BWT": matrix_bwt,
                "Tinit": None,
                "Tunseen": None,
            },
            "Standard PEFT CL": {
                "final average accuracy": final_average_accuracy,
                "forgetting": average_forgetting,
                "BWT": matrix_bwt,
            },
            "Dialogue NLG": {
                "BLEU-4": arper_bleu4,
                "SER": arper_ser,
                "forgetting": average_forgetting,
            },
        },
        "citb": {
            "ar": _mean(final_task_scores or final_scores),
            "fwt": matrix_fwt,
            "bwt": matrix_bwt if matrix_bwt is not None else -_optional_float(task_aware_forgetting, default=0.0),
            "tinit": None,
            "tunseen": None,
            "note": "AR/BWT use the exported ROUGE-L score matrix. FWT/Tinit/Tunseen remain null unless the run includes official pre-training or unseen-task probes.",
        },
        "dialogue_nlg": {
            "bleu4": arper_bleu4,
            "ser": arper_ser,
            "missing_slots": _optional_float(
                (
                    segment_metrics_rows[-1].get("eval.arper_woz3_miss")
                    if segment_metrics_rows[-1].get("eval.arper_woz3_miss") is not None
                    else segment_metrics_rows[-1].get("eval.slot_missing_count")
                )
                if segment_metrics_rows
                else None
            ),
            "required_slots": _optional_float(
                (
                    segment_metrics_rows[-1].get("eval.arper_woz3_total")
                    if segment_metrics_rows[-1].get("eval.arper_woz3_total") is not None
                    else segment_metrics_rows[-1].get("eval.slot_required_count")
                )
                if segment_metrics_rows
                else None
            ),
            "redundant_slots": _optional_float(
                segment_metrics_rows[-1].get("eval.arper_woz3_redunt") if segment_metrics_rows else None
            ),
        },
    }

    summary_path = out_dir / "summary.json"
    matrix_path = out_dir / "score_matrix.json"
    csv_path = out_dir / "per_task_time_metrics.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    matrix_path.write_text(
        json.dumps(
            {
                "task_ids": task_ids,
                "task_names": [task_names[sid] for sid in task_ids],
                "score_matrix": score_matrix,
                "task_aware_score_matrix": task_aware_matrix,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(csv_path, per_task_time_rows)

    return {
        "ccfa_postprocess_dir": str(out_dir),
        "ccfa_summary_json": str(summary_path),
        "ccfa_score_matrix_json": str(matrix_path),
        "ccfa_per_task_time_metrics_csv": str(csv_path),
        "ccfa_summary": summary,
    }


def _optional_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _csv_value(value: Any) -> Any:
    parsed = _optional_float(value)
    return "" if parsed is None else parsed


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _matrix_fwt(matrix: List[List[Optional[float]]]) -> Optional[float]:
    vals: List[float] = []
    for idx, row in enumerate(matrix[:-1]):
        if idx + 1 >= len(row) or row[idx + 1] is None:
            return None
        vals.append(float(row[idx + 1]))
    return _mean(vals)


def _matrix_bwt(matrix: List[List[Optional[float]]]) -> Optional[float]:
    if len(matrix) <= 1:
        return 0.0 if matrix else None
    vals: List[float] = []
    final_row = matrix[-1]
    for idx, row in enumerate(matrix[:-1]):
        if idx >= len(row) or idx >= len(final_row) or row[idx] is None or final_row[idx] is None:
            return None
        vals.append(float(final_row[idx]) - float(row[idx]))
    return _mean(vals)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "time_index",
        "trained_segment_id",
        "eval_segment_id",
        "eval_segment_name",
        "accuracy",
        "task_aware_accuracy",
        "forgetting",
        "task_aware_forgetting",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
