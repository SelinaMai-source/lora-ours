"""CITB continual-learning metrics.

The official ``collect_results.py`` computes:
- AR / average accuracy: mean of the final row over all stream tasks
- FWT: mean next-task score ``a_{i,i+1}`` for ``i < T``
- BWT: mean ``a_{T,i} - a_{i,i}`` for previous tasks

This module also exposes FR (forgetting rate) as the standard CL complement:
for each previous task, the best historical score minus the final score, then
averaged. FR is not written by the official script, so preflight records it as
implemented locally from the same task-score matrix instead of claiming an
official output field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


Score = float
ScoreMatrix = Sequence[Sequence[Optional[Score]]]


@dataclass(frozen=True)
class CITBMetricSummary:
    average_accuracy: float
    average_fwt: Optional[float]
    average_bwt: Optional[float]
    forgetting_rate: Optional[float]
    task_count: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mean(values: Iterable[float]) -> Optional[float]:
    clean = [float(value) for value in values]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _require_square_lower_task_matrix(results_matrix: ScoreMatrix) -> int:
    if not results_matrix:
        raise ValueError("results_matrix must not be empty")
    task_count = len(results_matrix)
    for row in results_matrix:
        if len(row) < task_count:
            raise ValueError("each results row must contain at least one column per task")
    return task_count


def compute_average_accuracy(results_matrix: ScoreMatrix) -> float:
    task_count = _require_square_lower_task_matrix(results_matrix)
    final_scores = results_matrix[-1][:task_count]
    if any(score is None for score in final_scores):
        raise ValueError("final row must contain all task scores for average accuracy")
    return float(_mean(score for score in final_scores if score is not None))


def compute_fwt(results_matrix: ScoreMatrix) -> Optional[float]:
    task_count = _require_square_lower_task_matrix(results_matrix)
    return _mean(
        float(row[task_idx + 1])
        for task_idx, row in enumerate(results_matrix)
        if task_idx < task_count - 1 and row[task_idx + 1] is not None
    )


def compute_bwt(results_matrix: ScoreMatrix) -> Optional[float]:
    task_count = _require_square_lower_task_matrix(results_matrix)
    if task_count <= 1:
        return None
    final_row = results_matrix[-1]
    return _mean(
        float(final_row[task_idx]) - float(results_matrix[task_idx][task_idx])
        for task_idx in range(task_count - 1)
        if final_row[task_idx] is not None and results_matrix[task_idx][task_idx] is not None
    )


def compute_forgetting_rate(results_matrix: ScoreMatrix) -> Optional[float]:
    task_count = _require_square_lower_task_matrix(results_matrix)
    if task_count <= 1:
        return None
    final_row = results_matrix[-1]
    forgetting_values: List[float] = []
    for task_idx in range(task_count - 1):
        final_score = final_row[task_idx]
        historical_scores = [
            row[task_idx]
            for row in results_matrix[task_idx:]
            if len(row) > task_idx and row[task_idx] is not None
        ]
        if final_score is None or not historical_scores:
            continue
        forgetting_values.append(max(float(score) for score in historical_scores) - float(final_score))
    return _mean(forgetting_values)


def compute_citb_metric_summary(results_matrix: ScoreMatrix) -> CITBMetricSummary:
    task_count = _require_square_lower_task_matrix(results_matrix)
    return CITBMetricSummary(
        average_accuracy=compute_average_accuracy(results_matrix),
        average_fwt=compute_fwt(results_matrix),
        average_bwt=compute_bwt(results_matrix),
        forgetting_rate=compute_forgetting_rate(results_matrix),
        task_count=task_count,
    )


def matrix_from_official_metric_jsons(metric_jsons: Sequence[Mapping[str, Any]], metric: str = "rougeL") -> List[List[Optional[float]]]:
    """Build a task-score matrix from official per-task ``metrics.json`` files.

    The expected keys match ``collect_results.py``:
    ``predict_ii_<metric>``, ``predict_seen_ij_<metric>``, and
    ``predict_next_i(i+1)_<metric>``.
    """

    task_count = len(metric_jsons)
    matrix: List[List[Optional[float]]] = [[None for _ in range(task_count)] for _ in range(task_count)]
    for row_idx, metrics in enumerate(metric_jsons):
        current_order = row_idx + 1
        current_key = f"predict_{current_order}{current_order}_{metric}"
        if current_key in metrics:
            matrix[row_idx][row_idx] = float(metrics[current_key])
        for previous_order in range(1, current_order):
            seen_key = f"predict_seen_{current_order}{previous_order}_{metric}"
            if seen_key in metrics:
                matrix[row_idx][previous_order - 1] = float(metrics[seen_key])
        next_order = current_order + 1
        next_key = f"predict_next_{current_order}{next_order}_{metric}"
        if row_idx < task_count - 1 and next_key in metrics:
            matrix[row_idx][row_idx + 1] = float(metrics[next_key])
    return matrix

