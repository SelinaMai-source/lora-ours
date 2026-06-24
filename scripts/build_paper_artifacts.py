from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LABELS = {
    "seq": "Sequential",
    "replay_b10": "Replay(10)",
    "replay_b50": "Replay(50)",
    "periodic_latest": "PeriodicLatest",
    "bank_no_router": "BankNoRouter",
    "router_only": "RouterOnly",
    "ours_full": "OursFull",
    "ours_no_drift": "OursNoDrift",
    "ours_no_router": "OursNoRouter",
    "ours_no_bank": "OursNoBank",
    "ours_no_overlap": "OursNoOverlap",
    "ours_M64": "OursM64",
    "ours_M128": "OursM128",
    "ours_M256": "OursM256",
    "ours_K50": "OursK50",
    "ours_K100": "OursK100",
    "ours_r8": "OursR8",
    "ours_r16": "OursR16",
    "ours_r32": "OursR32",
    "ours_br4": "OursBr4",
    "ours_br8": "OursBr8",
    "ours_beta001": "OursBeta001",
    "ours_beta003": "OursBeta003",
    "ours_beta006": "OursBeta006",
    "ours_no_meta_threshold": "OursNoMetaThreshold",
    "ours_reverse_curriculum": "OursReverseCurriculum",
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _method_label(variant_id: str) -> str:
    return LABELS.get(variant_id, variant_id)


def _default_matrix_csv(repo_root: Path) -> Path:
    executions = repo_root / "results" / "tables" / "paper_matrix_executions.csv"
    if executions.is_file():
        return executions
    return repo_root / "results" / "tables" / "paper_run_matrix.csv"


def _row_run_name(row: Dict[str, Any]) -> str:
    return str(row.get("actual_run_name") or row.get("run_name", ""))


def _entropy_from_utilization(util: Dict[str, Any]) -> float:
    ent = 0.0
    for p in util.values():
        p = _safe_float(p)
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    return float(ent)


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(sum(values) / len(values))
    if len(values) <= 1:
        return mean, 0.0
    var = float(sum((x - mean) ** 2 for x in values) / (len(values) - 1))
    return mean, float(math.sqrt(max(0.0, var)))


def _report_rank_key(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        _safe_float(row.get("eval.seen_avg_task_aware_score", row.get("eval.seen_avg_score"))),
        _safe_float(row.get("eval.token_f1_mean")),
        _safe_float(row.get("routing.oracle_agreement_rate")),
        -_safe_float(row.get("eval.forgetting")),
    )


def _effective_reporting_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    best_ours_by_benchmark: Dict[str, str] = {}
    for benchmark in sorted({str(r.get("benchmark", "")) for r in rows}):
        ours_rows = [r for r in rows if str(r.get("benchmark", "")) == benchmark and str(r.get("family", "")) == "ours"]
        if not ours_rows:
            continue
        declared_main = [
            r
            for r in ours_rows
            if str(r.get("category", "")) == "main" and str(r.get("variant_id", "")) == "ours_full"
        ]
        if declared_main:
            best_ours_by_benchmark[benchmark] = str(declared_main[0].get("resolved_run_name", ""))
        else:
            best_row = max(ours_rows, key=_report_rank_key)
            best_ours_by_benchmark[benchmark] = str(best_row.get("resolved_run_name", ""))

    effective: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        benchmark = str(new_row.get("benchmark", ""))
        is_ours = str(new_row.get("family", "")) == "ours"
        is_best_ours = str(new_row.get("resolved_run_name", "")) == best_ours_by_benchmark.get(benchmark, "")
        if is_ours and is_best_ours:
            new_row["category"] = "main"
            new_row["method_label"] = "Ours"
            new_row["method_variant"] = f"winner_{new_row.get('variant_id', 'ours')}"
        elif is_ours:
            if str(new_row.get("category", "")) == "main":
                new_row["category"] = "ablation"
        effective.append(new_row)
    return effective


def _aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    group_keys = [
        "benchmark",
        "category",
        "family",
        "variant_id",
        "method_variant",
        "method_label",
        "memory_budget",
    ]
    metric_keys = [
        "eval.current_score",
        "eval.seen_avg_score",
        "eval.current_task_aware_score",
        "eval.seen_avg_task_aware_score",
        "eval.forgetting",
        "eval.task_aware_forgetting",
        "eval.anytime_score",
        "eval.anytime_task_aware_score",
        "eval.task_aware_score_mean",
        "eval.token_f1_mean",
        "eval.lcs_overlap_mean",
        "routing.num_routed",
        "routing.oracle_agreement_rate",
        "routing.decision_confidence_mean",
        "routing.decision_entropy_mean",
        "routing.oracle_margin_mean",
        "routing.utilization_entropy",
        "overlap_mean_cosine",
        "drift.false_alarm_rate",
        "drift.miss_rate",
        "drift.detection_delay_mean",
    ]
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        merged = {k: v for k, v in zip(group_keys, key)}
        merged["num_runs"] = int(len(items))
        merged["seeds"] = ",".join(sorted({str(x.get("seed", "")) for x in items if str(x.get("seed", "")).strip()}))
        merged["resolved_run_names"] = ",".join(str(x.get("resolved_run_name", "")) for x in items)
        for metric in metric_keys:
            vals = [_safe_float(x.get(metric, 0.0)) for x in items]
            mean, std = _mean_std(vals)
            merged[f"{metric}.mean"] = mean
            merged[f"{metric}.std"] = std
        out.append(merged)
    return out


def _collect_run_rows(
    matrix_rows: List[Dict[str, str]],
    *,
    repo_root: Path,
    results_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    available: List[Dict[str, Any]] = []
    missing: List[Dict[str, str]] = []
    for row in matrix_rows:
        if _excluded_reason(row):
            continue
        run_name = _row_run_name(row)
        run_dir = results_dir / "runs" / run_name
        metrics_path = run_dir / "final_metrics.json"
        segment_table_path = results_dir / "tables" / f"{run_name}_segment_metrics.csv"
        if not metrics_path.is_file():
            if str(row.get("execution_status", "")).strip() != "running":
                missing.append(row)
            continue

        final_doc = _read_json(metrics_path)
        final = final_doc.get("final", {}) if isinstance(final_doc, dict) else {}
        drift_quality = final_doc.get("drift_quality", {}) if isinstance(final_doc.get("drift_quality", {}), dict) else {}
        routing_quality = final_doc.get("routing_quality", {}) if isinstance(final_doc.get("routing_quality", {}), dict) else {}
        current_score = _safe_float(final.get("eval.current_score"))
        seen_avg_score = _safe_float(final.get("eval.seen_avg_score"))
        forgetting = _safe_float(final.get("eval.forgetting"))
        row_out = {
            **row,
            "resolved_run_name": run_name,
            "reporting_status": "available",
            "method_label": _method_label(row["variant_id"]),
            "metrics_path": str(metrics_path.relative_to(repo_root)),
            "segment_table_path": str(segment_table_path.relative_to(repo_root)) if segment_table_path.is_file() else "",
            "has_segment_table": bool(segment_table_path.is_file()),
            "eval.current_score": current_score,
            "eval.seen_avg_score": seen_avg_score,
            "eval.current_task_aware_score": _safe_float(final.get("eval.current_task_aware_score", current_score)),
            "eval.seen_avg_task_aware_score": _safe_float(final.get("eval.seen_avg_task_aware_score", seen_avg_score)),
            "eval.forgetting": forgetting,
            "eval.task_aware_forgetting": _safe_float(final.get("eval.task_aware_forgetting", forgetting)),
            "eval.anytime_score": _safe_float(final.get("eval.anytime_score", seen_avg_score)),
            "eval.anytime_task_aware_score": _safe_float(
                final.get("eval.anytime_task_aware_score", final.get("eval.anytime_score", seen_avg_score))
            ),
            "eval.task_aware_score_mean": _safe_float(final.get("eval.task_aware_score_mean", seen_avg_score)),
            "eval.token_f1_mean": _safe_float(final.get("eval.token_f1_mean")),
            "eval.lcs_overlap_mean": _safe_float(final.get("eval.lcs_overlap_mean")),
            "train.router_num_labels": _safe_float(final.get("train.router_num_labels")),
            "routing.num_routed": _safe_float(routing_quality.get("num_routed")),
            "routing.oracle_agreement_rate": _safe_float(routing_quality.get("oracle_agreement_rate")),
            "routing.decision_confidence_mean": _safe_float(routing_quality.get("decision_confidence_mean")),
            "routing.decision_entropy_mean": _safe_float(routing_quality.get("decision_entropy_mean")),
            "routing.oracle_margin_mean": _safe_float(routing_quality.get("oracle_margin_mean")),
            "routing.utilization_entropy": _entropy_from_utilization(
                routing_quality.get("branch_utilization", {}) if isinstance(routing_quality, dict) else {}
            ),
            "overlap_mean_cosine": _safe_float(final.get("overlap_mean_cosine")),
            "drift.false_alarm_rate": _safe_float(drift_quality.get("false_alarm_rate")),
            "drift.miss_rate": _safe_float(drift_quality.get("miss_rate")),
            "drift.detection_delay_mean": _safe_float(drift_quality.get("detection_delay_mean")),
        }
        available.append(row_out)
    return available, missing


def _availability_rows(matrix_rows: List[Dict[str, str]], *, repo_root: Path, results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in matrix_rows:
        run_name = _row_run_name(row)
        metrics_path = results_dir / "runs" / run_name / "final_metrics.json"
        execution_status = str(row.get("execution_status", "")).strip()
        excluded_reason = _excluded_reason(row)
        has_final_metrics = metrics_path.is_file()
        if excluded_reason:
            reporting_status = "excluded"
        elif has_final_metrics:
            reporting_status = "available"
        elif execution_status == "running":
            reporting_status = "running"
        elif execution_status == "skipped_existing":
            reporting_status = "skipped_existing_missing_metrics"
        else:
            reporting_status = "missing"
        rows.append(
            {
                "benchmark": row.get("benchmark", ""),
                "category": row.get("category", ""),
                "family": row.get("family", ""),
                "variant_id": row.get("variant_id", ""),
                "seed": row.get("seed", ""),
                "run_name": run_name,
                "execution_status": execution_status,
                "has_final_metrics": has_final_metrics,
                "reporting_status": reporting_status,
                "excluded_reason": excluded_reason,
                "metrics_path": str(metrics_path.relative_to(repo_root)) if metrics_path.is_file() else "",
            }
        )
    return rows


def _consistency_warnings(*, available: List[Dict[str, Any]], availability: List[Dict[str, Any]]) -> List[str]:
    official_rows = [r for r in availability if not str(r.get("excluded_reason", "")).strip()]
    official_with_metrics = [r for r in official_rows if bool(r.get("has_final_metrics", False))]
    available_names = {str(r.get("resolved_run_name", "")) for r in available}
    missing_from_tables = [
        str(r.get("run_name", ""))
        for r in official_with_metrics
        if str(r.get("run_name", "")) not in available_names
    ]
    warnings: List[str] = []
    if official_with_metrics and len(available) < max(1, int(len(official_with_metrics) * 0.5)):
        warnings.append(
            "Artifact consistency warning: fewer than half of official runs with final_metrics entered tables."
        )
    if missing_from_tables:
        shown = ", ".join(missing_from_tables[:10])
        suffix = " ..." if len(missing_from_tables) > 10 else ""
        warnings.append(f"Official runs with final_metrics missing from tables: {shown}{suffix}")
    return warnings


def _is_pilot_or_override_row(row: Dict[str, str]) -> bool:
    """Paper artifacts should not promote smoke/mini override runs into main tables."""
    return bool(_excluded_reason(row))


def _excluded_reason(row: Dict[str, str]) -> str:
    """Return why a row is excluded from official tables, or empty string if official."""
    if str(row.get("run_name_suffix", "")).strip():
        return "run_name_suffix"
    generic_overrides = str(row.get("generic_overrides_json", "")).strip()
    if generic_overrides not in {"", "{}"}:
        try:
            overrides = json.loads(generic_overrides)
        except Exception:
            return "invalid_generic_overrides_json"
        if not isinstance(overrides, dict):
            return "generic_overrides_not_mapping"
        # Tracking-only overrides are part of official executions and must not
        # hide completed/skipped_existing runs from paper tables.
        non_tracking_keys = [str(k) for k in overrides.keys() if not str(k).startswith("output.tracking.")]
        if non_tracking_keys:
            return "semantic_generic_overrides:" + ",".join(sorted(non_tracking_keys))
    override_keys = [
        "max_segments_override",
        "max_train_examples_override",
        "max_eval_examples_override",
        "epochs_per_segment_override",
        "batch_size_override",
    ]
    active_override_keys = [k for k in override_keys if str(row.get(k, "")).strip()]
    if active_override_keys:
        return "semantic_overrides:" + ",".join(active_override_keys)
    return ""


def _plot_anytime(rows: List[Dict[str, Any]], *, repo_root: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_rows = [r for r in rows if r["category"] == "main"] or list(rows)
    if not plot_rows:
        return
    benchmarks = sorted({r["benchmark"] for r in plot_rows})
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(6 * len(benchmarks), 4.5), squeeze=False)
    for ax, benchmark in zip(axes[0], benchmarks):
        bench_rows = [r for r in plot_rows if r["benchmark"] == benchmark]
        for row in bench_rows:
            if not str(row.get("segment_table_path", "")).strip():
                continue
            seg_rows = _read_csv((repo_root / row["segment_table_path"]).resolve())
            xs = [_safe_float(x.get("segment_id")) for x in seg_rows]
            ys = [
                _safe_float(
                    x.get(
                        "eval.anytime_task_aware_score",
                        x.get("eval.anytime_score", x.get("eval.seen_avg_score", x.get("eval.current_score", 0.0))),
                    )
                )
                for x in seg_rows
            ]
            ax.plot(xs, ys, marker="o", label=row["method_label"])
        ax.set_title(f"{benchmark} Anytime")
        ax.set_xlabel("segment")
        ax.set_ylabel("task-aware score")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_forgetting(rows: List[Dict[str, Any]], *, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_rows = [r for r in rows if r["category"] == "main"] or list(rows)
    if not plot_rows:
        return
    benchmarks = sorted({r["benchmark"] for r in plot_rows})
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(6 * len(benchmarks), 4.5), squeeze=False)
    for ax, benchmark in zip(axes[0], benchmarks):
        bench_rows = [r for r in plot_rows if r["benchmark"] == benchmark]
        labels = [r["method_label"] for r in bench_rows]
        vals = [_safe_float(r["eval.forgetting"]) for r in bench_rows]
        ax.bar(range(len(labels)), vals)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"{benchmark} Forgetting")
        ax.set_ylabel("mean forgetting")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_budget(rows: List[Dict[str, Any]], *, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_rows = [r for r in rows if r["category"] == "main"] or list(rows)
    if not plot_rows:
        return
    benchmarks = sorted({r["benchmark"] for r in plot_rows})
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(6 * len(benchmarks), 4.5), squeeze=False)
    for ax, benchmark in zip(axes[0], benchmarks):
        bench_rows = sorted(
            [r for r in plot_rows if r["benchmark"] == benchmark],
            key=lambda x: (_safe_float(x["memory_budget"]), x["method_label"]),
        )
        xs = [_safe_float(r["memory_budget"]) for r in bench_rows]
        ys = [_safe_float(r.get("eval.seen_avg_task_aware_score", r["eval.seen_avg_score"])) for r in bench_rows]
        labels = [r["method_label"] for r in bench_rows]
        ax.scatter(xs, ys)
        for x, y, label in zip(xs, ys, labels):
            ax.text(x, y, label, fontsize=8)
        ax.set_title(f"{benchmark} Budget Sweep")
        ax.set_xlabel("memory budget")
        ax.set_ylabel("task-aware seen avg score")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_drift(rows: List[Dict[str, Any]], *, results_dir: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    target_rows = [r for r in rows if r.get("family") == "ours" and r.get("category") == "main"]
    if not target_rows:
        target_rows = [r for r in rows if r.get("family") == "ours"]
    if not target_rows:
        return
    benchmarks = sorted({r["benchmark"] for r in target_rows})
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(6 * len(benchmarks), 4.5), squeeze=False)
    for ax, benchmark in zip(axes[0], benchmarks):
        bench_rows = [r for r in target_rows if r["benchmark"] == benchmark]
        row = sorted(bench_rows, key=lambda x: _safe_float(x.get("eval.seen_avg_score", 0.0)), reverse=True)[0] if bench_rows else None
        if row is None:
            continue
        monitor_path = results_dir / "runs" / _row_run_name(row) / "anchor_monitor.jsonl"
        if not monitor_path.is_file():
            continue
        monitor_rows = _read_jsonl(monitor_path)
        xs = [_safe_float(r.get("monitor_step")) for r in monitor_rows]
        core = [_safe_float(r.get("core_mean_nll")) for r in monitor_rows]
        probe = [_safe_float(r.get("probe_mean_nll")) for r in monitor_rows]
        ax.plot(xs, core, marker="o", label="core_nll")
        ax.plot(xs, probe, marker="o", label="probe_nll")
        for r in monitor_rows:
            if bool(r.get("triggered", False)):
                ax.axvline(_safe_float(r.get("monitor_step")), linestyle="--", color="red", alpha=0.5)
        ax.set_title(f"{benchmark} Drift Monitor")
        ax.set_xlabel("monitor step")
        ax.set_ylabel("anchor NLL")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_router_behavior(rows: List[Dict[str, Any]], *, results_dir: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    target_rows = [r for r in rows if _safe_float(r["routing.oracle_agreement_rate"]) > 0]
    if not target_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [f"{r['benchmark']}:{r['method_label']}" for r in target_rows]
    agree = [_safe_float(r["routing.oracle_agreement_rate"]) for r in target_rows]
    entropy = [_safe_float(r["routing.decision_entropy_mean"]) for r in target_rows]
    xs = list(range(len(labels)))
    axes[0].bar(xs, agree)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Router Oracle Agreement")

    first_row = target_rows[0]
    final_doc = _read_json(results_dir / "runs" / _row_run_name(first_row) / "final_metrics.json")
    routing_quality = final_doc.get("routing_quality", {}) if isinstance(final_doc.get("routing_quality", {}), dict) else {}
    branch_counts = routing_quality.get("branch_counts", {}) if isinstance(routing_quality, dict) else {}
    b_labels = list(branch_counts.keys())
    b_vals = [int(branch_counts[k]) for k in b_labels]
    axes[1].bar(b_labels, b_vals)
    axes[1].set_title(f"Branch Utilization: {_row_run_name(first_row)}")
    axes[1].set_ylabel("count")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_pipeline(out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    ax.axis("off")
    stages = [
        ("Curriculum\nAnchors", 0.08),
        ("Meta CUSUM\nChange Point", 0.28),
        ("LoRA Bank\nAllocate/Merge", 0.48),
        ("Routed\nOnline Train", 0.68),
        ("Anti-Overlap\nSpecialize", 0.88),
    ]
    for label, x in stages:
        box = FancyBboxPatch((x - 0.09, 0.42), 0.18, 0.30, boxstyle="round,pad=0.02", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, 0.57, label, ha="center", va="center", fontsize=10.5, fontweight="bold")
    for i in range(len(stages) - 1):
        x0 = stages[i][1] + 0.09
        x1 = stages[i + 1][1] - 0.09
        ax.annotate("", xy=(x1, 0.57), xytext=(x0, 0.57), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.68, 0.40), xytext=(0.88, 0.40), arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.text(0.78, 0.31, "router weights and LoRA branches are regularized jointly", ha="center", fontsize=9)
    ax.text(0.08, 0.20, "easy core anchors\nhard probe anchors", ha="center", fontsize=9)
    ax.text(0.28, 0.20, "learn threshold from\nanchor volatility", ha="center", fontsize=9)
    ax.text(0.48, 0.20, "freeze old branches\nspawn or merge", ha="center", fontsize=9)
    ax.text(0.68, 0.20, "per-example branch\nassignment", ha="center", fontsize=9)
    ax.text(0.88, 0.20, "activation + weight\northogonality", ha="center", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_summary_markdown(
    *,
    available: List[Dict[str, Any]],
    aggregated: List[Dict[str, Any]],
    missing: List[Dict[str, str]],
    availability: List[Dict[str, Any]],
    warnings: List[str],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    status_counts: Dict[str, int] = {}
    for row in availability:
        status = str(row.get("reporting_status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    lines = [
        "# Paper Results Summary",
        "",
        f"- available runs: `{len(available)}`",
        f"- missing runs: `{len(missing)}`",
        f"- running runs: `{status_counts.get('running', 0)}`",
        f"- skipped_existing without metrics: `{status_counts.get('skipped_existing_missing_metrics', 0)}`",
        f"- excluded diagnostic runs: `{status_counts.get('excluded', 0)}`",
        "",
    ]
    if warnings:
        lines.extend(["## Consistency Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    lines.extend(["## Available Main Runs", ""])
    main_rows = [r for r in aggregated if r["category"] == "main"] or [r for r in available if r["category"] == "main"]
    if main_rows:
        lines.append("| benchmark | method | n | strict_seen_avg | task_aware_seen_avg | token_f1 | forgetting | oracle_agreement |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in main_rows:
            seen_avg = _safe_float(row.get("eval.seen_avg_score.mean", row.get("eval.seen_avg_score")))
            task_seen_avg = _safe_float(
                row.get("eval.seen_avg_task_aware_score.mean", row.get("eval.seen_avg_task_aware_score"))
            )
            token_f1 = _safe_float(row.get("eval.token_f1_mean.mean", row.get("eval.token_f1_mean")))
            forgetting = _safe_float(row.get("eval.forgetting.mean", row.get("eval.forgetting")))
            routed = _safe_float(row.get("routing.num_routed.mean", row.get("routing.num_routed")))
            oracle_agreement = (
                f"{_safe_float(row.get('routing.oracle_agreement_rate.mean', row.get('routing.oracle_agreement_rate'))):.4f}"
                if routed > 0
                else "N/A"
            )
            lines.append(
                f"| {row['benchmark']} | {row['method_label']} | "
                f"{int(row.get('num_runs', 1))} | "
                f"{seen_avg:.4f} | "
                f"{task_seen_avg:.4f} | "
                f"{token_f1:.4f} | "
                f"{forgetting:.4f} | "
                f"{oracle_agreement} |"
            )
    else:
        lines.append("_No main runs available yet._")
    lines.extend(["", "## Missing Runs", ""])
    if missing:
        for row in missing:
            lines.append(f"- `{_row_run_name(row)}` ({row['benchmark']} / {row['variant_id']})")
    else:
        lines.append("_No missing runs._")
    lines.extend(["", "## Run Availability", ""])
    if availability:
        lines.append("| status | count |")
        lines.append("|---|---:|")
        for status, count in sorted(status_counts.items()):
            lines.append(f"| {status} | {count} |")
    else:
        lines.append("_No matrix rows found._")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table1_markdown(*, aggregated_main: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Table 1 Main Comparison",
        "",
        "| benchmark | method | n | strict_seen_avg(mean±std) | task_aware_seen_avg(mean±std) | forgetting(mean±std) | token_f1(mean±std) | lcs(mean±std) | oracle_agreement(mean±std) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregated_main:
        seen_mean = _safe_float(row.get("eval.seen_avg_score.mean"))
        seen_std = _safe_float(row.get("eval.seen_avg_score.std"))
        task_seen_mean = _safe_float(row.get("eval.seen_avg_task_aware_score.mean"))
        task_seen_std = _safe_float(row.get("eval.seen_avg_task_aware_score.std"))
        forget_mean = _safe_float(row.get("eval.forgetting.mean"))
        forget_std = _safe_float(row.get("eval.forgetting.std"))
        token_mean = _safe_float(row.get("eval.token_f1_mean.mean"))
        token_std = _safe_float(row.get("eval.token_f1_mean.std"))
        lcs_mean = _safe_float(row.get("eval.lcs_overlap_mean.mean"))
        lcs_std = _safe_float(row.get("eval.lcs_overlap_mean.std"))
        routed = _safe_float(row.get("routing.num_routed.mean"))
        oracle_mean = _safe_float(row.get("routing.oracle_agreement_rate.mean"))
        oracle_std = _safe_float(row.get("routing.oracle_agreement_rate.std"))
        oracle_text = f"{oracle_mean:.4f}±{oracle_std:.4f}" if routed > 0 else "N/A"
        lines.append(
            f"| {row['benchmark']} | {row['method_label']} | {int(row.get('num_runs', 1))} | "
            f"{seen_mean:.4f}±{seen_std:.4f} | "
            f"{task_seen_mean:.4f}±{task_seen_std:.4f} | "
            f"{forget_mean:.4f}±{forget_std:.4f} | "
            f"{token_mean:.4f}±{token_std:.4f} | "
            f"{lcs_mean:.4f}±{lcs_std:.4f} | "
            f"{oracle_text} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paper-matrix runs into tables, figures, and summary docs.")
    parser.add_argument(
        "--matrix-csv",
        type=Path,
        default=None,
        help="Run manifest generated by build_paper_run_matrix.py",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Project results directory",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = (repo_root / args.results_dir).resolve()
    matrix_csv = (repo_root / args.matrix_csv).resolve() if args.matrix_csv is not None else _default_matrix_csv(repo_root)
    matrix_rows = _read_csv(matrix_csv)
    available, missing = _collect_run_rows(matrix_rows, repo_root=repo_root, results_dir=results_dir)
    availability = _availability_rows(matrix_rows, repo_root=repo_root, results_dir=results_dir)
    if not available:
        fallback_csv = repo_root / "results" / "tables" / "paper_matrix_executions.csv"
        if fallback_csv.is_file() and fallback_csv.resolve() != matrix_csv.resolve():
            matrix_csv = fallback_csv.resolve()
            matrix_rows = _read_csv(matrix_csv)
            available, missing = _collect_run_rows(matrix_rows, repo_root=repo_root, results_dir=results_dir)
            availability = _availability_rows(matrix_rows, repo_root=repo_root, results_dir=results_dir)
    if not available:
        _write_summary_markdown(
            available=[],
            aggregated=[],
            missing=matrix_rows,
            availability=availability,
            warnings=_consistency_warnings(available=[], availability=availability),
            out_path=results_dir / "paper_results_summary.md",
        )
        if availability:
            _write_csv(results_dir / "tables" / "paper_run_availability.csv", availability)
        return

    reporting_available = _effective_reporting_rows(available)
    main_rows = [r for r in reporting_available if r["category"] == "main"]
    ablation_rows = [r for r in reporting_available if r["category"] == "ablation"]
    sweep_rows = [r for r in reporting_available if r["category"] == "sweep"]
    router_rows = [r for r in reporting_available if _safe_float(r["routing.oracle_agreement_rate"]) > 0]
    aggregated_available = _aggregate_rows(reporting_available)
    aggregated_main = _aggregate_rows(main_rows)
    aggregated_ablation = _aggregate_rows(ablation_rows)
    aggregated_sweep = _aggregate_rows(sweep_rows)

    _write_csv(results_dir / "tables" / "paper_main_results.csv", main_rows or reporting_available)
    _write_csv(results_dir / "tables" / "paper_ablation_results.csv", ablation_rows or reporting_available)
    _write_csv(results_dir / "tables" / "results_budget_sweep.csv", sweep_rows or reporting_available)
    _write_csv(results_dir / "tables" / "router_metrics.csv", router_rows or reporting_available)
    _write_csv(results_dir / "tables" / "paper_run_availability.csv", availability)
    _write_csv(results_dir / "tables" / "paper_main_results_agg.csv", aggregated_main or aggregated_available)
    _write_csv(results_dir / "tables" / "paper_ablation_results_agg.csv", aggregated_ablation or aggregated_available)
    _write_csv(results_dir / "tables" / "results_budget_sweep_agg.csv", aggregated_sweep or aggregated_available)

    _plot_anytime(reporting_available, repo_root=repo_root, out_path=results_dir / "figures" / "fig1_anytime_performance.png")
    _plot_forgetting(reporting_available, out_path=results_dir / "figures" / "fig2_forgetting_summary.png")
    _plot_budget(reporting_available, out_path=results_dir / "figures" / "fig3_budget_vs_performance.png")
    _plot_drift(reporting_available, results_dir=results_dir, out_path=results_dir / "figures" / "fig4_drift_diagnostics.png")
    _plot_router_behavior(
        reporting_available,
        results_dir=results_dir,
        out_path=results_dir / "figures" / "fig5_router_behavior.png",
    )
    _plot_pipeline(results_dir / "figures" / "fig6_pipeline.png")
    _write_summary_markdown(
        available=reporting_available,
        aggregated=aggregated_available,
        missing=missing,
        availability=availability,
        warnings=_consistency_warnings(available=reporting_available, availability=availability),
        out_path=results_dir / "paper_results_summary.md",
    )
    _write_table1_markdown(
        aggregated_main=aggregated_main or aggregated_available,
        out_path=results_dir / "tables" / "table1_main_comparison.md",
    )


if __name__ == "__main__":
    main()
