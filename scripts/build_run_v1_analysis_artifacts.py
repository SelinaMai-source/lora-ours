from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _lookup_row(rows: Sequence[Dict[str, str]], *, benchmark: str, method_label: str) -> Dict[str, str]:
    for row in rows:
        if str(row.get("benchmark")) == benchmark and str(row.get("method_label")) == method_label:
            return dict(row)
    return {}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _annotate_bars(ax: plt.Axes, x: Sequence[float], values: Sequence[float]) -> None:
    ymax = max(values) if values else 0.0
    pad = max(0.003, ymax * 0.03)
    for xi, value in zip(x, values):
        ax.text(xi, value + pad, f"{value:.3f}", ha="center", va="bottom", fontsize=8)


def _plot_method_block_drift(main_rows: Sequence[Dict[str, str]], ablation_rows: Sequence[Dict[str, str]], out_path: Path) -> None:
    ours = _lookup_row(main_rows, benchmark="instrdialog", method_label="Ours")
    no_drift = _lookup_row(ablation_rows, benchmark="instrdialog", method_label="OursNoDrift")
    bank_no_router = _lookup_row(main_rows, benchmark="instrdialog", method_label="BankNoRouter")
    ours_full = _lookup_row(ablation_rows, benchmark="instrdialog", method_label="OursFull")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    labels_a = ["Ours", "OursNoDrift"]
    x_a = np.arange(len(labels_a))
    seen_vals = [_safe_float(ours.get("eval.seen_avg_score")), _safe_float(no_drift.get("eval.seen_avg_score"))]
    forget_vals = [_safe_float(ours.get("eval.forgetting")), _safe_float(no_drift.get("eval.forgetting"))]
    width = 0.36
    axes[0].bar(x_a - width / 2, seen_vals, width=width, label="Seen Avg", color="#4c72b0")
    axes[0].bar(x_a + width / 2, forget_vals, width=width, label="Forgetting", color="#dd8452")
    axes[0].set_title("Detector on vs off")
    axes[0].set_xticks(x_a)
    axes[0].set_xticklabels(labels_a)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    _annotate_bars(axes[0], x_a - width / 2, seen_vals)
    _annotate_bars(axes[0], x_a + width / 2, forget_vals)

    labels_b = ["Ours", "BankNoRouter", "OursFull"]
    x_b = np.arange(len(labels_b))
    miss_vals = [
        _safe_float(ours.get("drift.miss_rate")),
        _safe_float(bank_no_router.get("drift.miss_rate")),
        _safe_float(ours_full.get("drift.miss_rate")),
    ]
    delay_vals = [
        _safe_float(ours.get("drift.detection_delay_mean")),
        _safe_float(bank_no_router.get("drift.detection_delay_mean")),
        _safe_float(ours_full.get("drift.detection_delay_mean")),
    ]
    axes[1].bar(x_b - width / 2, miss_vals, width=width, label="Miss Rate", color="#c44e52")
    axes[1].bar(x_b + width / 2, delay_vals, width=width, label="Delay", color="#55a868")
    axes[1].set_title("Current drift quality is still weak")
    axes[1].set_xticks(x_b)
    axes[1].set_xticklabels(labels_b)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    _annotate_bars(axes[1], x_b - width / 2, miss_vals)
    _annotate_bars(axes[1], x_b + width / 2, delay_vals)

    fig.suptitle("Method Block: Drift Detection", fontsize=14)
    _save(fig, out_path)


def _plot_method_block_bank(main_rows: Sequence[Dict[str, str]], ablation_rows: Sequence[Dict[str, str]], out_path: Path) -> None:
    method_order = ["Sequential", "PeriodicLatest", "BankNoRouter", "Replay(50)", "Ours"]
    rows = [_lookup_row(main_rows, benchmark="instrdialog", method_label=m) for m in method_order]
    no_bank = _lookup_row(ablation_rows, benchmark="instrdialog", method_label="OursNoBank")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    labels = ["Sequential", "Periodic", "BankNoRouter", "Replay50", "Ours"]
    x = np.arange(len(labels))
    seen_vals = [_safe_float(row.get("eval.seen_avg_score")) for row in rows]
    forget_vals = [_safe_float(row.get("eval.forgetting")) for row in rows]

    axes[0].bar(x, seen_vals, color=["#55a868", "#c44e52", "#64b5cd", "#8172b2", "#dd8452"])
    axes[0].set_title("Seen Avg comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    _annotate_bars(axes[0], x, seen_vals)

    labels_b = ["Ours", "OursNoBank", "Replay50", "Sequential"]
    x_b = np.arange(len(labels_b))
    forget_vals_b = [
        _safe_float(rows[-1].get("eval.forgetting")),
        _safe_float(no_bank.get("eval.forgetting")),
        _safe_float(rows[3].get("eval.forgetting")),
        _safe_float(rows[0].get("eval.forgetting")),
    ]
    seen_vals_b = [
        _safe_float(rows[-1].get("eval.seen_avg_score")),
        _safe_float(no_bank.get("eval.seen_avg_score")),
        _safe_float(rows[3].get("eval.seen_avg_score")),
        _safe_float(rows[0].get("eval.seen_avg_score")),
    ]
    width = 0.36
    axes[1].bar(x_b - width / 2, seen_vals_b, width=width, label="Seen Avg", color="#4c72b0")
    axes[1].bar(x_b + width / 2, forget_vals_b, width=width, label="Forgetting", color="#dd8452")
    axes[1].set_title("Bank helps seen_avg, not yet forgetting")
    axes[1].set_xticks(x_b)
    axes[1].set_xticklabels(labels_b, rotation=15, ha="right")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    _annotate_bars(axes[1], x_b - width / 2, seen_vals_b)
    _annotate_bars(axes[1], x_b + width / 2, forget_vals_b)

    fig.suptitle("Method Block: LoRA Bank / Capacity Allocation", fontsize=14)
    _save(fig, out_path)


def _plot_method_block_router(main_rows: Sequence[Dict[str, str]], ablation_rows: Sequence[Dict[str, str]], out_path: Path) -> None:
    router_only = _lookup_row(main_rows, benchmark="instrdialog", method_label="RouterOnly")
    ours = _lookup_row(main_rows, benchmark="instrdialog", method_label="Ours")
    no_router = _lookup_row(ablation_rows, benchmark="instrdialog", method_label="OursNoRouter")
    no_drift = _lookup_row(ablation_rows, benchmark="instrdialog", method_label="OursNoDrift")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    points = [
        ("RouterOnly", router_only, "#8172b2"),
        ("Ours(main)", ours, "#dd8452"),
        ("OursNoDrift", no_drift, "#55a868"),
        ("OursNoRouter", no_router, "#4c72b0"),
    ]
    for label, row, color in points:
        x = _safe_float(row.get("routing.oracle_agreement_rate"))
        y = _safe_float(row.get("eval.seen_avg_score"))
        axes[0].scatter(x, y, s=150, color=color, alpha=0.9)
        axes[0].annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    axes[0].set_title("Oracle agreement does not map cleanly to utility")
    axes[0].set_xlabel("Oracle Agreement")
    axes[0].set_ylabel("Seen Avg")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    labels_b = ["Ours", "OursNoRouter"]
    x_b = np.arange(len(labels_b))
    seen_vals = [_safe_float(ours.get("eval.seen_avg_score")), _safe_float(no_router.get("eval.seen_avg_score"))]
    forget_vals = [_safe_float(ours.get("eval.forgetting")), _safe_float(no_router.get("eval.forgetting"))]
    width = 0.36
    axes[1].bar(x_b - width / 2, seen_vals, width=width, label="Seen Avg", color="#4c72b0")
    axes[1].bar(x_b + width / 2, forget_vals, width=width, label="Forgetting", color="#dd8452")
    axes[1].set_title("Router still helps inside Ours")
    axes[1].set_xticks(x_b)
    axes[1].set_xticklabels(labels_b)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    _annotate_bars(axes[1], x_b - width / 2, seen_vals)
    _annotate_bars(axes[1], x_b + width / 2, forget_vals)

    fig.suptitle("Method Block: Router / Task-Free Inference", fontsize=14)
    _save(fig, out_path)


def _plot_method_block_overlap(main_rows: Sequence[Dict[str, str]], ablation_rows: Sequence[Dict[str, str]], out_path: Path) -> None:
    ours = _lookup_row(main_rows, benchmark="instrdialog", method_label="Ours")
    no_overlap = _lookup_row(ablation_rows, benchmark="instrdialog", method_label="OursNoOverlap")
    metric_specs = [
        ("eval.seen_avg_score", "Seen Avg"),
        ("eval.forgetting", "Forgetting"),
        ("eval.token_f1_mean", "Token F1"),
        ("routing.oracle_agreement_rate", "Oracle Agr."),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.2))
    for ax, (metric_key, title) in zip(axes, metric_specs):
        labels = ["OursFull(main)", "OursNoOverlap"]
        values = [_safe_float(ours.get(metric_key)), _safe_float(no_overlap.get(metric_key))]
        x = np.arange(len(labels))
        ax.bar(x, values, color=["#55a868", "#dd8452"])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(["OursFull", "NoOverlap"], rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        _annotate_bars(ax, x, values)
    fig.suptitle("Method Block: Anti-Overlap Regularization", fontsize=14)
    _save(fig, out_path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    out_dir = results_dir / "analysis_figures"

    main_rows = _read_csv(results_dir / "tables" / "paper_main_results.csv")
    ablation_rows = _read_csv(results_dir / "tables" / "paper_ablation_results.csv")

    _plot_method_block_drift(main_rows, ablation_rows, out_dir / "method_block_drift.png")
    _plot_method_block_bank(main_rows, ablation_rows, out_dir / "method_block_bank.png")
    _plot_method_block_router(main_rows, ablation_rows, out_dir / "method_block_router.png")
    _plot_method_block_overlap(main_rows, ablation_rows, out_dir / "method_block_overlap.png")

    print(f"[run_v1_analysis] wrote method-block charts to {out_dir}")


if __name__ == "__main__":
    main()
