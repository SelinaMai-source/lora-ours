from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

BASELINE_RUN_SPECS = [
    {"run_name": "smoke_bosfix_sequential_lora", "label": "seq_smoke", "regime": "smoke"},
    {"run_name": "smoke_bosfix_replay_lora", "label": "replay_smoke", "regime": "smoke"},
    {"run_name": "smoke_bosfix_periodic_multilora", "label": "periodic_smoke", "regime": "smoke"},
    {"run_name": "smoke_bosfix_router_only", "label": "router_smoke", "regime": "smoke"},
    {"run_name": "smoke_bosfix_bank_no_router", "label": "bank_no_router_smoke", "regime": "smoke"},
    {"run_name": "baseline_recovery_mini_seq", "label": "seq_recovery_mini", "regime": "recovery"},
]

OURS_RUN_SPECS = [
    {"run_name": "smoke_bosfix_ours_full", "variant": "ours_full"},
    {"run_name": "smoke_bosfix_ours_no_drift", "variant": "ours_no_drift"},
    {"run_name": "smoke_bosfix_ours_no_bank", "variant": "ours_no_bank"},
    {"run_name": "smoke_bosfix_ours_no_router", "variant": "ours_no_router"},
    {"run_name": "smoke_bosfix_ours_no_overlap", "variant": "ours_no_overlap"},
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_last_csv_row(path: Path) -> Dict[str, str]:
    rows = _read_csv(path)
    if not rows:
        return {}
    return rows[-1]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_json_if_exists(path: Path) -> Any:
    if not path.is_file():
        return {}
    return _read_json(path)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _plot_line_compare(old_rows: List[Dict[str, str]], new_rows: List[Dict[str, str]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    steps_old = [_safe_int(r["step"]) for r in old_rows]
    steps_new = [_safe_int(r["step"]) for r in new_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(steps_old, [_safe_int(r["exact_match_count"]) for r in old_rows], label="old")
    axes[0].plot(steps_new, [_safe_int(r["exact_match_count"]) for r in new_rows], label="bosfix")
    axes[0].set_title("Overfit-8 Exact Match")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("count")
    axes[0].legend()

    axes[1].plot(steps_old, [_safe_float(r["prefix1_acc"]) for r in old_rows], label="old")
    axes[1].plot(steps_new, [_safe_float(r["prefix1_acc"]) for r in new_rows], label="bosfix")
    axes[1].plot(
        steps_old,
        [_safe_float(r["teacher_forced_shifted_token_acc_mean"]) for r in old_rows],
        linestyle="--",
        label="old_tf",
    )
    axes[1].plot(
        steps_new,
        [_safe_float(r["teacher_forced_shifted_token_acc_mean"]) for r in new_rows],
        linestyle="--",
        label="bosfix_tf",
    )
    axes[1].set_title("Prefix1 vs Teacher-Forced Acc")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("score")
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_quality_summary(rows: List[Dict[str, Any]], key_label: str, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    labels = [str(r[key_label]) for r in rows]
    token_f1 = [_safe_float(r["eval.token_f1_mean"]) for r in rows]
    prefix1 = [_safe_float(r["eval.prefix_1_match_mean"]) for r in rows]
    lcs = [_safe_float(r["eval.lcs_overlap_mean"]) for r in rows]
    train_acc = [_safe_float(r["train.train.answer_token_acc"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 1.6), 4.8))
    xs = list(range(len(labels)))

    axes[0].bar([x - 0.18 for x in xs], token_f1, width=0.36, label="token_f1")
    axes[0].bar([x + 0.18 for x in xs], prefix1, width=0.36, label="prefix_1")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Open-loop quality")
    axes[0].legend()

    axes[1].bar([x - 0.18 for x in xs], lcs, width=0.36, label="lcs_overlap")
    axes[1].bar([x + 0.18 for x in xs], train_acc, width=0.36, label="train_answer_acc")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Sequence overlap / train fit")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _segment_eval_path(run_dir: Path, segment_id: int) -> Path:
    return run_dir / f"segment_{segment_id:03d}" / "eval_metrics.json"


def _segment_state_path(run_dir: Path, segment_id: int, name: str) -> Path:
    return run_dir / f"segment_{segment_id:03d}" / name


def _collect_baseline_rows(results_dir: Path, repo_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in BASELINE_RUN_SPECS:
        run_name = spec["run_name"]
        run_dir = results_dir / "runs" / run_name
        metrics_path = run_dir / "final_metrics.json"
        table_path = results_dir / "tables" / f"{run_name}_segment_metrics.csv"
        if not metrics_path.is_file() or not table_path.is_file():
            continue

        final_doc = _read_json(metrics_path)
        final = final_doc.get("final", {})
        segment_id = _safe_int(final.get("segment_id"))
        eval_doc = _read_json_if_exists(_segment_eval_path(run_dir, segment_id))
        extra = eval_doc.get("extra", {}) if isinstance(eval_doc, dict) else {}
        routing = extra.get("routing", {}) if isinstance(extra, dict) else {}
        last_row = _read_last_csv_row(table_path)

        rows.append(
            {
                "label": spec["label"],
                "regime": spec["regime"],
                "baseline_name": final_doc.get("baseline_name") or final.get("baseline_name", ""),
                "run_name": run_name,
                "segment_id": segment_id,
                "segment_name": final.get("segment_name", ""),
                "eval.current_score": _safe_float(final.get("eval.current_score")),
                "eval.seen_avg_score": _safe_float(final.get("eval.seen_avg_score")),
                "eval.forgetting": _safe_float(final.get("eval.forgetting")),
                "eval.token_f1_mean": _safe_float(eval_doc.get("token_f1_mean", final.get("eval.token_f1_mean"))),
                "eval.lcs_overlap_mean": _safe_float(eval_doc.get("lcs_overlap_mean", final.get("eval.lcs_overlap_mean"))),
                "eval.prefix_1_match_mean": _safe_float(extra.get("prefix_1_match_mean")),
                "eval.prefix_3_match_mean": _safe_float(extra.get("prefix_3_match_mean")),
                "eval.prefix_5_match_mean": _safe_float(extra.get("prefix_5_match_mean")),
                "eval.num_bad_prefix_mismatch": _safe_int(extra.get("num_bad_prefix_mismatch")),
                "train.mean_batch_acc": _safe_float(
                    final.get("train.mean_batch_acc", last_row.get("train.mean_batch_acc"))
                ),
                "train.train.answer_token_acc": _safe_float(
                    final.get("train.train.answer_token_acc", last_row.get("train.train.answer_token_acc"))
                ),
                "routing.num_routed": _safe_int(routing.get("num_routed")),
                "active_adapter": final.get("active_adapter", ""),
                "metrics_path": str(metrics_path.relative_to(repo_root)),
                "segment_table_path": str(table_path.relative_to(repo_root)),
            }
        )
    return rows


def _collect_ours_rows(results_dir: Path, repo_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in OURS_RUN_SPECS:
        run_name = spec["run_name"]
        run_dir = results_dir / "runs" / run_name
        metrics_path = run_dir / "final_metrics.json"
        table_path = results_dir / "tables" / f"{run_name}_segment_metrics.csv"
        if not metrics_path.is_file() or not table_path.is_file():
            continue

        final_doc = _read_json(metrics_path)
        final = final_doc.get("final", {})
        segment_id = _safe_int(final.get("segment_id"))
        eval_doc = _read_json_if_exists(_segment_eval_path(run_dir, segment_id))
        extra = eval_doc.get("extra", {}) if isinstance(eval_doc, dict) else {}
        routing = extra.get("routing", {}) if isinstance(extra, dict) else {}
        last_row = _read_last_csv_row(table_path)
        bank_state = _read_json_if_exists(_segment_state_path(run_dir, segment_id, "bank_state.json"))
        drift_state = _read_json_if_exists(_segment_state_path(run_dir, segment_id, "drift_state.json"))
        router_state = _read_json_if_exists(_segment_state_path(run_dir, segment_id, "router_state.json"))

        branches = bank_state.get("branches", {}) if isinstance(bank_state, dict) else {}
        if not isinstance(branches, dict):
            branches = {}

        rows.append(
            {
                "variant": spec["variant"],
                "run_name": run_name,
                "segment_id": segment_id,
                "segment_name": final.get("segment_name", ""),
                "eval.current_score": _safe_float(final.get("eval.current_score")),
                "eval.seen_avg_score": _safe_float(final.get("eval.seen_avg_score")),
                "eval.forgetting": _safe_float(final.get("eval.forgetting")),
                "eval.token_f1_mean": _safe_float(eval_doc.get("token_f1_mean", final.get("eval.token_f1_mean"))),
                "eval.lcs_overlap_mean": _safe_float(eval_doc.get("lcs_overlap_mean", final.get("eval.lcs_overlap_mean"))),
                "eval.prefix_1_match_mean": _safe_float(extra.get("prefix_1_match_mean")),
                "eval.prefix_3_match_mean": _safe_float(extra.get("prefix_3_match_mean")),
                "eval.prefix_5_match_mean": _safe_float(extra.get("prefix_5_match_mean")),
                "eval.num_bad_prefix_mismatch": _safe_int(extra.get("num_bad_prefix_mismatch")),
                "train.train.answer_token_acc": _safe_float(
                    final.get("train.train.answer_token_acc", last_row.get("train.train.answer_token_acc"))
                ),
                "routing.num_routed": _safe_int(routing.get("num_routed")),
                "bank.active": bank_state.get("active", "") if isinstance(bank_state, dict) else "",
                "bank.num_branches": len(branches),
                "drift.ema": _safe_float(drift_state.get("ema")),
                "router.num_updates": _safe_int(router_state.get("num_updates")),
                "metrics_path": str(metrics_path.relative_to(repo_root)),
                "segment_table_path": str(table_path.relative_to(repo_root)),
            }
        )
    return rows


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"

    old_steps = _read_csv(results_dir / "runs" / "baseline_alignment_overfit" / "debug" / "overfit8" / "overfit8_steps.csv")
    new_steps = _read_csv(results_dir / "runs" / "baseline_alignment_overfit_bosfix" / "debug" / "overfit8" / "overfit8_steps.csv")
    old_fail = _read_csv(results_dir / "runs" / "baseline_alignment_overfit" / "debug" / "overfit8" / "step_0020_failure_summary.csv")[0]
    new_fail = _read_csv(results_dir / "runs" / "baseline_alignment_overfit_bosfix" / "debug" / "overfit8" / "step_0020_failure_summary.csv")[0]
    new_preds = _read_json(results_dir / "runs" / "baseline_alignment_overfit_bosfix" / "debug" / "overfit8" / "step_0020_predictions.json")

    overfit_rows = []
    for label, step_row, fail_row in [
        ("old", old_steps[-1], old_fail),
        ("bosfix", new_steps[-1], new_fail),
    ]:
        overfit_rows.append(
            {
                "run_variant": label,
                "exact_match_count": _safe_int(step_row["exact_match_count"]),
                "prefix1_acc": _safe_float(step_row["prefix1_acc"]),
                "prefix3_acc": _safe_float(step_row["prefix3_acc"]),
                "prefix5_acc": _safe_float(step_row["prefix5_acc"]),
                "teacher_forced_shifted_token_acc_mean": _safe_float(step_row["teacher_forced_shifted_token_acc_mean"]),
                "token_f1_mean": _safe_float(step_row["token_f1_mean"]),
                "lcs_overlap_mean": _safe_float(step_row["lcs_overlap_mean"]),
                "count_first_token_wrong": _safe_int(fail_row["count_first_token_wrong"]),
                "count_extra_preamble": _safe_int(fail_row["count_extra_preamble"]),
                "count_prefix_drift_1to5": _safe_int(fail_row["count_prefix_drift_1to5"]),
            }
        )
    _write_csv(tables_dir / "overfit_bosfix_comparison.csv", overfit_rows)
    _plot_line_compare(old_steps, new_steps, figures_dir / "overfit_bosfix_progress.png")

    error_rows: List[Dict[str, Any]] = []
    for row in new_preds:
        if row.get("exact_match"):
            continue
        error_rows.append(
            {
                "idx": row.get("idx"),
                "gold_output": row.get("gold_output", ""),
                "generated_output": row.get("generated_output", ""),
                "prefix_1_match": row.get("prefix_1_match"),
                "prefix_3_match": row.get("prefix_3_match"),
                "token_f1": row.get("token_f1"),
                "lcs_overlap": row.get("lcs_overlap"),
            }
        )
    _write_csv(tables_dir / "overfit_bosfix_error_examples.csv", error_rows)

    baseline_rows = _collect_baseline_rows(results_dir, repo_root)
    ours_rows = _collect_ours_rows(results_dir, repo_root)

    baseline_summary_path = tables_dir / "baseline_bosfix_comparison.csv"
    ours_summary_path = tables_dir / "ours_smoke_bosfix_ablation_summary.csv"

    if baseline_rows:
        _write_csv(baseline_summary_path, baseline_rows)
        _plot_quality_summary(
            baseline_rows,
            key_label="label",
            out_path=figures_dir / "baseline_bosfix_quality.png",
            title="Baseline BOS-Fix Comparison",
        )

    if ours_rows:
        _write_csv(ours_summary_path, ours_rows)
        _plot_quality_summary(
            ours_rows,
            key_label="variant",
            out_path=figures_dir / "ours_smoke_bosfix_ablation.png",
            title="Ours Smoke Ablation",
        )

    report_lines = [
        "# RP Alignment Progress Report",
        "",
        "## Behavior Gate",
        "",
        f"- Overfit exact match improved from `{overfit_rows[0]['exact_match_count']}/8` to `{overfit_rows[1]['exact_match_count']}/8` after removing duplicated BOS in generation-time prompt tokenization.",
        f"- First-token failures dropped from `{overfit_rows[0]['count_first_token_wrong']}` to `{overfit_rows[1]['count_first_token_wrong']}`; extra-preamble failures changed from `{overfit_rows[0]['count_extra_preamble']}` to `{overfit_rows[1]['count_extra_preamble']}`.",
        "- Remaining failures are still dominated by first-token misses and over-generation on a few samples, so the BOS fix removes a pipeline bug but does not fully solve exposure bias.",
        "",
    ]

    if baseline_rows:
        recovery_rows = [r for r in baseline_rows if r["regime"] == "recovery"]
        smoke_rows = [r for r in baseline_rows if r["regime"] == "smoke"]
        best_smoke = max(smoke_rows, key=lambda r: _safe_float(r["eval.token_f1_mean"])) if smoke_rows else None
        report_lines.extend(
            [
                "## Baseline Recovery",
                "",
                f"- Wrote baseline comparison table: `{baseline_summary_path.relative_to(repo_root)}`",
                f"- Wrote baseline quality figure: `{(figures_dir / 'baseline_bosfix_quality.png').relative_to(repo_root)}`",
            ]
        )
        if recovery_rows:
            recovery = recovery_rows[0]
            report_lines.append(
                f"- Recovery mini run `{recovery['run_name']}` reached `train_answer_acc={recovery['train.train.answer_token_acc']:.3f}`, `prefix1={recovery['eval.prefix_1_match_mean']:.3f}`, `prefix3={recovery['eval.prefix_3_match_mean']:.3f}`, `prefix5={recovery['eval.prefix_5_match_mean']:.3f}`, while `current_score` remained `{recovery['eval.current_score']:.3f}`."
            )
        if best_smoke is not None:
            report_lines.append(
                f"- Among smoke baselines, `{best_smoke['label']}` had the highest `token_f1={best_smoke['eval.token_f1_mean']:.3f}` with `prefix1={best_smoke['eval.prefix_1_match_mean']:.3f}`, but all smoke runs still stayed at `current_score=0.0`."
            )
        report_lines.append("")

    if ours_rows:
        full_row = next((r for r in ours_rows if r["variant"] == "ours_full"), None)
        best_ours = max(ours_rows, key=lambda r: _safe_float(r["eval.token_f1_mean"]))
        report_lines.extend(
            [
                "## Ours Smoke Ablation",
                "",
                f"- Wrote ablation table: `{ours_summary_path.relative_to(repo_root)}`",
                f"- Wrote ablation figure: `{(figures_dir / 'ours_smoke_bosfix_ablation.png').relative_to(repo_root)}`",
            ]
        )
        if full_row is not None:
            report_lines.append(
                f"- Full stack `{full_row['variant']}` produced `bank.num_branches={full_row['bank.num_branches']}`, `routing.num_routed={full_row['routing.num_routed']}`, `router.num_updates={full_row['router.num_updates']}`, and `drift.ema={full_row['drift.ema']:.4f}`."
            )
        report_lines.append(
            f"- Best ablation by token F1 was `{best_ours['variant']}` with `token_f1={best_ours['eval.token_f1_mean']:.3f}` and `prefix1={best_ours['eval.prefix_1_match_mean']:.3f}`."
        )
        report_lines.append("")

    report_lines.extend(
        [
            "## Artifacts",
            "",
            f"- Overfit comparison table: `{(tables_dir / 'overfit_bosfix_comparison.csv').relative_to(repo_root)}`",
            f"- Overfit progress figure: `{(figures_dir / 'overfit_bosfix_progress.png').relative_to(repo_root)}`",
            f"- Error analysis table: `{(tables_dir / 'overfit_bosfix_error_examples.csv').relative_to(repo_root)}`",
            "",
        ]
    )

    report_path = results_dir / "rp_alignment_progress_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
