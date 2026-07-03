#!/usr/bin/env python3
"""Write a lightweight status page for O-LoRA Standard official runs."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
TASKS = ["dbpedia", "amazon", "yahoo", "agnews"]
FAILURE_MARKERS = [
    "Traceback (most recent call last)",
    "RuntimeError:",
    "CUDA out of memory",
    "ROUND_EXIT_CODE:1",
    "ROUND_EXIT_CODE:2",
    "ROUND_EXIT_CODE:137",
    "ROUND_EXIT_CODE:143",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _gpu_snapshot() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"unavailable: {exc}"
    return out or "unavailable"


def _read_tail(path: Path, max_bytes: int = 12000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def _round_dirs(output_root: Path) -> list[Path]:
    dirs = [p for p in output_root.iterdir() if p.is_dir()] if output_root.is_dir() else []
    return sorted(dirs, key=lambda p: int(p.name.split("-", 1)[0]) if p.name.split("-", 1)[0].isdigit() else 999)


def _dataset_score(metrics: dict[str, Any], task: str, metric: str) -> float | None:
    key = f"predict_{metric}_for_{task}"
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _summarize(rounds: list[tuple[str, dict[str, Any]]]) -> tuple[list[str], dict[str, Any]]:
    lines: list[str] = []
    history: dict[str, list[float]] = {task: [] for task in TASKS}
    round_exact_values: list[float] = []
    round_rouge_values: list[float] = []

    for name, metrics in rounds:
        exact = metrics.get("predict_exact_match")
        rouge = metrics.get("predict_rougeL")
        samples = metrics.get("predict_samples")
        step = metrics.get("predict_global_step")
        lines.append(f"- {name}: step={step}, samples={samples}, exact={exact}, rougeL={rouge}")
        if isinstance(exact, (int, float)):
            round_exact_values.append(float(exact))
        if isinstance(rouge, (int, float)):
            round_rouge_values.append(float(rouge))
        for task in TASKS:
            score = _dataset_score(metrics, task, "exact_match")
            if score is not None:
                history[task].append(score)

    latest_scores = {task: values[-1] for task, values in history.items() if values}
    final_avg_exact = sum(latest_scores.values()) / len(latest_scores) if latest_scores else None
    forgetting_values = []
    for values in history.values():
        if len(values) >= 2:
            forgetting_values.append(max(values[:-1]) - values[-1])
    avg_forgetting = sum(forgetting_values) / len(forgetting_values) if forgetting_values else None
    summary = {
        "completed_rounds": len(rounds),
        "latest_task_exact": latest_scores,
        "round_avg_exact": sum(round_exact_values) / len(round_exact_values) if round_exact_values else None,
        "round_avg_rougeL": sum(round_rouge_values) / len(round_rouge_values) if round_rouge_values else None,
        "observed_avg_exact": final_avg_exact,
        "observed_avg_forgetting": avg_forgetting,
    }
    return lines, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor O-LoRA Standard official run outputs.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    run_dir = REPO / "results" / "runs" / args.run_name
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    output_root = Path(manifest.get("output_root") or run_dir / "official_outputs")
    if not output_root.is_absolute():
        output_root = REPO / output_root

    rounds = []
    for round_dir in _round_dirs(output_root):
        metrics = _read_json(round_dir / "all_results.json")
        if metrics:
            rounds.append((round_dir.name, metrics))

    round_lines, summary = _summarize(rounds)
    log_file = REPO / "results" / "logs" / f"{args.run_name}.log"
    log_tail = _read_tail(log_file)
    marker = next((item for item in FAILURE_MARKERS if item.lower() in log_tail.lower()), "")
    tmux_present = bool(
        subprocess.run(
            ["bash", "-lc", "tmux ls 2>/dev/null | rg -q 'olora-standard-official-base-formal'"],
            check=False,
        ).returncode
        == 0
    )
    sentinel = ""
    if marker or manifest.get("state") in {"failed", "blocked"}:
        sentinel = f"AGENT_LOOP_WAKE_LORA_OURS standard_olora_failed state={manifest.get('state')} marker={marker}"
    elif not tmux_present and manifest.get("state") not in {"completed", "failed", "blocked"}:
        sentinel = f"AGENT_LOOP_WAKE_LORA_OURS standard_olora_stopped_incomplete state={manifest.get('state')}"
    status_file = REPO / "results" / "logs" / f"{args.run_name}_monitor.md"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {args.run_name} monitor",
        "",
        f"- updated: {datetime.now().isoformat(timespec='seconds')}",
        f"- manifest_state: {manifest.get('state', 'missing')}",
        f"- label: {manifest.get('label', '')}",
        f"- note: {args.note}",
        f"- task_order: {' -> '.join(manifest.get('task_order', TASKS))}",
        f"- smoke_limits: {manifest.get('smoke_limits', {})}",
        f"- gpu: {_gpu_snapshot()}",
        f"- tmux_present: {tmux_present}",
        f"- log_marker: {marker}",
        f"- completed_rounds: {summary['completed_rounds']}/4",
        f"- round_avg_exact: {summary['round_avg_exact']}",
        f"- round_avg_rougeL: {summary['round_avg_rougeL']}",
        f"- observed_avg_exact: {summary['observed_avg_exact']}",
        f"- observed_avg_forgetting: {summary['observed_avg_forgetting']}",
        f"- latest_task_exact: {summary['latest_task_exact']}",
        "",
        "## Rounds",
        "",
    ]
    lines.extend(round_lines or ["- no completed round metrics yet"])
    lines.extend(
        [
            "",
            "## Comparability Boundary",
            "",
            "- This monitor reports local diagnostic metrics only.",
            "- A run with single-GPU runtime or reduced eval batch is not paper-comparable unless explicitly audited against the official setting.",
            "",
        ]
    )
    if sentinel:
        lines.extend(["", sentinel])
    status_file.write_text("\n".join(lines), encoding="utf-8")
    print(status_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
