#!/usr/bin/env python3
"""Generate published_setting_run_v2 manifest + status CSV from full (non-toy) configs."""
from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs/paper/published_setting"
MANIFEST = REPO / "results/tables/published_setting_run_v2_manifest.csv"
STATUS = REPO / "results/tables/published_setting_run_v2_status.csv"
WANDB_PROJECT = "lora-published-setting-run_v2"
TRACE_DATA = REPO / "data/processed/trace_cl_tasks_train50_eval10.json"

METHOD_ORDER = [
    "ours_published",
    "o_lora",
    "lb_cl",
    "progressive_prompts",
    "continual_t0",
    "sequential_lora",
    "replay_lora",
]
BENCH_ORDER = ["instrdialog", "instrdialogpp", "multiwoz_nlg", "trace"]

METHOD_MAP = {
    "ours": "ours_published",
    "o_lora": "o_lora",
    "lb_cl": "lb_cl",
    "progressive_prompts": "progressive_prompts",
    "continual_t0": "continual_t0",
    "sequential_lora": "sequential_lora",
    "replay_lora": "replay_lora",
}


def _parse_config(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"(\w+)__(\w+)__s123\.yaml$", path.name)
    if not m:
        return None
    bench_key, method_key = m.group(1), m.group(2)
    if bench_key == "instrdialogpp":
        benchmark = "instrdialogpp"
    elif bench_key == "instrdialog":
        benchmark = "instrdialog"
    elif bench_key == "multiwoz":
        benchmark = "multiwoz_nlg"
    elif bench_key == "trace":
        benchmark = "trace"
    else:
        return None
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError:
        cfg = {}
    run_name = (
        (cfg.get("output") or {}).get("run_name")
        or (cfg.get("logging") or {}).get("run_name")
        or path.stem
    )
    return {
        "run_name": run_name,
        "benchmark": benchmark,
        "method": METHOD_MAP.get(method_key, method_key),
        "seed": "123",
        "config_path": str(path.relative_to(REPO)),
        "wandb_project": WANDB_PROJECT,
        "wandb_group": f"v2_{benchmark}",
    }


def _run_dir(run_name: str) -> Path:
    return REPO / "results/runs" / run_name


def _has_final(run_name: str) -> bool:
    return (_run_dir(run_name) / "final_metrics.json").is_file()


def _has_partial(run_name: str) -> bool:
    rd = _run_dir(run_name)
    if not rd.is_dir():
        return False
    return any(rd.glob("segment_*"))


def _infer_status(row: dict) -> tuple[str, str]:
    run_name = row["run_name"]
    cfg = REPO / row["config_path"]
    if not cfg.is_file():
        return "blocked", "config 缺失"
    if row["benchmark"] == "trace" and not TRACE_DATA.is_file():
        return "blocked", "全量 processed 未就绪；TRACE raw/processed 待下载"
    if _has_final(run_name):
        return "completed", "已有 final_metrics.json"
    if _has_partial(run_name):
        return "skipped_partial", "存在 segment checkpoint；需人工决定续跑或清理"
    return "queued", "v2 full run 待排队"


def main() -> None:
    rows: list[dict] = []
    for path in sorted(CONFIG_DIR.glob("*__s123.yaml")):
        if "_toy" in path.name:
            continue
        parsed = _parse_config(path)
        if parsed:
            rows.append(parsed)

    def sort_key(r: dict) -> tuple:
        b = BENCH_ORDER.index(r["benchmark"]) if r["benchmark"] in BENCH_ORDER else 99
        m = METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 99
        return (b, m)

    rows.sort(key=sort_key)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fieldnames = [
        "run_name", "benchmark", "method", "seed", "config_path",
        "wandb_project", "wandb_group", "status", "notes", "exit_code", "updated_at",
    ]
    out_rows: list[dict] = []
    for row in rows:
        status, notes = _infer_status(row)
        out_rows.append({
            **row,
            "status": status,
            "notes": notes,
            "exit_code": "",
            "updated_at": now,
        })

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    status_fields = fieldnames + ["log_file"]
    for row in out_rows:
        row["log_file"] = f"results/logs/published_setting_run_v2/{row['run_name']}.log"
    with STATUS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=status_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    counts: dict[str, int] = {}
    for row in out_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"Wrote {len(out_rows)} rows -> {MANIFEST}")
    print("status counts:", counts)


if __name__ == "__main__":
    main()
