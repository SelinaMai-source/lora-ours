#!/usr/bin/env python3
"""Generate full 8×5 matrix gap audit CSV (seed=123, published_setting)."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/tables/full_matrix_gap_audit_s123.csv"
CONFIG_DIR = REPO / "configs/paper/published_setting"
RUNS = REPO / "results/runs"
SEQGLUE_DATA = REPO / "data/processed/seqglue_cl_tasks_train50_eval10.json"
LFPT5_CKPT = REPO / "assets/pretrained/lfpt5/lm_adapted_t5_large_torch/pytorch_model.bin"

METHODS = [
    ("sequential_lora", "Sequential LoRA"),
    ("replay_lora", "Replay LoRA"),
    ("o_lora", "O-LoRA"),
    ("lb_cl", "LB-CL"),
    ("progressive_prompts", "Progressive Prompts"),
    ("continual_t0", "Continual-T0"),
    ("lfpt5", "LFPT5"),
    ("ours", "Ours"),
]

BENCHMARKS = [
    ("instrdialog", "InstrDialog"),
    ("instrdialogpp", "InstrDialog++"),
    ("trace", "TRACE"),
    ("multiwoz", "MultiWOZ NLG"),
    ("seqglue", "Seq-GLUE"),
]

BENCH_FILE_PREFIX = {
    "instrdialog": "instrdialog",
    "instrdialogpp": "instrdialogpp",
    "trace": "trace",
    "multiwoz": "multiwoz",
    "seqglue": "seqglue",
}

METHOD_CONFIG_KEY = {
    "ours": "ours",
    "o_lora": "o_lora",
    "lb_cl": "lb_cl",
    "progressive_prompts": "progressive_prompts",
    "continual_t0": "continual_t0",
    "sequential_lora": "sequential_lora",
    "replay_lora": "replay_lora",
    "lfpt5": "lfpt5",
}

RUN_PREFIX = {
    "instrdialog": "published_instrdialog",
    "instrdialogpp": "published_instrdialogpp",
    "trace": "trace_full",
    "multiwoz": "full_multiwoz",
    "seqglue": "published_seqglue",
}


def _config_path(bench: str, method: str) -> Path:
    return CONFIG_DIR / f"{BENCH_FILE_PREFIX[bench]}__{METHOD_CONFIG_KEY[method]}__s123.yaml"


def _run_name_from_config(cfg_path: Path, bench: str, method: str) -> str:
    if cfg_path.is_file():
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            rn = (cfg.get("output") or {}).get("run_name")
            if rn:
                return str(rn)
        except yaml.YAMLError:
            pass
    m = "ours" if method == "ours" else method
    return f"{RUN_PREFIX[bench]}_{m}_s123"


def _has_final(run_name: str) -> bool:
    return (RUNS / run_name / "final_metrics.json").is_file()


def _infer_status(bench: str, method: str, cfg_path: Path, run_name: str) -> tuple[str, str]:
    if not cfg_path.is_file():
        return "missing", "config 缺失"
    if method == "lfpt5":
        if not LFPT5_CKPT.is_file():
            return "blocked", "LFPT5 需 LM-adapted T5-large checkpoint（未下载）"
        return "queued", "LFPT5 wrapper 可跑（T5 独立环境）"
    if bench == "seqglue" and not SEQGLUE_DATA.is_file():
        return "blocked", "Seq-GLUE processed 未就绪"
    if _has_final(run_name):
        return "completed", "已有 final_metrics.json"
    rd = RUNS / run_name
    if rd.is_dir() and any(rd.glob("segment_*")):
        return "queued", "存在 partial segment（续跑/队列 skipped_partial）"
    return "queued", "gap full run 待排队"


def main() -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: list[dict] = []
    for bench_key, bench_label in BENCHMARKS:
        for method_key, method_label in METHODS:
            cfg = _config_path(bench_key, method_key)
            run_name = _run_name_from_config(cfg, bench_key, method_key)
            status, notes = _infer_status(bench_key, method_key, cfg, run_name)
            rows.append({
                "method": method_label,
                "method_key": method_key,
                "benchmark": bench_label,
                "benchmark_key": bench_key,
                "seed": "123",
                "run_name": run_name,
                "config_path": str(cfg.relative_to(REPO)) if cfg.is_file() else "",
                "status": status,
                "notes": notes,
                "updated_at": now,
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "method_key", "benchmark", "benchmark_key", "seed",
        "run_name", "config_path", "status", "notes", "updated_at",
    ]
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"Wrote {len(rows)} cells -> {OUT}")
    print("status:", counts)


if __name__ == "__main__":
    main()
