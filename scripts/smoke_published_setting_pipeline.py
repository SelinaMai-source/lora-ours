#!/usr/bin/env python3
"""
CPU-only smoke validation for published-setting pipeline (no GPU training).

Checks:
  1. Published-setting YAML configs parse and reference existing processed data
  2. Continual streams load via core.data.load_continual_stream
  3. Published segment CSV final metrics match paper_experiment_plan appendix
  4. External baseline smoke (py_compile) — optional skip

Usage:
  python scripts/smoke_published_setting_pipeline.py
  python scripts/smoke_published_setting_pipeline.py --skip-external
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.data import load_continual_stream, _parse_stream_json

# Expected final metrics from published_setting runs (seed 123), source: results/tables/*_segment_metrics.csv
EXPECTED_FINAL: Dict[str, Tuple[float, float, float, int]] = {
    "instrdialog_ours": (0.3307, 0.3825, 0.4164, 19),
    "instrdialog_o_lora": (0.2053, 0.3246, 0.2620, 19),
    "instrdialog_lb_cl": (0.2254, 0.3351, 0.2810, 19),
    "instrdialog_progressive_prompts": (0.0316, 0.2518, 0.1104, 19),
    "instrdialog_continual_t0": (0.1263, 0.3044, 0.2121, 19),
    "instrdialogpp_ours": (0.2981, 0.3394, 0.4148, 38),
    "instrdialogpp_o_lora": (0.2243, 0.2822, 0.3167, 38),
    "instrdialogpp_lb_cl": (0.2365, 0.2839, 0.3346, 38),
    "instrdialogpp_progressive_prompts": (0.0211, 0.1943, 0.1531, 38),
    "instrdialogpp_continual_t0": (0.0217, 0.1792, 0.1737, 38),
}
TOL = 1e-3


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_configs() -> List[str]:
    errors: List[str] = []
    cfg_dir = REPO / "configs" / "paper" / "published_setting"
    for cfg_path in sorted(cfg_dir.glob("*.yaml")):
        try:
            cfg = _load_yaml(cfg_path)
            data_cfg = cfg.get("data", {})
            stream = load_continual_stream(
                mode=cfg.get("mode", "ours"),
                sample_stream_path=None,
                processed_stream_dir=str(REPO / cfg.get("paths", {}).get("processed_stream_dir", "data/processed")),
                processed_stream_file=str(cfg.get("paths", {}).get("processed_stream_file", "")),
                processed_stream_name=str(data_cfg.get("stream_name", "")),
                auto_prepare_processed=False,
                max_segments=int(data_cfg.get("max_segments", 1)),
                max_train_examples_per_segment=int(data_cfg.get("max_train_examples_per_segment", 2)),
                max_eval_examples_per_segment=int(data_cfg.get("max_eval_examples_per_segment", 2)),
            )
            if not stream.stream:
                errors.append(f"{cfg_path.name}: empty stream after truncate")
            else:
                print(f"OK config+data: {cfg_path.name} ({len(stream.stream)} segs loaded)")
        except Exception as exc:
            errors.append(f"{cfg_path.name}: {exc}")
    return errors


def check_processed_json() -> List[str]:
    errors: List[str] = []
    required = [
        "citb_cl_dialogue_tasks_train50_eval10.json",
        "citb_cl_38_random_tasks_train50_eval10.json",
        "trace_cl_tasks_train50_eval10_toy.json",
    ]
    for fn in required:
        p = REPO / "data" / "processed" / fn
        if not p.exists():
            errors.append(f"missing processed: {fn}")
            continue
        try:
            s = _parse_stream_json(str(p))
            print(f"OK processed: {fn} ({len(s.stream)} segments, benchmark={s.benchmark})")
        except Exception as exc:
            errors.append(f"parse failed {fn}: {exc}")
    return errors


def check_csv_metrics() -> List[str]:
    errors: List[str] = []
    tables = REPO / "results" / "tables"
    for key, (exp_seen, exp_ta, exp_f1, exp_segs) in EXPECTED_FINAL.items():
        matches = list(tables.glob(f"published_{key}_s123_segment_metrics.csv"))
        if not matches:
            # fuzzy match for method names with underscores
            alt = key.replace("_", "*")
            matches = list(tables.glob(f"published_{alt}_s123_segment_metrics.csv"))
        if not matches:
            errors.append(f"no CSV for {key}")
            continue
        csv_path = matches[0]
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        last = rows[-1]
        seen = float(last.get("eval.seen_avg_score", last.get("seen_avg_score", 0)))
        ta = float(last.get("eval.seen_avg_task_aware_score", last.get("seen_avg_task_aware_score", 0)))
        f1 = float(last.get("eval.token_f1_mean", last.get("token_f1_mean", 0)))
        segs = int(float(last.get("eval.num_seen_segments", last.get("num_seen_segments", 0))))
        ok = (
            abs(seen - exp_seen) <= TOL
            and abs(ta - exp_ta) <= TOL
            and abs(f1 - exp_f1) <= TOL
            and segs == exp_segs
        )
        if ok:
            print(f"OK CSV metrics: {csv_path.name} seen={seen:.4f} ta={ta:.4f} f1={f1:.4f} segs={segs}")
        else:
            errors.append(
                f"{csv_path.name}: seen={seen} (exp {exp_seen}), ta={ta} (exp {exp_ta}), "
                f"f1={f1} (exp {exp_f1}), segs={segs} (exp {exp_segs})"
            )
    return errors


def check_external_smoke() -> List[str]:
    script = REPO / "scripts" / "smoke_external_baselines.sh"
    proc = subprocess.run(["bash", str(script), "all"], capture_output=True, text=True, timeout=120)
    # exit 1 is expected (NEEDS_ENV for --help); py_compile passes matter
    compile_fails = [ln for ln in proc.stdout.splitlines() if "SKIP" in ln or ("FAIL" in ln and "NEEDS_ENV" not in ln)]
    if compile_fails:
        return compile_fails
    compile_pass = sum(1 for ln in proc.stdout.splitlines() if "PASS" in ln and "py_compile" in ln)
    print(f"OK external smoke: {compile_pass} py_compile passes (NEEDS_ENV for --help expected)")
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-external", action="store_true")
    args = parser.parse_args()

    all_errors: List[str] = []
    print("=== processed JSON ===")
    all_errors.extend(check_processed_json())
    print("=== published_setting configs + data load ===")
    all_errors.extend(check_configs())
    print("=== CSV final metrics ===")
    all_errors.extend(check_csv_metrics())
    if not args.skip_external:
        print("=== external baselines smoke ===")
        all_errors.extend(check_external_smoke())

    if all_errors:
        print("\nFAILED:")
        for e in all_errors:
            print(f"  - {e}")
        return 1
    print("\nALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
