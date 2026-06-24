#!/usr/bin/env python3
"""Diagnose LFPT5 protocol/metric failures from saved run artifacts.

The script is read-only and CPU-only. It inspects completed or partial LFPT5
runs for exact-match all-zero, token-F1 collapse, custom task token leakage,
target/prediction formatting problems, and config defaults that differ from the
paper-setting assumptions recorded in the tracker.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

REPO = Path(__file__).resolve().parents[1]
TASK_TOKEN_RE = re.compile(r"\bcitbtask\d+\b")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _iter_scored_examples(run_dir: Path, max_segments: int) -> Iterable[Dict[str, Any]]:
    seg_dirs = sorted(run_dir.glob("segment_*/eval_metrics.json"))
    if max_segments > 0:
        seg_dirs = seg_dirs[:max_segments]
    for path in seg_dirs:
        metrics = _read_json(path)
        for segment in metrics.get("extra", {}).get("scored_details_by_segment", []):
            for ex in segment.get("examples", []):
                item = dict(ex)
                item["source_eval_metrics"] = str(path)
                item["segment_id"] = segment.get("segment_id")
                yield item


def _score_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    exact_cols = [
        "eval.current_score",
        "eval.seen_avg_score",
        "eval.current_task_aware_score",
        "eval.seen_avg_task_aware_score",
    ]
    return {
        "num_metric_rows": len(rows),
        "all_exact_metric_values_zero": bool(rows)
        and all(float(row.get(col, 0.0)) == 0.0 for row in rows for col in exact_cols),
        "max_token_f1": max([float(row.get("eval.token_f1_mean", 0.0)) for row in rows] or [0.0]),
        "final_token_f1": float(rows[-1].get("eval.token_f1_mean", 0.0)) if rows else 0.0,
        "segments_seen": [row.get("segment_name") for row in rows],
    }


def diagnose(run_dir: Path, config_path: Path | None, max_segments: int, out_examples: int) -> Dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.jsonl"
    final_metrics_path = run_dir / "final_metrics.json"
    metrics_rows = _read_jsonl(metrics_path)
    examples = list(_iter_scored_examples(run_dir, max_segments=max_segments))

    prediction_counter: Counter[str] = Counter()
    target_counter: Counter[str] = Counter()
    task_token_examples = 0
    ans_token_examples = 0
    empty_prediction_examples = 0
    exact_matches = 0
    token_f1_values: List[float] = []

    for ex in examples:
        pred = str(ex.get("prediction", "") or "")
        gold = str(ex.get("gold_output", "") or "")
        prediction_counter.update(TASK_TOKEN_RE.findall(pred))
        target_counter.update(TASK_TOKEN_RE.findall(gold))
        if TASK_TOKEN_RE.search(pred):
            task_token_examples += 1
        if "__ans__" in pred:
            ans_token_examples += 1
        if not pred.strip():
            empty_prediction_examples += 1
        if bool(ex.get("exact_match")):
            exact_matches += 1
        token_f1_values.append(float(ex.get("token_f1", 0.0)))

    cfg: Dict[str, Any] = {}
    if config_path and config_path.is_file():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    lfpt5_cfg = cfg.get("lfpt5") or {}
    expected_defaults = {
        "lr": 5e-5,
        "kd_lamda": 0.05,
        "gradient_accumulation_steps": 1,
    }
    config_audit = {
        "config": str(config_path.relative_to(REPO)) if config_path and config_path.is_file() else None,
        "run_manifest_bridge_max_epoch": None,
        "lfpt5_values": {key: lfpt5_cfg.get(key, "<default>") for key in ["lr", "kd_lamda", "gradient_accumulation_steps", "bridge_max_epoch", "prompt_number", "batch_size_per_gpu"]},
        "expected_default_reference": expected_defaults,
        "mismatches": [],
    }
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        bridge_cmd = manifest.get("bridge_cmd") or []
        if "--max-epoch" in bridge_cmd:
            idx = bridge_cmd.index("--max-epoch")
            config_audit["run_manifest_bridge_max_epoch"] = bridge_cmd[idx + 1] if idx + 1 < len(bridge_cmd) else None
    for key, expected in expected_defaults.items():
        if key in lfpt5_cfg and lfpt5_cfg[key] != expected:
            config_audit["mismatches"].append({"key": key, "expected": expected, "observed": lfpt5_cfg[key]})

    n_examples = max(1, len(examples))
    score_summary = _score_summary(metrics_rows)
    root_cause_candidates: List[str] = []
    if score_summary["all_exact_metric_values_zero"]:
        root_cause_candidates.append("all exact-match metrics are zero across every logged segment")
    if task_token_examples / n_examples > 0.5:
        root_cause_candidates.append("predictions leak LFPT5 custom task tokens, suggesting generation/special-token masking is broken")
    if ans_token_examples / n_examples > 0.5:
        root_cause_candidates.append("predictions include __ans__, suggesting answer-token stripping or decoding is misaligned")
    if score_summary["max_token_f1"] < 0.05:
        root_cause_candidates.append("token F1 is near zero, so this is not only an exact-match normalization issue")
    if config_audit["mismatches"]:
        root_cause_candidates.append("config contains LFPT5 hparams that differ from the default paper-setting assumptions")
    if not examples:
        root_cause_candidates.append("no scored examples found; run may be partial or older bridge did not save eval details")

    sample_failures = []
    for ex in examples:
        if not ex.get("exact_match"):
            sample_failures.append(
                {
                    "segment_id": ex.get("segment_id"),
                    "gold_output": ex.get("gold_output"),
                    "prediction": ex.get("prediction"),
                    "normalized_gold": ex.get("normalized_gold"),
                    "normalized_prediction": ex.get("normalized_prediction"),
                    "token_f1": ex.get("token_f1"),
                }
            )
        if len(sample_failures) >= out_examples:
            break

    result = {
        "cell_hint": "LFPT5",
        "run_dir": str(run_dir.relative_to(REPO)),
        "status": "protocol-diagnosed-not-strict",
        "strict_allowed": False,
        "manifest_present": manifest_path.is_file(),
        "final_metrics_present": final_metrics_path.is_file(),
        "score_summary": score_summary,
        "example_summary": {
            "num_scored_examples_checked": len(examples),
            "exact_matches": exact_matches,
            "task_token_leak_examples": task_token_examples,
            "ans_token_leak_examples": ans_token_examples,
            "empty_prediction_examples": empty_prediction_examples,
            "mean_example_token_f1": sum(token_f1_values) / max(1, len(token_f1_values)),
            "top_leaked_task_tokens": prediction_counter.most_common(10),
            "gold_task_tokens": target_counter.most_common(10),
        },
        "config_audit": config_audit,
        "root_cause_candidates": root_cause_candidates,
        "minimum_fix_checks_before_rerun": [
            "mask or skip citbtask* and __ans__ tokens during generation decoding",
            "verify LFPT5 task-token/prompt initialization against the official Classification or Summarization path for this benchmark",
            "pin lr/kd_lamda/gradient_accumulation_steps/max_epoch from the paper or official scripts in the YAML",
            "run a one-segment CPU/GPU mini diagnostic and require non-task-token predictions before any full run",
            "only then rerun full LFPT5 InstrDialog++ or safely resume Seq-GLUE from a clean checkpoint plan",
        ],
        "sample_failures": sample_failures,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose saved LFPT5 metrics/predictions without training.")
    parser.add_argument("--run-dir", type=Path, default=Path("results/runs/published_instrdialogpp_lfpt5_s123"))
    parser.add_argument("--config", type=Path, default=Path("configs/paper/published_setting/instrdialogpp__lfpt5__s123.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/tables/lfpt5_instrdialogpp_protocol_diagnosis.json"))
    parser.add_argument("--max-segments", type=int, default=-1)
    parser.add_argument("--sample-failures", type=int, default=5)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO / args.run_dir
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    out_path = args.out if args.out.is_absolute() else REPO / args.out
    result = diagnose(run_dir, config_path if config_path.is_file() else None, args.max_segments, args.sample_failures)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
