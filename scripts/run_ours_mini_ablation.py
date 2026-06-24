from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml


ABLATIONS = {
    "ours_full": {
        "use_drift_detector": True,
        "use_lora_bank": True,
        "use_router": True,
        "use_overlap_loss": True,
    },
    "ours_no_drift": {
        "use_drift_detector": False,
        "use_lora_bank": True,
        "use_router": True,
        "use_overlap_loss": True,
    },
    "ours_no_bank": {
        "use_drift_detector": True,
        "use_lora_bank": False,
        "use_router": True,
        "use_overlap_loss": True,
    },
    "ours_no_router": {
        "use_drift_detector": True,
        "use_lora_bank": True,
        "use_router": False,
        "use_overlap_loss": True,
    },
    "ours_no_overlap": {
        "use_drift_detector": True,
        "use_lora_bank": True,
        "use_router": True,
        "use_overlap_loss": False,
    },
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_last_csv_row(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[-1]


def _latest_state(run_dir: Path, name: str) -> Dict[str, Any]:
    matches = sorted(run_dir.glob(f"segment_*/{name}"))
    if not matches:
        return {}
    return _read_json(matches[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small ours ablation sweep and summarize results.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ours_smoke_bosfix.yaml"),
        help="Base YAML config for the ours mini sweep.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(ABLATIONS.keys()),
        help="Comma-separated ablation variant names to run.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="smoke_bosfix",
        help="Prefix used when naming per-variant runs.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/tables/ours_smoke_bosfix_ablation_summary.csv"),
        help="Where to write the summarized per-variant results.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_cfg = _load_yaml((repo_root / args.config).resolve())
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    if not variants:
        raise ValueError("No variants requested.")

    results_dir = Path(str(base_cfg.get("paths", {}).get("results_dir", "results")))
    summary_rows: List[Dict[str, Any]] = []

    for variant in variants:
        if variant not in ABLATIONS:
            raise ValueError(f"Unknown variant: {variant}")

        cfg = copy.deepcopy(base_cfg)
        modules_cfg = cfg.setdefault("modules", {})
        if not isinstance(modules_cfg, dict):
            raise ValueError("modules config must be a mapping")
        modules_cfg.update(ABLATIONS[variant])
        output_cfg = cfg.setdefault("output", {})
        if not isinstance(output_cfg, dict):
            raise ValueError("output config must be a mapping")
        run_name = f"{args.run_prefix}_{variant}"
        output_cfg["run_name"] = run_name

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
            yaml.safe_dump(cfg, tmp, sort_keys=False, allow_unicode=True)
            tmp_path = Path(tmp.name)

        try:
            subprocess.run(
                [sys.executable, "core/train.py", "--config", str(tmp_path)],
                cwd=repo_root,
                check=True,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        run_dir = repo_root / results_dir / "runs" / run_name
        metrics_path = run_dir / "final_metrics.json"
        seg_table_path = repo_root / results_dir / "tables" / f"{run_name}_segment_metrics.csv"
        final_metrics = _read_json(metrics_path)
        last_row = _read_last_csv_row(seg_table_path)
        final_row = final_metrics.get("final", {}) if isinstance(final_metrics, dict) else {}
        bank_state = _latest_state(run_dir, "bank_state.json")
        drift_state = _latest_state(run_dir, "drift_state.json")
        router_state = _latest_state(run_dir, "router_state.json")

        summary_rows.append(
            {
                "variant": variant,
                "run_name": run_name,
                "segment_id": final_row.get("segment_id", last_row.get("segment_id", "")),
                "segment_name": final_row.get("segment_name", last_row.get("segment_name", "")),
                "eval.current_score": final_row.get("eval.current_score", last_row.get("eval.current_score", "")),
                "eval.seen_avg_score": final_row.get(
                    "eval.seen_avg_score", last_row.get("eval.seen_avg_score", "")
                ),
                "eval.forgetting": final_row.get("eval.forgetting", last_row.get("eval.forgetting", "")),
                "eval.num_seen_segments": final_row.get(
                    "eval.num_seen_segments", last_row.get("eval.num_seen_segments", "")
                ),
                "eval.token_f1_mean": final_row.get(
                    "eval.token_f1_mean", last_row.get("eval.token_f1_mean", "")
                ),
                "eval.lcs_overlap_mean": final_row.get(
                    "eval.lcs_overlap_mean", last_row.get("eval.lcs_overlap_mean", "")
                ),
                "train.train.loss": final_row.get("train.train.loss", last_row.get("train.train.loss", "")),
                "train.train.answer_token_acc": final_row.get(
                    "train.train.answer_token_acc", last_row.get("train.train.answer_token_acc", "")
                ),
                "routing.num_routed": last_row.get("train.routed_examples", ""),
                "bank.active": bank_state.get("active", ""),
                "bank.num_branches": len(bank_state.get("branches", {})),
                "drift.ema": drift_state.get("ema", ""),
                "router.num_updates": router_state.get("num_updates", ""),
                "metrics_path": str(metrics_path.relative_to(repo_root)),
                "segment_table_path": str(seg_table_path.relative_to(repo_root)),
                "mode": final_row.get("mode", final_metrics.get("mode", "")),
            }
        )

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
