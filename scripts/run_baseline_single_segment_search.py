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

DEFAULT_SEGMENTS = [3, 4, 14]
DEFAULT_VARIANTS: Dict[str, Dict[str, Any]] = {
    "greedy64_eosmask": {
        "gen_num_beams": 1,
        "gen_max_new_tokens": 64,
        "mask_eos_token_in_labels": True,
        "epochs_per_segment": 5,
    },
    "beam4_64_eosmask": {
        "gen_num_beams": 4,
        "gen_max_new_tokens": 64,
        "mask_eos_token_in_labels": True,
        "epochs_per_segment": 5,
    },
    "beam4_64_eoslearn": {
        "gen_num_beams": 4,
        "gen_max_new_tokens": 64,
        "mask_eos_token_in_labels": False,
        "epochs_per_segment": 5,
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


def _parse_int_list(text: str) -> List[int]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    return [int(x) for x in values]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _variant_names_arg(text: str) -> List[str]:
    names = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in names if x not in DEFAULT_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variant(s): {unknown}. Available: {sorted(DEFAULT_VARIANTS)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Search baseline recovery settings on single segments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_recovery_mini.yaml"),
        help="Base YAML config used to generate per-run temporary configs.",
    )
    parser.add_argument(
        "--segments",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SEGMENTS),
        help="Comma-separated segment indices to try via single-segment mode.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(DEFAULT_VARIANTS.keys()),
        help=f"Comma-separated variant names. Available: {','.join(DEFAULT_VARIANTS.keys())}",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/tables/baseline_single_segment_search.csv"),
        help="Where to write the summarized search results.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_cfg = _load_yaml((repo_root / args.config).resolve())
    segments = _parse_int_list(args.segments)
    variant_names = _variant_names_arg(args.variants)
    results_dir = Path(str(base_cfg.get("paths", {}).get("results_dir", "results")))

    summary_rows: List[Dict[str, Any]] = []

    for segment_index in segments:
        for variant_name in variant_names:
            variant = copy.deepcopy(DEFAULT_VARIANTS[variant_name])
            cfg = copy.deepcopy(base_cfg)

            data_cfg = cfg.setdefault("data", {})
            if not isinstance(data_cfg, dict):
                raise ValueError("data config must be a mapping")
            # Load the full ordered stream, then select one representative segment by index.
            data_cfg["max_segments"] = -1

            debug_tools = cfg.setdefault("debug_tools", {})
            if not isinstance(debug_tools, dict):
                raise ValueError("debug_tools config must be a mapping")
            debug_tools["enable_single_segment_mode"] = True
            debug_tools["single_segment_index"] = int(segment_index)
            debug_tools["enable_overfit_8_mode"] = False

            model_cfg = cfg.setdefault("model", {})
            if not isinstance(model_cfg, dict):
                raise ValueError("model config must be a mapping")
            model_cfg["gen_do_sample"] = False
            model_cfg["gen_num_beams"] = int(variant["gen_num_beams"])
            model_cfg["gen_max_new_tokens"] = int(variant["gen_max_new_tokens"])
            model_cfg["mask_eos_token_in_labels"] = bool(variant["mask_eos_token_in_labels"])

            train_cfg = cfg.setdefault("train", {})
            if not isinstance(train_cfg, dict):
                raise ValueError("train config must be a mapping")
            train_cfg["epochs_per_segment"] = int(variant["epochs_per_segment"])

            output_cfg = cfg.setdefault("output", {})
            if not isinstance(output_cfg, dict):
                raise ValueError("output config must be a mapping")
            run_name = f"baseline_single_seg{segment_index:02d}_{variant_name}"
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

            metrics_path = repo_root / results_dir / "runs" / run_name / "final_metrics.json"
            final_metrics = _read_json(metrics_path).get("final", {})
            summary_rows.append(
                {
                    "segment_index": int(segment_index),
                    "segment_id": final_metrics.get("segment_id", ""),
                    "segment_name": final_metrics.get("segment_name", ""),
                    "variant_name": variant_name,
                    "gen_num_beams": int(variant["gen_num_beams"]),
                    "gen_max_new_tokens": int(variant["gen_max_new_tokens"]),
                    "mask_eos_token_in_labels": bool(variant["mask_eos_token_in_labels"]),
                    "epochs_per_segment": int(variant["epochs_per_segment"]),
                    "eval.current_score": _safe_float(final_metrics.get("eval.current_score")),
                    "eval.token_f1_mean": _safe_float(final_metrics.get("eval.token_f1_mean")),
                    "eval.lcs_overlap_mean": _safe_float(final_metrics.get("eval.lcs_overlap_mean")),
                    "train.answer_token_acc": _safe_float(final_metrics.get("train.train.answer_token_acc")),
                    "run_name": run_name,
                    "metrics_path": str(metrics_path.relative_to(repo_root)),
                }
            )

    summary_rows.sort(
        key=lambda row: (
            -_safe_float(row.get("eval.current_score")),
            -_safe_float(row.get("eval.token_f1_mean")),
            -_safe_float(row.get("train.answer_token_acc")),
        )
    )
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {args.summary_csv}")
    if summary_rows:
        best = summary_rows[0]
        print(
            "Best:",
            json.dumps(
                {
                    "segment_index": best["segment_index"],
                    "segment_name": best["segment_name"],
                    "variant_name": best["variant_name"],
                    "eval.current_score": best["eval.current_score"],
                    "eval.token_f1_mean": best["eval.token_f1_mean"],
                    "train.answer_token_acc": best["train.answer_token_acc"],
                    "run_name": best["run_name"],
                },
                ensure_ascii=False,
            ),
        )


if __name__ == "__main__":
    main()
