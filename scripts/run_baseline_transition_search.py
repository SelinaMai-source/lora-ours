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


DEFAULT_SEGMENTS = [3, 4]
DEFAULT_VARIANTS: Dict[str, Dict[str, Any]] = {
    "ctrl_ep5_lr2e4": {
        "model": {
            "gen_num_beams": 1,
            "gen_max_new_tokens": 64,
            "gen_do_sample": False,
            "mask_eos_token_in_labels": True,
        },
        "train": {
            "epochs_per_segment": 5,
            "lr": 2.0e-4,
        },
    },
    "ep8_lr1e4": {
        "model": {
            "gen_num_beams": 1,
            "gen_max_new_tokens": 64,
            "gen_do_sample": False,
            "mask_eos_token_in_labels": True,
        },
        "train": {
            "epochs_per_segment": 8,
            "lr": 1.0e-4,
        },
    },
    "ep10_lr5e5": {
        "model": {
            "gen_num_beams": 1,
            "gen_max_new_tokens": 64,
            "gen_do_sample": False,
            "mask_eos_token_in_labels": True,
        },
        "train": {
            "epochs_per_segment": 10,
            "lr": 5.0e-5,
        },
    },
    "ep8_lr1e4_eoslearn": {
        "model": {
            "gen_num_beams": 1,
            "gen_max_new_tokens": 64,
            "gen_do_sample": False,
            "mask_eos_token_in_labels": False,
        },
        "train": {
            "epochs_per_segment": 8,
            "lr": 1.0e-4,
        },
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
    if not values:
        raise ValueError("At least one segment index is required.")
    return [int(x) for x in values]


def _variant_names_arg(text: str) -> List[str]:
    names = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in names if x not in DEFAULT_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variant(s): {unknown}. Available: {sorted(DEFAULT_VARIANTS)}")
    return names


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _subset_stream(raw_stream: Dict[str, Any], segment_indices: List[int]) -> Dict[str, Any]:
    stream = raw_stream.get("stream")
    if not isinstance(stream, list):
        raise ValueError("Processed stream JSON is missing a list `stream` field.")
    selected = []
    for idx in segment_indices:
        if idx < 0 or idx >= len(stream):
            raise IndexError(f"segment index {idx} out of range for stream of length {len(stream)}")
        selected.append(copy.deepcopy(stream[idx]))
    payload = copy.deepcopy(raw_stream)
    payload["stream"] = selected
    return payload


def _segment_metrics(run_dir: Path, segment_id: int) -> Dict[str, Any]:
    segment_dir = run_dir / f"segment_{segment_id:03d}"
    eval_metrics = _read_json(segment_dir / "eval_metrics.json")
    train_metrics = _read_json(segment_dir / "train_metrics.json")
    extra = eval_metrics.get("extra", {}) if isinstance(eval_metrics.get("extra", {}), dict) else {}
    return {
        "eval.current_score": _safe_float(eval_metrics.get("current_score")),
        "eval.seen_avg_score": _safe_float(eval_metrics.get("seen_avg_score")),
        "eval.forgetting": _safe_float(eval_metrics.get("forgetting")),
        "eval.token_f1_mean": _safe_float(eval_metrics.get("token_f1_mean")),
        "eval.lcs_overlap_mean": _safe_float(eval_metrics.get("lcs_overlap_mean")),
        "eval.prefix_1_match_mean": _safe_float(extra.get("prefix_1_match_mean")),
        "eval.prefix_3_match_mean": _safe_float(extra.get("prefix_3_match_mean")),
        "eval.prefix_5_match_mean": _safe_float(extra.get("prefix_5_match_mean")),
        "eval.num_bad_prefix_mismatch": int(extra.get("num_bad_prefix_mismatch", 0) or 0),
        "train.mean_batch_acc": _safe_float(train_metrics.get("mean_batch_acc")),
        "train.train.loss": _safe_float(train_metrics.get("train.loss")),
        "train.train.answer_token_acc": _safe_float(train_metrics.get("train.answer_token_acc")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sequential LoRA baseline sweeps on a compact transition subset of the CITB stream."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_recovery_main_greedy64.yaml"),
        help="Base YAML config used to generate per-run temporary configs.",
    )
    parser.add_argument(
        "--segments",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SEGMENTS),
        help="Comma-separated segment indices to keep in the temporary stream subset.",
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
        default=Path("results/tables/baseline_transition_search.csv"),
        help="Where to write the summarized sweep results.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_cfg = _load_yaml((repo_root / args.config).resolve())
    segment_indices = _parse_int_list(args.segments)
    variant_names = _variant_names_arg(args.variants)

    paths_cfg = base_cfg.get("paths", {}) if isinstance(base_cfg.get("paths", {}), dict) else {}
    processed_dir = repo_root / str(paths_cfg.get("processed_stream_dir", "data/processed"))
    processed_file = str(paths_cfg.get("processed_stream_file", ""))
    if not processed_file:
        raise ValueError("Base config must define paths.processed_stream_file")

    raw_stream = _read_json(processed_dir / processed_file)
    subset_payload = _subset_stream(raw_stream, segment_indices)
    subset_segment_ids = [int(seg["segment_id"]) for seg in subset_payload["stream"]]
    results_dir = Path(str(paths_cfg.get("results_dir", "results")))
    summary_rows: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="baseline_transition_subset_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        subset_file = tmp_dir / "subset_stream.json"
        subset_file.write_text(json.dumps(subset_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        for variant_name in variant_names:
            variant = copy.deepcopy(DEFAULT_VARIANTS[variant_name])
            cfg = copy.deepcopy(base_cfg)

            cfg_paths = cfg.setdefault("paths", {})
            if not isinstance(cfg_paths, dict):
                raise ValueError("paths config must be a mapping")
            cfg_paths["processed_stream_dir"] = str(tmp_dir)
            cfg_paths["processed_stream_file"] = subset_file.name

            data_cfg = cfg.setdefault("data", {})
            if not isinstance(data_cfg, dict):
                raise ValueError("data config must be a mapping")
            data_cfg["max_segments"] = -1

            debug_tools = cfg.setdefault("debug_tools", {})
            if not isinstance(debug_tools, dict):
                raise ValueError("debug_tools config must be a mapping")
            debug_tools["enable_single_segment_mode"] = False
            debug_tools["enable_overfit_8_mode"] = False

            model_cfg = cfg.setdefault("model", {})
            if not isinstance(model_cfg, dict):
                raise ValueError("model config must be a mapping")
            for key, value in variant.get("model", {}).items():
                model_cfg[key] = value

            train_cfg = cfg.setdefault("train", {})
            if not isinstance(train_cfg, dict):
                raise ValueError("train config must be a mapping")
            for key, value in variant.get("train", {}).items():
                train_cfg[key] = value

            output_cfg = cfg.setdefault("output", {})
            if not isinstance(output_cfg, dict):
                raise ValueError("output config must be a mapping")
            run_name = f"baseline_transition_seg{'_'.join(f'{x:02d}' for x in segment_indices)}_{variant_name}"
            output_cfg["run_name"] = run_name

            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp_cfg:
                yaml.safe_dump(cfg, tmp_cfg, sort_keys=False, allow_unicode=True)
                tmp_cfg_path = Path(tmp_cfg.name)

            try:
                subprocess.run(
                    [sys.executable, "core/train.py", "--config", str(tmp_cfg_path)],
                    cwd=repo_root,
                    check=True,
                )
            finally:
                tmp_cfg_path.unlink(missing_ok=True)

            run_dir = repo_root / results_dir / "runs" / run_name
            final_metrics = _read_json(run_dir / "final_metrics.json").get("final", {})

            row: Dict[str, Any] = {
                "subset_segment_indices": ",".join(str(x) for x in segment_indices),
                "subset_segment_ids": ",".join(str(x) for x in subset_segment_ids),
                "variant_name": variant_name,
                "epochs_per_segment": variant.get("train", {}).get("epochs_per_segment", ""),
                "lr": variant.get("train", {}).get("lr", ""),
                "gen_num_beams": variant.get("model", {}).get("gen_num_beams", ""),
                "gen_max_new_tokens": variant.get("model", {}).get("gen_max_new_tokens", ""),
                "mask_eos_token_in_labels": variant.get("model", {}).get("mask_eos_token_in_labels", ""),
                "run_name": run_name,
                "metrics_path": str((run_dir / "final_metrics.json").relative_to(repo_root)),
                "final.segment_id": final_metrics.get("segment_id", ""),
                "final.segment_name": final_metrics.get("segment_name", ""),
            }

            for seg_id in subset_segment_ids:
                metrics = _segment_metrics(run_dir, seg_id)
                for key, value in metrics.items():
                    row[f"segment_{seg_id:03d}.{key}"] = value

            summary_rows.append(row)

    last_seg_id = subset_segment_ids[-1]
    summary_rows.sort(
        key=lambda row: (
            -_safe_float(row.get(f"segment_{last_seg_id:03d}.eval.current_score")),
            -_safe_float(row.get(f"segment_{last_seg_id:03d}.train.mean_batch_acc")),
            -_safe_float(row.get(f"segment_{last_seg_id:03d}.eval.token_f1_mean")),
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
                    "variant_name": best["variant_name"],
                    f"segment_{last_seg_id:03d}.eval.current_score": best.get(
                        f"segment_{last_seg_id:03d}.eval.current_score"
                    ),
                    f"segment_{last_seg_id:03d}.train.mean_batch_acc": best.get(
                        f"segment_{last_seg_id:03d}.train.mean_batch_acc"
                    ),
                    f"segment_{last_seg_id:03d}.eval.token_f1_mean": best.get(
                        f"segment_{last_seg_id:03d}.eval.token_f1_mean"
                    ),
                    "run_name": best["run_name"],
                },
                ensure_ascii=False,
            ),
        )


if __name__ == "__main__":
    main()
