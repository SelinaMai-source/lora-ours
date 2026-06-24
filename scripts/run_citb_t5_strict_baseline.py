#!/usr/bin/env python3
"""CPU-safe preflight runner for CITB-strict T5 Sequential/Replay baselines.

This replaces the previous temptation to label local Llama LoRA Sequential/Replay
as strict. The script performs only CPU-safe checks, records the official CITB
command that would be launched, and writes a manifest with any remaining blockers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from baselines.citb_t5.citb_metrics import compute_citb_metric_summary
from baselines.citb_t5.data_splits import describe_split_mismatch, inspect_official_split, resolve_official_repo
from baselines.citb_t5.ft_init_trainer import build_official_continual_command
from baselines.citb_t5.replay_memory import replay_examples_from_policy


REQUIRED_MODULES = [
    "baselines/citb_t5/ft_init_trainer.py",
    "baselines/citb_t5/replay_memory.py",
    "baselines/citb_t5/citb_metrics.py",
    "baselines/citb_t5/data_splits.py",
]
REQUIRED_T5_ASSET_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def preflight(config_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    run_name = (cfg.get("output") or {}).get("run_name", config_path.stem)
    data_cfg = cfg.get("data") or {}
    model_cfg = cfg.get("model") or {}
    method_cfg = cfg.get("method") or {}
    paper_cfg = cfg.get("paper") or {}

    stream_path = None
    if data_cfg.get("processed_stream_file"):
        stream_path = REPO / str(data_cfg.get("processed_stream_file", ""))
        if not stream_path.is_file():
            stream_path = REPO / "data/processed" / str(data_cfg.get("processed_stream_file", ""))
    model_path = Path(str(model_cfg.get("model_name_or_path", "")))
    if not model_path.is_absolute():
        model_path = REPO / model_path
    official_repo = resolve_official_repo(data_cfg.get("official_repo"))

    missing_modules = [module for module in REQUIRED_MODULES if not (REPO / module).is_file()]
    missing_t5_asset_files = [name for name in REQUIRED_T5_ASSET_FILES if not (model_path / name).is_file()]
    errors: List[str] = []
    blockers: List[str] = []
    warnings: List[str] = []
    resolved: List[str] = []

    official_split = inspect_official_split(official_repo, str(data_cfg.get("benchmark", "InstrDialog")))
    run_plan = build_official_continual_command(cfg, config_path=config_path, official_repo=official_repo)

    if stream_path is not None and not stream_path.is_file():
        warnings.append(f"legacy processed_stream_file is not present and is ignored for official CITB preflight: {stream_path}")
    if model_cfg.get("strict_backbone") == "lm_adapted_t5_small" and missing_t5_asset_files:
        blockers.append(f"LM-adapted T5-small asset is incomplete at {model_path}; missing: {', '.join(missing_t5_asset_files)}")
    elif model_cfg.get("strict_backbone") == "lm_adapted_t5_small":
        resolved.append(f"LM-adapted T5-small asset present at {_rel(model_path)}")
    if missing_modules:
        blockers.append("missing CITB T5 implementation modules: " + ", ".join(missing_modules))
    else:
        resolved.append("CITB T5 implementation modules present: " + ", ".join(REQUIRED_MODULES))

    split_mismatch = describe_split_mismatch(str(data_cfg.get("split", "")))
    if split_mismatch:
        blockers.append(split_mismatch)
    elif official_split["strict_split_available"]:
        resolved.append(
            "official CITB split/order resources available for "
            f"{official_split['benchmark']} ({official_split['task_count']} tasks; orders {', '.join(official_split['available_orders'])})"
        )
    elif official_split["task_count"] and official_split["available_orders"]:
        resolved.append(
            "official CITB task split/order metadata available for "
            f"{official_split['benchmark']} ({official_split['task_count']} tasks; orders {', '.join(official_split['available_orders'])})"
        )
    if official_split["missing"]:
        blockers.append("official CITB repo resources missing: " + ", ".join(official_split["missing"]))
    if official_split["dataset_file_errors"]:
        blockers.append("official CITB materialized dataset files missing: " + "; ".join(official_split["dataset_file_errors"]))
    if official_split["order_errors"]:
        blockers.extend("official CITB task-order validation failed: " + error for error in official_split["order_errors"])

    if paper_cfg.get("metric_suite") != "AR_FWT_BWT_FR":
        blockers.append("CITB AR/FWT/BWT/FR metric suite is not implemented for this runner")
    else:
        smoke_metrics = compute_citb_metric_summary([[10.0, 20.0], [8.0, 30.0]]).as_dict()
        resolved.append("CITB AR/FWT/BWT/FR metric module import and smoke calculation passed")
    if method_cfg.get("name") == "citb_t5_replay":
        try:
            replay_num = replay_examples_from_policy(str(method_cfg.get("memory_policy")))
            resolved.append(f"CITB Replay memory policy validated: {method_cfg.get('memory_policy')} => {replay_num} examples/task")
        except ValueError as exc:
            blockers.append(str(exc))

    if run_plan.missing_inputs:
        blockers.append("official CITB full-run inputs missing: " + ", ".join(_rel(Path(path)) for path in run_plan.missing_inputs))

    blockers.append("full strict CITB T5 training is not launched by this CPU-safe preflight; run only after remaining inputs and compute budget are explicitly approved")

    manifest = {
        "run_name": run_name,
        "runner": "citb_t5_strict_preflight",
        "config": _rel(config_path),
        "method": method_cfg.get("name"),
        "benchmark": data_cfg.get("benchmark"),
        "status": (
            "strict-preflight-ready-needs-full-run"
            if not errors and not [b for b in blockers if "full strict CITB T5 training is not launched" not in b]
            else "strict-preflight-ready-with-blockers"
        ),
        "strict_allowed": False,
        "preflight_passed": not errors,
        "errors": errors,
        "blockers": blockers,
        "warnings": warnings,
        "resolved_checks": resolved,
        "official_split": official_split,
        "official_run_plan": run_plan.as_dict(),
        "paths": {
            "processed_stream": _rel(stream_path) if stream_path is not None else None,
            "official_repo": _rel(official_repo),
            "model_name_or_path": _rel(model_path),
            "run_dir": f"results/runs/{run_name}",
        },
        "implementation_plan": [
            {
                "file": "baselines/citb_t5/data_splits.py",
                "purpose": "implemented: load and validate official InstrDialog/InstrDialog++ task split and task order resources",
            },
            {
                "file": "baselines/citb_t5/ft_init_trainer.py",
                "purpose": "implemented: build official FT-init/FT-no-init/Replay full-finetune command plan without launching training",
            },
            {
                "file": "baselines/citb_t5/replay_memory.py",
                "purpose": "implemented: validate Replay(10)/(50) and select first-N examples per task from official seeded memory data",
            },
            {
                "file": "baselines/citb_t5/citb_metrics.py",
                "purpose": "implemented: compute AR, FWT, BWT, and FR from the official task-score matrix",
            },
            {
                "file": "scripts/run_citb_t5_strict_baseline.py",
                "purpose": "current file: CPU-safe preflight with official command preview; full training remains blocked until inputs and compute are approved",
            },
        ],
        "safe_commands": [
            f"python scripts/run_citb_t5_strict_baseline.py --config {_rel(config_path)} --preflight-only",
        ],
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="CITB T5 strict Sequential/Replay runner skeleton.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true", help="Only write blocker manifest; training is not implemented yet.")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else REPO / args.config
    manifest = preflight(config_path)
    run_dir = REPO / "results/runs" / str(manifest["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not args.preflight_only:
        print("[blocked] CITB T5 strict training was not launched by this CPU-safe preflight; rerun with --preflight-only for the safe check.")
        return 2
    return 0 if manifest["preflight_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
