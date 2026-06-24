#!/usr/bin/env python3
"""Run the official O-LoRA Seq-GLUE bridge for lora-run_v10.

The wrapper keeps the official O-LoRA training entry intact and only supplies
local paths, W&B settings, and a reproducible run manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Official O-LoRA Seq-GLUE runner")
    parser.add_argument(
        "--config",
        default="configs/paper/lora_run_v10/seqglue__o_lora_official__s123.yaml",
        help="v10 O-LoRA YAML config",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate/export and print command only")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    method = cfg["method"]
    data = cfg["data"]
    training = cfg["training"]
    output = cfg["output"]
    tracking = output.get("tracking", {})

    run_name = output["run_name"]
    run_dir = REPO / "results" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir = REPO / output["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    sequential_results_file = run_dir / "sequential_results.jsonl"

    source_dir = REPO / method.get("source_dir", "external_baselines/o_lora")
    entry = source_dir / "src" / "run_uie_lora.py"
    env_prefix = Path(method.get("env_prefix", "/root/autodl-tmp/conda_envs/lora_v10_o_lora"))
    bridge_script = REPO / data.get("bridge_script", "scripts/export_seqglue_to_olora.py")
    olora_root = REPO / data["olora_root"]

    manifest = {
        "run_name": run_name,
        "runner": "o_lora_official",
        "config": _rel(cfg_path),
        "source_dir": _rel(source_dir),
        "entry": _rel(entry),
        "env_prefix": str(env_prefix),
        "olora_root": _rel(olora_root),
        "status": "blocked",
        "block_reason": "",
    }

    blockers = []
    for path, label in [
        (source_dir, "official source_dir"),
        (entry, "official run_uie_lora.py"),
        (env_prefix, "official conda env"),
        (bridge_script, "Seq-GLUE export bridge"),
        (REPO / data["stream_json"], "Seq-GLUE stream"),
    ]:
        if not path.exists():
            blockers.append(f"{label} missing: {path}")
    if blockers:
        manifest["block_reason"] = "; ".join(blockers)
        _write_manifest(run_dir, manifest)
        print(f"[blocked] {manifest['block_reason']}", file=sys.stderr)
        return 2

    if not (olora_root / "manifest.json").is_file():
        export_cmd = [
            sys.executable,
            str(bridge_script),
            "--stream-json",
            str(REPO / data["stream_json"]),
            "--out-root",
            str(olora_root),
        ]
        print(f"[olora] exporting Seq-GLUE bridge: {' '.join(export_cmd)}")
        subprocess.run(export_cmd, cwd=str(REPO), check=True)
    validate_cmd = [
        sys.executable,
        str(bridge_script),
        "--out-root",
        str(olora_root),
        "--validate-only",
    ]
    print(f"[olora] validating Seq-GLUE bridge: {' '.join(validate_cmd)}")
    subprocess.run(validate_cmd, cwd=str(REPO), check=True)

    cmd = [
        "conda",
        "run",
        "-p",
        str(env_prefix),
        "--no-capture-output",
        "python",
        str(entry),
        "--do_train",
        "--do_predict",
        "--predict_with_generate",
        "--model_name_or_path",
        os.environ.get("MODEL_NAME_OR_PATH", cfg["model"]["model_name_or_path"]),
        "--data_dir",
        str(REPO / data["data_dir"]),
        "--task_config_dir",
        str(REPO / data["task_config_dir"]),
        "--instruction_file",
        str(REPO / data["instruction_file"]),
        "--instruction_strategy",
        "single",
        "--sequential_results_file",
        str(sequential_results_file),
        "--output_dir",
        str(output_dir),
        "--per_device_train_batch_size",
        str(training["per_device_train_batch_size"]),
        "--per_device_eval_batch_size",
        str(training["per_device_eval_batch_size"]),
        "--gradient_accumulation_steps",
        str(training["gradient_accumulation_steps"]),
        "--learning_rate",
        str(training["learning_rate"]),
        "--num_train_epochs",
        str(training["num_train_epochs"]),
        "--run_name",
        run_name,
        "--max_source_length",
        str(training["max_source_length"]),
        "--max_target_length",
        str(training["max_target_length"]),
        "--generation_max_length",
        str(training["generation_max_length"]),
        "--max_num_instances_per_task",
        str(training["max_num_instances_per_task"]),
        "--max_num_instances_per_eval_task",
        str(training["max_num_instances_per_eval_task"]),
        "--add_task_name",
        "True",
        "--add_dataset_name",
        "True",
        "--overwrite_output_dir",
        "--overwrite_cache",
        "--lr_scheduler_type",
        "constant",
        "--warmup_steps",
        "0",
        "--logging_strategy",
        "steps",
        "--logging_steps",
        "10",
        "--evaluation_strategy",
        "no",
        "--save_strategy",
        "no",
        "--lamda_1",
        str(training["lamda_1"]),
        "--lamda_2",
        str(training["lamda_2"]),
        "--lora_dim",
        str(training["lora_dim"]),
        "--seed",
        str(cfg["seed"]),
        "--report_to",
        tracking.get("report_to", "wandb"),
    ]

    env = os.environ.copy()
    env.update(
        {
            "WANDB_DISABLED": "False",
            "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", tracking.get("wandb_project", "lora-run_v10")),
            "WANDB_GROUP": os.environ.get("WANDB_GROUP", tracking.get("wandb_group", "lora-run_v10")),
            "WANDB_MODE": os.environ.get("WANDB_MODE", tracking.get("wandb_mode", "online")),
        }
    )

    manifest["status"] = "ready" if args.dry_run else "running"
    manifest["command"] = cmd
    manifest["log_file"] = output.get("log_file", "")
    _write_manifest(run_dir, manifest)

    print("[olora] command:")
    print(" ".join(cmd))
    if args.dry_run:
        return 0

    result = subprocess.run(cmd, cwd=str(source_dir), env=env)
    manifest["status"] = "completed" if result.returncode == 0 else "failed"
    manifest["exit_code"] = result.returncode
    if result.returncode != 0:
        manifest["block_reason"] = f"official O-LoRA runner exit code {result.returncode}"
    _write_manifest(run_dir, manifest)
    return int(result.returncode)


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
