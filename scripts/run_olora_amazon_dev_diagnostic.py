#!/usr/bin/env python3
"""Run an O-LoRA amazon/SC non-test prediction diagnostic.

The official order1 dev task configs are empty, although the dataset contains
`dev.json`. This diagnostic creates an isolated runtime under /root/autodl-tmp,
patches only that runtime so the prediction split reads a requested non-test
source split, and evaluates an existing adapter without training.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


REPO = Path("/root/lora-ours")
DEFAULT_OFFICIAL_ROOT = Path("/root/autodl-tmp/lora-baselines-run_v1/external_sources/o_lora")
DEFAULT_ENV_PREFIX = Path("/root/autodl-tmp/conda_envs/lora_v10_o_lora")
DEFAULT_BASE_MODEL = Path("/root/autodl-tmp/model_cache/hf_snapshots/t5-large")
DEFAULT_ADAPTER = (
    REPO
    / "results/runs/olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1"
    / "official_outputs/4-agnews/adapter"
)


def run(command: list[str], log_file: Path) -> int:
    with log_file.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"\nEXIT_CODE={proc.returncode}\n")
        return proc.returncode


def patch_runtime(runtime: Path, source_split: str) -> None:
    path = runtime / "src/uie_dataset_lora.py"
    text = path.read_text(encoding="utf-8")
    old = '''            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['test'],
                    "max_num_instances_per_task": None,  # default load total test samples to test
                    "subset": "test"
                }),
'''
    new = '''            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['test'],
                    "max_num_instances_per_task": None,
                    "subset": "SOURCE_SPLIT_PLACEHOLDER"
                }),
'''
    if old not in text:
        raise RuntimeError(f"expected test split block not found in {path}")
    path.write_text(text.replace(old, new.replace("SOURCE_SPLIT_PLACEHOLDER", source_split), 1), encoding="utf-8")


def write_task_config(task_config_dir: Path) -> None:
    task_config_dir.mkdir(parents=True, exist_ok=True)
    empty = {"SC": [], "TC": [], "NLI": [], "QQP": [], "WiC": [], "MultiRC": [], "COPA": [], "BoolQA": []}
    test = dict(empty)
    test["SC"] = [{"sampling strategy": "full", "dataset name": "amazon"}]
    for name, payload in {
        "train_tasks.json": empty,
        "dev_tasks.json": empty,
        "test_tasks.json": test,
    }.items():
        (task_config_dir / name).write_text(json.dumps(payload, indent=4), encoding="utf-8")


def slice_train_split(runtime: Path, offset: int, limit: int) -> dict[str, int]:
    if offset < 0:
        raise ValueError("--train-offset must be >= 0")
    if limit == 0 or limit < -1:
        raise ValueError("--train-limit must be -1 or a positive integer")
    path = runtime / "CL_Benchmark/SC/amazon/train.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    end = None if limit < 0 else offset + limit
    sliced = rows[offset:end]
    if not sliced:
        raise ValueError(f"empty train slice: offset={offset}, limit={limit}, total={len(rows)}")
    path.write_text(json.dumps(sliced, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"original_train_size": len(rows), "train_offset": offset, "train_limit": limit, "sliced_train_size": len(sliced)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--max-predict-samples", type=int, default=-1)
    parser.add_argument("--source-split", choices=["dev", "train"], default="dev")
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=-1)
    parser.add_argument("--official-root", default=str(DEFAULT_OFFICIAL_ROOT))
    parser.add_argument("--env-prefix", default=str(DEFAULT_ENV_PREFIX))
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    args = parser.parse_args()

    run_root = Path("/root/autodl-tmp/lora-ours-devdiag") / args.run_name
    runtime = run_root / "official_runtime"
    output_dir = run_root / "outputs"
    task_config_dir = run_root / "task_config_amazon_dev_as_test"
    log_file = REPO / "results/logs" / f"{args.run_name}.log"
    manifest_file = REPO / "results/logs" / f"{args.run_name}.manifest.json"
    status_file = REPO / "results/logs" / f"{args.run_name}.status.md"

    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(args.official_root, runtime)
    train_slice = None
    if args.source_split == "train" and (args.train_offset or args.train_limit >= 0):
        train_slice = slice_train_split(runtime, args.train_offset, args.train_limit)
    patch_runtime(runtime, args.source_split)
    write_task_config(task_config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "conda",
        "run",
        "-p",
        args.env_prefix,
        "--no-capture-output",
        "python",
        str(runtime / "src/run_uie_lora.py"),
        "--do_predict",
        "--predict_with_generate",
        "--model_name_or_path",
        args.adapter,
        "--data_dir",
        str(runtime / "CL_Benchmark"),
        "--task_config_dir",
        str(task_config_dir),
        "--instruction_file",
        str(runtime / "configs/instruction_config.json"),
        "--instruction_strategy",
        "single",
        "--output_dir",
        str(output_dir),
        "--per_device_eval_batch_size",
        "16",
        "--max_source_length",
        "512",
        "--max_target_length",
        "50",
        "--generation_max_length",
        "50",
        "--add_task_name",
        "True",
        "--add_dataset_name",
        "True",
        "--overwrite_output_dir",
        "--overwrite_cache",
        "--seed",
        "1",
        "--report_to",
        "none",
    ]
    if args.max_predict_samples >= 0:
        command += ["--max_predict_samples", str(args.max_predict_samples)]

    manifest = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "purpose": f"{args.source_split}_only_amazon_sc_prediction_diagnostic",
        "source_split": args.source_split,
        "uses_test_json": False,
        "adapter": args.adapter,
        "runtime": str(runtime),
        "output_dir": str(output_dir),
        "task_config_dir": str(task_config_dir),
        "log_file": str(log_file),
        "max_predict_samples": args.max_predict_samples,
        "train_slice": train_slice,
    }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    status_file.write_text(
        "\n".join(
            [
                f"# {args.run_name}",
                "",
                "- state: running",
                f"- diagnostic: {args.source_split}-only amazon/SC prediction",
                "- no training in this diagnostic; no test.json; v69 final adapter",
                f"- train_slice: `{train_slice}`",
                f"- output_dir: `{output_dir}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    code = run(command, log_file)
    manifest["exit_code"] = code
    manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    status_file.write_text(status_file.read_text(encoding="utf-8").replace("- state: running", f"- state: exit {code}"), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
