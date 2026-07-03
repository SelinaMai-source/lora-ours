#!/usr/bin/env python3
"""Prepare an isolated CITB runtime tree for paper-target split runs.

The official Stage-2 script exposes a single eval cap, so the script-strict
launcher yields 500/50/50 for InstrDialog. This helper creates a separate copy of
the official tree and patches only the local split utility to use distinct
train/dev/test caps. The source checkout is left unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ignore(_: str, names: list[str]) -> set[str]:
    ignored = {
        ".git",
        "__pycache__",
        "cache",
        "wandb",
        "runs",
        "outputs",
    }
    return {name for name in names if name in ignored or name.endswith(".pyc")}


def _patch_utils(path: Path, *, dev: int, test: int) -> None:
    text = path.read_text(encoding="utf-8")
    original = text
    text = text.replace(
        "def train_dev_test_split_by_task(raw_datasets, max_num_instances_per_task, max_num_instances_per_eval_task, continual=False):\n",
        "def train_dev_test_split_by_task(raw_datasets, max_num_instances_per_task, max_num_instances_per_eval_task, continual=False):\n"
        f"    paper_target_dev_instances = {dev}\n"
        f"    paper_target_test_instances = {test}\n",
        1,
    )
    text = text.replace(
        "            test_instances[task_name].extend(instances[:max_num_instances_per_eval_task])\n"
        "            dev_instances[task_name].extend(instances[max_num_instances_per_eval_task:max_num_instances_per_eval_task*2])\n"
        "\n"
        "            # make sure per task training instances not exceeding the limit\n"
        "            remaining_instances = instances[max_num_instances_per_eval_task*2:]\n",
        "            test_instances[task_name].extend(instances[:paper_target_test_instances])\n"
        "            dev_instances[task_name].extend(instances[paper_target_test_instances:paper_target_test_instances + paper_target_dev_instances])\n"
        "\n"
        "            # make sure per task training instances not exceeding the limit\n"
        "            remaining_instances = instances[paper_target_test_instances + paper_target_dev_instances:]\n",
        1,
    )
    text = text.replace(
        "            test_instances.extend(instances[:max_num_instances_per_eval_task])\n"
        "            dev_instances.extend(instances[max_num_instances_per_eval_task:max_num_instances_per_eval_task*2])\n"
        "\n"
        "            # make sure per task training instances not exceeding the limit\n"
        "            remaining_instances = instances[max_num_instances_per_eval_task*2:]\n",
        "            test_instances.extend(instances[:paper_target_test_instances])\n"
        "            dev_instances.extend(instances[paper_target_test_instances:paper_target_test_instances + paper_target_dev_instances])\n"
        "\n"
        "            # make sure per task training instances not exceeding the limit\n"
        "            remaining_instances = instances[paper_target_test_instances + paper_target_dev_instances:]\n",
        1,
    )
    if text == original:
        raise RuntimeError(f"Did not patch expected split blocks in {path}")
    path.write_text(text, encoding="utf-8")


def prepare(source: Path, output_base: Path, *, train: int, dev: int, test: int, force: bool) -> Dict[str, str]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    runtime = output_base / f"citb_official_paper_target_train{train}_dev{dev}_test{test}"
    if runtime.exists():
        if not force:
            manifest = runtime / "paper_target_manifest.json"
            return {"runtime": str(runtime), "manifest": str(manifest), "reused": "true"}
        shutil.rmtree(runtime)
    shutil.copytree(source, runtime, ignore=_ignore)
    utils_path = runtime / "continual_learning" / "utils.py"
    before_hash = _sha256(utils_path)
    _patch_utils(utils_path, dev=dev, test=test)
    after_hash = _sha256(utils_path)
    manifest_path = runtime / "paper_target_manifest.json"
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source),
        "runtime": str(runtime),
        "policy": "paper_target_500_50_100",
        "train_instances_per_task": train,
        "dev_instances_per_task": dev,
        "test_instances_per_task": test,
        "patched_files": {
            str(utils_path.relative_to(runtime)): {
                "before_sha256": before_hash,
                "after_sha256": after_hash,
                "reason": "Separate dev and test caps for CITB paper-target InstrDialog split.",
            }
        },
        "claim_boundary": "Local paper-target split patch; do not mix with official-script 500/50/50 results.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"runtime": str(runtime), "manifest": str(manifest_path), "reused": "false"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--train", type=int, default=500)
    parser.add_argument("--dev", type=int, default=50)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--print-path-only", action="store_true")
    args = parser.parse_args()
    result = prepare(
        source=Path(args.source),
        output_base=Path(args.output_base),
        train=args.train,
        dev=args.dev,
        test=args.test,
        force=args.force,
    )
    if args.print_path_only:
        print(result["runtime"])
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
