#!/usr/bin/env python3
"""No-GPU split-count preflight for CITB official Stage-2 task streams."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _load_order(order_file: Path) -> List[str]:
    return [line.strip() for line in order_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _task_count(task_path: Path) -> int:
    data = json.loads(task_path.read_text(encoding="utf-8"))
    return len(data.get("Instances") or [])


def build_report(
    *,
    citb_root: Path,
    order: int,
    task_order_dir: Path,
    train_cap: int,
    dev_cap: int,
    test_cap: int,
) -> Dict[str, Any]:
    order_file = task_order_dir / f"order{order}.txt"
    tasks = _load_order(order_file)
    rows: List[Dict[str, Any]] = []
    short_tasks: List[Dict[str, Any]] = []
    for index, task_name in enumerate(tasks):
        total = _task_count(citb_root / "data" / "tasks" / f"{task_name}.json")
        test = min(test_cap, total)
        dev = min(dev_cap, max(0, total - test))
        train = min(train_cap, max(0, total - test - dev))
        row = {
            "index": index,
            "task": task_name,
            "total_instances": total,
            "train": train,
            "dev": dev,
            "test": test,
            "target_train": train_cap,
            "target_dev": dev_cap,
            "target_test": test_cap,
            "meets_target": train == train_cap and dev == dev_cap and test == test_cap,
        }
        rows.append(row)
        if not row["meets_target"]:
            short_tasks.append(row)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "citb_root": str(citb_root),
        "order_file": str(order_file),
        "policy": f"{train_cap}/{dev_cap}/{test_cap}",
        "task_count": len(rows),
        "short_task_count": len(short_tasks),
        "all_tasks_meet_target": not short_tasks,
        "tasks": rows,
        "short_tasks": short_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--citb-root", default="/root/autodl-tmp/Lora-code/external_baselines/citb_official")
    parser.add_argument(
        "--task-order-dir",
        default="data/CIT_data/task_orders/stream=cl_dialogue_tasks",
        help="Path relative to citb-root, or an absolute path, containing orderN.txt.",
    )
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--train", type=int, default=500)
    parser.add_argument("--dev", type=int, default=50)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    citb_root = Path(args.citb_root)
    task_order_dir = Path(args.task_order_dir)
    if not task_order_dir.is_absolute():
        task_order_dir = citb_root / task_order_dir
    report = build_report(
        citb_root=citb_root,
        order=args.order,
        task_order_dir=task_order_dir,
        train_cap=args.train,
        dev_cap=args.dev,
        test_cap=args.test,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["all_tasks_meet_target"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
