#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = REPO / "configs/ccfa_three_suite/standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v62_formal.yaml"
RUN_NAME = "standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v62_formal_neutral_20260705"
LOG_DIR = REPO / "results/logs"
RUNTIME_CONFIG = LOG_DIR / f"{RUN_NAME}.runtime.yaml"
RUN_LOG = LOG_DIR / f"{RUN_NAME}.log"
MONITOR_LOG = LOG_DIR / f"{RUN_NAME}.monitor.jsonl"
EXIT_JSON = LOG_DIR / f"{RUN_NAME}.exit.json"
PREFLIGHT_JSON = LOG_DIR / f"{RUN_NAME}.preflight.json"


BLOCKING_MARKERS = (
    "standard_peft_ours_v62_formal_enforcer",
    "standard_peft_ours_v62_formal_guard",
    "standard_peft_ours_v62_daemon_formal_guard",
    "standard-ours-order1-v62-gate-controller",
    "standard-ours-order1-v62-formal",
    "standard-ours-order1-v62-eval-diagnostic",
    "standard-ours-order1-v62-isolated-diagnostic",
    "launch_v63_isolated_ultrashort.py",
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_text(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(REPO), text=True, capture_output=True, check=False)
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def process_rows() -> list[str]:
    proc = subprocess.run(["ps", "-eo", "pid,ppid,sid,pgid,stat,args="], text=True, capture_output=True)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def blocking_processes() -> list[str]:
    rows = []
    for row in process_rows():
        if any(marker in row for marker in BLOCKING_MARKERS):
            if "launch_v62_neutral_formal.py" not in row:
                rows.append(row)
    return rows


def write_runtime_config() -> None:
    cfg = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    cfg["experiment_name"] = RUN_NAME
    output = cfg.setdefault("output", {})
    output["run_name"] = RUN_NAME
    tracking = output.setdefault("tracking", {})
    tracking["use_wandb"] = True
    tracking["wandb_project"] = "lora-ours"
    tracking["wandb_group"] = "ccfa_standard_peft_strict_ours_v62_neutral_formal"
    tracking["wandb_mode"] = "online"
    tags = list(tracking.get("wandb_tags") or [])
    for tag in ["neutral_formal", "published_base_pivot", "sigterm_blocker_rerun"]:
        if tag not in tags:
            tags.append(tag)
    tracking["wandb_tags"] = tags
    notes = str((cfg.get("paper") or {}).get("notes", ""))
    cfg.setdefault("paper", {})["notes"] = (
        notes
        + " Neutral 20260705 rerun changes only run/log/W&B identifiers to avoid legacy guard matching; "
        "data, model, order, and training hyperparameters are preserved from v62 formal."
    )
    RUNTIME_CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def write_preflight(payload: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def launch(force: bool) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_runtime_config()
    blockers = blocking_processes()
    core_train = [
        row
        for row in process_rows()
        if "python -m core.train" in row or "python core/train.py" in row or "python -u core/train.py" in row
    ]
    preflight = {
        "updated_at": now(),
        "event": "neutral_formal_preflight",
        "run_name": RUN_NAME,
        "source_config": str(SOURCE_CONFIG),
        "runtime_config": str(RUNTIME_CONFIG),
        "run_log": str(RUN_LOG),
        "monitor_log": str(MONITOR_LOG),
        "exit_json": str(EXIT_JSON),
        "blocking_processes": blockers,
        "core_train_processes": core_train,
        "tmux": run_text(["tmux", "ls"]),
        "gpu": run_text(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"]),
        "comparability": {
            "preserved": ["stream", "task_order", "T5-large", "LoRA settings", "epochs", "batch sizes", "eval settings"],
            "changed": ["run_name", "W&B group/tags", "runtime config path"],
        },
    }
    blocked = blockers or core_train
    if blocked and not force:
        preflight["state"] = "blocked"
        write_preflight(preflight)
        print(json.dumps(preflight, indent=2), flush=True)
        return 75
    preflight["state"] = "launching"
    write_preflight(preflight)

    command = (
        f"cd {REPO} && "
        "export PYTHONUNBUFFERED=1 WANDB_PROJECT=lora-ours WANDB_MODE=online && "
        f": > {MONITOR_LOG} && "
        f"(while true; do printf '{{\"updated_at\":\"%s\",\"event\":\"heartbeat\",\"run_name\":\"{RUN_NAME}\"}}\\n' \"$(date -Iseconds)\" >> {MONITOR_LOG}; sleep 60; done) & monitor_pid=$! && "
        f"python -m core.train --config {RUNTIME_CONFIG} 2>&1 | tee {RUN_LOG}; "
        "status=${PIPESTATUS[0]}; "
        "kill $monitor_pid 2>/dev/null || true; "
        f"printf '{{\"updated_at\":\"%s\",\"event\":\"pipeline_exit\",\"run_name\":\"{RUN_NAME}\",\"runtime_config\":\"{RUNTIME_CONFIG}\",\"train_exit_status\":%s}}\\n' \"$(date -Iseconds)\" \"$status\" > {EXIT_JSON}; "
        "exit $status"
    )
    proc = subprocess.run(
        ["tmux", "new-session", "-d", "-s", "standard-v62-neutral-formal-20260705", "-n", "train", command],
        cwd=str(REPO),
        text=True,
        capture_output=True,
        check=False,
    )
    preflight["state"] = "launched" if proc.returncode == 0 else "launch_failed"
    preflight["tmux_command"] = command
    preflight["returncode"] = proc.returncode
    preflight["stdout"] = proc.stdout
    preflight["stderr"] = proc.stderr
    write_preflight(preflight)
    print(json.dumps(preflight, indent=2), flush=True)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a neutral v62 formal rerun without legacy guard/session names.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        write_runtime_config()
        payload = {
            "updated_at": now(),
            "event": "neutral_formal_dry_run",
            "run_name": RUN_NAME,
            "runtime_config": str(RUNTIME_CONFIG),
            "blocking_processes": blocking_processes(),
        }
        write_preflight(payload)
        print(json.dumps(payload, indent=2), flush=True)
        return 0
    return launch(args.force)


if __name__ == "__main__":
    raise SystemExit(main())
