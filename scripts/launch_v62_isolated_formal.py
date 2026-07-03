#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUN_NAME = "standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v62_formal"
CONFIG = "configs/ccfa_three_suite/standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v62_formal.yaml"
RUNTIME_CONFIG = Path("/tmp/lora_ours_approved_standard_order1.yaml")
LOG_DIR = REPO / "results/logs"
ARTIFACT_PATH = LOG_DIR / f"{RUN_NAME}.isolated_launcher.json"
PID_PATH = LOG_DIR / f"{RUN_NAME}.pid"
SUPERVISOR_LOG = LOG_DIR / f"{RUN_NAME}.isolated_launcher.log"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_text(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), text=True, capture_output=True, timeout=15)
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": repr(exc)}


def write_artifact(payload: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def core_train_processes() -> list[str]:
    proc = subprocess.run(["ps", "-eo", "pid,ppid,sid,pgid,stat,args="], text=True, capture_output=True)
    rows = []
    for line in proc.stdout.splitlines():
        if "python -m core.train" in line or "python core/train.py" in line or "python -u core/train.py" in line:
            rows.append(line.strip())
    return rows


def isolated_child() -> None:
    os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)


def launch(config: str, force: bool) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source_config = (REPO / config).resolve() if not Path(config).is_absolute() else Path(config)
    RUNTIME_CONFIG.write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
    existing_train = core_train_processes()
    if existing_train and not force:
        write_artifact(
            {
                "updated_at": now(),
                "event": "blocked_existing_core_train",
                "run_name": RUN_NAME,
                "existing_core_train": existing_train,
            }
        )
        print("Refusing isolated formal launch while core.train is already running.", file=sys.stderr)
        return 75

    env = os.environ.copy()
    env.update(
        {
            "ALLOW_V62_FORMAL": "1",
            "PYTHONUNBUFFERED": "1",
            "WANDB_MODE": "online",
            "V62_ISOLATED_FORMAL": "1",
        }
    )
    stdout = SUPERVISOR_LOG.open("ab", buffering=0)
    command = ["bash", "scripts/run_ours_v1_strict_iteration.sh", str(RUNTIME_CONFIG)]
    proc = subprocess.Popen(
        command,
        cwd=str(REPO),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        preexec_fn=isolated_child,
        close_fds=True,
    )
    PID_PATH.write_text(f"{proc.pid}\n", encoding="utf-8")
    payload = {
        "updated_at": now(),
        "event": "isolated_formal_child_started",
        "run_name": RUN_NAME,
        "config": str(RUNTIME_CONFIG),
        "source_config": str(source_config),
        "launcher_pid": os.getpid(),
        "launcher_ppid": os.getppid(),
        "launcher_sid": os.getsid(0),
        "launcher_pgid": os.getpgid(0),
        "child_pid": proc.pid,
        "child_sid": os.getsid(proc.pid),
        "child_pgid": os.getpgid(proc.pid),
        "pid_path": str(PID_PATH),
        "supervisor_log": str(SUPERVISOR_LOG),
        "train_log": str(LOG_DIR / f"{RUN_NAME}.log"),
        "run_exit_artifact": str(LOG_DIR / f"{RUN_NAME}.exit.json"),
        "tmux": run_text(["tmux", "ls"]),
        "gpu": run_text(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "gpu_compute": run_text(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ]
        ),
    }
    write_artifact(payload)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch v62 formal without inheriting tmux/session SIGHUP.")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--force", action="store_true", help="launch even if another core.train process is visible")
    args = parser.parse_args()
    return launch(args.config, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
