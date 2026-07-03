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

import yaml


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/ccfa_three_suite/standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v63_ultrashort_probe.yaml"
LOG_DIR = REPO / "results/logs"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def resolve_run_name(config: str) -> str:
    path = Path(config)
    if not path.is_absolute():
        path = REPO / path
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    output = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    return str(output.get("run_name") or cfg.get("experiment_name") or "v63_isolated_diagnostic")


def run_text(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), text=True, capture_output=True, timeout=15)
        return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": repr(exc)}


def core_train_processes() -> list[str]:
    proc = subprocess.run(["ps", "-eo", "pid,ppid,sid,pgid,stat,args="], text=True, capture_output=True)
    markers = ("python -m core.train", "python core/train.py", "python -u core/train.py")
    return [line.strip() for line in proc.stdout.splitlines() if any(marker in line for marker in markers)]


def isolated_child() -> None:
    os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)


def launch(config: str, force: bool) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_name = resolve_run_name(config)
    artifact_path = LOG_DIR / f"{run_name}.isolated_launcher.json"
    pid_path = LOG_DIR / f"{run_name}.pid"
    supervisor_log = LOG_DIR / f"{run_name}.isolated_launcher.log"
    existing_train = core_train_processes()
    if existing_train and not force:
        artifact_path.write_text(
            json.dumps(
                {
                    "updated_at": now(),
                    "event": "blocked_existing_core_train",
                    "run_name": run_name,
                    "existing_core_train": existing_train,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("Refusing isolated launch while core.train is already running.", file=sys.stderr)
        return 75

    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "WANDB_MODE": "online", "V63_ISOLATED_DIAGNOSTIC": "1"})
    stdout = supervisor_log.open("ab", buffering=0)
    proc = subprocess.Popen(
        ["bash", "scripts/run_ours_v1_strict_iteration.sh", config],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        preexec_fn=isolated_child,
        close_fds=True,
    )
    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    payload = {
        "updated_at": now(),
        "event": "isolated_child_started",
        "run_name": run_name,
        "config": config,
        "launcher_pid": os.getpid(),
        "launcher_ppid": os.getppid(),
        "launcher_sid": os.getsid(0),
        "launcher_pgid": os.getpgid(0),
        "child_pid": proc.pid,
        "child_sid": os.getsid(proc.pid),
        "child_pgid": os.getpgid(proc.pid),
        "pid_path": str(pid_path),
        "supervisor_log": str(supervisor_log),
        "train_log": str(LOG_DIR / f"{run_name}.log"),
        "run_exit_artifact": str(LOG_DIR / f"{run_name}.exit.json"),
        "tmux": run_text(["tmux", "ls"]),
        "gpu": run_text(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"]),
        "gpu_compute": run_text(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]),
    }
    artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a v63 diagnostic in an isolated session.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true", help="launch even if another core.train process is visible")
    args = parser.parse_args()
    return launch(args.config, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
