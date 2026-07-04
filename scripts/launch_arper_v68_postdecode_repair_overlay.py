#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUN_ID = "arper_woz3_official_sclstm_v66_published_base_plus_ours_postdecode_repair_overlay_v68"
SESSION = "arper-woz3-v68-postdecode-repair-overlay"
LOG_DIR = REPO / "results/logs"
PREFLIGHT_JSON = LOG_DIR / f"{RUN_ID}_preflight.json"
RUN_LOG = LOG_DIR / f"{RUN_ID}.log"
STATUS_JSON = LOG_DIR / f"{RUN_ID}_status.json"
STATUS_MD = LOG_DIR / f"{RUN_ID}_status.md"
BASE_CONFIG = REPO / "results/logs/arper_woz3_official_sclstm_formal_v66.cfg"
CHECKPOINT = (
    REPO
    / "results/runs/arper_woz3_official_sclstm_formal_v66/1111/"
    / "exemplar_ewc_loss_250_formal_v66_0.5_300000.0/lm/model/"
    / "exemplar_ewc_loss_250_formal_v66_0.5_300000.0_1706428.pt"
)
ARPER_ROOT = REPO / "baselines/advanced_baselines/arper_dialog_nlg/external"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_text(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return (proc.stdout + proc.stderr).strip()


def gpu_compute_processes() -> str:
    return run_text(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])


def write_preflight(payload: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch ARPER v68 post-decode repair overlay in tmux.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    tmux_ls = run_text(["tmux", "list-sessions"])
    gpu = gpu_compute_processes()
    checks = {
        "base_config_exists": BASE_CONFIG.exists(),
        "checkpoint_exists": CHECKPOINT.exists(),
        "official_arper_root_exists": ARPER_ROOT.exists(),
        "tmux_session_absent": SESSION not in tmux_ls,
        "gpu_compute_empty": not gpu.strip(),
        "wandb": "not_used_by_official_ARPER_Path_B; overlay logs local status only",
        "label": "published-base + ours overlay; inference-time repair",
        "ground_truth_changed": False,
        "scoring_changed": False,
    }
    required_checks = [
        "base_config_exists",
        "checkpoint_exists",
        "official_arper_root_exists",
        "tmux_session_absent",
        "gpu_compute_empty",
    ]
    payload = {
        "updated_at": now(),
        "run_id": RUN_ID,
        "session": SESSION,
        "run_log": str(RUN_LOG.relative_to(REPO)),
        "status_json": str(STATUS_JSON.relative_to(REPO)),
        "status_md": str(STATUS_MD.relative_to(REPO)),
        "checks": checks,
        "tmux": tmux_ls,
        "gpu_compute": gpu,
        "max_batches": args.max_batches,
    }
    blocked = [name for name in required_checks if not checks.get(name)]
    if blocked and not args.force:
        payload["state"] = "blocked"
        payload["blocked_checks"] = blocked
        write_preflight(payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        return 75

    if args.dry_run:
        payload["state"] = "dry_run_ready" if not blocked else "dry_run_with_force_needed"
        payload["blocked_checks"] = blocked
        write_preflight(payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        return 0

    command = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        SESSION,
        "-n",
        "repair",
        (
            f"cd {REPO} && "
            "PYTHONUNBUFFERED=1 WANDB_MODE=online "
            "python scripts/arper_woz3_v68_postdecode_repair_overlay.py "
            f"--base-config {BASE_CONFIG} "
            f"--status-json {STATUS_JSON} "
            f"--status-md {STATUS_MD} "
            + (f"--max-batches {args.max_batches} " if args.max_batches is not None else "")
            + f"> {RUN_LOG} 2>&1"
        ),
    ]
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    payload["state"] = "launched" if proc.returncode == 0 else "launch_failed"
    payload["tmux_command"] = command[-1]
    payload["returncode"] = proc.returncode
    payload["stdout"] = proc.stdout
    payload["stderr"] = proc.stderr
    write_preflight(payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
