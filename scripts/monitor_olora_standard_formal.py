#!/usr/bin/env python3
"""Write a compact status file for O-LoRA standard formal runs."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "RuntimeError:",
    "CUDA out of memory",
    "ValueError:",
    "FORMAL_EXIT_CODE:",
)


def run_text(command: str) -> str:
    completed = subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip()


def read_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"state": "starting", "reason": ""}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"state": "manifest_error", "reason": str(exc)}


def read_tail(path: Path, max_bytes: int = 12000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def build_status(run_name: str, session: str) -> str:
    root = Path.cwd()
    manifest_path = root / "results" / "runs" / run_name / "run_manifest.json"
    log_path = root / "results" / "logs" / f"{run_name}.log"
    manifest = read_manifest(manifest_path)
    state = manifest.get("state", "unknown")
    reason = manifest.get("reason", "")

    train_processes = run_text("pgrep -af 'run_uie_lora|python .*official_runtime' || true")
    gpu = run_text("nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true")
    tmux_state = "present" if run_text(f"tmux has-session -t {session} 2>/dev/null && echo present || true") else "absent"

    log_age = None
    if log_path.exists():
        log_age = max(0.0, datetime.now().timestamp() - log_path.stat().st_mtime)

    tail = read_tail(log_path)
    marker = next((item for item in FAILURE_MARKERS if item.lower() in tail.lower()), "")
    sentinel = ""
    if state in {"failed", "blocked"} or marker:
        sentinel = f"AGENT_LOOP_WAKE_LORA_OURS standard_olora_formal_failed state={state} marker={marker} reason={reason}"
    elif log_age is not None and log_age > 1800 and train_processes:
        sentinel = f"AGENT_LOOP_WAKE_LORA_OURS standard_olora_formal_stalled log_age_seconds={log_age:.0f}"
    elif tmux_state == "absent" and state not in {"completed", "failed", "blocked"}:
        sentinel = f"AGENT_LOOP_WAKE_LORA_OURS standard_olora_formal_stopped_incomplete state={state}"

    lines = [
        f"# {run_name} monitor",
        "",
        f"- updated: {datetime.now().isoformat(timespec='seconds')}",
        f"- state: {state}",
        f"- reason: {reason}",
        f"- tmux: {tmux_state}",
        f"- gpu: {gpu}",
        f"- log_age_seconds: {log_age}",
        f"- log_marker: {marker}",
        "- train_processes:",
        train_processes or "none",
        "",
    ]
    if sentinel:
        lines.append(sentinel)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--session", default="olora-standard-official-base-formal")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    status = build_status(args.run_name, args.session)
    Path(args.out).write_text(status, encoding="utf-8")


if __name__ == "__main__":
    main()
