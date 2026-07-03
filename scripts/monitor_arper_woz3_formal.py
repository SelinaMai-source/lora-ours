#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def run_text(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return proc.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor an ARPER WOZ3 official SCLSTM run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--status-basename", required=True)
    args = parser.parse_args()

    log_path = Path(args.log_path)
    status_json = Path("results/logs") / f"{args.status_basename}.json"
    status_md = Path("results/logs") / f"{args.status_basename}.md"
    status_json.parent.mkdir(parents=True, exist_ok=True)

    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    lines = text.splitlines()
    proc_lines = [
        line
        for line in run_text(["pgrep", "-af", f"run_woz3.py.*{args.run_id}"]).splitlines()
        if "pgrep -af" not in line and "monitor_arper_woz3_formal.py" not in line
    ]
    state = "running" if proc_lines else ("completed_or_stopped" if text else "not_started")
    signal_lines = [
        line
        for line in lines[-120:]
        if (
            "Current task:" in line
            or "Task " in line
            or "Train Loss:" in line
            or "valid Loss:" in line
            or "test Loss=" in line
            or "Slot error:" in line
            or "BLEU4:" in line
            or "Traceback" in line
            or "RuntimeError" in line
            or "SIGTERM" in line
            or "CUDA out of memory" in line
        )
    ]
    errors = [
        line
        for line in lines
        if any(token in line for token in ("Traceback", "RuntimeError", "SIGTERM", "CUDA out of memory", "Killed"))
    ]
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": args.run_id,
        "state": state,
        "processes": proc_lines,
        "log_path": str(log_path),
        "log_size": log_path.stat().st_size if log_path.exists() else 0,
        "last_signals": signal_lines[-30:],
        "error_count": len(errors),
        "last_errors": errors[-10:],
    }
    status_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    status_md.write_text(
        "\n".join(
            [
                "# ARPER WOZ3 Official SCLSTM Formal v66",
                "",
                f"- State: `{state}`",
                f"- Log: `{log_path}`",
                f"- Error count: `{len(errors)}`",
                "",
                "## Recent Signals",
                "",
                *[f"- `{line}`" for line in signal_lines[-12:]],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"state": state, "last": signal_lines[-3:], "errors": len(errors)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
