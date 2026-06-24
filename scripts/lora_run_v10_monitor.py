#!/usr/bin/env python3
"""Monitor lora-run_v10 paper-aligned queue and write a compact status report."""
from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "results/tables/lora_run_v10_manifest.csv"
STATUS = REPO / "results/tables/lora_run_v10_status.csv"
OUT = REPO / "results/logs/lora_run_v10_status_report.md"
INTERVAL_SEC = 300


def _tmux_sessions() -> list[str]:
    try:
        out = subprocess.check_output(["tmux", "ls"], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [line.split(":", 1)[0] for line in out.splitlines() if line.strip()]


def _active_processes() -> list[str]:
    patterns = "run_lora_run_v10_queue.sh|run_lfpt5_published_setting.py|citb_bridge/run_continual.py"
    try:
        out = subprocess.check_output(["pgrep", "-af", patterns], text=True)
    except subprocess.CalledProcessError:
        return []
    return [
        line.strip()
        for line in out.splitlines()
        if "pgrep -af" not in line and "lora_run_v10_monitor.py" not in line
    ]


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _latest_log(run_name: str, rows: list[dict[str, str]]) -> Path | None:
    for row in rows:
        if row.get("run_name") == run_name and row.get("log_file"):
            p = REPO / row["log_file"]
            if p.is_file():
                return p
    candidates = sorted(
        (REPO / "results/logs/lora_run_v10").glob(f"{run_name}.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _log_progress(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    segment_line = next(
        (line for line in reversed(lines) if "=== LFPT5 segment" in line or "=== Segment" in line),
        "",
    )
    segment = "?"
    match = re.search(r"segment\s+(\d+)", segment_line, flags=re.IGNORECASE)
    if match:
        segment = match.group(1)
    wandb_url = next((line.strip() for line in lines if "https://wandb.ai/" in line), "")
    return {
        "log": str(path.relative_to(REPO)),
        "segment": segment,
        "last_event": segment_line.strip(),
        "wandb": wandb_url,
    }


def _run_manifest(run_name: str) -> dict[str, object]:
    path = REPO / "results/runs" / run_name / "run_manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_report() -> None:
    manifest_rows = _rows(MANIFEST)
    status_rows = _rows(STATUS) or manifest_rows
    active = next((row for row in status_rows if row.get("status") == "running"), None)
    run_name = active.get("run_name", "") if active else ""
    log_info = _log_progress(_latest_log(run_name, status_rows)) if run_name else {}
    run_manifest = _run_manifest(run_name) if run_name else {}
    sessions = _tmux_sessions()
    processes = _active_processes()
    final_path = REPO / "results/runs" / run_name / "final_metrics.json" if run_name else None

    counts: dict[str, int] = {}
    for row in status_rows:
        counts[row.get("status", "unknown")] = counts.get(row.get("status", "unknown"), 0) + 1

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# lora-run_v10 Status Report",
        "",
        f"Updated: {now} CST",
        "",
        "## Active Run",
        "",
        f"- run: `{run_name or '-'}`",
        f"- status counts: `{counts}`",
        f"- tmux `lora_run_v10_queue`: {'alive' if 'lora_run_v10_queue' in sessions else 'missing'}",
        f"- tmux `lora_run_v10_monitor`: {'alive' if 'lora_run_v10_monitor' in sessions else 'missing'}",
        f"- process count: {len(processes)}",
        f"- final metrics: {'present' if final_path and final_path.is_file() else 'not yet'}",
        "",
        "## Progress",
        "",
        f"- latest log: `{log_info.get('log', '-')}`",
        f"- latest segment: `{log_info.get('segment', '?')}`",
        f"- latest event: `{log_info.get('last_event', '-')}`",
        f"- W&B: {log_info.get('wandb', '-')}",
        f"- checkpoint ready: `{run_manifest.get('checkpoint_ready', '-')}`",
        f"- exported segments: `{run_manifest.get('n_segments', '-')}`",
        "",
        "## Guardrail",
        "",
        "- Do not treat unified-entry O-LoRA/LB-CL/Progressive Prompts/Continual-T0 scaffold results as strict paper-aligned outputs.",
        "- Do not edit published_setting_run_v2 gap manifest/status files from this monitor.",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


def main() -> None:
    while True:
        write_report()
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
