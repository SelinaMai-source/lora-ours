#!/usr/bin/env python3
"""Ours v10 campaign monitor — write status report every 10 minutes."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/logs/ours_v10_status_report.md"
GAP_JSON = REPO / "results/tables/ours_v10_sota_gap_s123.json"
INTERVAL_SEC = 600


def _tmux_sessions() -> list[str]:
    try:
        out = subprocess.check_output(["tmux", "ls"], text=True, stderr=subprocess.DEVNULL)
        return [ln.split(":")[0] for ln in out.strip().splitlines() if ln]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _train_pid() -> str:
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "core/train.py.*ours_v10"],
            text=True,
        ).strip()
        if out:
            return out.split("\n")[0].strip()
    except subprocess.CalledProcessError:
        pass
    return "none"


def _parse_log_tail() -> dict:
    logs = sorted(
        (REPO / "results/logs").glob("ours_v10*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not logs:
        return {}
    text = logs[0].read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    seg_line = next((ln for ln in reversed(lines) if "=== Segment" in ln), "")
    m_seg = re.search(r"Segment (\d+):", seg_line)
    seen_m = re.search(r'"seen_avg_score":\s*([\d.]+)', text[::-1])
    # search from end for last eval metrics
    seen = ta = f1 = None
    for ln in reversed(lines[-200:]):
        if '"seen_avg_score"' in ln:
            m = re.search(r'"seen_avg_score":\s*([\d.]+)', ln)
            if m:
                seen = m.group(1)
        if '"seen_avg_task_aware_score"' in ln and ta is None:
            m = re.search(r'"seen_avg_task_aware_score":\s*([\d.]+)', ln)
            if m:
                ta = m.group(1)
        if '"token_f1_mean"' in ln and f1 is None:
            m = re.search(r'"token_f1_mean":\s*([\d.]+)', ln)
            if m:
                f1 = m.group(1)
    return {
        "log": str(logs[0].relative_to(REPO)),
        "segment": m_seg.group(1) if m_seg else "?",
        "seen": seen,
        "ta": ta,
        "f1": f1,
    }


def write_report() -> None:
    subprocess.run(
        ["python3", "scripts/ours_v10_sota_gap_audit.py"],
        cwd=str(REPO),
        check=False,
    )
    gap = {}
    if GAP_JSON.is_file():
        gap = json.loads(GAP_JSON.read_text(encoding="utf-8"))
    sessions = _tmux_sessions()
    train = _parse_log_tail()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Ours v10 Status Report",
        "",
        f"**Updated**: {now}",
        "",
        "## tmux",
        "",
        f"- `lora_ours_v10_train`: {'🟢' if 'lora_ours_v10_train' in sessions else '—'}",
        f"- `lora_ours_v10_monitor`: {'🟢' if 'lora_ours_v10_monitor' in sessions else '—'}",
        "",
        "## Training",
        "",
        f"- PID/cmd: `{_train_pid()}`",
        f"- Log: `{train.get('log', '—')}`",
        f"- Segment: **{train.get('segment', '?')}**",
        f"- Latest seen/ta/f1: {train.get('seen', '?')} / {train.get('ta', '?')} / {train.get('f1', '?')}",
        "",
        "## SOTA gap",
        "",
        f"- Weakest benchmark: **{gap.get('weakest_benchmark', '?')}**",
        f"- Total fail cells: **{gap.get('total_fail_cells', '?')}** / 20",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


def main() -> None:
    import time

    while True:
        write_report()
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
