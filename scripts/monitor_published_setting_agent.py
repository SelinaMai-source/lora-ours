#!/usr/bin/env python3
"""Monitor published-setting experiments and wake the Cursor agent on changes.

The script is intentionally local and conservative:
- it never kills or restarts training jobs;
- it writes concise status into docs/;
- it queues actionable issues for the agent instead of dumping raw logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATUS_CSV = ROOT / "results" / "tables" / "published_setting_single_seed_status.csv"
MANIFEST_CSV = ROOT / "results" / "tables" / "published_setting_single_seed_manifest.csv"
STATE_DIR = ROOT / ".cursor" / "experiment-monitor"
STATE_PATH = STATE_DIR / "published_setting_state.json"
ERROR_QUEUE = ROOT / ".cursor" / "error-autofix" / "queue.jsonl"
LATEST_ISSUE = ROOT / ".cursor" / "error-autofix" / "latest_experiment_issue.md"
REPORT_PATH = ROOT / "docs" / "published_setting_status.md"
TMUX_SESSION = "lora_published_setting"
WANDB_PROJECT = "lora-published-setting"
STALL_SECONDS = 90 * 60


@dataclass
class Snapshot:
    rows: list[dict[str, str]]
    counts: dict[str, int]
    running: list[dict[str, str]]
    failed: list[dict[str, str]]
    completed: list[dict[str, str]]
    tmux_alive: bool
    train_alive: bool
    stalled_running: list[dict[str, Any]]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_quiet(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _tmux_alive() -> bool:
    return _run_quiet(["tmux", "has-session", "-t", TMUX_SESSION]).returncode == 0


def _train_alive() -> bool:
    proc = _run_quiet(["pgrep", "-f", r"core/train.py --config configs/paper/published_setting"])
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _snapshot() -> Snapshot:
    rows = _load_csv(STATUS_CSV)
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "") or "unknown"
        counts[status] = counts.get(status, 0) + 1

    running = [r for r in rows if r.get("status") == "running"]
    failed = [r for r in rows if r.get("status") == "failed"]
    completed = [r for r in rows if r.get("status") == "completed"]
    stalled: list[dict[str, Any]] = []
    now = time.time()
    for row in running:
        log_file = row.get("log_file", "")
        if not log_file:
            continue
        path = ROOT / log_file
        if path.exists():
            age = now - path.stat().st_mtime
            if age > STALL_SECONDS:
                stalled.append({**row, "log_silent_seconds": int(age)})
    return Snapshot(
        rows=rows,
        counts=counts,
        running=running,
        failed=failed,
        completed=completed,
        tmux_alive=_tmux_alive(),
        train_alive=_train_alive(),
        stalled_running=stalled,
    )


def _compact_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "no status rows"
    order = ["completed", "running", "queued", "failed", "skipped_existing", "unknown"]
    parts = [f"{k}={counts[k]}" for k in order if k in counts]
    parts.extend(f"{k}={v}" for k, v in sorted(counts.items()) if k not in order)
    return ", ".join(parts)


def _write_report(snap: Snapshot) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    current = snap.running[0].get("run_name", "none") if snap.running else "none"
    lines = [
        "# Published-Setting Experiment Status",
        "",
        f"- Last check: `{_now()}`",
        f"- W&B project: `{WANDB_PROJECT}`",
        f"- tmux session: `{TMUX_SESSION}` ({'alive' if snap.tmux_alive else 'not running'})",
        f"- active train process: `{'yes' if snap.train_alive else 'no'}`",
        f"- status: `{_compact_counts(snap.counts)}`",
        f"- current run: `{current}`",
        "",
        "## Runs",
        "",
        "| run | benchmark | method | status | log |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in snap.rows:
        lines.append(
            "| {run} | {benchmark} | {method} | {status} | {log} |".format(
                run=row.get("run_name", ""),
                benchmark=row.get("benchmark", ""),
                method=row.get("method", ""),
                status=row.get("status", ""),
                log=row.get("log_file", ""),
            )
        )
    if snap.failed:
        lines.extend(["", "## Needs Agent Attention", ""])
        for row in snap.failed:
            lines.append(f"- Failed: `{row.get('run_name', '')}`; log: `{row.get('log_file', '')}`")
    if snap.stalled_running:
        lines.extend(["", "## Possible Stalls", ""])
        for row in snap.stalled_running:
            minutes = int(row.get("log_silent_seconds", 0)) // 60
            lines.append(f"- Silent for ~{minutes} min: `{row.get('run_name', '')}`")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _queue_issue(kind: str, summary: str, details: dict[str, Any]) -> None:
    ERROR_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "pending",
        "source": "published_setting_monitor",
        "kind": kind,
        "summary": summary,
        "details": details,
    }
    with ERROR_QUEUE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    LATEST_ISSUE.write_text(
        "# Latest Published-Setting Monitor Issue\n\n"
        f"- Time: `{_now()}`\n"
        f"- Kind: `{kind}`\n"
        f"- Summary: {summary}\n"
        f"- Status report: `{REPORT_PATH.relative_to(ROOT)}`\n",
        encoding="utf-8",
    )


def _emit(kind: str, payload: dict[str, Any]) -> None:
    print(f"AGENT_LOOP_{kind}_PUBLISHED_SETTING {json.dumps(payload, ensure_ascii=False, sort_keys=True)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heartbeat", action="store_true", help="Emit an update even if nothing changed.")
    args = parser.parse_args()

    snap = _snapshot()
    _write_report(snap)

    state = _load_state()
    previous_signature = state.get("signature")
    signature = {
        "counts": snap.counts,
        "running": [r.get("run_name", "") for r in snap.running],
        "tmux_alive": snap.tmux_alive,
        "train_alive": snap.train_alive,
    }
    alerted = set(state.get("alerted", []))

    issue_payloads: list[tuple[str, str, dict[str, Any]]] = []
    for row in snap.failed:
        key = f"failed:{row.get('run_name', '')}:{row.get('exit_code', '')}"
        if key not in alerted:
            issue_payloads.append(("WAKE", f"Published-setting run failed: {row.get('run_name', '')}", dict(row)))
            alerted.add(key)
    if snap.rows and not snap.tmux_alive and not snap.train_alive:
        unfinished = [r for r in snap.rows if r.get("status") in {"queued", "running"}]
        if unfinished:
            key = "stopped-with-unfinished"
            if key not in alerted:
                issue_payloads.append(
                    (
                        "WAKE",
                        "Published-setting queue stopped while runs remain unfinished.",
                        {"unfinished": [r.get("run_name", "") for r in unfinished]},
                    )
                )
                alerted.add(key)
    for row in snap.stalled_running:
        key = f"stalled:{row.get('run_name', '')}"
        if key not in alerted:
            issue_payloads.append(
                ("WAKE", f"Published-setting run may be stalled: {row.get('run_name', '')}", dict(row))
            )
            alerted.add(key)

    for kind, summary, details in issue_payloads:
        _queue_issue(kind.lower(), summary, details)
        _emit(
            "WAKE",
            {
                "prompt": "检查 published-setting 实验失败/卡住原因，先自动诊断和修复，不要把原始报错直接发给用户。",
                "summary": summary,
                "report": str(REPORT_PATH.relative_to(ROOT)),
            },
        )

    changed = previous_signature != signature
    if changed or args.heartbeat:
        _emit(
            "UPDATE",
            {
                "prompt": "向用户简短汇报 published-setting 实验最新状态。",
                "status": _compact_counts(snap.counts),
                "current": snap.running[0].get("run_name", "none") if snap.running else "none",
                "report": str(REPORT_PATH.relative_to(ROOT)),
            },
        )

    state.update({"signature": signature, "alerted": sorted(alerted), "last_check": _now()})
    _save_state(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
