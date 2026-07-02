#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
RUN_ID = "citb_instrdialog_order1_seed1_ours_v18_smoke_strict"
LOG_PATH = REPO / "results" / "logs" / f"{RUN_ID}.log"
RUN_DIR = REPO / "results" / "runs" / RUN_ID
STATUS_JSON = REPO / "results" / "logs" / "ours_v18_strict_status.json"
STATUS_MD = REPO / "results" / "logs" / "ours_v18_strict_status.md"

STALE_SECONDS = 30 * 60
LOW_SCORE_MIN_SEGMENT = 3
LOW_SEEN_AVG_FLOOR = 0.10
LOW_TASK_AWARE_FLOOR = 0.10


def _run(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(REPO), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def _tmux_summary() -> Dict[str, Any]:
    out = _run(["tmux", "list-panes", "-a", "-F", "#{session_name}:#{window_index}.#{pane_index} #{pane_current_command} #{pane_current_path}"])
    panes = [line for line in out.splitlines() if line and not line.startswith("ERROR:")]
    return {
        "panes": panes,
        "v18_panes": [line for line in panes if "lora-ours" in line or "ours_v18" in line],
    }


def _train_processes() -> List[str]:
    out = _run(["pgrep", "-af", "core.train|core/train.py"])
    if out.startswith("ERROR:"):
        return []
    return [line for line in out.splitlines() if "ours_v18" in line or RUN_ID in line]


def _last_eval_from_log() -> Dict[str, Any]:
    if not LOG_PATH.is_file():
        return {}
    text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    latest_segment: Optional[int] = None
    for line in reversed(lines):
        match = re.search(r"=== Segment (\d+):", line)
        if match:
            latest_segment = int(match.group(1))
            break
    latest_eval: Dict[str, Any] = {}
    for line in reversed(lines):
        if "Eval metrics:" not in line:
            continue
        payload = line.split("Eval metrics:", 1)[1].strip()
        try:
            latest_eval = json.loads(payload)
        except json.JSONDecodeError:
            latest_eval = {"parse_error": payload[:500]}
        break
    return {
        "latest_segment": latest_segment,
        "latest_eval": latest_eval,
        "log_mtime": LOG_PATH.stat().st_mtime,
        "log_age_seconds": max(0.0, time.time() - LOG_PATH.stat().st_mtime),
        "log_path": str(LOG_PATH),
    }


def _run_artifacts() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in ["final_metrics.json", "stop_and_diagnose.json"]:
        path = RUN_DIR / name
        if path.is_file():
            try:
                out[name] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                out[name] = {"error": repr(exc)}
    return out


def _classify(log_info: Dict[str, Any], artifacts: Dict[str, Any], processes: List[str]) -> Dict[str, Any]:
    if "stop_and_diagnose.json" in artifacts:
        return {"state": "stopped_low_or_failed", "reason": artifacts["stop_and_diagnose.json"].get("reason", "stop_and_diagnose")}
    if "final_metrics.json" in artifacts:
        return {"state": "completed", "reason": "final_metrics_present"}
    if LOG_PATH.is_file() and log_info.get("log_age_seconds", 0.0) > STALE_SECONDS and processes:
        return {"state": "stale", "reason": f"log older than {STALE_SECONDS}s while process exists"}
    latest_eval = log_info.get("latest_eval", {}) if isinstance(log_info.get("latest_eval"), dict) else {}
    latest_segment = int(log_info.get("latest_segment") or -1)
    if latest_segment >= LOW_SCORE_MIN_SEGMENT:
        seen = float(latest_eval.get("seen_avg_score", 0.0) or 0.0)
        task_seen = float(latest_eval.get("seen_avg_task_aware_score", seen) or 0.0)
        if seen < LOW_SEEN_AVG_FLOOR or task_seen < LOW_TASK_AWARE_FLOOR:
            return {
                "state": "low_score_gate",
                "reason": f"segment {latest_segment}: seen={seen:.4f}, task_aware={task_seen:.4f}",
            }
    if processes:
        return {"state": "running", "reason": "v18 train process present"}
    if LOG_PATH.is_file():
        return {"state": "unknown_not_running", "reason": "log exists but process/final status absent"}
    return {"state": "not_started", "reason": "no v18 log found"}


def build_status() -> Dict[str, Any]:
    tmux = _tmux_summary()
    processes = _train_processes()
    log_info = _last_eval_from_log()
    artifacts = _run_artifacts()
    classification = _classify(log_info, artifacts, processes)
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": RUN_ID,
        "classification": classification,
        "processes": processes,
        "tmux": tmux,
        "log": log_info,
        "artifacts": artifacts,
    }


def write_status() -> None:
    status = build_status()
    STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cls = status["classification"]
    latest_eval = status.get("log", {}).get("latest_eval", {})
    lines = [
        "# Ours v18 Strict Status",
        "",
        f"- Updated: `{status['updated_at']}`",
        f"- Run: `{status['run_id']}`",
        f"- State: `{cls.get('state')}`",
        f"- Reason: `{cls.get('reason')}`",
        f"- Latest segment: `{status.get('log', {}).get('latest_segment', 'n/a')}`",
        f"- Latest seen/task-aware: `{latest_eval.get('seen_avg_score', 'n/a')}` / `{latest_eval.get('seen_avg_task_aware_score', 'n/a')}`",
        f"- Train processes: `{len(status.get('processes', []))}`",
        f"- Log: `{LOG_PATH}`",
    ]
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(status["classification"], ensure_ascii=False))


def main() -> None:
    while True:
        write_status()
        time.sleep(300)


if __name__ == "__main__":
    main()
