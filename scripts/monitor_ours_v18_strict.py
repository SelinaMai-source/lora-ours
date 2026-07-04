#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("OURS_MONITOR_RUN_ID", "citb_instrdialog_order1_seed1_ours_v18_smoke_strict")
STATUS_BASENAME = os.environ.get("OURS_MONITOR_STATUS_BASENAME", "ours_v18_strict_status")
STATUS_TITLE = os.environ.get("OURS_MONITOR_STATUS_TITLE", "Ours v18 Strict Status")
LOG_PATH = REPO / "results" / "logs" / f"{RUN_ID}.log"
RUN_DIR = Path(os.environ.get("OURS_MONITOR_RUN_DIR", str(REPO / "results" / "runs" / RUN_ID)))
STATUS_JSON = REPO / "results" / "logs" / f"{STATUS_BASENAME}.json"
STATUS_MD = REPO / "results" / "logs" / f"{STATUS_BASENAME}.md"
LOG_CANDIDATES = [
    LOG_PATH,
    RUN_DIR / "wandb" / "latest-run" / "files" / "output.log",
]

STALE_SECONDS = 30 * 60
MONITOR_INTERVAL_SECONDS = 60
LOW_SCORE_MIN_SEGMENT = 3
LOW_SEEN_AVG_FLOOR = 0.10
LOW_TASK_AWARE_FLOOR = 0.10
STANDARD_ORDER1_TASKS = ["dbpedia", "amazon", "yahoo", "agnews"]


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
        "matching_panes": [line for line in panes if "lora-ours" in line or "ours_v" in line],
    }


def _train_processes() -> List[str]:
    out = _run(["pgrep", "-af", "core.train|core/train.py"])
    if out.startswith("ERROR:"):
        return []
    train_markers = ("python -m core.train", "python core/train.py", "python -u core/train.py")
    return [
        line
        for line in out.splitlines()
        if RUN_ID in line and any(marker in line for marker in train_markers)
    ]


def _last_eval_from_log() -> Dict[str, Any]:
    log_paths = [path for path in LOG_CANDIDATES if path.is_file()]
    log_paths.extend(sorted((RUN_DIR / "wandb").glob("run-*/files/output.log")))
    seen_paths = []
    unique_log_paths = []
    for path in log_paths:
        real_path = path.resolve()
        if real_path in seen_paths:
            continue
        seen_paths.append(real_path)
        unique_log_paths.append(path)
    if not unique_log_paths:
        return {}

    lines: List[tuple[Path, str]] = []
    latest_path = max(unique_log_paths, key=lambda path: path.stat().st_mtime)
    latest_mtime = latest_path.stat().st_mtime
    for path in unique_log_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines.extend((path, line) for line in text.splitlines())

    latest_segment: Optional[int] = None
    for _, line in reversed(lines):
        match = re.search(r"=== Segment (\d+):", line)
        if match:
            latest_segment = int(match.group(1))
            break
    latest_eval: Dict[str, Any] = {}
    latest_eval_path: Optional[Path] = None
    for path, line in reversed(lines):
        if "Eval metrics:" not in line:
            continue
        payload = line.split("Eval metrics:", 1)[1].strip()
        try:
            latest_eval = json.loads(payload)
        except json.JSONDecodeError:
            latest_eval = {"parse_error": payload[:500]}
        latest_eval_path = path
        break
    return {
        "latest_segment": latest_segment,
        "latest_eval": latest_eval,
        "log_mtime": latest_mtime,
        "log_age_seconds": max(0.0, time.time() - latest_mtime),
        "log_path": str(latest_path),
        "latest_eval_log_path": str(latest_eval_path) if latest_eval_path else None,
        "log_candidates": [str(path) for path in unique_log_paths],
    }


def _run_artifacts() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in ["final_metrics.json", "stop_and_diagnose.json", "process_exit.json"]:
        path = RUN_DIR / name
        if path.is_file():
            try:
                out[name] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                out[name] = {"error": repr(exc)}
    eval_statuses: List[Dict[str, Any]] = []
    for path in sorted(RUN_DIR.glob("segment_*/eval_status.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload = {"path": str(path), "error": repr(exc)}
        else:
            payload["path"] = str(path)
        eval_statuses.append(payload)
    if eval_statuses:
        out["eval_statuses"] = eval_statuses
        out["latest_eval_status"] = eval_statuses[-1]
    eval_failures: List[Dict[str, Any]] = []
    for pattern in ["segment_*/eval_timeout.json", "segment_*/eval_exception.json"]:
        for path in sorted(RUN_DIR.glob(pattern)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                payload = {"path": str(path), "error": repr(exc)}
            else:
                payload["path"] = str(path)
            eval_failures.append(payload)
    if eval_failures:
        out["eval_failures"] = eval_failures
    return out


def _classify(log_info: Dict[str, Any], artifacts: Dict[str, Any], processes: List[str]) -> Dict[str, Any]:
    if "stop_and_diagnose.json" in artifacts:
        return {"state": "stopped_low_or_failed", "reason": artifacts["stop_and_diagnose.json"].get("reason", "stop_and_diagnose")}
    if artifacts.get("eval_failures"):
        latest_failure = artifacts["eval_failures"][-1]
        return {
            "state": "eval_failed",
            "reason": f"{latest_failure.get('event', 'eval_failure')}: {latest_failure.get('error', '')}",
        }
    if processes:
        if LOG_PATH.is_file() and log_info.get("log_age_seconds", 0.0) > STALE_SECONDS:
            return {
                "state": "running",
                "reason": f"train process present; log older than {STALE_SECONDS}s during possible silent train phase",
            }
        return {"state": "running", "reason": "train process present"}
    if "final_metrics.json" in artifacts:
        return {"state": "completed", "reason": "final_metrics_present"}
    if "process_exit.json" in artifacts:
        process_exit = artifacts["process_exit.json"]
        return {
            "state": "exited_with_artifact",
            "reason": f"{process_exit.get('status')} at {process_exit.get('phase')}",
        }
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
    latest_extra = latest_eval.get("extra", {}) if isinstance(latest_eval, dict) else {}
    per_segment = latest_extra.get("per_segment_accuracy", []) if isinstance(latest_extra, dict) else []
    per_segment_task = latest_extra.get("per_segment_task_aware_accuracy", []) if isinstance(latest_extra, dict) else []
    segment_scores = {
        int(item.get("segment_id")): item.get("accuracy")
        for item in per_segment
        if isinstance(item, dict) and item.get("segment_id") is not None
    }
    task_segment_scores = {
        int(item.get("segment_id")): item.get("task_aware_accuracy")
        for item in per_segment_task
        if isinstance(item, dict) and item.get("segment_id") is not None
    }
    watched_lines = []
    for idx, task_name in enumerate(STANDARD_ORDER1_TASKS):
        if idx in segment_scores or idx in task_segment_scores:
            watched_lines.append(
                f"- {task_name}: exact=`{segment_scores.get(idx, 'n/a')}`, task-aware=`{task_segment_scores.get(idx, 'n/a')}`"
            )
    forgetting = latest_eval.get("forgetting", "n/a") if isinstance(latest_eval, dict) else "n/a"
    task_forgetting = latest_eval.get("task_aware_forgetting", "n/a") if isinstance(latest_eval, dict) else "n/a"
    approx_bwt = -float(forgetting) if isinstance(forgetting, (int, float)) else "n/a"
    approx_task_bwt = -float(task_forgetting) if isinstance(task_forgetting, (int, float)) else "n/a"
    final_metrics = status.get("artifacts", {}).get("final_metrics.json", {})
    latest_eval_status = status.get("artifacts", {}).get("latest_eval_status", {})
    process_exit = status.get("artifacts", {}).get("process_exit.json", {})
    ccfa_summary = final_metrics.get("ccfa_summary", {}) if isinstance(final_metrics, dict) else {}
    final_bwt = ccfa_summary.get("bwt", "n/a") if isinstance(ccfa_summary, dict) else "n/a"
    lines = [
        f"# {STATUS_TITLE}",
        "",
        f"- Updated: `{status['updated_at']}`",
        f"- Run: `{status['run_id']}`",
        f"- State: `{cls.get('state')}`",
        f"- Reason: `{cls.get('reason')}`",
        f"- Latest segment: `{status.get('log', {}).get('latest_segment', 'n/a')}`",
        f"- Latest seen/task-aware: `{latest_eval.get('seen_avg_score', 'n/a')}` / `{latest_eval.get('seen_avg_task_aware_score', 'n/a')}`",
        f"- Latest current/task-aware: `{latest_eval.get('current_score', 'n/a')}` / `{latest_eval.get('current_task_aware_score', 'n/a')}`",
        f"- Forgetting/task-aware: `{forgetting}` / `{task_forgetting}`",
        f"- Approx BWT/task-aware BWT: `{approx_bwt}` / `{approx_task_bwt}`",
        f"- Final BWT: `{final_bwt}`",
        f"- Train processes: `{len(status.get('processes', []))}`",
        f"- Log: `{status.get('log', {}).get('log_path', LOG_PATH)}`",
        f"- Latest eval source: `{status.get('log', {}).get('latest_eval_log_path', 'n/a')}`",
        f"- Latest eval heartbeat: `{latest_eval_status.get('event', 'n/a')}` segment `{latest_eval_status.get('segment_id', 'n/a')}` elapsed `{latest_eval_status.get('elapsed_seconds', 'n/a')}`",
        f"- Process exit artifact: `{process_exit.get('status', 'n/a')}` event `{process_exit.get('event', 'n/a')}` phase `{process_exit.get('phase', 'n/a')}`",
        "",
        "## Watched Standard Order1 Metrics",
        "",
        *(watched_lines or ["- no completed eval metrics yet"]),
    ]
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(status["classification"], ensure_ascii=False))
    state = str(cls.get("state", ""))
    if state not in {"running", "completed", "not_started"}:
        print(f"AGENT_LOOP_WAKE_LORA_OURS run={RUN_ID} state={state} reason={cls.get('reason')}")


def main() -> None:
    while True:
        write_status()
        time.sleep(MONITOR_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
