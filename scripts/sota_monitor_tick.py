#!/usr/bin/env python3
"""SOTA monitor tick with mandatory failure-analysis gate before next version launch."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
LOG_PATH = REPO / "SOTA_MONITOR.log"
STATE_PATH = REPO / "SOTA_MONITOR_STATE.json"
PROGRESS_PATH = REPO / "SOTA_PROGRESS.md"
AGENT_WAKE_FLAG = REPO / "SOTA_AGENT_WAKE.flag"
AGENT_WAKE_LOG = REPO / "SOTA_AGENT_WAKE.log"

AGENT_PROMPT = (
    "Check SOTA_MONITOR.log and SOTA_MONITOR_STATE.json. "
    "If AWAITING_AGENT or pending_action needs work, execute failure analysis → "
    "implement → launch per SOTA_PROGRESS.md Iteration Protocol. "
    "Do not kill healthy training. Update SOTA_PROGRESS.md."
)

SOTA_TARGETS = {
    "seen_avg": ("eval.seen_avg_score", ">=", 0.6),
    "task_aware": ("eval.seen_avg_task_aware_score", ">=", 0.6),
    "forgetting": ("eval.forgetting", "<=", 0.0333),
    "ta_forgetting": ("eval.task_aware_forgetting", "<=", 0.077),
    "token_f1": ("eval.token_f1_mean", ">=", 0.6),
    "lcs": ("eval.lcs_overlap_mean", ">=", 0.6),
}

V62_TRAJECTORY = {
    0: 0.0, 1: 0.05, 2: 0.0, 3: 0.25, 4: 0.34, 5: 0.325, 6: 0.426,
    7: 0.373, 8: 0.378, 9: 0.322, 10: 0.274, 11: 0.311, 12: 0.295,
    13: 0.275, 14: 0.297, 15: 0.328, 16: 0.35, 17: 0.381, 18: 0.35,
}
TRAJ_MARGIN = 0.05
MAX_SEGMENT = 18

CHAMPION_VERSION = "v8_sota_5"
CHAMPION_SEEN = 0.353
CHAMPION_TASK_AWARE = 0.404

NEXT_ON_FAIL: Dict[str, str] = {
    "v8_sota_1": "v8_sota_2",
    "v8_sota_2": "v8_sota_3",
    "v8_sota_3": "v8_sota_4",
    "v8_sota_4": "v8_sota_5",
    "v8_sota_5": "v8_sota_6",
    "v8_sota_6": "v8_sota_7",
    "v8_sota_7": "v8_sota_8",
    "v8_sota_8": "v8_sota_9",
    "v8_sota_9": "v8_sota_10",
    "v7_sota_1": "v8_sota_1",
    "v7_sota_2": "v8_sota_2",
    "v6_sota_2": "v8_sota_1",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _signal_agent_wake(
    *,
    action: str,
    failed: str,
    next_ver: str,
    detail: str,
) -> None:
    """Emit grep-friendly AWAITING_AGENT line + flag file for sota_agent_loop."""
    line = f"AWAITING_AGENT: {action} {next_ver} | failed={failed} | {detail}"
    _log(line)
    payload = {
        "action": action,
        "failed_version": failed,
        "next_version": next_ver,
        "detail": detail,
        "prompt": AGENT_PROMPT,
        "timestamp": _now(),
    }
    AGENT_WAKE_FLAG.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with AGENT_WAKE_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{_now()}] {line}\n")


def _clear_agent_wake() -> None:
    if AGENT_WAKE_FLAG.is_file():
        AGENT_WAKE_FLAG.unlink()


def _default_state() -> Dict[str, Any]:
    return {
        "launched_versions": [],
        "current_version": None,
        "pending_action": None,
        "failed_version": None,
        "next_version": None,
        "last_action": None,
        "last_verdict": None,
    }


def _load_state() -> Dict[str, Any]:
    if STATE_PATH.is_file():
        try:
            st = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            for k, v in _default_state().items():
                st.setdefault(k, v)
            return st
        except json.JSONDecodeError:
            pass
    return _default_state()


def _save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _tmux_sessions() -> List[str]:
    try:
        out = subprocess.check_output(["tmux", "list-sessions", "-F", "#{session_name}"], text=True)
        return [s.strip() for s in out.splitlines() if s.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _train_processes() -> List[str]:
    try:
        out = subprocess.check_output(["pgrep", "-af", "core/train.py"], text=True)
        return [ln for ln in out.splitlines() if "train.py" in ln]
    except subprocess.CalledProcessError:
        return []


def _parse_version_from_config_line(line: str) -> Optional[str]:
    m = re.search(r"configs/paper/(v\d+_sota_\d+)\.yaml", line)
    return m.group(1) if m else None


def _version_train_alive(version: str) -> bool:
    return any(f"{version}.yaml" in p for p in _train_processes())


def _detect_active_version(sessions: List[str], procs: List[str], state: Dict[str, Any]) -> Optional[str]:
    for proc in procs:
        ver = _parse_version_from_config_line(proc)
        if ver:
            return ver
    for s in sessions:
        if s in {"sota_monitor", "sota_agent"}:
            continue
        m = re.match(r"(v\d+_sota_\d+)", s)
        if m:
            return m.group(1)
    return state.get("current_version")


def _run_dir(version: str) -> Path:
    return REPO / "results/runs" / f"paper_instrdialog_ours_full_s123_{version}"


def _read_metrics_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def _read_all_metrics(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _read_final(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "final_metrics.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("final") or data
    except json.JSONDecodeError:
        return None


def _log_has_early_stop(version: str) -> bool:
    log = REPO / f"{version}.log"
    if not log.is_file():
        return False
    tail = log.read_text(encoding="utf-8", errors="replace")[-8000:]
    return "Early stopping triggered" in tail or "Trajectory early stopping" in tail


def _check_sota(row: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    gaps: Dict[str, Any] = {}
    all_met = True
    for name, (key, op, target) in SOTA_TARGETS.items():
        val = float(row.get(key, 0) or 0)
        if op == ">=":
            met = val >= target
            gap = target - val
        else:
            met = val <= target
            gap = val - target
        gaps[name] = {"value": val, "target": target, "met": met, "gap": gap}
        if not met:
            all_met = False
    return all_met, gaps


def _classify_status(
    version: str,
    run_dir: Path,
    train_alive: bool,
    tmux_alive: bool,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    final = _read_final(run_dir)
    latest = _read_metrics_jsonl(run_dir / "metrics.jsonl")
    early = _log_has_early_stop(version)

    if final and int(final.get("segment_id", -1)) >= MAX_SEGMENT:
        return "completed", final
    if final:
        return "completed", final
    if early and not train_alive:
        return "early_stopped", latest
    if train_alive or (tmux_alive and latest is not None):
        return "running", latest
    if latest and not train_alive and not tmux_alive:
        if early:
            return "early_stopped", latest
        seg = int(latest.get("segment_id", 0))
        if seg >= MAX_SEGMENT:
            return "completed", latest
        return "crashed", latest
    if tmux_alive and not train_alive:
        return "crashed", latest
    return "idle", latest


def _failure_analysis_path(next_version: str) -> Path:
    return REPO / "archive" / next_version / "FAILURE_ANALYSIS.md"


def _analysis_ready(next_version: str) -> bool:
    path = _failure_analysis_path(next_version)
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if re.search(r"^status:\s*ready\s*$", text, re.MULTILINE):
        return True
    if re.search(r"agent_signoff:\s*true", text, re.MULTILINE | re.IGNORECASE):
        return True
    return False


def _implementation_ready(next_version: str) -> bool:
    cfg = REPO / "configs/paper" / f"{next_version}.yaml"
    script = REPO / f"run_{next_version}.sh"
    changes = REPO / "archive" / next_version / "CHANGES.md"
    return cfg.is_file() and script.is_file() and changes.is_file()


def _run_failure_analysis(failed: str, next_ver: str) -> Path:
    subprocess.run(
        [sys.executable, str(REPO / "scripts/sota_failure_analysis.py"), "--failed", failed, "--next", next_ver],
        cwd=str(REPO),
        check=False,
    )
    return _failure_analysis_path(next_ver)


def _launch_version(version: str, state: Dict[str, Any]) -> bool:
    if _version_train_alive(version):
        _log(f"SKIP launch {version}: train already alive")
        return False
    if any(_parse_version_from_config_line(p) for p in _train_processes()):
        _log("SKIP launch: another train.py running")
        return False
    if not _analysis_ready(version):
        _log(f"BLOCKED launch {version}: FAILURE_ANALYSIS.md not ready (status: ready required)")
        return False
    if not _implementation_ready(version):
        _log(f"BLOCKED launch {version}: missing yaml/run.sh/CHANGES.md")
        return False
    script = REPO / f"run_{version}.sh"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", version, f"bash {script}"],
        check=False,
        cwd=str(REPO),
    )
    launched = state.setdefault("launched_versions", [])
    if version not in launched:
        launched.append(version)
    state["current_version"] = version
    state["pending_action"] = None
    state["failed_version"] = None
    state["next_version"] = None
    state["last_action"] = f"launched_{version}"
    _save_state(state)
    _log(f"LAUNCHED {version} (post failure-analysis gate)")
    return True


def _queue_failure_workflow(state: Dict[str, Any], failed: str, status: str) -> None:
    if state.get("pending_action") and state.get("failed_version") == failed:
        return
    nxt = NEXT_ON_FAIL.get(failed)
    if not nxt:
        m = re.match(r"v(\d+)_sota_(\d+)", failed)
        if m:
            nxt = f"v{m.group(1)}_sota_{int(m.group(2)) + 1}"
        else:
            nxt = "v8_sota_2"
    state["pending_action"] = "failure_analysis"
    state["failed_version"] = failed
    state["next_version"] = nxt
    state["last_action"] = f"queued_failure_analysis_{failed}_to_{nxt}"
    _save_state(state)
    _log(f"WORKFLOW queued failure_analysis: {failed} -> {nxt} (status={status})")


def _process_pending_workflow(state: Dict[str, Any]) -> str:
    action = state.get("pending_action")
    if not action:
        return "no pending workflow"

    failed = state.get("failed_version") or ""
    nxt = state.get("next_version") or ""
    if not failed or not nxt:
        state["pending_action"] = None
        _save_state(state)
        return "cleared invalid pending workflow"

    if action == "failure_analysis":
        fa_path = _failure_analysis_path(nxt)
        if not fa_path.is_file():
            _run_failure_analysis(failed, nxt)
            _log(f"WORKFLOW generated draft {fa_path}")
        if _analysis_ready(nxt):
            state["pending_action"] = "implement"
            state["last_action"] = f"analysis_ready_{nxt}"
            _save_state(state)
            _signal_agent_wake(
                action="implement",
                failed=failed,
                next_ver=nxt,
                detail=f"analysis ready; implement {nxt}",
            )
            return f"analysis ready -> implement {nxt}"
        state["last_action"] = f"awaiting_agent_analysis_{nxt}"
        _save_state(state)
        _signal_agent_wake(
            action="failure_analysis",
            failed=failed,
            next_ver=nxt,
            detail=f"review archive/{nxt}/FAILURE_ANALYSIS.md, set status: ready",
        )
        return f"AWAITING_AGENT: failure_analysis {nxt} | failed={failed}"

    if action == "implement":
        if not _analysis_ready(nxt):
            state["pending_action"] = "failure_analysis"
            _save_state(state)
            return f"reverted to failure_analysis (doc not ready)"
        if _implementation_ready(nxt):
            state["pending_action"] = "launch"
            state["last_action"] = f"implement_ready_{nxt}"
            _save_state(state)
            return f"implement complete -> launch {nxt}"
        state["last_action"] = f"awaiting_agent_implement_{nxt}"
        _save_state(state)
        _signal_agent_wake(
            action="implement",
            failed=failed,
            next_ver=nxt,
            detail=f"create yaml + run_{nxt}.sh + archive/{nxt}/CHANGES.md",
        )
        return f"AWAITING_AGENT: implement {nxt} | failed={failed}"

    if action == "launch":
        if _version_train_alive(nxt):
            state["pending_action"] = None
            state["current_version"] = nxt
            _clear_agent_wake()
            _save_state(state)
            return f"{nxt} already training"
        if _launch_version(nxt, state):
            _clear_agent_wake()
            return f"launched {nxt}"
        _signal_agent_wake(
            action="launch",
            failed=failed,
            next_ver=nxt,
            detail="files ready; ensure tmux session started",
        )
        return f"AWAITING_AGENT: launch {nxt} | launch blocked, agent verify"

    return f"unknown action {action}"


def _update_progress_live(
    *,
    status: str,
    version: Optional[str],
    latest: Optional[Dict[str, Any]],
    gaps: Optional[Dict[str, Any]],
    next_action: str,
    pending_action: Optional[str],
) -> None:
    seg = int(latest.get("segment_id", -1)) if latest else -1
    seen = float(latest.get("eval.seen_avg_score", 0) or 0) if latest else 0.0
    ta = float(latest.get("eval.seen_avg_task_aware_score", 0) or 0) if latest else 0.0
    forget = float(latest.get("eval.forgetting", 0) or 0) if latest else 0.0
    oracle = latest.get("routing.oracle_agreement_rate", "—") if latest else "—"
    floor = "—"
    if seg >= 3 and seg in V62_TRAJECTORY:
        floor = f"{V62_TRAJECTORY[seg] - TRAJ_MARGIN:.3f}"

    gap_lines = ""
    if gaps:
        gap_lines = "\n".join(
            f"| {k} | {v['value']:.4f} | {v['target']} | {'✅' if v['met'] else '❌'} |"
            for k, v in gaps.items()
        )

    pending_line = pending_action or "—"

    block = f"""<!-- LIVE_STATUS_START -->
## Live Status（自动监控）

| 项 | 值 |
|---|---|
| **最后检查** | {_now()} |
| **监控 tmux** | `sota_monitor` |
| **训练 tmux** | `{version or '—'}` |
| **状态** | {status} |
| **workflow** | `{pending_line}` |
| **当前 segment** | {seg} |
| **seen_avg** | {seen:.4f} |
| **task_aware** | {ta:.4f} |
| **forgetting** | {forget:.4f} |
| **oracle_agree** | {oracle} |
| **轨迹地板 (seg≥3)** | {floor} |
| **下一动作** | {next_action} |

### 距 SOTA 目标差距（基于最新指标）
| 指标 | 当前 | 目标 | 达标 |
|------|------|------|------|
{gap_lines or '| — | — | — | — |'}
<!-- LIVE_STATUS_END -->"""

    text = PROGRESS_PATH.read_text(encoding="utf-8") if PROGRESS_PATH.is_file() else ""
    if "<!-- LIVE_STATUS_START -->" in text:
        text = re.sub(
            r"<!-- LIVE_STATUS_START -->.*?<!-- LIVE_STATUS_END -->",
            block,
            text,
            flags=re.DOTALL,
        )
    elif "## Iteration Protocol" in text:
        text = text.replace("## Iteration Protocol", block + "\n\n## Iteration Protocol", 1)
    else:
        text = block + "\n\n" + text
    PROGRESS_PATH.write_text(text, encoding="utf-8")


def main() -> int:
    sessions = _tmux_sessions()
    procs = _train_processes()
    state = _load_state()

    # Process queued workflow first (failure_analysis / implement / launch)
    workflow_msg = _process_pending_workflow(state)
    if state.get("pending_action"):
        _log(f"WORKFLOW {workflow_msg}")

    version = _detect_active_version(sessions, procs, state) or state.get("current_version", "v8_sota_1")
    state["current_version"] = version

    run_dir = _run_dir(version)
    train_alive = _version_train_alive(version)
    tmux_alive = version in sessions
    status, row = _classify_status(version, run_dir, train_alive, tmux_alive)

    seg_s = int(row.get("segment_id", -1)) if row else -1
    seen_s = float(row.get("eval.seen_avg_score", 0) or 0) if row else 0.0
    _log(
        f"TICK version={version} status={status} seg={seg_s} seen={seen_s:.4f} "
        f"train_alive={train_alive} pending={state.get('pending_action')}"
    )

    gaps: Optional[Dict[str, Any]] = None
    next_action = workflow_msg if state.get("pending_action") else f"monitor {version} (seg {seg_s})"

    if row:
        _, gaps = _check_sota(row)

    if status == "running":
        if seg_s >= 3 and seen_s < 0.2:
            _log(f"WARN {version} seg{seg_s} seen={seen_s:.4f} below 0.2")
        elif seg_s >= 3 and seg_s in V62_TRAJECTORY and seen_s < V62_TRAJECTORY[seg_s] - TRAJ_MARGIN:
            _log(
                f"WARN {version} seg{seg_s} seen={seen_s:.4f} below trajectory floor "
                f"{V62_TRAJECTORY[seg_s] - TRAJ_MARGIN:.3f}"
            )
        if not state.get("pending_action"):
            next_action = f"monitor {version} (seg {seg_s})"

    elif status in {"early_stopped", "crashed", "completed"} and not train_alive:
        check_row = _read_final(run_dir) or row or {}
        all_met, gaps = _check_sota(check_row)
        verdict = "SOTA_ALL_MET" if all_met else "SOTA_NOT_MET"
        _log(
            f"VERDICT {version} status={status} {verdict} "
            f"seen={check_row.get('eval.seen_avg_score')} seg={check_row.get('segment_id')}"
        )
        state["last_verdict"] = {"version": version, "status": status, "all_met": all_met}
        if all_met:
            state["pending_action"] = None
            state["last_action"] = "sota_complete"
            next_action = "ALL TARGETS MET — stop chase"
        elif not state.get("pending_action"):
            _queue_failure_workflow(state, version, status)
            fa_path = _failure_analysis_path(state["next_version"])
            if not fa_path.is_file():
                _run_failure_analysis(version, state["next_version"])
            next_action = _process_pending_workflow(state)
            _log(f"WORKFLOW after verdict: {next_action}")
        else:
            next_action = workflow_msg
        _save_state(state)

    if not state.get("pending_action") and status == "running":
        _clear_agent_wake()

    _update_progress_live(
        status=status,
        version=version,
        latest=row,
        gaps=gaps,
        next_action=next_action,
        pending_action=state.get("pending_action"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
