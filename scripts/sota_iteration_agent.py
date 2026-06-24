#!/usr/bin/env python3
"""Headless SOTA iteration: monitor -> early-stop -> analyze -> launch next version."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "results" / "logs"
ARCHIVES = REPO / "archives"
CONFIG_DIR = REPO / "configs" / "paper" / "sota_campaign"
AGENT_LOG = LOG_DIR / "sota_iteration_agent.log"
PROGRESS = REPO / "sota_progress.md"
STATE_PATH = REPO / "experiments" / "sota_campaign" / "ITERATION_AGENT_STATE.json"

# Primary success: +33% vs advanced LB-CL baseline (user decision 2026-06-16).
REL_SEEN = 0.3153
REL_TA = 0.3785
REL_F1 = 0.4461
ABS_SEEN = 0.6001  # informational only; no longer drives iteration
EARLY_MIN_SEEN = 0.28
POLL_SEC = 120
SUCCESS_FLAG = REPO / "experiments" / "sota_campaign" / "SOTA_RELATIVE_ACHIEVED"
VERDICT_PATH = REPO / "experiments" / "sota_campaign" / "SOTA_VERDICT.json"


def _log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}"
    print(line, flush=True)
    with AGENT_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _tmux() -> List[str]:
    try:
        out = subprocess.check_output(["tmux", "ls"], text=True, stderr=subprocess.DEVNULL)
        return [ln.split(":")[0] for ln in out.strip().splitlines() if ln]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _train_pid(version: int) -> Optional[int]:
    needle = f"sota_v{version}_instrdialogpp"
    try:
        out = subprocess.check_output(["pgrep", "-af", "python3 core/train.py"], text=True)
    except subprocess.CalledProcessError:
        return None
    for line in out.splitlines():
        if needle in line:
            m = re.match(r"^(\d+)", line.strip())
            if m:
                return int(m.group(1))
    return None


def _active_version(sessions: List[str]) -> Optional[int]:
    for v in (5, 4, 3, 2, 1):
        if f"sota-v{v}" in sessions or _train_pid(v):
            return v
    return None


def _log_path(v: int) -> Path:
    return LOG_DIR / f"paper_instrdialogpp_sota_v{v}_ours_s123.log"


def _parse(v: int) -> Tuple[List[Dict[str, Any]], bool]:
    path = _log_path(v)
    if not path.is_file():
        return [], False
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = 0
    for i, line in enumerate(lines):
        if f"sota-v{v} train start" in line:
            start = i
    traj: List[Dict[str, Any]] = []
    finished = False
    for line in lines[start:]:
        if "intentional stop (early-stop)" in line or "Trajectory early stopping" in line:
            finished = True
        if "Eval metrics:" not in line:
            continue
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        d = json.loads(m.group())
        traj.append(
            {
                "segment_id": int(d.get("num_seen_segments", 1)) - 1,
                "seen": float(d["seen_avg_score"]),
                "ta": float(d.get("seen_avg_task_aware_score", 0)),
                "f1": float(d.get("token_f1_mean", 0)),
            }
        )
    return traj, finished


@dataclass
class Verdict:
    peak_seen: float
    peak_ta: float
    peak_f1: float
    latest_seen: float
    latest_seg: int
    relative_ok: bool
    absolute_ok: bool
    early_stop: bool
    reason: str


def _relative_ok(traj: List[Dict[str, Any]]) -> Tuple[float, float, float, bool]:
    peak_seen = max(t["seen"] for t in traj)
    peak_ta = max(t["ta"] for t in traj)
    peak_f1 = max(t["f1"] for t in traj)
    ok = peak_seen >= REL_SEEN and peak_ta >= REL_TA and peak_f1 >= REL_F1
    return peak_seen, peak_ta, peak_f1, ok


def _verdict(traj: List[Dict[str, Any]], finished: bool) -> Verdict:
    if not traj:
        return Verdict(0, 0, 0, 0, -1, False, False, False, "waiting for metrics")
    latest = traj[-1]
    ps, pta, pf1, rel = _relative_ok(traj)
    ls, seg = float(latest["seen"]), int(latest["segment_id"])
    abs_ok = ps >= ABS_SEEN and ls >= ABS_SEEN * 0.9
    early = False
    reason = ""
    if seg >= 3 and ls < EARLY_MIN_SEEN:
        early, reason = True, f"seg{seg} seen {ls:.3f} < {EARLY_MIN_SEEN}"
    elif seg >= 6 and ps >= REL_SEEN and ls < ps * 0.55:
        early, reason = True, f"collapse {ps:.3f}->{ls:.3f}"
    elif finished and not rel:
        reason = "ended without +33% relative SOTA"
    return Verdict(ps, pta, pf1, ls, seg, rel, abs_ok, early, reason)


def _kill(v: int) -> None:
    pid = _train_pid(v)
    if pid:
        subprocess.run(["kill", str(pid)], check=False)
    if f"sota-v{v}" in _tmux():
        subprocess.run(["tmux", "kill-session", "-t", f"sota-v{v}"], check=False)


def _launch(v: int) -> None:
    pause = REPO / "experiments/sota_campaign" / f"PAUSE_SOTA_V{v}"
    if pause.is_file():
        _log(f"PAUSE_SOTA_V{v} set; skip launch")
        return
    script = REPO / "scripts" / f"launch_sota_v{v}.sh"
    if not script.is_file():
        _log(f"no launch script for v{v}")
        return
    for old in range(1, v):
        if f"sota-v{old}" in _tmux() or _train_pid(old):
            _kill(old)
    subprocess.run(["bash", str(script)], cwd=REPO, check=False)
    _log(f"launched sota-v{v}")


def _latest_config_version() -> int:
    versions = []
    for p in CONFIG_DIR.glob("sota_v*_instrdialogpp_s123.yaml"):
        m = re.search(r"sota_v(\d+)", p.name)
        if m:
            versions.append(int(m.group(1)))
    return max(versions) if versions else 1


def _note(v: int, traj: List[Dict[str, Any]], vd: Verdict) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    block = (
        f"\n## 迭代代理 {now}\n"
        f"- sota-v{v}: seg{vd.latest_seg} seen={vd.latest_seen:.4f} "
        f"peak_seen={vd.peak_seen:.4f} peak_ta={vd.peak_ta:.4f} peak_f1={vd.peak_f1:.4f}\n"
        f"- +33% 主目标: {'✅ SUCCESS' if vd.relative_ok else '❌ 未达标'} "
        f"(seen≥{REL_SEEN}, ta≥{REL_TA}, f1≥{REL_F1})\n"
        f"- 绝对 0.6（参考）: {'OK' if vd.absolute_ok else 'NO'}\n"
        f"- {vd.reason}\n"
    )
    if traj:
        for t in traj[-4:]:
            block += f"  - seg{t['segment_id']}: seen={t['seen']:.4f}\n"
    if PROGRESS.is_file():
        text = PROGRESS.read_text(encoding="utf-8")
        if "## 迭代代理" in text:
            text = text.split("## 迭代代理")[0].rstrip() + block
        else:
            text = text.rstrip() + "\n\n---\n" + block
    else:
        text = "# SOTA\n" + block
    PROGRESS.write_text(text, encoding="utf-8")


def _write_verdict(v: int, traj: List[Dict[str, Any]], vd: Verdict) -> None:
    payload = {
        "verdict": "SUCCESS" if vd.relative_ok else "PENDING",
        "achieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "user_decision": "2026-06-16: +33% margin vs advanced baseline is sufficient; absolute 0.6 not required",
        "version": v,
        "peak": {
            "seen_avg_score": vd.peak_seen,
            "seen_avg_task_aware_score": vd.peak_ta,
            "token_f1_mean": vd.peak_f1,
        },
        "targets_relative_33pct": {
            "seen_avg_score": REL_SEEN,
            "seen_avg_task_aware_score": REL_TA,
            "token_f1_mean": REL_F1,
        },
        "absolute_reference": {"seen_avg_score": ABS_SEEN},
        "relative_ok": vd.relative_ok,
        "absolute_ok": vd.absolute_ok,
        "trajectory": traj,
    }
    VERDICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERDICT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    SUCCESS_FLAG.touch()


def tick() -> Dict[str, Any]:
    sessions = _tmux()
    v = _active_version(sessions) or _latest_config_version()
    traj, finished = _parse(v)
    vd = _verdict(traj, finished)
    _note(v, traj, vd)
    out = {"version": v, **vd.__dict__, "finished": finished}
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if SUCCESS_FLAG.is_file():
        _log(f"relative +33% SOTA already achieved (v{v}); no further version bumps")
        return out

    if vd.relative_ok:
        _write_verdict(v, traj, vd)
        _log(f"relative +33% SOTA achieved on v{v}; stopping iteration")
        return out

    if vd.early_stop or (finished and not vd.relative_ok):
        dest = ARCHIVES / f"sota_v{v}"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "FAILURE.md").write_text(vd.reason + "\n", encoding="utf-8")
        nxt = min(v + 1, 5)
        if nxt > v and (REPO / "scripts" / f"launch_sota_v{nxt}.sh").is_file():
            _launch(nxt)
            out["launched"] = nxt
    return out


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--poll-sec", type=int, default=POLL_SEC)
    args = p.parse_args()
    _log("start")
    if args.once:
        tick()
        return 0
    while True:
        try:
            tick()
        except Exception as exc:
            _log(f"error: {exc}")
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
