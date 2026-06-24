#!/usr/bin/env python3
"""Continuous SOTA campaign monitor: parse logs, update sota_progress.md, detect transitions."""
from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "results/logs/paper_instrdialogpp_v8_sota5_ours_s123_v8s5camp.log"
PROGRESS = REPO / "sota_progress.md"
STATE = REPO / "experiments/sota_campaign/MONITOR_STATE.json"
MONITOR_LOG = REPO / "results/logs/sota_campaign_monitor.log"

SOTA_MARGIN = {
    "seen_avg_score": 0.3153,
    "seen_avg_task_aware_score": 0.3785,
    "token_f1_mean": 0.4461,
}
# User decision 2026-06-16: campaign success = +33% margin only (see archives/.../new_true_sota_targets.json).
VERDICT_PATH = REPO / "experiments/sota_campaign/SOTA_VERDICT.json"
SUCCESS_FLAG = REPO / "experiments/sota_campaign/SOTA_RELATIVE_ACHIEVED"
V62_FLOOR = {
    0: 0.0, 1: 0.05, 2: 0.0, 3: 0.25, 4: 0.34, 5: 0.325, 6: 0.426, 7: 0.373,
    8: 0.378, 9: 0.322, 10: 0.274, 11: 0.311, 12: 0.295, 13: 0.275, 14: 0.297,
    15: 0.328, 16: 0.35, 17: 0.381, 18: 0.35,
}
TRAJ_MARGIN = 0.05


def _log(msg: str) -> None:
    MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with MONITOR_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _sota_train_pid() -> Optional[int]:
    for pat in ("sota_v3_instrdialogpp", "sota_v2_instrdialogpp", "sota_v1_instrdialogpp"):
        try:
            out = subprocess.check_output(
                ["pgrep", "-af", f"python3 core/train.py.*{pat}"],
                text=True,
            ).strip()
            for line in out.splitlines():
                if re.search(r"python\d*(?:\.\d+)?\s+.*core/train\.py", line):
                    m = re.match(r"^(\d+)", line.strip())
                    if m:
                        return int(m.group(1))
        except subprocess.CalledProcessError:
            continue
    return None


def _ensure_sota_running(sessions: List[str]) -> None:
    if REPO.joinpath("experiments/sota_campaign/PAUSE_SOTA_V1").is_file():
        sessions = [s for s in sessions if s != "sota-v1"]
    if REPO.joinpath("experiments/sota_campaign/PAUSE_SOTA_V2").is_file():
        sessions = [s for s in sessions if s != "sota-v2"]
    if REPO.joinpath("experiments/sota_campaign/PAUSE_SOTA_V3").is_file():
        sessions = [s for s in sessions if s != "sota-v3"]
    active = [s for s in ("sota-v3", "sota-v2", "sota-v1") if s in sessions]
    if active or _sota_train_pid():
        return
    pause = REPO / "experiments/sota_campaign/PAUSE_V8S5CAMP_SERIAL"
    if not pause.is_file():
        return
    for label, script in (("v3", "launch_sota_v3.sh"), ("v2", "launch_sota_v2.sh")):
        if label == "v2" and REPO.joinpath("experiments/sota_campaign/PAUSE_SOTA_V2").is_file():
            continue
        if label == "v3" and REPO.joinpath("experiments/sota_campaign/PAUSE_SOTA_V3").is_file():
            continue
        sh = REPO / "scripts" / script
        if sh.is_file():
            _log(f"sota train dead; auto-relaunching sota-{label}")
            subprocess.run(["bash", str(sh)], cwd=str(REPO), check=False)
            break


def _train_pid() -> Optional[int]:
    pid = _sota_train_pid()
    if pid:
        return pid
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "core/train.py.*v8_sota5|core/train.py --config.*sota"],
            text=True,
        ).strip()
        if out:
            return int(out.split("\n")[0])
    except subprocess.CalledProcessError:
        pass
    try:
        out = subprocess.check_output(["pgrep", "-f", "core/train.py"], text=True).strip()
        if out:
            return int(out.split("\n")[0])
    except subprocess.CalledProcessError:
        return None
    return None


def _tmux_sessions() -> List[str]:
    try:
        out = subprocess.check_output(["tmux", "ls"], text=True, stderr=subprocess.DEVNULL)
        return [line.split(":")[0] for line in out.strip().splitlines() if line]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _parse_latest_run_trajectory(log_path: Path) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    if not log_path.is_file():
        return [], None, None
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    # last Segment 0 marks latest restart
    start = 0
    for i, line in enumerate(lines):
        if "=== Segment 0:" in line:
            start = i
    traj: List[Dict[str, Any]] = []
    early_stop: Optional[str] = None
    current_seg: Optional[str] = None
    for line in lines[start:]:
        if "=== Segment" in line:
            current_seg = line.strip()
        if "Trajectory early stopping" in line or "Early stopping triggered" in line:
            early_stop = line.strip()
        if "Eval metrics:" in line:
            m = re.search(r"\{.*\}", line)
            if m:
                d = json.loads(m.group())
                sid = int(d.get("num_seen_segments", 1)) - 1
                traj.append({
                    "segment_id": sid,
                    "seen": float(d["seen_avg_score"]),
                    "ta": float(d["seen_avg_task_aware_score"]),
                    "f1": float(d["token_f1_mean"]),
                    "forget": float(d.get("forgetting", 0)),
                    "oracle": float(
                        (d.get("extra") or {}).get("routing", {}).get("oracle_agreement_rate", 0) or 0
                    ),
                })
    return traj, early_stop, current_seg


def _parse_sota_campaign_log(version: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    p = REPO / f"results/logs/paper_instrdialogpp_sota_{version}_ours_s123.log"
    if not p.is_file():
        return [], None
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = 0
    marker = f"sota-{version} train start"
    for i, line in enumerate(lines):
        if marker in line:
            start = i
    traj: List[Dict[str, Any]] = []
    current_seg: Optional[str] = None
    for line in lines[start:]:
        if "=== Segment" in line:
            current_seg = line.strip()
        if "Eval metrics:" in line:
            m = re.search(r"\{.*\}", line)
            if m:
                d = json.loads(m.group())
                traj.append({
                    "segment_id": int(d.get("num_seen_segments", 1)) - 1,
                    "seen": float(d["seen_avg_score"]),
                    "ta": float(d["seen_avg_task_aware_score"]),
                    "f1": float(d["token_f1_mean"]),
                    "oracle": float(
                        (d.get("extra") or {}).get("routing", {}).get("oracle_agreement_rate", 0) or 0
                    ),
                })
    return traj, current_seg


def _parse_sota_v1_log() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    return _parse_sota_campaign_log("v1")


def _parse_sota_v2_log() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    return _parse_sota_campaign_log("v2")


def _parse_sota_v3_log() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    return _parse_sota_campaign_log("v3")


def _margin_status(traj: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not traj:
        return {"peak": {}, "relative_ok": False}
    peak_seen = max(t["seen"] for t in traj)
    peak_ta = max(t["ta"] for t in traj)
    peak_f1 = max(t["f1"] for t in traj)
    rel = (
        peak_seen >= SOTA_MARGIN["seen_avg_score"]
        and peak_ta >= SOTA_MARGIN["seen_avg_task_aware_score"]
        and peak_f1 >= SOTA_MARGIN["token_f1_mean"]
    )
    return {
        "peak": {"seen": peak_seen, "ta": peak_ta, "f1": peak_f1},
        "relative_ok": rel,
    }


def _maybe_write_verdict(exp_label: str, traj: List[Dict[str, Any]], ms: Dict[str, Any]) -> None:
    if not ms.get("relative_ok"):
        return
    peak = ms["peak"]
    payload = {
        "verdict": "SUCCESS",
        "achieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "user_decision": "2026-06-16: +33% margin vs advanced baseline is sufficient; absolute 0.6 not required",
        "experiment": exp_label,
        "peak": {
            "seen_avg_score": peak["seen"],
            "seen_avg_task_aware_score": peak["ta"],
            "token_f1_mean": peak["f1"],
        },
        "targets_relative_33pct": SOTA_MARGIN,
        "relative_ok": True,
        "trajectory": traj,
    }
    VERDICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERDICT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    SUCCESS_FLAG.touch()


def _floor(seg: int) -> Optional[float]:
    if seg not in V62_FLOOR:
        return None
    return V62_FLOOR[seg] - TRAJ_MARGIN


def _update_progress(
    traj: List[Dict[str, Any]],
    early_stop: Optional[str],
    current_seg: Optional[str],
    pid: Optional[int],
    sessions: List[str],
    sota_v1_traj: List[Dict[str, Any]],
    sota_current_seg: Optional[str] = None,
    sota_v2_traj: Optional[List[Dict[str, Any]]] = None,
    sota_v2_current_seg: Optional[str] = None,
    sota_v3_traj: Optional[List[Dict[str, Any]]] = None,
    sota_v3_current_seg: Optional[str] = None,
) -> None:
    sota_v2_traj = sota_v2_traj or []
    sota_v3_traj = sota_v3_traj or []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    v3_active = "sota-v3" in sessions or bool(sota_v3_traj)
    v2_active = (not v3_active) and ("sota-v2" in sessions or bool(sota_v2_traj))
    v1_active = (not v3_active and not v2_active) and ("sota-v1" in sessions or bool(sota_v1_traj)) and not REPO.joinpath(
        "experiments/sota_campaign/PAUSE_SOTA_V1"
    ).is_file()
    sota_active = v3_active or v2_active or v1_active
    if v3_active and sota_v3_traj:
        campaign_traj, campaign_seg = sota_v3_traj, sota_v3_current_seg
    elif v2_active and sota_v2_traj:
        campaign_traj, campaign_seg = sota_v2_traj, sota_v2_current_seg
    elif v1_active:
        campaign_traj, campaign_seg = sota_v1_traj, sota_current_seg
    else:
        campaign_traj, campaign_seg = [], None
    if sota_active:
        current_seg = campaign_seg or current_seg
    if sota_active:
        latest = campaign_traj[-1] if campaign_traj else {}
        meaningful_src = campaign_traj
    else:
        latest = campaign_traj[-1] if campaign_traj else (traj[-1] if traj else {})
        meaningful_src = campaign_traj if campaign_traj else traj
    meaningful = [t for t in meaningful_src if t["segment_id"] >= 3] or meaningful_src
    peak = max(meaningful, key=lambda x: x["seen"]) if meaningful else {}
    ms = _margin_status(meaningful_src)
    peak_all = ms["peak"]
    rel_ok = ms["relative_ok"]
    exp_label = "sota-v3" if v3_active else ("sota-v2" if v2_active else ("sota-v1" if v1_active else "v8s5camp"))
    if rel_ok and sota_active:
        _maybe_write_verdict(exp_label, campaign_traj, ms)
    lines = [
        "# SOTA 实验进度报告",
        "",
        f"**最后更新**：{now}  ",
        f"**工作目录**：`/root/autodl-tmp/Lora-code`  ",
        "**监控**：`scripts/sota_campaign_monitor.py`（tmux `sota-monitor`）",
        "",
        "---",
        "",
        "## 实时状态",
        "",
        f"**当前实验**：`{exp_label}`  ",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 训练 PID | `{pid or 'none'}` |",
        f"| 当前 segment 行 | `{current_seg or 'unknown'}` |",
    ]
    if latest:
        sid = latest["segment_id"]
        fl = _floor(sid)
        floor_s = f"{fl:.3f}" if fl is not None else "—"
        ok = "✅" if fl is None or latest["seen"] >= fl else "❌ 将/已早停"
        lines += [
            f"| 最新 eval seg | **{sid}** |",
            f"| seen / ta / f1 | **{latest['seen']:.4f}** / {latest['ta']:.4f} / {latest['f1']:.4f} |",
            f"| oracle_agreement | {latest.get('oracle', 0):.3f} |",
            f"| 轨迹地板 seg{sid} | {floor_s} {ok} |",
            f"| 峰值 seen | **{peak['seen']:.4f}** @seg{peak['segment_id']} |",
            f"| +33% seen≥{SOTA_MARGIN['seen_avg_score']} | {'✅' if peak_all.get('seen', 0) >= SOTA_MARGIN['seen_avg_score'] else '❌'} (peak {peak_all.get('seen', 0):.4f}) |",
            f"| +33% ta≥{SOTA_MARGIN['seen_avg_task_aware_score']} | {'✅' if peak_all.get('ta', 0) >= SOTA_MARGIN['seen_avg_task_aware_score'] else '❌'} (peak {peak_all.get('ta', 0):.4f}) |",
            f"| +33% f1≥{SOTA_MARGIN['token_f1_mean']} | {'✅' if peak_all.get('f1', 0) >= SOTA_MARGIN['token_f1_mean'] else '❌'} (peak {peak_all.get('f1', 0):.4f}) |",
            f"| **相对 SOTA 主目标** | {'✅ **已达成**' if rel_ok else '❌ 未达成'} |",
        ]
    if rel_ok:
        lines += [
            "",
            "## 🏆 相对 SOTA 终局判定",
            "",
            f"**宣告**：相对 advanced baseline **+33% 已达标**（用户决策：无需绝对 0.6）。详见 `experiments/sota_campaign/SOTA_VERDICT.json`。",
            "",
        ]
    if early_stop:
        lines.append(f"| 早停 | `{early_stop[-120:]}` |")
    lines += [
        "",
        "### tmux",
        "",
        "| Session | 状态 |",
        "|---------|------|",
    ]
    for name, desc in [
        ("v8s5camp_serial", "campaign 训练"),
        ("sota-v3", "sota-v3 实验（主战役）"),
        ("sota-v2", "sota-v2 实验（已暂停/完成）"),
        ("sota-v1", "sota-v1 实验（已暂停）"),
        ("sota-monitor", "本监控脚本"),
    ]:
        st = "🟢 活跃" if name in sessions else "—"
        lines.append(f"| `{name}` | {st} {desc} |")

    if campaign_traj:
        lines += ["", f"### {exp_label} 轨迹", ""]
        for pt in campaign_traj[-6:]:
            lines.append(
                f"- seg{pt['segment_id']}: seen={pt['seen']:.4f} ta={pt['ta']:.4f} "
                f"f1={pt['f1']:.4f} oracle={pt.get('oracle', 0):.3f}"
            )

  # preserve static sections from existing file if present
    static_tail = ""
    campaign_notes = ""
    if PROGRESS.is_file():
        text = PROGRESS.read_text(encoding="utf-8")
        if "## 战役摘要" in text and "## SOTA 目标定义" in text:
            a = text.index("## 战役摘要")
            b = text.index("## SOTA 目标定义")
            campaign_notes = text[a:b].rstrip() + "\n\n"
        marker = "## SOTA 目标定义"
        if marker in text:
            static_tail = text[text.index(marker):]

    body = "\n".join(lines) + "\n\n---\n\n" + campaign_notes + (static_tail if static_tail else _default_static())
    PROGRESS.write_text(body, encoding="utf-8")


def _default_static() -> str:
    return """## SOTA 目标定义

> **用户决策 2026-06-16**：战役成功条件已收窄为相对 advanced baseline **+33% margin**；**不再**以绝对 seen/f1/lcs≥0.6 作为停止或升版条件。

### 主目标：+33% vs LB-CL（advanced baseline）
| 指标 | 阈值 |
|------|------|
| seen_avg_score | ≥ 0.3153 |
| seen_avg_task_aware_score | ≥ 0.3785 |
| token_f1_mean | ≥ 0.4461 |

### 参考（非停止条件）
绝对目标 seen/f1/lcs≥0.6001 仅作历史对照；`new_true_sota_targets.json` 已归档，见 `archive_candidates/.../state/`。

---
*本文件由 sota_campaign_monitor.py 自动维护*
"""


def main() -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    _log("SOTA monitor started")
    last_pid: Optional[int] = None
    while True:
        pid = _train_pid()
        sessions = _tmux_sessions()
        traj, early_stop, current_seg = _parse_latest_run_trajectory(LOG)
        sota_v1_traj, sota_current_seg = _parse_sota_v1_log()
        sota_v2_traj, sota_v2_current_seg = _parse_sota_v2_log()
        sota_v3_traj, sota_v3_current_seg = _parse_sota_v3_log()
        _update_progress(
            traj,
            early_stop,
            current_seg,
            pid,
            sessions,
            sota_v1_traj,
            sota_current_seg,
            sota_v2_traj,
            sota_v2_current_seg,
            sota_v3_traj,
            sota_v3_current_seg,
        )

        state = {
            "updated": datetime.now().isoformat(),
            "train_pid": pid,
            "latest_segment": traj[-1] if traj else None,
            "early_stop": early_stop,
            "tmux": sessions,
        }
        STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")

        if last_pid and not pid:
            _log(f"Train PID {last_pid} exited; checking sota relaunch")
            _ensure_sota_running(sessions)
        elif not any(s in sessions for s in ("sota-v1", "sota-v2", "sota-v3")) and not _sota_train_pid():
            _ensure_sota_running(sessions)
        last_pid = pid
        time.sleep(90)


if __name__ == "__main__":
    main()
