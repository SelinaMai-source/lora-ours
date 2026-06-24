#!/usr/bin/env python3
"""Supervisor for lora-published-setting-run_v2: health polling, status reports, agent wake."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "results/tables/published_setting_run_v2_manifest.csv"
STATUS_CSV = REPO / "results/tables/published_setting_run_v2_status.csv"
LOG_DIR = REPO / "results/logs/published_setting_run_v2"
QUEUE_LOG = REPO / "results/logs/published_setting_run_v2_queue.log"
SUPERVISOR_LOG = REPO / "results/logs/published_setting_run_v2_supervisor.log"
STATE_PATH = REPO / "results/logs/published_setting_run_v2_supervisor_state.json"
REPORT_PATH = REPO / "results/logs/published_setting_run_v2_status_report.md"
WAKE_JSON = REPO / "results/logs/AGENT_WAKE_PUBLISHED_SETTING_V2.json"
QUEUE_SESSION = "lora_published_setting_run_v2"
SUPERVISOR_SESSION = "lora_published_setting_run_v2_supervisor"

SENTINEL = "AGENT_LOOP_WAKE_PUBLISHED_SETTING_V2"
WAKE_COOLDOWN_HOURS = 24
STALE_SEG_SEC = 7200
TERMINAL = frozenset({
    "completed", "skipped_existing", "skipped_partial", "blocked",
    "failed", "early_stopped", "paused",
})
OOM_PATTERNS = ["CUDA out of memory", "OutOfMemoryError", "out of memory"]
CUDA_ERR_PATTERNS = ["CUDA error", "NCCL error", "cuDNN error"]

METHODS = [
    "ours_published", "o_lora", "lb_cl", "progressive_prompts",
    "continual_t0", "sequential_lora", "replay_lora",
]
BENCHMARKS = ["instrdialog", "instrdialogpp", "multiwoz_nlg", "trace"]


def _log(msg: str) -> None:
    SUPERVISOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with SUPERVISOR_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _train_processes() -> List[Tuple[int, str]]:
    try:
        out = subprocess.check_output(["pgrep", "-af", "core/train.py"], text=True).strip()
    except subprocess.CalledProcessError:
        return []
    procs: List[Tuple[int, str]] = []
    for line in out.splitlines():
        if "core/train.py" not in line:
            continue
        m = re.match(r"^(\d+)\s+(.*)", line.strip())
        if m:
            procs.append((int(m.group(1)), m.group(2)))
    return procs


def _config_frag(cmd: str) -> Optional[str]:
    m = re.search(r"--config\s+(\S+)", cmd)
    return Path(m.group(1)).name if m else None


def _manifest_row(run_name: str) -> Optional[Dict[str, str]]:
    for row in _read_csv(MANIFEST):
        if row.get("run_name") == run_name:
            return row
    return None


def _run_name_for_frag(frag: str) -> Optional[str]:
    for row in _read_csv(MANIFEST):
        cfg = row.get("config_path") or ""
        if frag in cfg or Path(cfg).name == frag:
            return row.get("run_name")
    return None


def _resolve_active() -> Tuple[Optional[str], Optional[int], Optional[str]]:
    for pid, cmd in _train_processes():
        frag = _config_frag(cmd)
        if frag:
            rn = _run_name_for_frag(frag)
            if rn:
                return rn, pid, frag
    for row in _read_csv(STATUS_CSV):
        if row.get("status") == "running":
            return row.get("run_name"), None, None
    return None, None, None


def _tail_lines(path: Path, n: int) -> List[str]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]


def _last_metrics(run_name: str) -> Optional[Dict[str, Any]]:
    mp = REPO / "results/runs" / run_name / "metrics.jsonl"
    if not mp.is_file():
        return None
    last = None
    for line in mp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                pass
    return last


def _metrics_mtime(run_name: str) -> Optional[float]:
    mp = REPO / "results/runs" / run_name / "metrics.jsonl"
    return mp.stat().st_mtime if mp.is_file() else None


def _manifest_stats() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in _read_csv(MANIFEST):
        st = row.get("status") or "unknown"
        counts[st] = counts.get(st, 0) + 1
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    return counts


def _matrix_rows() -> List[str]:
    rows_map: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in _read_csv(MANIFEST):
        rows_map[(row.get("benchmark", ""), row.get("method", ""))] = row
    lines = ["| benchmark | method | run_name | status | notes |", "| --- | --- | --- | --- | --- |"]
    for bench in BENCHMARKS:
        for method in METHODS:
            row = rows_map.get((bench, method))
            if not row:
                lines.append(f"| {bench} | {method} | — | missing | config 未生成 |")
                continue
            notes = (row.get("notes") or "")[:60]
            lines.append(
                f"| {bench} | {method} | `{row['run_name']}` | {row.get('status','?')} | {notes} |"
            )
    return lines


def _scan_log_issues(run_name: Optional[str]) -> List[str]:
    issues: List[str] = []
    paths: List[Path] = []
    if run_name:
        paths.append(LOG_DIR / f"{run_name}.log")
        paths.append(REPO / "results/logs" / f"{run_name}.log")
    paths.append(QUEUE_LOG)
    seen: set[str] = set()
    for p in paths:
        if not p.is_file():
            continue
        text = "\n".join(_tail_lines(p, 50))
        for pat in OOM_PATTERNS + CUDA_ERR_PATTERNS:
            if pat.lower() in text.lower():
                key = f"log:{pat}"
                if key not in seen:
                    seen.add(key)
                    issues.append(f"检测到 {pat}（{p.relative_to(REPO)}）")
    return issues


def _queue_all_done() -> bool:
    return any("[all-done]" in ln for ln in _tail_lines(QUEUE_LOG, 30))


def _tmux_alive(name: str) -> bool:
    try:
        subprocess.run(["tmux", "has-session", "-t", name], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _issue_hash(kind: str, detail: str) -> str:
    return hashlib.sha256(f"{kind}|{detail}".encode()).hexdigest()[:16]


def _should_wake(state: Dict[str, Any], issue_hash: str) -> bool:
    wakes = state.get("last_wake_issue_hash") or {}
    prev = wakes.get(issue_hash)
    if not prev:
        return True
    try:
        t = datetime.fromisoformat(prev)
        return datetime.now() - t > timedelta(hours=WAKE_COOLDOWN_HOURS)
    except ValueError:
        return True


def _record_wake(state: Dict[str, Any], issue_hash: str) -> None:
    wakes = state.setdefault("last_wake_issue_hash", {})
    wakes[issue_hash] = datetime.now().isoformat(timespec="seconds")


def _emit_wake(prompt: str, state: Dict[str, Any], kind: str, detail: str) -> None:
    ih = _issue_hash(kind, detail)
    if not _should_wake(state, ih):
        _log(f"wake suppressed (24h dedup): {kind} {detail[:80]}")
        return
    payload = {
        "prompt": prompt,
        "kind": kind,
        "detail": detail,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "manifest": str(MANIFEST.relative_to(REPO)),
            "status": str(STATUS_CSV.relative_to(REPO)),
            "queue_log": str(QUEUE_LOG.relative_to(REPO)),
            "supervisor_state": str(STATE_PATH.relative_to(REPO)),
        },
    }
    WAKE_JSON.parent.mkdir(parents=True, exist_ok=True)
    WAKE_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{SENTINEL} " + json.dumps({"prompt": prompt}, ensure_ascii=False), flush=True)
    _record_wake(state, ih)
    _log(f"AGENT_WAKE kind={kind} detail={detail[:120]}")
    _maybe_cursor_api(prompt)


def _maybe_cursor_api(prompt: str) -> None:
    api_key = os.environ.get("CURSOR_API_KEY", "").strip()
    if not api_key:
        return
    body = json.dumps({
        "prompt": {"text": prompt},
        "source": {"repository": str(REPO), "ref": "main"},
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.cursor.com/v0/agents",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            _log(f"Cursor API agent created: {resp.status}")
    except (urllib.error.URLError, TimeoutError) as e:
        _log(f"Cursor API optional call failed: {e}")


def _build_report(
    state: Dict[str, Any],
    active_run: Optional[str],
    train_pid: Optional[int],
    metrics: Optional[Dict[str, Any]],
    issues: List[str],
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats = _manifest_stats()
    lines = [
        f"# published_setting_run_v2 状态报告",
        f"",
        f"**更新时间**: {ts}",
        f"",
        f"## 汇总",
        f"",
        f"| 指标 | 值 |",
        f"| --- | ---: |",
    ]
    for k in sorted(stats.keys()):
        lines.append(f"| {k} | {stats[k]} |")
    lines += [
        f"",
        f"## 当前运行",
        f"",
        f"- **active_run**: `{active_run or '—'}`",
        f"- **train PID**: {train_pid or '—'}",
        f"- **queue tmux**: `{QUEUE_SESSION}` ({'alive' if _tmux_alive(QUEUE_SESSION) else 'missing'})",
        f"- **supervisor tmux**: `{SUPERVISOR_SESSION}` ({'alive' if _tmux_alive(SUPERVISOR_SESSION) else 'missing'})",
    ]
    if metrics:
        sid = metrics.get("segment_id", metrics.get("eval.num_seen_segments", 1))
        if isinstance(sid, int) and "eval.num_seen_segments" in metrics:
            sid = int(metrics["eval.num_seen_segments"]) - 1
        seen = metrics.get("eval.seen_avg_score", metrics.get("seen_avg_score", 0))
        ta = metrics.get("eval.seen_avg_task_aware_score", metrics.get("seen_avg_task_aware_score", 0))
        f1 = metrics.get("eval.token_f1_mean", metrics.get("token_f1_mean", 0))
        lines += [
            f"- **末次 eval**: seg={sid} seen={float(seen):.4f} ta={float(ta):.4f} f1={float(f1):.4f}",
        ]
    else:
        lines.append(f"- **末次 eval**: 无 metrics.jsonl")
    if issues:
        lines += ["", "## 待处理 Issues", ""]
        for i in issues:
            lines.append(f"- {i}")
    lines += ["", "## 矩阵 (7 methods × 4 benchmarks)", ""] + _matrix_rows()
    lines += [
        "",
        "## Agent 唤醒",
        "",
        f"- JSON: `{WAKE_JSON.relative_to(REPO)}`",
        f"- Sentinel: `{SENTINEL}`（供 Cursor `/loop` 或 monitored shell 捕获）",
        f"- 设置 `CURSOR_API_KEY` 可自动调用 Cursor Cloud Agent API",
    ]
    return "\n".join(lines) + "\n"


def _detect_issues(
    active_run: Optional[str],
    train_pid: Optional[int],
    state: Dict[str, Any],
) -> List[Tuple[str, str, str]]:
    """Return list of (kind, detail, prompt)."""
    found: List[Tuple[str, str, str]] = []
    stats = _manifest_stats()

    if active_run and not train_pid:
        detail = f"status=running 但无 train PID（{active_run}）"
        prompt = (
            f"v2 队列 run `{active_run}` 标记 running 但 pgrep 无 core/train.py。"
            f"请检查 {STATUS_CSV}、{QUEUE_LOG}，必要时修正 status 或重启队列 tmux `{QUEUE_SESSION}`。"
        )
        found.append(("stale_running", detail, prompt))

    if not _tmux_alive(QUEUE_SESSION) and stats.get("queued", 0) > 0:
        detail = "queue tmux 缺失且有 queued runs"
        prompt = (
            f"tmux `{QUEUE_SESSION}` 不存在但 manifest 仍有 queued。"
            f"请 `bash scripts/run_published_setting_run_v2_queue.sh` 于 tmux 中重启。"
        )
        found.append(("queue_down", detail, prompt))

    for row in _read_csv(MANIFEST):
        if row.get("status") in {"failed", "skipped_partial"}:
            rn = row["run_name"]
            detail = f"{rn} status={row['status']}: {row.get('notes','')}"
            prompt = (
                f"published_setting_run_v2 中 `{rn}` 为 {row['status']}，需人工处理。"
                f"日志: results/logs/published_setting_run_v2/{rn}.log"
            )
            found.append(("needs_human", detail, prompt))

    if active_run and train_pid:
        mt = _metrics_mtime(active_run)
        if mt and (time.time() - mt) > STALE_SEG_SEC:
            detail = f"{active_run} metrics.jsonl {int(time.time()-mt)}s 无更新"
            prompt = (
                f"训练 PID {train_pid} 仍在但 `{active_run}` 的 metrics.jsonl "
                f"超过 {STALE_SEG_SEC//3600}h 无新 eval。请检查 GPU/日志。"
            )
            found.append(("stale_eval", detail, prompt))

    for issue in _scan_log_issues(active_run):
        prompt = f"v2 训练日志异常: {issue}。请检查 OOM/ CUDA 并调整 batch 或重启 run。"
        found.append(("cuda_oom", issue, prompt))

    if _queue_all_done() and stats.get("queued", 0) == 0 and stats.get("running", 0) == 0:
        blocked = stats.get("blocked", 0)
        partial = stats.get("skipped_partial", 0)
        if blocked or partial:
            detail = f"queue all-done; blocked={blocked} skipped_partial={partial}"
            prompt = (
                f"v2 队列已 [all-done]。仍有 blocked={blocked}、skipped_partial={partial}。"
                f"请更新 manifest 或人工清理 partial run 后重新入队。"
            )
            found.append(("all_done_with_gaps", detail, prompt))

    procs = _train_processes()
    if len(procs) > 1:
        detail = f"多个 train 进程: {[p[0] for p in procs]}"
        prompt = "检测到多个 core/train.py（违反单 GPU 约束）。请保留 v2 队列进程并终止其余。"
        found.append(("multi_train", detail, prompt))

    return found


def tick(loop: bool, interval_min: int, interval_max: int) -> int:
    state: Dict[str, Any] = {}
    if STATE_PATH.is_file():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}

    while True:
        active_run, train_pid, _ = _resolve_active()
        metrics = _last_metrics(active_run) if active_run else None
        issues_raw = _detect_issues(active_run, train_pid, state)
        issue_texts = [d for _, d, _ in issues_raw]

        for kind, detail, prompt in issues_raw:
            _emit_wake(prompt, state, kind, detail)

        stats = _manifest_stats()
        state.update({
            "last_tick": datetime.now().isoformat(timespec="seconds"),
            "active_run": active_run,
            "train_pid": train_pid,
            "manifest_stats": stats,
            "issues": issue_texts,
            "queue_tmux_alive": _tmux_alive(QUEUE_SESSION),
            "metrics_tail": metrics,
        })
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        REPORT_PATH.write_text(
            _build_report(state, active_run, train_pid, metrics, issue_texts),
            encoding="utf-8",
        )

        if not loop:
            break
        sleep_sec = random.randint(interval_min, interval_max)
        time.sleep(sleep_sec)

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Supervisor for published_setting_run_v2")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--interval-min", type=int, default=90)
    p.add_argument("--interval-max", type=int, default=120)
    args = p.parse_args()
    return tick(loop=args.loop, interval_min=args.interval_min, interval_max=args.interval_max)


if __name__ == "__main__":
    sys.exit(main())
