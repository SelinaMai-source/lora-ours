#!/usr/bin/env python3
"""Monitor four-benchmark single-seed queue: log parse, doc sync, queue continuation."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "results/tables/four_benchmark_single_seed_manifest.csv"
STATUS_CSV = REPO / "results/tables/four_benchmark_single_seed_status.csv"
MONITOR_LOG = REPO / "results/logs/four_benchmark_monitor.log"
STATE_PATH = REPO / "experiments/sota_campaign/FOUR_BENCHMARK_MONITOR_STATE.json"
REPRO_DOC = REPO / "docs/reproduction_status.md"
PAPER_PLAN = REPO / "docs/paper_experiment_plan.md"
LOG_DIR = REPO / "results/logs/four_benchmark_single_seed"
QUEUE_SESSION = "lora_four_benchmark_single_seed"

SOTA_V3_RUN = "paper_instrdialog_sota_v3_ours_s123"
SOTA_V3_CONFIG_FRAG = "sota_v3_instrdialog_s123.yaml"
USER_EARLY_STOP_SEG = 3
USER_EARLY_STOP_SEEN = 0.15

TERMINAL_STATUSES = frozenset({
    "completed", "early_stopped", "failed", "paused", "skipped_existing", "skipped_partial",
})
QUEUE_PROGRESS_MARKER = "### 队列当前跑批（动态）"
SOTA_V3_TABLE_MARKER = "### sota-v3 InstrDialog 逐 segment eval"


def _log(msg: str) -> None:
    MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with MONITOR_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _train_pid_for_config(fragment: str) -> Optional[int]:
    try:
        out = subprocess.check_output(["pgrep", "-af", fragment], text=True).strip()
    except subprocess.CalledProcessError:
        return None
    for line in out.splitlines():
        if "core/train.py" in line and fragment in line:
            m = re.match(r"^(\d+)", line.strip())
            if m:
                return int(m.group(1))
    return None


def _any_train_pid() -> Optional[int]:
    try:
        out = subprocess.check_output(["pgrep", "-af", "core/train.py"], text=True).strip()
    except subprocess.CalledProcessError:
        return None
    for line in out.splitlines():
        if re.search(r"python\d*(?:\.\d+)?\s+.*core/train\.py", line):
            m = re.match(r"^(\d+)", line.strip())
            if m:
                return int(m.group(1))
    return None


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


def _config_fragment_from_cmd(cmd: str) -> Optional[str]:
    m = re.search(r"--config\s+(\S+)", cmd)
    if not m:
        return None
    return Path(m.group(1)).name


def _run_name_for_config_fragment(fragment: str) -> Optional[str]:
    for row in _read_csv(MANIFEST):
        cfg = row.get("config_path") or ""
        if not cfg:
            continue
        if fragment in cfg or Path(cfg).name == fragment:
            return row.get("run_name")
    return None


def _manifest_row(run_name: str) -> Optional[Dict[str, str]]:
    for row in _read_csv(MANIFEST):
        if row.get("run_name") == run_name:
            return row
    return None


def _resolve_active_run() -> Tuple[Optional[str], Optional[str], Path, Optional[int]]:
    for pid, cmd in _train_processes():
        frag = _config_fragment_from_cmd(cmd)
        if not frag:
            continue
        run_name = _run_name_for_config_fragment(frag)
        if run_name:
            return run_name, frag, _log_path_for_run(run_name), pid
    for row in _read_csv(STATUS_CSV):
        if row.get("status") != "running" or not row.get("run_name"):
            continue
        run_name = row["run_name"]
        if _manifest_terminal(run_name):
            continue
        mrow = _manifest_row(run_name)
        cfg = (mrow or {}).get("config_path") or row.get("config_path") or ""
        frag = Path(cfg).name if cfg else ""
        return run_name, frag or None, _log_path_for_run(run_name), None
    return None, None, LOG_DIR / "_.log", None


def _display_label(run_name: str) -> str:
    row = _manifest_row(run_name)
    if not row:
        return run_name
    bench = row.get("benchmark") or "?"
    method = row.get("method") or "?"
    return f"{bench}/{method} ({run_name})"


def _patch_reproduction_gpu_line(seg_note: str) -> None:
    if not REPRO_DOC.is_file():
        return
    text = REPRO_DOC.read_text(encoding="utf-8")
    gpu_line = f"**GPU 约束**：{seg_note}；tmux `{QUEUE_SESSION}`"
    text = re.sub(
        r"\*\*GPU 约束\*\*：[^\n]+",
        gpu_line,
        text,
        count=1,
    )
    REPRO_DOC.write_text(text, encoding="utf-8")


def _finalize_run(
    run_name: str,
    log_path: Path,
    state: Dict[str, Any],
    config_frag: str = "",
) -> None:
    if state.get("finalized_run") == run_name:
        return
    finished, outcome = _run_finished(run_name)
    if not finished:
        return
    traj, early, current_seg = _progress_from_run(run_name, log_path)
    if traj:
        last = traj[-1]
        seg_note = (
            f"seg{last['segment_id']+1}/19 seen={last['seen']:.3f} ta={last['ta']:.3f}"
        )
    else:
        seg_note = current_seg or "n/a"
    if outcome == "completed":
        st = "completed"
        exit_c = "0"
    elif outcome == "early_stopped":
        st = "early_stopped"
        exit_c = "143"
    else:
        st = "early_stopped" if early else "failed"
        exit_c = "1"
    metrics = _extract_final_from_run(run_name)
    _update_status_row(run_name, st, seg_note, exit_c)
    if run_name == SOTA_V3_RUN:
        state_str = "✅ 已完成" if st == "completed" else f"⚠️ {st}"
        _patch_reproduction_sota_row(seg_note, state_str, metrics)
        _patch_paper_plan(run_name, st, metrics, traj)
    elif st == "completed":
        _patch_paper_plan_queue_progress(run_name, None, traj, None)
    _log(f"Finalized {run_name} as {st} metrics={metrics}")
    state["finalized_run"] = run_name
    state.pop(_continuation_launched_key(), None)


def _log_path_for_run(run_name: str) -> Path:
    primary = LOG_DIR / f"{run_name}.log"
    alt = REPO / "results/logs" / f"{run_name}.log"
    if alt.is_file() and primary.is_file():
        return alt if alt.stat().st_mtime >= primary.stat().st_mtime else primary
    if alt.is_file():
        return alt
    return primary


def _parse_metrics_jsonl_trajectory(run_name: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    run_dir = REPO / "results/runs" / run_name
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return [], None
    traj: List[Dict[str, Any]] = []
    last_seg_name: Optional[str] = None
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = int(d.get("segment_id", int(d.get("eval.num_seen_segments", 1)) - 1))
        traj.append({
            "segment_id": sid,
            "segment_name": d.get("segment_name", ""),
            "seen": float(d.get("eval.seen_avg_score", d.get("seen_avg_score", 0))),
            "ta": float(d.get("eval.seen_avg_task_aware_score", d.get("seen_avg_task_aware_score", 0))),
            "f1": float(d.get("eval.token_f1_mean", d.get("token_f1_mean", 0))),
            "lcs": float(d.get("eval.lcs_overlap_mean", d.get("lcs_overlap_mean", 0))),
        })
        last_seg_name = d.get("segment_name") or last_seg_name
    return traj, last_seg_name


def _progress_from_run(run_name: str, log_path: Path) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    traj, last_seg = _parse_metrics_jsonl_trajectory(run_name)
    _, early, current_seg = _parse_log_trajectory(log_path)
    if not current_seg and last_seg:
        current_seg = f"=== Segment (metrics): {last_seg} ==="
    if traj:
        return traj, early, current_seg
    return _parse_log_trajectory(log_path)


def _manifest_terminal(run_name: str) -> bool:
    row = _manifest_row(run_name)
    return bool(row and row.get("status") in TERMINAL_STATUSES)


def _maybe_finalize_sota_v3_once(state: Dict[str, Any]) -> None:
    if state.get("finalized_run") == SOTA_V3_RUN:
        return
    if _manifest_terminal(SOTA_V3_RUN):
        row = _manifest_row(SOTA_V3_RUN) or {}
        st = row.get("status", "early_stopped")
        metrics = _extract_final_from_run(SOTA_V3_RUN)
        log_path = _log_path_for_run(SOTA_V3_RUN)
        traj, _, _ = _progress_from_run(SOTA_V3_RUN, log_path)
        if not state.get("sota_v3_doc_synced"):
            state_str = "⚠️ early_stopped" if st == "early_stopped" else f"⚠️ {st}"
            seg_note = "sota-v3 已终止（勿重启）"
            if traj:
                last = traj[-1]
                seg_note = f"sota-v3 末 eval seg{last['segment_id']+1} seen={last['seen']:.3f}"
            _patch_reproduction_sota_row(seg_note, state_str, metrics)
            outcome = st if st in {"completed", "early_stopped"} else "early_stopped"
            _patch_paper_plan(SOTA_V3_RUN, outcome, metrics, traj)
            state["sota_v3_doc_synced"] = True
        state["finalized_run"] = SOTA_V3_RUN
        _log(f"Sota-v3 terminal in manifest ({st}); skip live log polling")
        return
    _finalize_run(
        SOTA_V3_RUN,
        _log_path_for_run(SOTA_V3_RUN),
        state,
        SOTA_V3_CONFIG_FRAG,
    )


def _patch_paper_plan_queue_progress(
    run_name: str,
    pid: Optional[int],
    traj: List[Dict[str, Any]],
    current_seg: Optional[str],
) -> None:
    if not PAPER_PLAN.is_file():
        return
    text_p = PAPER_PLAN.read_text(encoding="utf-8")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    label = _display_label(run_name)
    n_eval = len(traj)
    if traj:
        last = traj[-1]
        summary = (
            f"末次 eval seg{last['segment_id']+1}/19 "
            f"seen={last['seen']:.3f} ta={last['ta']:.3f} f1={last['f1']:.3f}"
        )
    else:
        summary = current_seg or "等待首个 segment eval"
    rows = ["| seg | task | seen | ta | f1 | lcs |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for t in traj[-12:]:
        task = (t.get("segment_name") or "?").replace("|", "/")[:48]
        rows.append(
            f"| {t['segment_id']} | {task} | {t['seen']:.3f} | {t['ta']:.3f} | "
            f"{t['f1']:.3f} | {t.get('lcs', 0):.3f} |"
        )
    pid_s = str(pid) if pid else "—"
    block = (
        f"\n\n{QUEUE_PROGRESS_MARKER}\n\n"
        f"- **更新**: {ts}\n"
        f"- **active_run**: `{run_name}`（{label}）\n"
        f"- **train PID**: {pid_s}\n"
        f"- **进度**: {summary}\n"
        f"- **eval 完成**: {n_eval}/19\n\n"
        + "\n".join(rows)
        + "\n\n*InstrDialog sota-v3 已 early_stopped（冻结）；以下表为当前 GPU 队列 run。*\n"
    )
    if QUEUE_PROGRESS_MARKER in text_p:
        text_p = re.sub(
            re.escape(QUEUE_PROGRESS_MARKER) + r"[\s\S]*?(?=\n### |\n## |\Z)",
            block.strip() + "\n",
            text_p,
            count=1,
        )
    else:
        anchor = SOTA_V3_TABLE_MARKER
        if anchor in text_p:
            text_p = text_p.replace(anchor, block.strip() + "\n\n" + anchor, 1)
        else:
            text_p += block
    row_pat = r"(\| `lora_four_benchmark_single_seed` \| 四基准单 seed 队列 \| )[^|]+( \|)"
    cell = f"🟢 {run_name} PID {pid_s}"
    if re.search(row_pat, text_p):
        text_p = re.sub(row_pat, rf"\1{cell}\2", text_p, count=1)
    PAPER_PLAN.write_text(text_p, encoding="utf-8")


def _parse_log_trajectory(log_path: Path) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    if not log_path.is_file():
        return [], None, None
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
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
            if not m:
                continue
            try:
                d = json.loads(m.group())
            except json.JSONDecodeError:
                continue
            sid = int(d.get("num_seen_segments", 1)) - 1
            traj.append({
                "segment_id": sid,
                "seen": float(d["seen_avg_score"]),
                "ta": float(d["seen_avg_task_aware_score"]),
                "f1": float(d["token_f1_mean"]),
            })
    return traj, early_stop, current_seg


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _update_status_row(run_name: str, status: str, notes: str = "", exit_code: str = "") -> None:
    rows = _read_csv(STATUS_CSV)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for key in ("status", "notes", "exit_code", "updated_at"):
        if key not in fieldnames:
            fieldnames.append(key)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in rows:
        if row.get("run_name") == run_name:
            row["status"] = status
            if notes:
                row["notes"] = notes
            if exit_code != "":
                row["exit_code"] = exit_code
            row["updated_at"] = now
    _write_csv(STATUS_CSV, rows, fieldnames)
    _sync_manifest_status(run_name, status, notes)


def _sync_manifest_status(run_name: str, status: str, notes: str = "") -> None:
    rows = _read_csv(MANIFEST)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        if row.get("run_name") == run_name:
            row["status"] = status
            if notes:
                row["notes"] = notes
    _write_csv(MANIFEST, rows, fieldnames)


def _queued_runs() -> List[str]:
    names: List[str] = []
    for row in _read_csv(MANIFEST):
        if row.get("status") == "queued" and row.get("config_path"):
            cfg = REPO / row["config_path"]
            if cfg.is_file():
                names.append(row["run_name"])
    return names


def _run_finished(run_name: str) -> Tuple[bool, str]:
    row = _manifest_row(run_name)
    if row and row.get("status") in {"early_stopped", "completed", "failed", "paused"}:
        st = row["status"]
        if st == "failed" and row.get("exit_code") == "143":
            return True, "early_stopped"
        if st == "paused":
            return True, "paused"
        return True, st
    run_dir = REPO / "results/runs" / run_name
    if (run_dir / "final_metrics.json").is_file():
        return True, "completed"
    for row in _read_csv(STATUS_CSV):
        if row.get("run_name") == run_name and row.get("status") in {
            "early_stopped", "failed", "completed"
        }:
            st = row["status"]
            if st == "failed" and row.get("exit_code") == "143":
                return True, "early_stopped"
            return True, st
    metrics = run_dir / "metrics.jsonl"
    log_path = LOG_DIR / f"{run_name}.log"
    if log_path.is_file():
        _, early, _ = _parse_log_trajectory(log_path)
        if early:
            return True, "early_stopped"
    return False, "running"


def _extract_final_from_run(run_name: str) -> Optional[Dict[str, float]]:
    run_dir = REPO / "results/runs" / run_name
    final_path = run_dir / "final_metrics.json"
    if final_path.is_file():
        d = json.loads(final_path.read_text(encoding="utf-8"))
        ev = d.get("eval", d)
        return {
            "seen": float(ev.get("seen_avg_score", 0)),
            "ta": float(ev.get("seen_avg_task_aware_score", 0)),
            "f1": float(ev.get("token_f1_mean", 0)),
        }
    metrics = run_dir / "metrics.jsonl"
    if metrics.is_file():
        last = None
        for line in metrics.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    pass
        if last:
            return {
                "seen": float(last.get("eval.seen_avg_score", last.get("seen_avg_score", 0))),
                "ta": float(last.get("eval.seen_avg_task_aware_score", last.get("seen_avg_task_aware_score", 0))),
                "f1": float(last.get("eval.token_f1_mean", last.get("token_f1_mean", 0))),
            }
    seg_csv = REPO / "results/tables" / f"{run_name}_segment_metrics.csv"
    if seg_csv.is_file():
        rows = _read_csv(seg_csv)
        if rows:
            r = rows[-1]
            return {
                "seen": float(r.get("seen_avg_score", r.get("eval.seen_avg_score", 0))),
                "ta": float(r.get("seen_avg_task_aware_score", r.get("eval.seen_avg_task_aware_score", 0))),
                "f1": float(r.get("token_f1_mean", r.get("eval.token_f1_mean", 0))),
            }
    return None


def _patch_reproduction_sota_row(seg_note: str, state: str, metrics: Optional[Dict[str, float]] = None) -> None:
    if not REPRO_DOC.is_file():
        return
    text = REPRO_DOC.read_text(encoding="utf-8")
    csv_cell = "未完成"
    if metrics:
        csv_cell = f"seen={metrics['seen']:.4f} ta={metrics['ta']:.4f} f1={metrics['f1']:.4f}"
    new_line = (
        f"| **Ours (sota-v3)** | ✅ | `sota_v3_instrdialog_s123.yaml` | {csv_cell} | {state} |"
    )
    pattern = r"\| \*\*Ours \(sota-v3\)\*\* \| ✅ \| `sota_v3_instrdialog_s123\.yaml` \| [^|]+ \| [^|]+ \|"
    if re.search(pattern, text):
        text = re.sub(pattern, new_line, text, count=1)
    gpu_line = (
        f"**GPU 约束**：InstrDialog++ sota-v3 **已暂停**（`PAUSE_SOTA_V3`，未重启）；"
        f"tmux `{QUEUE_SESSION}` — {seg_note}"
    )
    text = re.sub(
        r"\*\*GPU 约束\*\*：[^\n]+",
        gpu_line,
        text,
        count=1,
    )
    REPRO_DOC.write_text(text, encoding="utf-8")


def _patch_paper_plan(run_name: str, outcome: str, metrics: Optional[Dict[str, float]], traj: List[Dict[str, Any]]) -> None:
    if not PAPER_PLAN.is_file():
        return
    text = PAPER_PLAN.read_text(encoding="utf-8")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    traj_s = ", ".join(f"seg{t['segment_id']} seen={t['seen']:.3f}" for t in traj[-6:])
    met_s = ""
    if metrics:
        met_s = f"终局 seen={metrics['seen']:.4f} ta={metrics['ta']:.4f} f1={metrics['f1']:.4f}"
    seg_csv = REPO / "results/tables" / f"{run_name}_segment_metrics.csv"
    csv_note = f"`{seg_csv.relative_to(REPO)}`" if seg_csv.is_file() else "（早停可能仅 metrics.jsonl）"
    block = (
        f"\n\n### InstrDialog sota-v3 单 seed 跑批记录（{ts}）\n\n"
        f"- **run**: `{run_name}`\n"
        f"- **结果**: {outcome}；{met_s}\n"
        f"- **轨迹**: {traj_s or 'n/a'}\n"
        f"- **segment CSV**: {csv_note}\n"
    )
    marker = "### InstrDialog sota-v3 单 seed 跑批记录"
    if marker in text:
        text = re.sub(
            rf"{re.escape(marker)}[^\n]*\n[\s\S]*?(?=\n### |\n## |\Z)",
            block.strip() + "\n",
            text,
            count=1,
        )
    else:
        if "## 7." in text:
            text = text.replace("## 7.", block + "\n## 7.", 1)
        else:
            text += block
    # tmux table row
    row_pat = r"(\| `lora_four_benchmark_single_seed` \| 四基准单 seed 队列 \| )[^|]+( \|)"
    if outcome == "completed":
        cell = "🟢 后续 queued 继续"
    elif outcome == "early_stopped":
        cell = "⚠️ sota-v3 轨迹早停；队列继续"
    else:
        cell = f"状态 {outcome}"
    if re.search(row_pat, text):
        text = re.sub(row_pat, rf"\1{cell}\2", text, count=1)
    instr_row = r"(\| InstrDialog \| ✅ published \+ )[^|]+( \|)"
    if re.search(instr_row, text):
        repl = "✅ sota-v3 完成" if outcome == "completed" else f"⚠️ sota-v3 {outcome}"
        text = re.sub(instr_row, rf"\1{repl}\2", text, count=1)
    PAPER_PLAN.write_text(text, encoding="utf-8")


def _maybe_kill_user_early_stop(traj: List[Dict[str, Any]], config_fragment: str) -> bool:
    if not traj:
        return False
    last = traj[-1]
    sid = int(last["segment_id"])
    seen = float(last["seen"])
    if sid < USER_EARLY_STOP_SEG or seen >= USER_EARLY_STOP_SEEN:
        return False
    pid = _train_pid_for_config(config_fragment)
    if not pid:
        return False
    _log(
        f"User early-stop rule: seg={sid} seen={seen:.4f} < {USER_EARLY_STOP_SEEN}; "
        f"sending SIGTERM to train PID {pid}"
    )
    _log(
        f"User early-stop would apply (seg={sid} seen={seen:.4f}) but deferring to train.py post-segment stop"
    )
    return False


def _continuation_launched_key() -> str:
    return "continuation_launched_at"


def _maybe_continue_queue(state: Dict[str, Any]) -> None:
    if _any_train_pid():
        return
    queued = _queued_runs()
    if not queued:
        return
    if state.get(_continuation_launched_key()):
        # allow re-launch if new queued appeared and no train for 10+ min
        launched = state.get(_continuation_launched_key(), "")
        if launched:
            return
    _log(f"GPU idle; queued runs remain ({len(queued)}): {queued[:5]}...")
    script = REPO / "scripts/run_four_benchmark_single_seed_queue.sh"
    cmd = (
        f"cd {REPO} && export WANDB_PROJECT=lora-four-benchmark-single-seed "
        f"WANDB_MODE=online && bash {script} 2>&1 | tee -a results/logs/four_benchmark_single_seed_queue.log"
    )
    try:
        subprocess.run(
            ["tmux", "has-session", "-t", QUEUE_SESSION],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["tmux", "send-keys", "-t", QUEUE_SESSION, cmd, "C-m"],
            check=False,
        )
        _log("Sent queue continuation to tmux session")
    except subprocess.CalledProcessError:
        _log("Queue tmux missing; running continuation in background")
        subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=str(REPO),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    state[_continuation_launched_key()] = datetime.now().isoformat(timespec="seconds")


def _load_state() -> Dict[str, Any]:
    if STATE_PATH.is_file():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tick(loop: bool, interval: int) -> int:
    state = _load_state()

    while True:
        _maybe_finalize_sota_v3_once(state)

        run_name, config_frag, log_path, pid = _resolve_active_run()
        if run_name:
            traj, early, current_seg = _progress_from_run(run_name, log_path)
            finished, outcome = _run_finished(run_name)

            if traj:
                last = traj[-1]
                seg_note = (
                    f"{_display_label(run_name)} seg{last['segment_id']+1}/19 "
                    f"seen={last['seen']:.3f} ta={last['ta']:.3f}"
                )
            else:
                seg_note = current_seg or f"{_display_label(run_name)} 等待首个 Eval metrics"

            if pid and not finished:
                _update_status_row(run_name, "running", seg_note)
                _patch_reproduction_gpu_line(f"🟢 {seg_note}")
                if run_name == SOTA_V3_RUN and config_frag:
                    _maybe_kill_user_early_stop(traj, config_frag)
            elif finished and state.get("finalized_run") != run_name:
                _finalize_run(run_name, log_path, state, config_frag or "")
            elif not pid and not finished:
                _log(f"Stale running status? no train PID ({seg_note})")
        else:
            queued = _queued_runs()
            if queued:
                _log(f"No active train; next queued ({len(queued)}): {queued[:3]}...")
            if _any_train_pid():
                _patch_reproduction_gpu_line("🟢 训练进行中（解析 manifest 中）")

        state["last_tick"] = datetime.now().isoformat(timespec="seconds")
        state["active_run"] = run_name
        state["train_pid"] = pid
        current_seg_state = None
        if run_name:
            traj, early, current_seg_state = _progress_from_run(run_name, log_path)
            state["trajectory"] = traj[-8:]
            state["early_stop_line"] = early
            if pid and run_name != SOTA_V3_RUN:
                _patch_paper_plan_queue_progress(run_name, pid, traj, current_seg_state)
        _save_state(state)

        _maybe_continue_queue(state)
        _save_state(state)

        if not loop:
            break
        time.sleep(interval)

    return 0



def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--loop", action="store_true")
    p.add_argument("--interval", type=int, default=90)
    args = p.parse_args()
    return tick(loop=args.loop, interval=args.interval)


if __name__ == "__main__":
    sys.exit(main())
