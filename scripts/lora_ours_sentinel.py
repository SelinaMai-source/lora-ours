#!/usr/bin/env python3
"""Aggregate three-suite experiment status and wake the Cursor agent on blockers.

Scans ``results/logs/*_status.json`` and ``*_status.md`` for CITB, Standard, and
Dialogue suites. Writes a consolidated report and optionally emits
``SOTA_AGENT_WAKE.flag`` for ``scripts/sota_agent_loop.sh``.

The monitor is intentionally conservative: it never kills training jobs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO / "results" / "logs"
STATUS_JSON = LOGS_DIR / "lora_ours_sentinel_status.json"
STATUS_MD = LOGS_DIR / "lora_ours_sentinel_status.md"
MONITOR_LOG = LOGS_DIR / "lora_ours_sentinel.monitor.log"
AGENT_WAKE_FLAG = REPO / "SOTA_AGENT_WAKE.flag"
AGENT_WAKE_LOG = REPO / "SOTA_AGENT_WAKE.log"

SENTINEL_PREFIX = "AGENT_LOOP_WAKE_LORA_OURS"

SOTA_TARGETS = {
    "CITB": {"metric": "rouge_l_ar", "baseline": 40.4, "target": 53.87, "unit": "ROUGE-L AR"},
    "Standard": {"metric": "avg_em", "baseline": 76.7, "target": 84.5, "unit": "avg EM"},
    "Dialogue": {"metric": "bleu4", "baseline": 0.701, "target": 0.935, "unit": "BLEU-4"},
}

SUITE_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "CITB": ("citb_", "citb-", "instrdialog"),
    "Standard": ("olora_", "standard_peft_", "standard_olora", "published_base_olora"),
    "Dialogue": ("dialogue_", "arper_", "todcl_"),
}

FAILURE_STATES = {
    "failed",
    "incomplete",
    "crashed",
    "stalled",
    "stopped_incomplete",
    "unknown_not_running",
    "early_stopped",
    "blocked",
    "stopped_low_or_failed",
    "completed_no_gain",
    "completed_all_zero",
    "completed_or_stopped",
    "exited_with_artifact",
    "not_started",
}

RUNNING_STATES = {"running", "training", "active"}

AGENT_PROMPT = (
    "Read results/logs/lora_ours_sentinel_status.md and lora_ours_sentinel_status.json. "
    "For any suite with wake_reason or pending_action, run failure analysis per "
    "docs/official_alignment/ccfa_experiment_gate.md: diagnose → single-mechanism vN+1 "
    "patch → smoke gate → formal. Do not kill healthy training. Update "
    "docs/official_alignment/status.md and docs/experiments/* as needed."
)


@dataclass
class StatusRecord:
    suite: str
    basename: str
    path: Path
    mtime: float
    state: str
    run_id: str
    reason: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    source: str = "json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(msg: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with MONITOR_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _classify_suite(name: str) -> Optional[str]:
    lowered = name.lower()
    for suite, patterns in SUITE_PATTERNS.items():
        if any(lowered.startswith(p) or p in lowered for p in patterns):
            return suite
    return None


def _extract_state(data: Dict[str, Any]) -> Tuple[str, str]:
    for key in ("state",):
        if isinstance(data.get(key), str):
            return data[key], str(data.get("reason") or data.get("diagnosis", {}).get("reason") or "")
    classification = data.get("classification")
    if isinstance(classification, dict):
        return str(classification.get("state") or "unknown"), str(classification.get("reason") or "")
    return "unknown", ""


def _extract_run_id(data: Dict[str, Any], basename: str) -> str:
    for key in ("run_id", "run_name"):
        if data.get(key):
            return str(data[key])
    return basename.replace("_status", "")


def _parse_md_state(text: str) -> Tuple[str, str, Dict[str, Any]]:
    state = "unknown"
    reason = ""
    metrics: Dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        m = re.match(r"^[-*]\s*(?:state|State)\s*[:=]\s*`?([^`\s]+)`?", stripped, re.I)
        if m:
            state = m.group(1).lower()
        m = re.match(r"^[-*]\s*(?:reason|Reason)\s*[:=]\s*(.+)$", stripped, re.I)
        if m:
            reason = m.group(1).strip().strip("`")
        m = re.match(r"^[-*]\s*`predict_official_rougeL`\s*[:=]\s*([0-9.]+)", stripped)
        if m:
            metrics["rouge_l_ar"] = float(m.group(1))
        m = re.match(r"^[-*]\s*(?:observed_avg_exact|round_avg_exact|final avg)\s*[:=]\s*([0-9.]+)", stripped, re.I)
        if m:
            metrics["avg_em"] = float(m.group(1))
        m = re.match(r"^[-*]\s*(?:BLEU4?|bleu4?)\s*[:=]\s*([0-9.]+)", stripped, re.I)
        if m:
            metrics["bleu4"] = float(m.group(1))
        m = re.match(r"^[-*]\s*SER\s*[:=]\s*([0-9.]+)", stripped, re.I)
        if m:
            metrics["ser"] = float(m.group(1))
    return state, reason, metrics


def _extract_metrics(data: Dict[str, Any], suite: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    log_block = data.get("log") or {}
    latest_eval = log_block.get("latest_eval") or {}
    artifacts = data.get("artifacts") or {}
    final_block = artifacts.get("final_metrics.json") or {}
    final = final_block.get("final") or final_block

    if suite == "CITB":
        for key in ("predict_official_rougeL", "rouge_l_ar"):
            if key in data:
                metrics["rouge_l_ar"] = float(data[key])
        if "rouge_l_ar" not in metrics and isinstance(final, dict):
            val = final.get("eval.seen_avg_task_aware_score") or final.get("rouge_l_ar")
            if val is not None:
                metrics["rouge_l_ar"] = float(val)
        if "rouge_l_ar" not in metrics:
            ccfa = (final_block.get("ccfa_summary") or {}) if isinstance(final_block, dict) else {}
            citb = ccfa.get("citb") or {}
            if citb.get("ar") is not None:
                metrics["rouge_l_ar"] = float(citb["ar"]) * (100.0 if float(citb["ar"]) <= 1.0 else 1.0)

    if suite == "Standard":
        for key in ("observed_avg_exact", "round_avg_exact", "final_avg_em"):
            if key in data:
                metrics["avg_em"] = float(data[key])
        if "avg_em" not in metrics:
            val = latest_eval.get("seen_avg_score") or latest_eval.get("current_score")
            if val is not None:
                metrics["avg_em"] = float(val) * (100.0 if float(val) <= 1.0 else 1.0)
        if "avg_em" not in metrics and isinstance(final, dict):
            val = final.get("eval.seen_avg_score")
            if val is not None:
                metrics["avg_em"] = float(val) * (100.0 if float(val) <= 1.0 else 1.0)

    if suite == "Dialogue":
        ccfa = {}
        if isinstance(final_block, dict):
            ccfa = final_block.get("ccfa_summary") or {}
        dialogue = ccfa.get("dialogue_nlg") or {}
        if dialogue.get("bleu4") is not None:
            metrics["bleu4"] = float(dialogue["bleu4"])
        if dialogue.get("ser") is not None:
            metrics["ser"] = float(dialogue["ser"])
        if "bleu4" not in metrics:
            val = latest_eval.get("bleu_mean") or (final or {}).get("eval.bleu_mean")
            if val is not None:
                metrics["bleu4"] = float(val)
        if "ser" not in metrics:
            val = latest_eval.get("slot_error_rate") or (final or {}).get("eval.slot_error_rate")
            if val is not None:
                metrics["ser"] = float(val)

    seg = log_block.get("latest_segment")
    if seg is not None:
        metrics["latest_segment"] = seg
    metrics["log_age_seconds"] = log_block.get("log_age_seconds")
    return metrics


def _scan_status_files() -> List[Path]:
    files: List[Path] = []
    if not LOGS_DIR.is_dir():
        return files
    for path in LOGS_DIR.iterdir():
        if path.name.startswith("lora_ours_sentinel"):
            continue
        if path.name.endswith("_status.json") or path.name.endswith("_status.md"):
            files.append(path)
    return files


def _collect_records() -> Dict[str, StatusRecord]:
    by_suite: Dict[str, List[StatusRecord]] = {suite: [] for suite in SUITE_PATTERNS}
    for path in _scan_status_files():
        suite = _classify_suite(path.name)
        if not suite:
            continue
        basename = path.name
        mtime = path.stat().st_mtime
        if path.suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            state, reason = _extract_state(data)
            record = StatusRecord(
                suite=suite,
                basename=basename,
                path=path,
                mtime=mtime,
                state=state,
                run_id=_extract_run_id(data, basename),
                reason=reason,
                metrics=_extract_metrics(data, suite),
                source="json",
            )
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            state, reason, metrics = _parse_md_state(text)
            record = StatusRecord(
                suite=suite,
                basename=basename,
                path=path,
                mtime=mtime,
                state=state,
                run_id=basename.replace("_status.md", ""),
                reason=reason,
                metrics=metrics,
                source="md",
            )
        by_suite[suite].append(record)

    latest: Dict[str, StatusRecord] = {}
    for suite, records in by_suite.items():
        if not records:
            continue
        json_records = [r for r in records if r.source == "json"]
        pool = json_records or records
        latest[suite] = max(pool, key=lambda r: r.mtime)
    return latest


def _gpu_compute_empty() -> bool:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return not out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def _train_processes() -> List[str]:
    try:
        out = subprocess.check_output(["pgrep", "-af", r"core/train\.py"], text=True)
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except subprocess.CalledProcessError:
        return []


def _needs_wake(record: StatusRecord) -> Tuple[bool, str]:
    state = record.state.lower()
    if state in FAILURE_STATES:
        return True, f"suite={record.suite} state={state}"
    if state not in RUNNING_STATES and state not in {"completed", "complete", "done"}:
        if record.reason and any(x in record.reason.lower() for x in ("traceback", "oom", "blocked", "early")):
            return True, f"suite={record.suite} reason={record.reason[:120]}"
    return False, ""


def _gap_line(suite: str, metrics: Dict[str, Any]) -> str:
    target_info = SOTA_TARGETS.get(suite)
    if not target_info:
        return "—"
    metric_key = target_info["metric"]
    val = metrics.get(metric_key)
    if val is None:
        return f"no {target_info['unit']} yet (target {target_info['target']})"
    gap = target_info["target"] - float(val)
    return f"{val:.4f} / target {target_info['target']} (gap {gap:+.4f})"


def _build_report(records: Dict[str, StatusRecord]) -> Dict[str, Any]:
    wake_items: List[Dict[str, Any]] = []
    suite_rows: List[Dict[str, Any]] = []
    for suite in ("CITB", "Standard", "Dialogue"):
        record = records.get(suite)
        if not record:
            suite_rows.append(
                {
                    "suite": suite,
                    "state": "no_status_file",
                    "run_id": "—",
                    "status_file": "—",
                    "metrics_summary": "—",
                    "wake": False,
                }
            )
            continue
        needs, detail = _needs_wake(record)
        if needs:
            wake_items.append(
                {
                    "suite": suite,
                    "state": record.state,
                    "run_id": record.run_id,
                    "detail": detail or record.reason,
                    "status_file": str(record.path.relative_to(REPO)),
                }
            )
        suite_rows.append(
            {
                "suite": suite,
                "state": record.state,
                "run_id": record.run_id,
                "status_file": str(record.path.relative_to(REPO)),
                "metrics_summary": _gap_line(suite, record.metrics),
                "metrics": record.metrics,
                "wake": needs,
            }
        )

    gpu_empty = _gpu_compute_empty()
    train_procs = _train_processes()
    pending_action = None
    wake_reason = None
    if wake_items:
        pending_action = "failure_analysis"
        wake_reason = "; ".join(item["detail"] for item in wake_items)

    return {
        "updated_at": _now(),
        "repo": str(REPO),
        "suites": suite_rows,
        "gpu_compute_empty": gpu_empty,
        "train_process_count": len(train_procs),
        "train_processes": train_procs[:5],
        "pending_action": pending_action,
        "wake_reason": wake_reason,
        "wake_items": wake_items,
        "sota_targets": SOTA_TARGETS,
        "prompt": AGENT_PROMPT,
    }


def _write_markdown(report: Dict[str, Any]) -> None:
    lines = [
        "# Lora-Ours Three-Suite Sentinel",
        "",
        f"- updated: `{report['updated_at']}`",
        f"- gpu_compute_empty: `{report['gpu_compute_empty']}`",
        f"- train_process_count: `{report['train_process_count']}`",
        f"- pending_action: `{report.get('pending_action') or '—'}`",
        f"- wake_reason: `{report.get('wake_reason') or '—'}`",
        "",
        "## Suites",
        "",
        "| Suite | State | Run | SOTA gap | Status file | Wake |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["suites"]:
        lines.append(
            "| {suite} | {state} | `{run_id}` | {metrics_summary} | `{status_file}` | {wake} |".format(
                suite=row["suite"],
                state=row["state"],
                run_id=row["run_id"],
                metrics_summary=row["metrics_summary"],
                status_file=row["status_file"],
                wake="yes" if row.get("wake") else "no",
            )
        )
    if report.get("wake_items"):
        lines.extend(["", "## Needs Agent Attention", ""])
        for item in report["wake_items"]:
            lines.append(
                f"- **{item['suite']}**: `{item['state']}` — {item['detail']} (`{item['status_file']}`)"
            )
    lines.extend(
        [
            "",
            "## Monitor",
            "",
            f"- JSON: `{STATUS_JSON.relative_to(REPO)}`",
            f"- log: `{MONITOR_LOG.relative_to(REPO)}`",
            f"- wake flag: `{AGENT_WAKE_FLAG.relative_to(REPO)}`",
            f"- agent loop tmux: `lora-ours-agent-loop`",
            "",
        ]
    )
    STATUS_MD.write_text("\n".join(lines), encoding="utf-8")


def _signal_agent_wake(report: Dict[str, Any]) -> None:
    payload = {
        "action": report.get("pending_action") or "review",
        "wake_reason": report.get("wake_reason") or "sentinel_tick",
        "wake_items": report.get("wake_items") or [],
        "suites": report.get("suites") or [],
        "gpu_compute_empty": report.get("gpu_compute_empty"),
        "prompt": AGENT_PROMPT,
        "timestamp": report["updated_at"],
        "status_file": str(STATUS_MD.relative_to(REPO)),
    }
    AGENT_WAKE_FLAG.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with AGENT_WAKE_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{report['updated_at']}] AWAITING_AGENT: {payload['wake_reason']}\n")
    _log(f"{SENTINEL_PREFIX} " + json.dumps(payload, ensure_ascii=False))


def _clear_agent_wake() -> None:
    if AGENT_WAKE_FLAG.is_file():
        AGENT_WAKE_FLAG.unlink()


def tick(*, emit_wake: bool, clear_when_idle: bool) -> Dict[str, Any]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    records = _collect_records()
    report = _build_report(records)
    STATUS_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(report)

    if report.get("wake_items"):
        if emit_wake:
            _signal_agent_wake(report)
    elif clear_when_idle:
        _clear_agent_wake()

    summary = ", ".join(f"{row['suite']}={row['state']}" for row in report["suites"])
    _log(f"SENTINEL_TICK suites=[{summary}] gpu_empty={report['gpu_compute_empty']} wake={bool(report.get('wake_items'))}")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-wake", action="store_true", help="Write SOTA_AGENT_WAKE.flag when blockers found.")
    parser.add_argument(
        "--clear-when-idle",
        action="store_true",
        default=True,
        help="Remove wake flag when no blockers (default: true).",
    )
    parser.add_argument("--loop", action="store_true", help="Run forever with sleep between ticks.")
    parser.add_argument("--poll-sec", type=int, default=int(__import__("os").environ.get("SENTINEL_POLL_SEC", "120")))
    args = parser.parse_args(argv)

    if args.loop:
        import time

        while True:
            tick(emit_wake=args.emit_wake, clear_when_idle=args.clear_when_idle)
            time.sleep(max(15, args.poll_sec))
    else:
        report = tick(emit_wake=args.emit_wake, clear_when_idle=args.clear_when_idle)
        if report.get("wake_items") and args.emit_wake:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
