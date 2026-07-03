#!/usr/bin/env python3
"""Write a compact status report for the CITB official-base reproduction."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN_NAME = "citb_instrdialog_order1_seed1_official_script_500_50_50_tie_fixed_ft_instr_stage1_v54"
DEFAULT_OUTPUT_BASE = Path("/root/autodl-tmp/citb_official_base_repro")
FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "RuntimeError:",
    "ValueError:",
    "TypeError:",
    "CUDA out of memory",
    "CITB/Tk-Instruct collator mismatch",
    "Blocked paper_target_500_50_100",
    "FORMAL_RETRY_EXIT_CODE:3",
    "error:",
)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": repr(exc)}


def _result_dirs(output_dir: Path) -> List[Path]:
    results = output_dir / "results"
    if not results.is_dir():
        return []

    def sort_key(path: Path) -> tuple[int, str]:
        prefix = path.name.split("_", 1)[0]
        return (int(prefix), path.name) if prefix.isdigit() else (10**9, path.name)

    return sorted([p for p in results.iterdir() if p.is_dir()], key=sort_key)


def _read_tail(path: Path, max_bytes: int = 20000) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _latest_activity(output_dir: Path, log_paths: Optional[List[Path]] = None) -> Dict[str, Any]:
    log_paths = [path for path in (log_paths or []) if path.is_file()]
    if not output_dir.exists():
        if not log_paths:
            return {"path": "", "mtime": 0.0, "age_seconds": None}
        latest_log = max(log_paths, key=lambda p: p.stat().st_mtime)
        age = max(0.0, datetime.now().timestamp() - latest_log.stat().st_mtime)
        return {"path": str(latest_log), "mtime": latest_log.stat().st_mtime, "age_seconds": age}
    candidates = [p for p in output_dir.rglob("*") if p.is_file()] + log_paths
    if not candidates:
        return {"path": "", "mtime": 0.0, "age_seconds": None}
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    age = max(0.0, datetime.now().timestamp() - latest.stat().st_mtime)
    return {"path": str(latest), "mtime": latest.stat().st_mtime, "age_seconds": age}


def _failure_signals(output_dir: Path, log_paths: Optional[List[Path]] = None) -> List[Dict[str, str]]:
    signals: List[Dict[str, str]] = []
    candidate_paths: List[Path] = []
    if output_dir.exists():
        candidate_paths.extend(sorted(output_dir.rglob("*")))
    candidate_paths.extend(path for path in (log_paths or []) if path.is_file())
    for path in candidate_paths:
        if not path.is_file() or path.suffix not in {".log", ".txt", ".err", ".out"}:
            continue
        tail = _read_tail(path)
        lowered = tail.lower()
        for marker in FAILURE_MARKERS:
            if marker.lower() in lowered:
                signals.append({"path": str(path), "marker": marker})
                break
        if len(signals) >= 5:
            break
    return signals


def _derive_state(
    *,
    output_dir: Path,
    result_dirs: List[Path],
    all_results: Dict[str, Any],
    trainer_state: Dict[str, Any],
    diagnosis: Dict[str, Any],
    failure_signals: List[Dict[str, str]],
    latest_activity: Dict[str, Any],
    stale_minutes: Optional[float],
    expected_tasks: Optional[int],
    train_process_count: int,
) -> str:
    if diagnosis:
        return "failed"
    if failure_signals:
        return "failed"
    if train_process_count > 0:
        return "running_or_recently_active"
    if expected_tasks is not None and len(result_dirs) >= expected_tasks and all_results:
        return "completed"
    if expected_tasks is not None and result_dirs and len(result_dirs) < expected_tasks:
        return "stopped_incomplete"
    if all_results:
        age_seconds = latest_activity.get("age_seconds")
        if stale_minutes is not None and age_seconds is not None and age_seconds <= stale_minutes * 60:
            return "running_or_recently_active"
        return "completed"
    if trainer_state.get("global_step"):
        return "training_or_evaluating"
    if result_dirs:
        return "has_results"
    if stale_minutes is not None:
        age_seconds = latest_activity.get("age_seconds")
        if age_seconds is not None and age_seconds > stale_minutes * 60:
            return "stalled"
    return "pending_or_starting"


def _train_process_count(output_dir: Path) -> int:
    try:
        result = subprocess.run(
            ["pgrep", "-af", str(output_dir)],
            check=False,
            text=True,
            capture_output=True,
        )
    except Exception:
        return 0
    if result.returncode not in {0, 1}:
        return 0
    return sum(1 for line in result.stdout.splitlines() if "run_continual_instruct_tuning.py" in line)


def build_status(
    output_dir: Path,
    run_name: str,
    stale_minutes: Optional[float] = None,
    expected_tasks: Optional[int] = None,
    log_paths: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    result_dirs = _result_dirs(output_dir)
    latest_dir = result_dirs[-1] if result_dirs else None
    latest_metrics = _load_json(latest_dir / "metrics.json") if latest_dir else {}
    train_state = _load_json(output_dir / "trainer_state.json")
    all_results = _load_json(output_dir / "all_results.json")
    diagnosis = _load_json(output_dir / "manual_stop_diagnosis.json") or _load_json(output_dir / "stop_and_diagnose.json")
    activity = _latest_activity(output_dir, log_paths)
    failures = _failure_signals(output_dir, log_paths)
    train_processes = _train_process_count(output_dir)
    state = _derive_state(
        output_dir=output_dir,
        result_dirs=result_dirs,
        all_results=all_results,
        trainer_state=train_state,
        diagnosis=diagnosis,
        failure_signals=failures,
        latest_activity=activity,
        stale_minutes=stale_minutes,
        expected_tasks=expected_tasks,
        train_process_count=train_processes,
    )
    return {
        "updated": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "output_dir": str(output_dir),
        "state": state,
        "num_result_dirs": len(result_dirs),
        "expected_tasks": expected_tasks,
        "train_process_count": train_processes,
        "latest_result_dir": str(latest_dir) if latest_dir else "",
        "latest_activity": activity,
        "failure_signals": failures,
        "diagnosis": diagnosis,
        "latest_metric_keys": sorted(latest_metrics.keys())[:40],
        "latest_metrics": {
            key: latest_metrics.get(key)
            for key in sorted(latest_metrics)
            if key.endswith("_rougeL")
            or key.endswith("_exact_match")
            or key.endswith("_samples")
            or key in {"train_runtime", "train_samples", "epoch"}
        },
        "trainer_state_exists": bool(train_state),
        "all_results": all_results,
        "red_flags": [
            "CITB paper states InstrDialog uses 500/50/100 train/dev/test instances, but the official short-stream FT_INSTR script sets max_num_instances_per_eval_task=50.",
            "Official CITB Stage-2 code uses one max_num_instances_per_eval_task for both dev and per-task test split size, yielding script-strict 500/50/50 unless split code is patched.",
            "Default launcher keeps the official-script 500/50/50 smoke policy; requested 500/50/100 remains blocked unless split code is patched and audited.",
            "Official dry-run currently requires the lora_v10_citb Python 3.9 environment; base Python lacks datasets.load_metric.",
            "Stage-1 tokenizer files are incompatible with the old official stack, so the launcher uses the local google__t5-small-lm-adapt tokenizer as a compatibility override.",
            "Tk-Instruct metric code imports AutoTokenizer.from_pretrained('gpt2'); launcher redirects only that tokenizer lookup to a local GPT-2 tokenizer cache via project-local sitecustomize because this environment cannot reach Hugging Face.",
            "The hyintell/CITB checkout has an untracked Tk-Instruct copy whose collator lacks add_task_id; launcher now defaults to the tracked local citb_official tree whose collator matches the CL entrypoint.",
        ],
    }


def write_status(status: Dict[str, Any], basename: str) -> None:
    logs_dir = REPO / "results/logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_path = logs_dir / f"{basename}.json"
    md_path = logs_dir / f"{basename}.md"
    json_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CITB Official Base Repro Status",
        "",
        f"- Updated: `{status['updated']}`",
        f"- Run: `{status['run_name']}`",
        f"- State: `{status['state']}`",
        f"- Output dir: `{status['output_dir']}`",
        f"- Result dirs: `{status['num_result_dirs']}`",
        f"- Expected tasks: `{status.get('expected_tasks')}`",
        f"- Train processes: `{status.get('train_process_count')}`",
        f"- Latest result dir: `{status['latest_result_dir']}`",
        f"- Latest activity: `{status.get('latest_activity', {}).get('path', '')}`",
        f"- Latest activity age seconds: `{status.get('latest_activity', {}).get('age_seconds')}`",
        "",
        "## Latest Metrics",
    ]
    latest_metrics = status.get("latest_metrics") or {}
    if latest_metrics:
        for key, value in latest_metrics.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No per-task metrics yet.")
    failures = status.get("failure_signals") or []
    if failures:
        lines.extend(["", "## Failure Signals"])
        for signal in failures:
            lines.append(f"- `{signal.get('marker')}` in `{signal.get('path')}`")
    diagnosis = status.get("diagnosis") or {}
    if diagnosis:
        lines.extend(["", "## Diagnosis"])
        for key in ("reason", "root_cause", "action", "evidence"):
            value = diagnosis.get(key)
            if value:
                lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## RED FLAG"])
    for flag in status["red_flags"]:
        lines.append(f"- {flag}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=os.environ.get("CITB_OFFICIAL_OUTPUT_DIR", ""))
    parser.add_argument("--run-name", default=os.environ.get("CITB_OFFICIAL_RUN_NAME", DEFAULT_RUN_NAME))
    parser.add_argument("--basename", default="citb_official_base_repro_v53_status")
    parser.add_argument("--stale-minutes", type=float, default=10.0)
    parser.add_argument("--expected-tasks", type=int, default=None)
    parser.add_argument("--log-path", action="append", default=[])
    parser.add_argument("--emit-sentinel", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_BASE / args.run_name
    status = build_status(
        output_dir=output_dir,
        run_name=args.run_name,
        stale_minutes=args.stale_minutes,
        expected_tasks=args.expected_tasks,
        log_paths=[Path(path) for path in args.log_path],
    )
    write_status(status, args.basename)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if args.emit_sentinel and status["state"] in {"failed", "stalled", "stopped_incomplete"}:
        payload = {
            "scope": "citb_official_base_repro",
            "state": status["state"],
            "run_name": status["run_name"],
            "output_dir": status["output_dir"],
            "status_file": f"results/logs/{args.basename}.md",
        }
        print("AGENT_LOOP_WAKE_LORA_OURS " + json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
