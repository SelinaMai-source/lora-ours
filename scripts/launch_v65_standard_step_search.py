#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[1]
MANIFEST = "configs/ccfa_three_suite/standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v65_step_search.yaml"
LOG_DIR = REPO / "results/logs"
CONFIG_DIR = REPO / "configs/ccfa_three_suite"
STATUS_JSON = LOG_DIR / "standard_peft_ours_v65_step_search_status.json"
STATUS_MD = LOG_DIR / "standard_peft_ours_v65_step_search_status.md"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_text(cmd: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), text=True, capture_output=True, timeout=timeout)
        return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": repr(exc)}


def core_train_processes() -> list[str]:
    proc = subprocess.run(["ps", "-eo", "pid,ppid,sid,pgid,stat,args="], text=True, capture_output=True)
    markers = ("python -m core.train", "python core/train.py", "python -u core/train.py")
    return [line.strip() for line in proc.stdout.splitlines() if any(marker in line for marker in markers)]


def isolated_child() -> None:
    os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": repr(exc), "path": str(path)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"parse_error": line[:500]})
    return rows


def nested_set(cfg: dict[str, Any], keys: list[str], value: Any) -> None:
    cur: dict[str, Any] = cfg
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def build_candidate_config(base: dict[str, Any], candidate: dict[str, Any]) -> tuple[str, Path, dict[str, Any]]:
    cid = str(candidate["id"])
    train_n = int(candidate["max_train_examples_per_segment"])
    eval_n = int(candidate["max_eval_examples_per_segment"])
    accum = int(candidate["gradient_accumulation_steps"])
    run_name = f"standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v65_step_search_{cid}"
    cfg = copy.deepcopy(base)
    cfg["experiment_name"] = run_name
    nested_set(cfg, ["data", "max_segments"], 2)
    nested_set(cfg, ["data", "max_train_examples_per_segment"], train_n)
    nested_set(cfg, ["data", "max_eval_examples_per_segment"], eval_n)
    nested_set(cfg, ["train", "gradient_accumulation_steps"], accum)
    nested_set(cfg, ["train", "diagnostics", "train_heartbeat_every_batches"], 1)
    nested_set(cfg, ["diagnostics", "per_segment_eval_timeout_seconds"], 1200)
    nested_set(cfg, ["output", "run_name"], run_name)
    nested_set(cfg, ["output", "tracking", "wandb_project"], "lora-ours")
    nested_set(cfg, ["output", "tracking", "wandb_group"], "ccfa_standard_peft_strict_ours_v65_step_search")
    nested_set(cfg, ["output", "tracking", "wandb_mode"], "online")
    tags = list(((cfg.get("output") or {}).get("tracking") or {}).get("wandb_tags") or [])
    tags = [tag for tag in tags if str(tag) not in {"v64", "safe_short"}]
    tags.extend(["v65", "step_search", cid, f"train{train_n}", f"eval{eval_n}", f"accum{accum}"])
    nested_set(cfg, ["output", "tracking", "wandb_tags"], tags)
    nested_set(
        cfg,
        ["paper", "notes"],
        (
            f"v65 Standard safe-short step-search candidate {cid}: {train_n} train / "
            f"{eval_n} eval examples per segment, gradient_accumulation_steps={accum}. "
            "This is an adapted Ours diagnostic run, not an official O-LoRA reproduction "
            "and not paper-comparable."
        ),
    )
    cfg["v65_step_search"] = {
        "candidate_id": cid,
        "reason": candidate.get("reason", ""),
        "manifest": MANIFEST,
        "target": "dbpedia/amazon eval not all zero without SIGTERM",
    }
    path = CONFIG_DIR / f"{run_name}_probe.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return run_name, path, cfg


def run_dir_for(cfg: dict[str, Any], run_name: str) -> Path:
    results_dir = Path(((cfg.get("paths") or {}).get("results_dir") or REPO / "results"))
    return results_dir / "runs" / run_name


def summarize_memory(heartbeats: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"heartbeat_rows": len(heartbeats)}
    if not heartbeats:
        return summary
    for key in ["rss_max_mb", "cuda_allocated_mb", "cuda_reserved_mb", "cuda_max_allocated_mb"]:
        values: list[float] = []
        for row in heartbeats:
            memory = row.get("memory") if isinstance(row.get("memory"), dict) else {}
            if key in memory:
                try:
                    values.append(float(memory[key]))
                except (TypeError, ValueError):
                    pass
        if values:
            summary[f"max_{key}"] = max(values)
    summary["last_heartbeat"] = heartbeats[-1]
    return summary


def summarize_run(run_name: str, cfg: dict[str, Any], config_path: Path, returncode: int, elapsed: float) -> dict[str, Any]:
    run_dir = run_dir_for(cfg, run_name)
    final_metrics = read_json(run_dir / "final_metrics.json")
    process_exit = read_json(run_dir / "process_exit.json")
    launcher_exit = read_json(LOG_DIR / f"{run_name}.exit.json")
    heartbeats = read_jsonl(run_dir / "train_heartbeat.jsonl")
    eval_statuses = []
    for path in sorted(run_dir.glob("segment_*/eval_status.json")):
        payload = read_json(path)
        if payload:
            payload["path"] = str(path)
            eval_statuses.append(payload)
    final = final_metrics.get("final") if isinstance(final_metrics.get("final"), dict) else {}
    ccfa_summary = final_metrics.get("ccfa_summary") if isinstance(final_metrics.get("ccfa_summary"), dict) else {}
    if not ccfa_summary and isinstance(final.get("ccfa_summary"), dict):
        ccfa_summary = final["ccfa_summary"]
    score_matrix = ccfa_summary.get("score_matrix") or ccfa_summary.get("task_aware_score_matrix")
    flat_scores: list[float] = []
    if isinstance(score_matrix, list):
        for row in score_matrix:
            if isinstance(row, list):
                for value in row:
                    if value is not None:
                        try:
                            flat_scores.append(float(value))
                        except (TypeError, ValueError):
                            pass
    sigterm = (
        "SIGTERM" in str(process_exit.get("event", ""))
        or "SIGTERM" in str(launcher_exit.get("event", ""))
        or process_exit.get("exit_code") == 143
        or returncode == 143
    )
    return {
        "run_name": run_name,
        "config": str(config_path.relative_to(REPO)),
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "run_dir": str(run_dir),
        "process_exit": process_exit,
        "launcher_exit": launcher_exit,
        "eval_statuses": eval_statuses,
        "memory": summarize_memory(heartbeats),
        "train_optimizer_steps": final.get("train.optimizer_steps"),
        "train_batches": final.get("train.batches"),
        "score_matrix": score_matrix,
        "final_average_accuracy": ccfa_summary.get("final_average_accuracy"),
        "final_average_task_aware_accuracy": ccfa_summary.get("final_average_task_aware_accuracy"),
        "all_zero": bool(flat_scores) and all(value == 0.0 for value in flat_scores),
        "has_nonzero_score": any(value != 0.0 for value in flat_scores),
        "sigterm": sigterm,
        "gpu_after": run_text(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }


def write_status(payload: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Standard PEFT Ours v65 Step Search Status",
        "",
        f"- Updated: `{payload.get('updated_at')}`",
        f"- State: `{payload.get('state')}`",
        f"- Decision: `{payload.get('decision')}`",
        f"- Best safe candidate: `{payload.get('best_safe_candidate') or 'n/a'}`",
        f"- Best safe train-cap candidate: `{payload.get('best_safe_train_cap_candidate') or 'n/a'}`",
        f"- First non-zero candidate: `{payload.get('first_nonzero_candidate') or 'n/a'}`",
        "",
        "## Candidate Results",
        "",
    ]
    for item in payload.get("results", []):
        memory = item.get("memory", {})
        lines.extend(
            [
                f"- `{item.get('candidate_id')}` / `{item.get('run_name')}`",
                f"  - state: return `{item.get('returncode')}`, sigterm `{item.get('sigterm')}`, all_zero `{item.get('all_zero')}`, nonzero `{item.get('has_nonzero_score')}`",
                f"  - train/eval/accum: `{item.get('train_cap')}` / `{item.get('eval_cap')}` / `{item.get('accum')}`",
                f"  - batches/optimizer_steps: `{item.get('train_batches')}` / `{item.get('train_optimizer_steps')}`",
                f"  - elapsed: `{item.get('elapsed_seconds')}` sec",
                f"  - max cuda allocated/reserved: `{memory.get('max_cuda_max_allocated_mb', 'n/a')}` / `{memory.get('max_cuda_reserved_mb', 'n/a')}` MB",
                f"  - score_matrix: `{item.get('score_matrix')}`",
            ]
        )
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def launch_candidate(run_name: str, config_path: Path) -> tuple[int, float]:
    supervisor_log = LOG_DIR / f"{run_name}.isolated_launcher.log"
    stdout = supervisor_log.open("ab", buffering=0)
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "WANDB_MODE": "online", "V65_STANDARD_STEP_SEARCH": "1"})
    started = time.time()
    proc = subprocess.Popen(
        ["bash", "scripts/run_ours_v1_strict_iteration.sh", str(config_path)],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        preexec_fn=isolated_child,
        close_fds=True,
    )
    (LOG_DIR / f"{run_name}.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
    launcher_payload = {
        "updated_at": now(),
        "event": "v65_isolated_child_started",
        "run_name": run_name,
        "config": str(config_path),
        "launcher_pid": os.getpid(),
        "launcher_ppid": os.getppid(),
        "child_pid": proc.pid,
        "child_sid": os.getsid(proc.pid),
        "child_pgid": os.getpgid(proc.pid),
        "supervisor_log": str(supervisor_log),
        "gpu_before": run_text(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "gpu_compute_before": run_text(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ]
        ),
    }
    (LOG_DIR / f"{run_name}.isolated_launcher.json").write_text(
        json.dumps(launcher_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    rc = proc.wait()
    return rc, time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v65 Standard safe-short step search serially.")
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--force", action="store_true", help="run even if a core.train process is visible")
    parser.add_argument("--max-candidates", type=int, default=0, help="optional limit for testing the launcher")
    parser.add_argument("--summarize-existing", action="store_true", help="rebuild status from existing candidate artifacts without launching")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = (REPO / args.manifest).resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    base_path = (REPO / manifest["base_config"]).resolve()
    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    candidates = list(manifest.get("candidates") or [])
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    def rebuild_existing_status() -> dict[str, Any]:
        rebuilt: list[dict[str, Any]] = []
        best_safe_existing: str | None = None
        best_safe_train_cap_existing: str | None = None
        best_safe_train_cap_value = -1
        first_nonzero_existing: str | None = None
        sigterm_existing = False
        for candidate in candidates:
            cid = str(candidate["id"])
            run_name = f"standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v65_step_search_{cid}"
            config_path = CONFIG_DIR / f"{run_name}_probe.yaml"
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
            launcher_exit = read_json(LOG_DIR / f"{run_name}.exit.json")
            rc = launcher_exit.get("train_exit_status")
            if rc is None:
                rc = launcher_exit.get("launcher_exit_status", 0)
            summary = summarize_run(run_name, cfg, config_path, int(rc or 0), 0.0)
            if summary.get("launcher_exit", {}).get("started_at") and summary.get("launcher_exit", {}).get("updated_at"):
                try:
                    start = datetime.fromisoformat(str(summary["launcher_exit"]["started_at"]))
                    end = datetime.fromisoformat(str(summary["launcher_exit"]["updated_at"]))
                    if start.tzinfo is not None and end.tzinfo is None:
                        end = end.replace(tzinfo=start.tzinfo)
                    summary["elapsed_seconds"] = (end - start).total_seconds()
                except Exception:
                    pass
            train_cap = int(candidate["max_train_examples_per_segment"])
            accum = int(candidate["gradient_accumulation_steps"])
            summary.update(
                {
                    "candidate_id": cid,
                    "train_cap": train_cap,
                    "eval_cap": int(candidate["max_eval_examples_per_segment"]),
                    "accum": accum,
                    "reason": candidate.get("reason", ""),
                }
            )
            rebuilt.append(summary)
            if summary.get("sigterm"):
                sigterm_existing = True
            elif int(summary.get("returncode") or 0) == 0:
                best_safe_existing = cid
                if accum == 8 and train_cap > best_safe_train_cap_value:
                    best_safe_train_cap_existing = cid
                    best_safe_train_cap_value = train_cap
            if summary.get("has_nonzero_score") and first_nonzero_existing is None:
                first_nonzero_existing = cid
        if sigterm_existing:
            state = "stopped_after_sigterm"
            decision = "SIGTERM observed; artifact recorded and search stopped"
        elif first_nonzero_existing:
            state = "completed"
            decision = "found non-zero learning signal without SIGTERM"
        else:
            state = "completed_all_zero"
            decision = "no SIGTERM, but all tested dbpedia/amazon eval scores remained zero"
        return {
            "updated_at": now(),
            "state": state,
            "decision": decision,
            "best_safe_candidate": best_safe_existing,
            "best_safe_train_cap_candidate": best_safe_train_cap_existing,
            "first_nonzero_candidate": first_nonzero_existing,
            "manifest": str(manifest_path.relative_to(REPO)),
            "base_config": str(base_path.relative_to(REPO)),
            "results": rebuilt,
        }

    if args.summarize_existing:
        payload = rebuild_existing_status()
        write_status(payload)
        print(json.dumps(payload, indent=2), flush=True)
        return 0

    existing = core_train_processes()
    if existing and not args.force:
        payload = {
            "updated_at": now(),
            "state": "blocked",
            "decision": "existing core.train process visible",
            "existing_core_train": existing,
            "results": [],
        }
        write_status(payload)
        print(json.dumps(payload, indent=2), flush=True)
        return 75

    results: list[dict[str, Any]] = []
    best_safe: str | None = None
    best_safe_train_cap: str | None = None
    best_safe_train_cap_value = -1
    first_nonzero: str | None = None
    sigterm_seen = False
    accum8_nonzero = False

    for candidate in candidates:
        cid = str(candidate["id"])
        accum = int(candidate["gradient_accumulation_steps"])
        if sigterm_seen:
            break
        if accum != 8 and accum8_nonzero:
            continue
        existing = core_train_processes()
        if existing and not args.force:
            results.append({"candidate_id": cid, "state": "blocked_existing_core_train", "existing_core_train": existing})
            break
        run_name, config_path, cfg = build_candidate_config(base_cfg, candidate)
        print(f"[{now()}] launching {cid}: {run_name}", flush=True)
        rc, elapsed = launch_candidate(run_name, config_path)
        summary = summarize_run(run_name, cfg, config_path, rc, elapsed)
        summary.update(
            {
                "candidate_id": cid,
                "train_cap": int(candidate["max_train_examples_per_segment"]),
                "eval_cap": int(candidate["max_eval_examples_per_segment"]),
                "accum": accum,
                "reason": candidate.get("reason", ""),
            }
        )
        results.append(summary)
        if not summary.get("sigterm") and rc == 0:
            best_safe = cid
            if accum == 8 and int(candidate["max_train_examples_per_segment"]) > best_safe_train_cap_value:
                best_safe_train_cap = cid
                best_safe_train_cap_value = int(candidate["max_train_examples_per_segment"])
        if summary.get("has_nonzero_score") and first_nonzero is None:
            first_nonzero = cid
            if accum == 8:
                accum8_nonzero = True
        if summary.get("sigterm"):
            sigterm_seen = True
        write_status(
            {
                "updated_at": now(),
                "state": "running" if not sigterm_seen else "stopped_after_sigterm",
                "decision": "partial results recorded",
                "best_safe_candidate": best_safe,
                "best_safe_train_cap_candidate": best_safe_train_cap,
                "first_nonzero_candidate": first_nonzero,
                "results": results,
            }
        )

    if sigterm_seen:
        state = "stopped_after_sigterm"
        decision = "SIGTERM observed; artifact recorded and search stopped"
    elif first_nonzero:
        state = "completed"
        decision = "found non-zero learning signal without SIGTERM"
    else:
        state = "completed_all_zero"
        decision = "no SIGTERM, but all tested dbpedia/amazon eval scores remained zero"
    payload = {
        "updated_at": now(),
        "state": state,
        "decision": decision,
        "best_safe_candidate": best_safe,
        "best_safe_train_cap_candidate": best_safe_train_cap,
        "first_nonzero_candidate": first_nonzero,
        "manifest": str(manifest_path.relative_to(REPO)),
        "base_config": str(base_path.relative_to(REPO)),
        "results": results,
    }
    write_status(payload)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
