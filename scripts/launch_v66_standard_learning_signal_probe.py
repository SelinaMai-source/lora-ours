#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[1]
MANIFEST = "configs/ccfa_three_suite/standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v66_learning_signal_probe.yaml"
LOG_DIR = REPO / "results/logs"
CONFIG_DIR = REPO / "configs/ccfa_three_suite"
STATUS_JSON = LOG_DIR / "standard_peft_ours_v66_learning_signal_probe_status.json"
STATUS_MD = LOG_DIR / "standard_peft_ours_v66_learning_signal_probe_status.md"


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


def gpu_compute_processes() -> list[str]:
    info = run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )
    if info.get("returncode") != 0:
        return []
    return [line.strip() for line in str(info.get("stdout", "")).splitlines() if line.strip()]


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


def run_dir_for(cfg: dict[str, Any], run_name: str) -> Path:
    results_dir = Path(((cfg.get("paths") or {}).get("results_dir") or REPO / "results"))
    return results_dir / "runs" / run_name


def build_candidate_config(base: dict[str, Any], candidate: dict[str, Any]) -> tuple[str, Path, dict[str, Any]]:
    cid = str(candidate["id"])
    train_n = int(candidate["max_train_examples_per_segment"])
    eval_n = int(candidate["max_eval_examples_per_segment"])
    epochs = int(candidate["epochs_per_segment"])
    accum = int(candidate["gradient_accumulation_steps"])
    run_name = f"standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v66_learning_signal_{cid}"
    cfg = copy.deepcopy(base)
    cfg["experiment_name"] = run_name
    nested_set(cfg, ["data", "max_segments"], 2)
    nested_set(cfg, ["data", "max_train_examples_per_segment"], train_n)
    nested_set(cfg, ["data", "max_eval_examples_per_segment"], eval_n)
    nested_set(cfg, ["train", "epochs_per_segment"], epochs)
    nested_set(cfg, ["train", "gradient_accumulation_steps"], accum)
    nested_set(cfg, ["train", "diagnostics", "train_heartbeat_every_batches"], 1)
    nested_set(cfg, ["diagnostics", "per_segment_eval_timeout_seconds"], 1200)
    nested_set(cfg, ["modules", "use_overlap_loss"], bool(candidate.get("use_overlap_loss", False)))
    nested_set(cfg, ["spectral_replay", "replay_ratio"], float(candidate.get("replay_ratio", 0.0)))
    nested_set(cfg, ["spectral_replay", "min_replay_per_segment"], int(candidate.get("min_replay_per_segment", 0)))
    nested_set(cfg, ["eval_normalization", "enable_teacher_forced_eval"], True)
    nested_set(cfg, ["eval_normalization", "teacher_forced_eval_max_examples"], eval_n)
    nested_set(cfg, ["eval_normalization", "debug_examples_per_segment"], eval_n)
    nested_set(cfg, ["eval_normalization", "debug_examples_max_total"], eval_n * 2)
    nested_set(cfg, ["output", "run_name"], run_name)
    nested_set(cfg, ["output", "tracking", "wandb_project"], "lora-ours")
    nested_set(cfg, ["output", "tracking", "wandb_group"], "ccfa_standard_peft_strict_ours_v66_learning_signal")
    nested_set(cfg, ["output", "tracking", "wandb_mode"], "online")
    tags = list(((cfg.get("output") or {}).get("tracking") or {}).get("wandb_tags") or [])
    tags = [tag for tag in tags if str(tag) not in {"v64", "v65", "safe_short", "step_search"}]
    tags.extend(["v66", "learning_signal", cid, f"train{train_n}", f"eval{eval_n}", f"ep{epochs}", f"accum{accum}"])
    nested_set(cfg, ["output", "tracking", "wandb_tags"], tags)
    nested_set(
        cfg,
        ["paper", "notes"],
        (
            f"v66 Standard learning-signal probe {cid}: {train_n} train / {eval_n} eval, "
            f"epochs_per_segment={epochs}, gradient_accumulation_steps={accum}, "
            "replay/overlap pressure reduced for first-signal diagnosis. This is not paper-comparable."
        ),
    )
    cfg["v66_learning_signal_probe"] = {
        "candidate_id": cid,
        "manifest": MANIFEST,
        "reason": candidate.get("reason", ""),
        "target": "non-zero eval or clear teacher-forced/train learning signal",
    }
    path = CONFIG_DIR / f"{run_name}_probe.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return run_name, path, cfg


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


def summarize_debug(run_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted((run_dir / "eval_debug").glob("eval_segment_*.json")):
        payload = read_json(path)
        if isinstance(payload.get("examples"), list):
            rows.extend(payload["examples"])
        elif isinstance(payload, list):
            rows.extend(payload)
    tf_accs = []
    for row in rows:
        value = row.get("teacher_forced_answer_token_acc") if isinstance(row, dict) else None
        if value is not None:
            try:
                tf_accs.append(float(value))
            except (TypeError, ValueError):
                pass
    return {
        "debug_examples": len(rows),
        "teacher_forced_answer_token_acc_mean": (sum(tf_accs) / len(tf_accs)) if tf_accs else None,
    }


def summarize_train(run_dir: Path) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for path in sorted(run_dir.glob("segment_*/train_metrics.json")):
        payload = read_json(path)
        segment = path.parent.name
        summaries[segment] = {
            "loss": payload.get("loss") or payload.get("train.loss"),
            "mean_batch_acc": payload.get("mean_batch_acc"),
            "answer_token_acc": payload.get("train.answer_token_acc"),
            "optimizer_steps": payload.get("optimizer_steps") or payload.get("train.optimizer_steps"),
            "batches": payload.get("batches") or payload.get("train.batches"),
        }
    return summaries


def summarize_run(run_name: str, cfg: dict[str, Any], config_path: Path, returncode: int, elapsed: float) -> dict[str, Any]:
    run_dir = run_dir_for(cfg, run_name)
    final_metrics = read_json(run_dir / "final_metrics.json")
    process_exit = read_json(run_dir / "process_exit.json")
    launcher_exit = read_json(LOG_DIR / f"{run_name}.exit.json")
    heartbeats = read_jsonl(run_dir / "train_heartbeat.jsonl")
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
        "elapsed_seconds": round(elapsed, 1),
        "run_dir": str(run_dir),
        "process_exit": process_exit,
        "launcher_exit": launcher_exit,
        "memory": summarize_memory(heartbeats),
        "train": summarize_train(run_dir),
        "debug": summarize_debug(run_dir),
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
        "# Standard PEFT Ours v66 Learning-Signal Probe Status",
        "",
        f"- Updated: `{payload.get('updated_at')}`",
        f"- State: `{payload.get('state')}`",
        f"- Decision: `{payload.get('decision')}`",
        f"- First non-zero candidate: `{payload.get('first_nonzero_candidate') or 'n/a'}`",
        "",
        "## Candidate Results",
        "",
    ]
    for item in payload.get("results", []):
        memory = item.get("memory", {})
        debug = item.get("debug", {})
        lines.extend(
            [
                f"- `{item.get('candidate_id')}` / `{item.get('run_name')}`",
                f"  - state: return `{item.get('returncode')}`, sigterm `{item.get('sigterm')}`, all_zero `{item.get('all_zero')}`, nonzero `{item.get('has_nonzero_score')}`",
                f"  - train/eval/epochs/accum: `{item.get('train_cap')}` / `{item.get('eval_cap')}` / `{item.get('epochs')}` / `{item.get('accum')}`",
                f"  - elapsed: `{item.get('elapsed_seconds')}` sec",
                f"  - max cuda allocated/reserved: `{memory.get('max_cuda_max_allocated_mb', 'n/a')}` / `{memory.get('max_cuda_reserved_mb', 'n/a')}` MB",
                f"  - debug examples / teacher-forced acc mean: `{debug.get('debug_examples')}` / `{debug.get('teacher_forced_answer_token_acc_mean')}`",
                f"  - score_matrix: `{item.get('score_matrix')}`",
                f"  - train: `{item.get('train')}`",
            ]
        )
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def launch_candidate(run_name: str, config_path: Path) -> tuple[int, float]:
    supervisor_log = LOG_DIR / f"{run_name}.isolated_launcher.log"
    stdout = supervisor_log.open("ab", buffering=0)
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "WANDB_MODE": "online", "V66_STANDARD_LEARNING_SIGNAL": "1"})
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
    (LOG_DIR / f"{run_name}.isolated_launcher.json").write_text(
        json.dumps(
            {
                "updated_at": now(),
                "event": "v66_isolated_child_started",
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
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rc = proc.wait()
    return rc, time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v66 Standard learning-signal probe serially.")
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--force", action="store_true", help="run even if a core.train or GPU process is visible")
    parser.add_argument("--max-candidates", type=int, default=0)
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = (REPO / args.manifest).resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    base_path = (REPO / manifest["base_config"]).resolve()
    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    candidates = list(manifest.get("candidates") or [])
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    existing = core_train_processes()
    existing_gpu = gpu_compute_processes()
    if (existing or existing_gpu) and not args.force:
        payload = {
            "updated_at": now(),
            "state": "blocked",
            "decision": "existing training or GPU compute process visible",
            "existing_core_train": existing,
            "existing_gpu_compute": existing_gpu,
            "results": [],
        }
        write_status(payload)
        print(json.dumps(payload, indent=2), flush=True)
        return 75

    results: list[dict[str, Any]] = []
    first_nonzero: str | None = None
    sigterm_seen = False
    for candidate in candidates:
        if sigterm_seen or first_nonzero:
            break
        existing = core_train_processes()
        existing_gpu = gpu_compute_processes()
        if (existing or existing_gpu) and not args.force:
            results.append(
                {
                    "candidate_id": str(candidate["id"]),
                    "state": "blocked_existing_process",
                    "existing_core_train": existing,
                    "existing_gpu_compute": existing_gpu,
                }
            )
            break
        run_name, config_path, cfg = build_candidate_config(base_cfg, candidate)
        print(f"[{now()}] launching {candidate['id']}: {run_name}", flush=True)
        rc, elapsed = launch_candidate(run_name, config_path)
        summary = summarize_run(run_name, cfg, config_path, rc, elapsed)
        summary.update(
            {
                "candidate_id": str(candidate["id"]),
                "train_cap": int(candidate["max_train_examples_per_segment"]),
                "eval_cap": int(candidate["max_eval_examples_per_segment"]),
                "epochs": int(candidate["epochs_per_segment"]),
                "accum": int(candidate["gradient_accumulation_steps"]),
                "reason": candidate.get("reason", ""),
            }
        )
        results.append(summary)
        if summary.get("has_nonzero_score"):
            first_nonzero = str(candidate["id"])
        if summary.get("sigterm"):
            sigterm_seen = True
        write_status(
            {
                "updated_at": now(),
                "state": "running" if not (sigterm_seen or first_nonzero) else "stopping",
                "decision": "partial results recorded",
                "first_nonzero_candidate": first_nonzero,
                "results": results,
            }
        )

    if sigterm_seen:
        state = "stopped_after_sigterm"
        decision = "SIGTERM observed; artifact recorded and probe stopped"
    elif first_nonzero:
        state = "completed"
        decision = "found non-zero learning signal without SIGTERM"
    else:
        state = "completed_all_zero"
        decision = "no SIGTERM, but all tested eval scores remained zero"
    payload = {
        "updated_at": now(),
        "state": state,
        "decision": decision,
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
