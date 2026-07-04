#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

import launch_v66_standard_learning_signal_probe as v66


REPO = Path(__file__).resolve().parents[1]
MANIFEST = "configs/ccfa_three_suite/standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v67_replay_retention_probe.yaml"
LOG_DIR = REPO / "results/logs"
CONFIG_DIR = REPO / "configs/ccfa_three_suite"
STATUS_JSON = LOG_DIR / "standard_peft_ours_v67_replay_retention_probe_status.json"
STATUS_MD = LOG_DIR / "standard_peft_ours_v67_replay_retention_probe_status.md"
BASELINE_FINAL_AVG = 0.4375


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


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
    run_name = f"standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v67_replay_retention_{cid}"
    cfg = copy.deepcopy(base)
    cfg["experiment_name"] = run_name
    nested_set(cfg, ["data", "max_segments"], 2)
    nested_set(cfg, ["data", "max_train_examples_per_segment"], int(candidate["max_train_examples_per_segment"]))
    nested_set(cfg, ["data", "max_eval_examples_per_segment"], int(candidate["max_eval_examples_per_segment"]))
    nested_set(cfg, ["train", "epochs_per_segment"], int(candidate["epochs_per_segment"]))
    nested_set(cfg, ["train", "gradient_accumulation_steps"], int(candidate["gradient_accumulation_steps"]))
    nested_set(cfg, ["train", "diagnostics", "train_heartbeat_every_batches"], 1)
    nested_set(cfg, ["diagnostics", "per_segment_eval_timeout_seconds"], 1200)
    nested_set(cfg, ["modules", "use_overlap_loss"], bool(candidate.get("use_overlap_loss", False)))
    nested_set(cfg, ["spectral_replay", "replay_ratio"], float(candidate.get("replay_ratio", 0.0)))
    nested_set(cfg, ["spectral_replay", "min_replay_per_segment"], int(candidate.get("min_replay_per_segment", 0)))
    nested_set(cfg, ["spectral_replay", "selection_strategy"], "label_balanced")
    nested_set(cfg, ["eval_normalization", "enable_teacher_forced_eval"], True)
    nested_set(cfg, ["eval_normalization", "teacher_forced_eval_max_examples"], int(candidate["max_eval_examples_per_segment"]))
    nested_set(cfg, ["eval_normalization", "debug_examples_per_segment"], int(candidate["max_eval_examples_per_segment"]))
    nested_set(cfg, ["eval_normalization", "debug_examples_max_total"], int(candidate["max_eval_examples_per_segment"]) * 2)
    nested_set(cfg, ["output", "run_name"], run_name)
    nested_set(cfg, ["output", "tracking", "wandb_project"], "lora-ours")
    nested_set(cfg, ["output", "tracking", "wandb_group"], "ccfa_standard_peft_strict_ours_v67_replay_retention")
    nested_set(cfg, ["output", "tracking", "wandb_mode"], "online")
    tags = list(((cfg.get("output") or {}).get("tracking") or {}).get("wandb_tags") or [])
    tags = [tag for tag in tags if str(tag) not in {"v66", "learning_signal", "t20_e16_ep3_accum2_noreplay"}]
    tags.extend(["v67", "replay_retention", cid])
    nested_set(cfg, ["output", "tracking", "wandb_tags"], tags)
    nested_set(
        cfg,
        ["paper", "notes"],
        (
            f"v67 Standard replay-retention probe {cid}: fixed v66 t20/e16/ep3/accum2 budget, "
            f"replay_ratio={candidate.get('replay_ratio')}, min_replay_per_segment={candidate.get('min_replay_per_segment')}. "
            "This is not paper-comparable."
        ),
    )
    cfg["v67_replay_retention_probe"] = {
        "candidate_id": cid,
        "manifest": MANIFEST,
        "baseline": "v66 t20_e16_ep3_accum2_noreplay final avg/task-aware acc 0.4375",
        "reason": candidate.get("reason", ""),
    }
    path = CONFIG_DIR / f"{run_name}_probe.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return run_name, path, cfg


def score_cells(score_matrix: Any) -> dict[str, float | None]:
    if not isinstance(score_matrix, list) or len(score_matrix) < 2:
        return {"dbpedia_after_amazon": None, "amazon_current": None, "final_seen_avg": None}
    row = score_matrix[1] if isinstance(score_matrix[1], list) else []
    dbpedia = float(row[0]) if len(row) > 0 and row[0] is not None else None
    amazon = float(row[1]) if len(row) > 1 and row[1] is not None else None
    values = [value for value in [dbpedia, amazon] if value is not None]
    return {
        "dbpedia_after_amazon": dbpedia,
        "amazon_current": amazon,
        "final_seen_avg": (sum(values) / len(values)) if values else None,
    }


def write_status(payload: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Standard PEFT Ours v67 Replay-Retention Probe Status",
        "",
        f"- Updated: `{payload.get('updated_at')}`",
        f"- State: `{payload.get('state')}`",
        f"- Decision: `{payload.get('decision')}`",
        f"- Baseline final avg/task-aware acc: `{BASELINE_FINAL_AVG}`",
        f"- Best candidate: `{payload.get('best_candidate') or 'n/a'}`",
        "",
        "## Candidate Results",
        "",
    ]
    for item in payload.get("results", []):
        cells = item.get("score_cells") or {}
        memory = item.get("memory", {})
        lines.extend(
            [
                f"- `{item.get('candidate_id')}` / `{item.get('run_name')}`",
                f"  - state: return `{item.get('returncode')}`, sigterm `{item.get('sigterm')}`, nonzero `{item.get('has_nonzero_score')}`",
                f"  - replay ratio/min: `{item.get('replay_ratio')}` / `{item.get('min_replay_per_segment')}`",
                f"  - final avg/task-aware: `{item.get('final_average_accuracy')}` / `{item.get('final_average_task_aware_accuracy')}`",
                f"  - dbpedia retention / amazon current / seen avg: `{cells.get('dbpedia_after_amazon')}` / `{cells.get('amazon_current')}` / `{cells.get('final_seen_avg')}`",
                f"  - elapsed: `{item.get('elapsed_seconds')}` sec; max cuda allocated/reserved: `{memory.get('max_cuda_max_allocated_mb', 'n/a')}` / `{memory.get('max_cuda_reserved_mb', 'n/a')}` MB",
                f"  - score_matrix: `{item.get('score_matrix')}`",
            ]
        )
    lines.extend(["", "## Next Gate", "", f"- {payload.get('next_gate', 'n/a')}"])
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v67 Standard replay-retention probe serially.")
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--force", action="store_true", help="run even if a core.train or GPU process is visible")
    args = parser.parse_args()

    manifest_path = (REPO / args.manifest).resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    base_path = (REPO / manifest["base_config"]).resolve()
    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    results: list[dict[str, Any]] = []

    existing = v66.core_train_processes()
    existing_gpu = v66.gpu_compute_processes()
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

    for candidate in list(manifest.get("candidates") or []):
        run_name, config_path, cfg = build_candidate_config(base_cfg, candidate)
        print(f"[{now()}] launching {candidate['id']}: {run_name}", flush=True)
        started = time.time()
        rc, elapsed = v66.launch_candidate(run_name, config_path)
        summary = v66.summarize_run(run_name, cfg, config_path, rc, elapsed)
        summary.update(
            {
                "candidate_id": str(candidate["id"]),
                "replay_ratio": float(candidate.get("replay_ratio", 0.0)),
                "min_replay_per_segment": int(candidate.get("min_replay_per_segment", 0)),
                "score_cells": score_cells(summary.get("score_matrix")),
                "started_at_epoch": round(started, 3),
            }
        )
        results.append(summary)
        write_status(
            {
                "updated_at": now(),
                "state": "running",
                "decision": "partial results recorded",
                "best_candidate": None,
                "results": results,
            }
        )
        if summary.get("sigterm"):
            break

    healthy = [item for item in results if not item.get("sigterm") and item.get("returncode") == 0]
    best = max(
        healthy,
        key=lambda item: float(item.get("final_average_task_aware_accuracy") or item.get("final_average_accuracy") or -1.0),
        default=None,
    )
    best_value = float((best or {}).get("final_average_task_aware_accuracy") or (best or {}).get("final_average_accuracy") or -1.0)
    sigterm_seen = any(item.get("sigterm") for item in results)
    if sigterm_seen:
        state = "stopped_after_sigterm"
        decision = "SIGTERM observed; stop before 4-task extension"
        next_gate = "Do not extend; inspect the failed candidate's process_exit and memory artifacts."
    elif best and best_value > BASELINE_FINAL_AVG:
        state = "completed_healthy"
        decision = "best candidate improved over the v66 no-replay baseline"
        next_gate = f"Prepare a 4-task extension from `{best.get('candidate_id')}` under the same serial GPU policy."
    else:
        state = "completed_no_gain"
        decision = "no candidate improved over the v66 no-replay baseline"
        next_gate = "Record replay/current-task trade-off; avoid 4-task extension until the two-task average improves."
    payload = {
        "updated_at": now(),
        "state": state,
        "decision": decision,
        "manifest": str(manifest_path.relative_to(REPO)),
        "base_config": str(base_path.relative_to(REPO)),
        "baseline_final_avg": BASELINE_FINAL_AVG,
        "best_candidate": best.get("candidate_id") if best else None,
        "next_gate": next_gate,
        "results": results,
    }
    write_status(payload)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
