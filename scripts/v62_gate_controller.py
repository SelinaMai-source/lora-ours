#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EARLY_RUN = "standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v62_earlygate"
EARLY_STATUS = REPO / "results/logs/standard_peft_ours_v62_earlygate_status.json"
FORMAL_RUN = "standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v62_formal"
FORMAL_CONFIG = "configs/ccfa_three_suite/standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v62_formal.yaml"
DECISION_PATH = REPO / "results/logs/standard_peft_ours_v62_gate_decision.json"
PASS_THRESHOLDS = {
    "min_final_avg": 0.70,
    "min_dbpedia_retention": 0.90,
    "min_amazon_current": 0.45,
    "max_forgetting": 0.10,
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    print(f"[{now()}] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(REPO), text=True, **kwargs)


def write_decision(payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = now()
    DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DECISION_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def refresh_status() -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "OURS_MONITOR_RUN_ID": EARLY_RUN,
            "OURS_MONITOR_STATUS_BASENAME": "standard_peft_ours_v62_earlygate_status",
            "OURS_MONITOR_STATUS_TITLE": "Standard PEFT Ours v62 Earlygate Status",
        }
    )
    subprocess.run(["timeout", "20s", "python", "scripts/monitor_ours_v18_strict.py"], cwd=str(REPO), env=env, text=True)
    if EARLY_STATUS.is_file():
        return json.loads(EARLY_STATUS.read_text(encoding="utf-8"))
    return {}


def final_payload(status: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any, Any]:
    artifacts = status.get("artifacts") or {}
    final = artifacts.get("final_metrics.json") or {}
    summary = final.get("ccfa_summary") or {}
    final_metrics = final.get("final") or {}
    matrix = summary.get("score_matrix") or []
    dbpedia = None
    amazon = None
    if len(matrix) >= 2 and isinstance(matrix[1], list) and len(matrix[1]) >= 2:
        dbpedia = matrix[1][0]
        amazon = matrix[1][1]
    return final, summary, final_metrics, dbpedia, amazon


def is_pass(status: dict[str, Any]) -> tuple[bool, list[str]]:
    final, summary, final_metrics, dbpedia, amazon = final_payload(status)
    if not final:
        return False, ["missing final_metrics.json"]
    final_avg = summary.get("final_average_task_aware_accuracy", summary.get("final_average_accuracy"))
    forgetting = summary.get("average_forgetting", final_metrics.get("eval.forgetting"))
    checks = [
        ("final_avg", final_avg, ">=", PASS_THRESHOLDS["min_final_avg"]),
        ("dbpedia_retention", dbpedia, ">=", PASS_THRESHOLDS["min_dbpedia_retention"]),
        ("amazon_current", amazon, ">=", PASS_THRESHOLDS["min_amazon_current"]),
        ("forgetting", forgetting, "<=", PASS_THRESHOLDS["max_forgetting"]),
    ]
    ok = True
    reasons = []
    for name, value, op, threshold in checks:
        if value is None:
            ok = False
            reasons.append(f"{name}=missing")
            continue
        passed = value >= threshold if op == ">=" else value <= threshold
        ok = ok and passed
        reasons.append(f"{name}={value} {op} {threshold}")
    return ok, reasons


def any_core_train() -> bool:
    out = subprocess.check_output(["ps", "-eo", "args="], text=True)
    markers = ("python -m core.train", "python core/train.py", "python -u core/train.py")
    return any(any(marker in line for marker in markers) for line in out.splitlines())


def launch_formal() -> bool:
    if any_core_train():
        write_decision({"state": "pass_waiting_gpu", "run_id": EARLY_RUN, "formal_run": FORMAL_RUN})
        return False
    if subprocess.run(["tmux", "has-session", "-t", "standard-ours-order1-v62-formal"], cwd=str(REPO)).returncode == 0:
        write_decision({"state": "formal_already_running", "formal_run": FORMAL_RUN})
        return True

    train_cmd = f"cd {REPO} && PYTHONUNBUFFERED=1 bash scripts/run_ours_v1_strict_iteration.sh {FORMAL_CONFIG}"
    run(["tmux", "new-session", "-d", "-s", "standard-ours-order1-v62-formal", "-n", "train", train_cmd])
    monitor_cmd = (
        f"cd {REPO} && while true; do "
        f"OURS_MONITOR_RUN_ID={FORMAL_RUN} "
        f"OURS_MONITOR_STATUS_BASENAME=standard_peft_ours_v62_formal_status "
        f"OURS_MONITOR_STATUS_TITLE='Standard PEFT Ours v62 Formal Status' "
        f"python scripts/monitor_ours_v18_strict.py; sleep 60; done"
    )
    run(["tmux", "new-window", "-t", "standard-ours-order1-v62-formal", "-n", "monitor", monitor_cmd])
    write_decision({"state": "formal_started", "formal_run": FORMAL_RUN, "formal_config": FORMAL_CONFIG})
    return True


def main() -> None:
    print(f"[{now()}] v62 gate controller started", flush=True)
    write_decision({"state": "watching", "run_id": EARLY_RUN, "thresholds": PASS_THRESHOLDS})
    while True:
        status = refresh_status()
        cls = status.get("classification") or {}
        state = cls.get("state")
        reason = cls.get("reason")
        print(f"[{now()}] earlygate state={state} reason={reason}", flush=True)
        if state == "completed":
            passed, reasons = is_pass(status)
            _, summary, _, dbpedia, amazon = final_payload(status)
            write_decision(
                {
                    "state": "earlygate_passed" if passed else "earlygate_failed_gate",
                    "passed": passed,
                    "reasons": reasons,
                    "run_id": EARLY_RUN,
                    "summary": summary,
                    "dbpedia_retention": dbpedia,
                    "amazon_current": amazon,
                }
            )
            print(f"[{now()}] gate passed={passed} reasons={reasons}", flush=True)
            if passed:
                while not launch_formal():
                    print(f"[{now()}] pass detected; waiting for GPU to launch formal", flush=True)
                    time.sleep(60)
            break
        if state in {"stopped_low_or_failed", "eval_failed", "low_score_gate", "unknown_not_running", "exited_with_artifact"}:
            write_decision({"state": "earlygate_blocked", "run_id": EARLY_RUN, "classification": cls, "status": status})
            break
        time.sleep(60)
    print(f"[{now()}] v62 gate controller exiting", flush=True)


if __name__ == "__main__":
    main()
