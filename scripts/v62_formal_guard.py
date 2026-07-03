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
FORMAL_SESSION = "standard-ours-order1-v62-formal"
DIAGNOSTIC_SESSION = "standard-ours-order1-v62-eval-diagnostic"
LEGACY_GATE_SESSION = "standard-ours-order1-v62-gate-controller"
DECISION_PATH = REPO / "results/logs/standard_peft_ours_v62_gate_decision.json"
GUARD_LOG = REPO / "results/logs/standard_peft_ours_v62_formal_guard.log"
PASS_THRESHOLDS = {
    "min_final_avg": 0.70,
    "min_dbpedia_retention": 0.90,
    "min_amazon_current": 0.45,
    "max_forgetting": 0.10,
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    GUARD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with GUARD_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(REPO), text=True, **kwargs)


def command_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(REPO), text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def tmux_sessions() -> set[str]:
    out = command_output(["tmux", "list-sessions", "-F", "#{session_name}"])
    return {line.strip() for line in out.splitlines() if line.strip() and not line.startswith("no server")}


def write_decision(payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = now()
    DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DECISION_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def session_exists(name: str) -> bool:
    return name in tmux_sessions()


def pane_text(name: str) -> str:
    if not session_exists(name):
        return ""
    return command_output(["tmux", "capture-pane", "-pt", f"{name}:0.0", "-S", "-40"])


def kill_session(name: str, reason: str) -> None:
    if session_exists(name):
        log(f"killing {name}: {reason}")
        subprocess.run(["tmux", "kill-session", "-t", name], cwd=str(REPO), text=True)


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
    out = command_output(["ps", "-eo", "args="])
    markers = ("python -m core.train", "python core/train.py", "python -u core/train.py")
    return any(any(marker in line for marker in markers) for line in out.splitlines())


def protect_formal_path() -> None:
    kill_session(DIAGNOSTIC_SESSION, "formal guard blocks v62 diagnostic before formal gate decision")
    kill_session(LEGACY_GATE_SESSION, "formal guard is the authoritative v62 gate controller")


def launch_formal() -> bool:
    if any_core_train():
        write_decision({"state": "pass_waiting_gpu", "run_id": EARLY_RUN, "formal_run": FORMAL_RUN})
        return False
    if session_exists(FORMAL_SESSION):
        write_decision({"state": "formal_already_running", "run_id": EARLY_RUN, "formal_run": FORMAL_RUN})
        return True

    train_cmd = f"cd {REPO} && PYTHONUNBUFFERED=1 bash scripts/run_ours_v1_strict_iteration.sh {FORMAL_CONFIG}"
    run(["tmux", "new-session", "-d", "-s", FORMAL_SESSION, "-n", "train", train_cmd])
    monitor_cmd = (
        f"cd {REPO} && while true; do "
        f"OURS_MONITOR_RUN_ID={FORMAL_RUN} "
        f"OURS_MONITOR_STATUS_BASENAME=standard_peft_ours_v62_formal_status "
        f"OURS_MONITOR_STATUS_TITLE='Standard PEFT Ours v62 Formal Status' "
        f"python scripts/monitor_ours_v18_strict.py; sleep 60; done"
    )
    run(["tmux", "new-window", "-t", FORMAL_SESSION, "-n", "monitor", monitor_cmd])
    write_decision({"state": "formal_started", "formal_run": FORMAL_RUN, "formal_config": FORMAL_CONFIG})
    return True


def main() -> None:
    log("v62 formal guard started")
    write_decision({"state": "watching_formal_guard", "run_id": EARLY_RUN, "thresholds": PASS_THRESHOLDS})
    while True:
        protect_formal_path()
        status = refresh_status()
        cls = status.get("classification") or {}
        state = cls.get("state")
        reason = cls.get("reason")
        log(f"earlygate state={state} reason={reason}")
        if session_exists(FORMAL_SESSION):
            write_decision({"state": "formal_already_running", "run_id": EARLY_RUN, "formal_run": FORMAL_RUN})
            time.sleep(60)
            continue
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
            log(f"gate passed={passed} reasons={reasons}")
            if passed:
                while not launch_formal():
                    log("pass detected; waiting for GPU to launch formal")
                    time.sleep(30)
            return
        if state in {"stopped_low_or_failed", "eval_failed", "low_score_gate", "unknown_not_running", "exited_with_artifact"}:
            write_decision({"state": "earlygate_blocked", "run_id": EARLY_RUN, "classification": cls, "status": status})
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
