#!/usr/bin/env python3
"""Evaluate official-metric gates from suite_state + scorer outputs. Never invent metrics."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
STATE_PATH = Path(__file__).resolve().parent / "suite_state.json"
LOG_DIR = REPO / "results" / "logs"
RUNS = REPO / "results" / "runs"


def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def plus_one_third(base: float, higher: bool, bounded_0_100: bool) -> float:
    if higher:
        if bounded_0_100:
            return base + (100.0 - base) / 3.0
        return base * (4.0 / 3.0)
    return base * (2.0 / 3.0)


def grep_float(path: Path, patterns: list[str]) -> float | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    for pat in patterns:
        hits = re.findall(pat, text, flags=re.IGNORECASE)
        if hits:
            try:
                return float(hits[-1])
            except ValueError:
                continue
    return None


def read_json(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def metric_standard_em(run_name: str) -> float | None:
    for cand in [
        RUNS / run_name / "ccfa_postprocess" / "summary.json",
        RUNS / run_name / "ours_overlay_manifest.json",
        RUNS / run_name / "run_manifest.json",
    ]:
        data = read_json(cand)
        for key in ("final_average_em", "average_em", "EM", "final_em"):
            if data.get(key) is not None:
                return float(data[key])
    return grep_float(
        LOG_DIR / f"{run_name}.log",
        [r"final.*EM[=:\s]+([0-9]+\.[0-9]+)", r"average.*EM[=:\s]+([0-9]+\.[0-9]+)"],
    )


def metric_citb_ar(run_name: str) -> dict[str, float | None]:
    summary = RUNS / run_name / "ccfa_postprocess" / "summary.json"
    data = read_json(summary)
    ar = bwt = None
    for key in ("average_retention_rougeL", "AR", "rougeL_ar", "avg_rougeL"):
        if data.get(key) is not None:
            ar = float(data[key])
            break
    for key in ("BWT", "bwt", "backward_transfer"):
        if data.get(key) is not None:
            bwt = float(data[key])
            break
    if ar is None:
        ar = grep_float(LOG_DIR / f"{run_name}.log", [r"AR[=:\s]+([0-9]+\.[0-9]+)"])
    return {"AR": ar, "BWT": bwt}


def metric_arper(run_id: str) -> dict[str, float | None]:
    status = read_json(LOG_DIR / f"{run_id}_status.json")
    bleu = ser = None
    for line in status.get("last_signals") or []:
        m = re.search(r"BLEU4:\s*([0-9.]+)", line)
        if m:
            bleu = float(m.group(1))
        m = re.search(r"Slot error:\s*([0-9.]+)", line)
        if m:
            ser = float(m.group(1))
    log = Path(f"/root/autodl-tmp/lora-ours-logs/{run_id}.log")
    if bleu is None:
        bleu = grep_float(log, [r"BLEU4:\s*([0-9.]+)"])
    if ser is None:
        ser = grep_float(log, [r"Slot error:\s*([0-9.]+)"])
    return {"BLEU4": bleu, "SER": ser}


def metric_todcl(run_id: str) -> dict[str, float | None]:
    log = Path(f"/root/autodl-tmp/lora-ours-logs/{run_id}.log")
    return {
        "BLEU": grep_float(log, [r"BLEU[=:\s]+([0-9.]+)", r"avg.*bleu.*?([0-9.]+)"]),
        "EER": grep_float(log, [r"EER[=:\s]+([0-9.]+)", r"entity.*?error.*?([0-9.]+)"]),
    }


def judge(value: float | None, target: float | None, higher: bool) -> str:
    if value is None or target is None:
        return "pending"
    if higher:
        return "PASS" if value >= target else "FAIL"
    return "PASS" if value <= target else "FAIL"


def evaluate_suite(suite: str, run_key: str) -> dict[str, Any]:
    state = load_state()
    suite_st = state["suites"][suite]
    results: dict[str, Any] = {"suite": suite, "run_key": run_key, "metrics": {}, "all_pass": False}

    if suite == "standard":
        got = {"EM": metric_standard_em(run_key)}
    elif suite.startswith("citb"):
        got = metric_citb_ar(run_key)
    elif suite == "arper":
        got = metric_arper(run_key)
    elif suite == "todcl":
        got = metric_todcl(run_key)
    else:
        raise SystemExit(f"unknown suite {suite}")

    all_required_pass = True
    any_pending = False
    for name, spec in suite_st["metrics"].items():
        if spec.get("base") is None and spec.get("target") is None:
            # optional / blocked metric
            results["metrics"][name] = {"ours": got.get(name), "verdict": "n/a"}
            continue
        target = spec.get("target")
        if target is None and spec.get("base") is not None:
            target = plus_one_third(
                float(spec["base"]),
                bool(spec.get("higher_better", True)),
                bool(spec.get("bounded_0_100", False)),
            )
            spec["target"] = target
        ours = got.get(name)
        verdict = judge(ours, target, bool(spec.get("higher_better", True)))
        spec["ours"] = ours
        results["metrics"][name] = {
            "base": spec.get("base"),
            "target": target,
            "ours": ours,
            "verdict": verdict,
            "higher_better": spec.get("higher_better", True),
        }
        if verdict == "pending":
            any_pending = True
            all_required_pass = False
        elif verdict == "FAIL":
            all_required_pass = False

    results["all_pass"] = all_required_pass and not any_pending
    if results["all_pass"]:
        suite_st["status"] = "completed"
        suite_st["best_result"] = results
    elif any_pending:
        suite_st["status"] = suite_st.get("status") or "pending"
    else:
        suite_st["status"] = "failed_metrics"
    save_state(state)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--run-key", required=True, help="run_name / run_id / status basename key")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    out = evaluate_suite(args.suite, args.run_key)
    text = json.dumps(out, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    return 0 if out.get("all_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
