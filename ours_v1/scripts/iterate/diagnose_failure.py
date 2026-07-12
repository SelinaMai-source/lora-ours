#!/usr/bin/env python3
"""First-principles single-cause diagnosis from failure evidence. Writes failure_vN.md."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
STATE_PATH = Path(__file__).resolve().parent / "suite_state.json"
DOCS = REPO / "docs" / "experiments"


RP_MECHANISMS = [
    "ssrg_class_coverage",
    "assess_retention_gate",
    "assess_retention_threshold",
    "lora_bank_router",
    "drift_detection",
    "anti_overlap",
    "replay_budget",
    "amazon_round_assess_pause",
]


def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def diagnose_standard(evidence: dict[str, Any]) -> dict[str, Any]:
    """Prefer retention vs current-task underlearning vs routing/overlap."""
    dbpedia = evidence.get("dbpedia_em")
    amazon = evidence.get("amazon_em")
    reason = (evidence.get("reason") or "").lower()
    version = evidence.get("version", "ours-v1")

    if "early gate" in reason and amazon is not None and dbpedia is not None:
        if dbpedia >= 90 and amazon < 50:
            primary = "current_task_underlearning"
            mechanism_suspect = "assess_retention_gate_interaction_under_smoke_caps"
            # v1 already had assess on; if amazon round pause already applied, escalate.
            if "skip" in reason or evidence.get("assess_skip_amazon"):
                primary = "current_task_underlearning_after_assess_pause"
                mechanism_suspect = "ssrg_class_coverage_or_replay_budget"
            recommended = (
                "amazon_round_assess_pause"
                if version.endswith("v1")
                else "replay_budget"
            )
            return {
                "primary_cause": primary,
                "mechanism_suspect": mechanism_suspect,
                "recommended_next_delta": recommended,
                "rationale": (
                    f"dbpedia EM {dbpedia} PASS with amazon EM {amazon} FAIL indicates "
                    "SC current-task underlearning after TC→SC transition, not retention collapse."
                ),
            }

    if "assess-retention" in reason:
        return {
            "primary_cause": "retention_gate_block",
            "mechanism_suspect": "assess_retention_threshold",
            "recommended_next_delta": "assess_retention_threshold",
            "rationale": "Assess gate rejected adapter promotion; isolate threshold before stacking mechanisms.",
        }

    return {
        "primary_cause": "unknown_or_infra",
        "mechanism_suspect": "queue_or_env",
        "recommended_next_delta": "replay_budget",
        "rationale": "Insufficient metric pattern; prefer smallest RP overlay knob (replay budget) after infra check.",
    }


def diagnose_citb(evidence: dict[str, Any]) -> dict[str, Any]:
    ar = evidence.get("AR")
    target = evidence.get("AR_target")
    if ar is not None and target is not None and ar < target:
        return {
            "primary_cause": "insufficient_retention_or_replay_coverage",
            "mechanism_suspect": "ssrg_ratio_or_replay_budget",
            "recommended_next_delta": "replay_budget",
            "rationale": f"AR {ar} < target {target}; adjust SSRG/replay budget only (one delta).",
        }
    return {
        "primary_cause": "run_incomplete_or_infra",
        "mechanism_suspect": "launcher_or_data",
        "recommended_next_delta": "ssrg_class_coverage",
        "rationale": "No complete AR; fix entry/data then re-smoke before mechanism change.",
    }


def diagnose_arper(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_cause": "exemplar_selection_quality",
        "mechanism_suspect": "ssrg_exemplar_selection",
        "recommended_next_delta": "ssrg_class_coverage",
        "rationale": "Keep Path B SCLSTM; only tune SSRG exemplar selector hyperparameters as next delta.",
        "forbidden_reminder": evidence.get("forbidden") or ["Fisher CPU", "post-decode repair"],
    }


def diagnose_todcl(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_cause": "assess_orthogonal_strength",
        "mechanism_suspect": "assess_threshold_or_orthogonal_penalty",
        "recommended_next_delta": "assess_retention_threshold",
        "rationale": "Adjust Assess-then-Update threshold/penalty one at a time on ADAPTER base.",
    }


def write_report(suite: str, version: str, diagnosis: dict[str, Any], evidence: dict[str, Any]) -> Path:
    DOCS.mkdir(parents=True, exist_ok=True)
    ver_num = re.sub(r"[^0-9]", "", version) or "N"
    path = DOCS / f"{suite}_failure_v{ver_num}_{datetime.now().strftime('%Y%m%d')}.md"
    # Stable name preferred by plan: {suite}_failure_vN.md
    stable = DOCS / f"{suite}_failure_v{ver_num}.md"
    body = [
        f"# {suite} failure — {version}",
        "",
        f"Updated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Evidence",
        "",
        "```json",
        json.dumps(evidence, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Diagnosis (single primary cause)",
        "",
        f"- **primary_cause:** {diagnosis['primary_cause']}",
        f"- **mechanism_suspect:** {diagnosis['mechanism_suspect']}",
        f"- **recommended_next_delta:** {diagnosis['recommended_next_delta']}",
        f"- **rationale:** {diagnosis['rationale']}",
        "",
        "## Policy",
        "",
        "- One mechanism only in next version.",
        "- Do not change official scorer/split/task order/model.",
        "- Published-base + minimal overlay only.",
        "",
    ]
    text = "\n".join(body)
    path.write_text(text, encoding="utf-8")
    stable.write_text(text, encoding="utf-8")
    return stable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--evidence-json", default="", help="path to evidence json or inline omitted")
    ap.add_argument("--dbpedia-em", type=float, default=None)
    ap.add_argument("--amazon-em", type=float, default=None)
    ap.add_argument("--reason", default="")
    ap.add_argument("--ar", type=float, default=None)
    ap.add_argument("--ar-target", type=float, default=None)
    ap.add_argument("--assess-skip-amazon", action="store_true")
    args = ap.parse_args()

    evidence: dict[str, Any] = {"version": args.version, "suite": args.suite}
    if args.evidence_json:
        p = Path(args.evidence_json)
        if p.is_file():
            evidence.update(json.loads(p.read_text(encoding="utf-8")))
    if args.dbpedia_em is not None:
        evidence["dbpedia_em"] = args.dbpedia_em
    if args.amazon_em is not None:
        evidence["amazon_em"] = args.amazon_em
    if args.reason:
        evidence["reason"] = args.reason
    if args.ar is not None:
        evidence["AR"] = args.ar
    if args.ar_target is not None:
        evidence["AR_target"] = args.ar_target
    if args.assess_skip_amazon:
        evidence["assess_skip_amazon"] = True

    # Auto-pull from known v1 smoke status if standard and empty
    if args.suite == "standard" and "amazon_em" not in evidence:
        status = read_text(REPO / "results/logs/olora_official_base_ours_overlay_v1_20260708_smoke_order1_seed1_status.md")
        m = re.search(r"amazon: EM ([0-9.]+)", status)
        if m:
            evidence["amazon_em"] = float(m.group(1))
        m = re.search(r"dbpedia.*?EM ([0-9.]+)|reason:.*?dbpedia", status)
        # parse reason line
        rm = re.search(r"reason:\s*(.*)", status)
        if rm:
            evidence.setdefault("reason", rm.group(1).strip())
        # also from failure doc
        fail = read_text(REPO / "docs/experiments/standard_failure_v1_20260712.md")
        m = re.search(r"dbpedia.*?EM \*\*([0-9.]+)", fail)
        if m:
            evidence["dbpedia_em"] = float(m.group(1))
        m = re.search(r"amazon.*?EM \*\*([0-9.]+)", fail)
        if m:
            evidence["amazon_em"] = float(m.group(1))

    if args.suite == "standard":
        diagnosis = diagnose_standard(evidence)
    elif args.suite.startswith("citb"):
        diagnosis = diagnose_citb(evidence)
    elif args.suite == "arper":
        diagnosis = diagnose_arper(evidence)
    elif args.suite == "todcl":
        diagnosis = diagnose_todcl(evidence)
    else:
        raise SystemExit(f"unknown suite {args.suite}")

    report = write_report(args.suite, args.version, diagnosis, evidence)
    state = load_state()
    suite_st = state["suites"][args.suite]
    suite_st["last_failure"] = {
        "version": args.version,
        "path": str(report.relative_to(REPO)),
        "diagnosis": diagnosis,
        "evidence": evidence,
    }
    suite_st["status"] = "failed_diagnosed"
    save_state(state)

    out = {"report": str(report), "diagnosis": diagnosis}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
