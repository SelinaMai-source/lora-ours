#!/usr/bin/env python3
"""Validate and summarize lora-run_v10 strict-alignment status.

This script is intentionally CPU/read-only with respect to training. It checks
that every requested Method x Benchmark cell is represented, that no row is
marked strict without a completed/evidence-backed status, and that known bridge
or data blockers are materialized in local manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

METHODS = [
    "Sequential LoRA",
    "Replay LoRA",
    "O-LoRA",
    "LB-CL",
    "Progressive Prompts",
    "Continual-T0",
    "LFPT5",
    "Ours",
]

BENCHMARKS = [
    "InstrDialog",
    "InstrDialog++",
    "TRACE",
    "MultiWOZ NLG",
    "Seq-GLUE",
]

STRICT_STATUSES = {"strict", "strict-complete", "official-strict-complete"}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_key(row: Dict[str, str]) -> tuple[str, str]:
    return str(row.get("method", "")), str(row.get("benchmark", ""))


def _is_strict(status: str) -> bool:
    lowered = status.strip().lower()
    return lowered in STRICT_STATUSES


def _category(status: str) -> str:
    lowered = status.lower()
    if _is_strict(status):
        return "strict"
    if any(
        token in lowered
        for token in [
            "runner-bridge",
            "official-runner-ready",
            "strict-preflight-ready",
            "strict-runner-skeleton",
            "parser-implemented",
            "completed-needs",
            "protocol-diagnosed",
            "protocol-repaired",
            "user-stopped-partial",
        ]
    ):
        return "near_strict"
    if "blocked" in lowered or "required" in lowered or "needs" in lowered or "not-strict" in lowered or "mismatch" in lowered:
        return "blocked_or_needs_audit"
    return "other"


def _summarize_rows(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    categories = Counter(_category(row.get("status", "")) for row in rows)
    statuses = Counter(row.get("status", "") for row in rows)
    by_method: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_method[row["method"]][_category(row["status"])] += 1
    return {
        "num_cells": len(rows),
        "category_counts": dict(categories),
        "status_counts": dict(statuses),
        "by_method": {method: dict(counts) for method, counts in sorted(by_method.items())},
    }


def _validate_matrix(rows: Sequence[Dict[str, str]], repo: Path) -> List[str]:
    errors: List[str] = []
    expected = {(method, benchmark) for method in METHODS for benchmark in BENCHMARKS}
    observed = {_cell_key(row) for row in rows}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        errors.append("missing cells: " + ", ".join(f"{m}/{b}" for m, b in missing))
    if extra:
        errors.append("unexpected cells: " + ", ".join(f"{m}/{b}" for m, b in extra))

    for row in rows:
        if None in row:
            errors.append(
                f"{row.get('method', '<unknown>')} / {row.get('benchmark', '<unknown>')} has extra CSV columns; quote or remove commas"
            )
        status = row.get("status", "")
        if _is_strict(status):
            evidence = " ".join([row.get("current_evidence", ""), row.get("strict_gap", ""), row.get("next_action", "")]).lower()
            if any(token in evidence for token in ["need", "missing", "blocked", "mismatch", "suspicious", "not "]):
                errors.append(f"{row['method']} / {row['benchmark']} has strict-like status but unresolved evidence: {status}")

    olora_manifest = _read_json(repo / "results/runs/lora_run_v10_seqglue_o_lora_official_s123/run_manifest.json")
    if olora_manifest.get("status") not in {"ready", "running", "completed"}:
        errors.append("O-LoRA Seq-GLUE official runner manifest is not ready/running/completed")

    return errors


def _selected_rows(rows: Iterable[Dict[str, str]], methods: Sequence[str]) -> List[Dict[str, str]]:
    wanted = set(methods)
    return [row for row in rows if row.get("method") in wanted]


def _evidence_chain_for_row(row: Dict[str, str]) -> Dict[str, Any]:
    """Convert the human tracker row into a conservative strict-gate record."""
    status = row.get("status", "")
    status_l = status.lower()
    evidence_l = row.get("current_evidence", "").lower()
    benchmark = row.get("benchmark", "")

    gates: Dict[str, Any] = {
        "official_or_paper_source": "needs_audit",
        "paper_setting_hparams": "needs_audit",
        "data_processing_and_split": "needs_audit",
        "label_instruction_mapping": "needs_audit",
        "metric_protocol": "needs_audit",
        "official_or_equivalent_runner": "needs_audit",
        "full_run_or_verified_equivalent": False,
        "strict_allowed": False,
    }

    if "official env" in evidence_l or "official source" in evidence_l or "official-source" in status_l:
        gates["official_or_paper_source"] = True
    if "citb paper source found" in evidence_l or "paper source found" in evidence_l:
        gates["official_or_paper_source"] = True
    if "no dedicated official" in evidence_l or "faithful-port" in status_l:
        gates["official_or_paper_source"] = False
        gates["official_or_equivalent_runner"] = False
    if "local-adaptation-not-strict-paper-mismatch" in status_l:
        gates["paper_setting_hparams"] = False
        gates["official_or_equivalent_runner"] = False
        gates["metric_protocol"] = False
    if "blocked-data" in status_l:
        gates["data_processing_and_split"] = False
    elif "config/data available" in evidence_l or "exports 8 segments" in evidence_l or "stream" in evidence_l:
        gates["data_processing_and_split"] = True
    if "runner-bridge" in status_l or "official-runner-ready" in status_l or "strict-preflight-ready" in status_l:
        gates["official_or_equivalent_runner"] = True
    if "ar/fwt/bwt/fr" in evidence_l or "metric module" in evidence_l:
        gates["metric_protocol"] = True
    if "completed" in status_l:
        gates["full_run_or_verified_equivalent"] = True
    if "user-stopped-partial" in status_l:
        gates["full_run_or_verified_equivalent"] = False
    if "suspicious" in status_l or "all-zero" in row.get("strict_gap", "").lower():
        gates["metric_protocol"] = False
    if "strict-runner-skeleton" in status_l:
        gates["official_or_equivalent_runner"] = "skeleton_only"
        gates["full_run_or_verified_equivalent"] = False
    if row.get("method") == "O-LoRA" and row.get("benchmark") == "Seq-GLUE":
        gates.update(
            {
                "official_or_paper_source": True,
                "data_processing_and_split": True,
                "label_instruction_mapping": "bridge_validates_local_labels_needs_paper_equivalence",
                "official_or_equivalent_runner": True,
                "paper_setting_hparams": "config_recorded_needs_paper_citation",
                "metric_protocol": False,
                "full_run_or_verified_equivalent": "completed" in status_l,
            }
        )
    if row.get("method") == "LFPT5" and row.get("benchmark") == "InstrDialog++":
        gates.update(
            {
                "official_or_paper_source": True,
                "data_processing_and_split": True,
                "official_or_equivalent_runner": True,
                "full_run_or_verified_equivalent": True,
                "paper_setting_hparams": "needs_lfpt5_paper_citation",
                "metric_protocol": False,
            }
        )

    blocker_kind = "needs_audit"
    if "blocked-data" in status_l:
        blocker_kind = "missing_data"
    elif "faithful-port" in status_l:
        blocker_kind = "missing_official_or_faithful_implementation"
    elif "mismatch" in status_l or "not-strict" in status_l:
        blocker_kind = "paper_setting_mismatch"
    elif "official-source-needs-bridge" in status_l or "official-source-needs-benchmark-bridge" in status_l:
        blocker_kind = "missing_official_bridge"
    elif "strict-preflight-ready" in status_l:
        blocker_kind = "preflight_ready_needs_full_run_and_audit"
    elif "strict-runner-skeleton" in status_l:
        blocker_kind = "runner_skeleton_needs_assets_or_implementation"
    elif "official-runner-ready" in status_l:
        blocker_kind = "runner_ready_needs_full_run_and_audit"
    elif "suspicious-zero" in status_l or "protocol-diagnosed" in status_l:
        blocker_kind = "completed_but_metric_suspicious"
    elif "user-stopped-partial" in status_l:
        blocker_kind = "partial_user_stopped"
    elif row.get("method") == "Ours":
        blocker_kind = "project_method_not_official_baseline"

    return {
        "method": row.get("method", ""),
        "benchmark": row.get("benchmark", ""),
        "priority": row.get("priority", ""),
        "status": status,
        "category": _category(status),
        "blocker_kind": blocker_kind,
        "gates": gates,
        "current_evidence": row.get("current_evidence", ""),
        "strict_gap": row.get("strict_gap", ""),
        "next_action": row.get("next_action", ""),
    }


def _write_markdown(
    path: Path,
    *,
    rows: Sequence[Dict[str, str]],
    summary: Dict[str, Any],
    errors: Sequence[str],
    olora_manifest: Dict[str, Any],
) -> None:
    strict_rows = [row for row in rows if _category(row.get("status", "")) == "strict"]
    near_rows = [row for row in rows if _category(row.get("status", "")) == "near_strict"]
    blocked_rows = [row for row in rows if _category(row.get("status", "")) == "blocked_or_needs_audit"]

    lines = [
        "# lora-run_v10 Strict Alignment Report",
        "",
        "Updated: 2026-06-21 02:46 CST",
        "",
        "Rule: no Method x Benchmark cell is strict unless the official source or paper setting, data processing, metric, runner, and full-run/equivalent evidence are all present. This report only downgrades or preserves status; it does not promote scaffold outputs to strict.",
        "",
        "## Summary",
        "",
        f"- Matrix coverage: {summary['num_cells']} / {len(METHODS) * len(BENCHMARKS)} cells.",
        f"- Strict-complete cells: {len(strict_rows)}.",
        f"- Near-strict / runner-ready diagnostic cells: {len(near_rows)}.",
        f"- Blocked or still audit-needed cells: {len(blocked_rows)}.",
        f"- Validation status: {'PASS' if not errors else 'FAIL'}",
        "",
        "## Strict-complete",
        "",
    ]
    if strict_rows:
        for row in strict_rows:
            lines.append(f"- {row['method']} / {row['benchmark']}: `{row['status']}`")
    else:
        lines.append("- None. No current cell has enough evidence to be marked strict.")

    lines.extend(
        [
            "",
            "## Near Strict / Useful Diagnostic Evidence",
            "",
        ]
    )
    if near_rows:
        for row in near_rows:
            lines.append(f"- {row['method']} / {row['benchmark']}: `{row['status']}`. Evidence: {row['current_evidence']} Gap: {row['strict_gap']}")
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Method-level Status",
            "",
        ]
    )
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        statuses = Counter(row["status"] for row in method_rows)
        status_text = "; ".join(f"`{status}` x{count}" for status, count in sorted(statuses.items()))
        lines.append(f"- {method}: {status_text}")

    lines.extend(
        [
            "",
            "## Blocking Evidence",
            "",
            "- Sequential LoRA / Replay LoRA: CITB official source, split/order files, LM-adapted T5-small asset, CPU-safe command planning, Replay(10/50) policy, and AR/FWT/BWT/FR metric helpers are present for InstrDialog. Stage-1 seed=50 full run is now running in `tmux` session `citb_stage1_seed50`; both cells remain non-strict until that checkpoint completes and the official Stage-2 full runs, metric parser, and paper comparison all pass.",
            "- O-LoRA: official source, Seq-GLUE runner bridge, one official diagnostic full run, and paper mismatch audit are present; the local T5-small 8-segment bridge still does not match the published T5-large O-LoRA protocol or later Seq-GLUE-7 target.",
            "- LB-CL: no dedicated official source was found after web searches; `results/tables/lb_cl_official_source_audit.json` records the evidence, so every tracked cell requires an audited faithful port before strict.",
            "- Progressive Prompts / Continual-T0: PP now has an official command preflight and `results_dict.npy` parser, but the local 8-task bridge is not the published 15-task order; Continual-T0 has source plus CT0 tokenizer/config assets and an active full-checkpoint HF download, while Drive processed assets remain permission-blocked.",
            "- LFPT5: task-token decoding leakage has a bridge-level protocol repair and smoke evidence, but prior runs remain diagnostic; clean one-segment smoke and full rerun are still required before any strict claim.",
            "- TOD37: retired from the tracked benchmark matrix by user decision; it is no longer counted as a strict-alignment blocker.",
            "",
            "## O-LoRA Seq-GLUE Bridge Check",
            "",
            f"- Manifest status: `{olora_manifest.get('status', 'missing')}`.",
            f"- Official entry: `{olora_manifest.get('entry', 'missing')}`.",
            f"- Command includes official runner: `{bool(olora_manifest.get('command'))}`.",
            "- Data bridge validation is enforced by `scripts/export_seqglue_to_olora.py --validate-only` and by the official runner dry-run path.",
            "- Per-cell strict gates are also written to `results/tables/lora_run_v10_strict_alignment_evidence.json` for interruption-safe follow-up.",
            "",
            "## Validation Errors",
            "",
        ]
    )
    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("- None.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check lora-run_v10 strict-alignment tracker coverage and blockers.")
    parser.add_argument("--tracker", type=Path, default=Path("results/tables/lora_run_v10_strict_alignment_tracker.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("results/tables/lora_run_v10_strict_alignment_summary.json"))
    parser.add_argument("--evidence-json", type=Path, default=Path("results/tables/lora_run_v10_strict_alignment_evidence.json"))
    parser.add_argument("--report-md", type=Path, default=Path("docs/lora_run_v10_strict_alignment_report.md"))
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    tracker = args.tracker if args.tracker.is_absolute() else repo / args.tracker
    rows = _read_csv(tracker)
    summary = _summarize_rows(rows)
    errors = _validate_matrix(rows, repo)
    olora_manifest = _read_json(repo / "results/runs/lora_run_v10_seqglue_o_lora_official_s123/run_manifest.json")

    payload = {
        "summary": summary,
        "validation_errors": errors,
        "strict_cells": [row for row in rows if _category(row.get("status", "")) == "strict"],
        "near_strict_cells": [row for row in rows if _category(row.get("status", "")) == "near_strict"],
        "blocked_cells": [row for row in rows if _category(row.get("status", "")) == "blocked_or_needs_audit"],
    }

    evidence_matrix = {
        "rule": "strict_allowed is false unless source/paper setting, data processing, label/instruction mapping, metric protocol, runner, and full-run/equivalent evidence all pass.",
        "cells": [_evidence_chain_for_row(row) for row in rows],
    }

    summary_path = args.summary_json if args.summary_json.is_absolute() else repo / args.summary_json
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    evidence_path = args.evidence_json if args.evidence_json.is_absolute() else repo / args.evidence_json
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence_matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_path = args.report_md if args.report_md.is_absolute() else repo / args.report_md
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(report_path, rows=rows, summary=summary, errors=errors, olora_manifest=olora_manifest)

    print(json.dumps({"summary_json": str(summary_path), "evidence_json": str(evidence_path), "report_md": str(report_path), "errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
