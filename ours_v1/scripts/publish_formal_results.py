#!/usr/bin/env python3
"""Publish ours-v1 three-suite formal results table with +1/3 targets and blockers."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO / "results" / "manifests"
TABLE_DIR = REPO / "results" / "tables"
LOG_DIR = REPO / "results" / "logs"

TARGETS = {
    "citb_instrdialog": {"metric": "AR", "base": 39.98, "target": 53.87, "higher_better": True},
    "citb_instrdialogpp": {"metric": "AR", "base": None, "target": None, "higher_better": True},
    "standard": {"metric": "EM", "base": 76.81, "target": 84.5, "higher_better": True},
    "arper": {"metrics": [("BLEU4", 0.59890, 0.935, True), ("SER", 5.938, 2.72, False)]},
    "todcl": {"metrics": [("BLEU", 21.7719, 29.0, True), ("EER", 0.163975, 0.123, False)]},
}


def plus_one_third(base: float, higher: bool) -> float:
    if higher:
        if base <= 100:
            return base + (100.0 - base) / 3.0
        return base * (4.0 / 3.0)
    return base * (2.0 / 3.0)


def read_json(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def grep_metric(log_path: Path, patterns: list[str]) -> float | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for pat in patterns:
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m[-1])
            except ValueError:
                continue
    return None


def citb_ar(run_name: str) -> float | None:
    summary = REPO / "results" / "runs" / run_name / "ccfa_postprocess" / "summary.json"
    if summary.is_file():
        data = read_json(summary)
        for key in ("average_retention_rougeL", "AR", "rougeL_ar", "avg_rougeL"):
            if key in data and data[key] is not None:
                return float(data[key])
    log = LOG_DIR / f"{run_name}.log"
    return grep_metric(log, [r"AR[=:\s]+([0-9]+\.[0-9]+)", r"average.*rouge.*?([0-9]+\.[0-9]+)"])


def standard_em(run_name: str) -> float | None:
    run_dir = REPO / "results" / "runs" / run_name
    for candidate in [
        run_dir / "ccfa_postprocess" / "summary.json",
        run_dir / "ours_overlay_manifest.json",
    ]:
        if candidate.is_file():
            data = read_json(candidate)
            for key in ("final_average_em", "average_em", "EM"):
                if key in data and data[key] is not None:
                    return float(data[key])
    log = LOG_DIR / f"{run_name}.log"
    return grep_metric(log, [r"final.*EM[=:\s]+([0-9]+\.[0-9]+)", r"average.*EM.*?([0-9]+\.[0-9]+)"])


def arper_metrics(run_id: str) -> dict[str, float | None]:
    status = read_json(LOG_DIR / f"{run_id}_status.json")
    bleu = ser = None
    for line in status.get("last_signals") or []:
        if "BLEU4" in line:
            m = re.search(r"BLEU4:\s*([0-9.]+)", line)
            if m:
                bleu = float(m.group(1))
        if "Slot error" in line or "SER" in line:
            m = re.search(r"Slot error:\s*([0-9.]+)", line)
            if m:
                ser = float(m.group(1))
    log = Path(f"/root/autodl-tmp/lora-ours-logs/{run_id}.log")
    if bleu is None:
        bleu = grep_metric(log, [r"BLEU4:\s*([0-9.]+)"])
    if ser is None:
        ser = grep_metric(log, [r"Slot error:\s*([0-9.]+)"])
    return {"BLEU4": bleu, "SER": ser}


def todcl_metrics(run_id: str) -> dict[str, float | None]:
    log = Path(f"/root/autodl-tmp/lora-ours-logs/{run_id}.log")
    return {
        "BLEU": grep_metric(log, [r"BLEU[=:\s]+([0-9.]+)", r"avg.*bleu.*?([0-9.]+)"]),
        "EER": grep_metric(log, [r"EER[=:\s]+([0-9.]+)", r"entity.*?error.*?([0-9.]+)"]),
    }


def evaluate(higher: bool, ours: float | None, target: float) -> str:
    if ours is None:
        return "pending"
    return "pass" if (ours >= target if higher else ours <= target) else "fail"


def main() -> int:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []

    specs = [
        ("citb_instrdialog", "citb_instrdialog_order1_seed1_ours_v1_20260708_formal_strict", citb_ar),
        ("citb_instrdialogpp", "citb_instrdialogpp_order1_seed1_ours_v1_20260708_formal_strict", citb_ar),
        ("standard", "olora_official_base_ours_overlay_v1_20260708_formal_order1_seed1", standard_em),
    ]
    for suite, run_name, fn in specs:
        manifest = read_json(MANIFEST_DIR / f"{suite}_ours_v1_formal_manifest.json")
        if manifest.get("status") == "blocked":
            blockers.append({"suite": suite, "blockers": manifest.get("blockers")})
        ours = fn(run_name)
        t = TARGETS[suite]
        base = t.get("base")
        target = t.get("target")
        if base is not None and target is None:
            target = plus_one_third(base, t["higher_better"])
        rows.append({
            "suite": suite,
            "official_base": base,
            "ours_v1": ours,
            "delta": (ours - base) if ours is not None and base is not None else None,
            "target_plus_one_third": target,
            "pass_fail": evaluate(t["higher_better"], ours, target) if target else ("pending" if ours is None else "n/a"),
            "manifest": str(MANIFEST_DIR / f"{suite}_ours_v1_formal_manifest.json"),
            "run_name": run_name,
        })

    for suite, run_id in [
        ("arper", "arper_woz3_sclstm_v89_ssrg_exemplar_overlay_v1_20260708_formal"),
        ("todcl", "todcl_adapter_nlg_ours_assess_overlay_v1_20260708_formal"),
    ]:
        manifest = read_json(MANIFEST_DIR / f"{suite}_ours_v1_formal_manifest.json")
        if manifest.get("status") == "blocked":
            blockers.append({"suite": suite, "blockers": manifest.get("blockers")})
        metrics = arper_metrics(run_id) if suite == "arper" else todcl_metrics(run_id)
        for name, base, target, higher in TARGETS[suite]["metrics"]:
            ours = metrics.get(name)
            rows.append({
                "suite": f"{suite}:{name}",
                "official_base": base,
                "ours_v1": ours,
                "delta": (ours - base) if ours is not None else None,
                "target_plus_one_third": target,
                "pass_fail": evaluate(higher, ours, target),
                "manifest": str(MANIFEST_DIR / f"{suite}_ours_v1_formal_manifest.json"),
                "run_id": run_id,
            })

    out_json = TABLE_DIR / "ours_v1_three_suite_formal.json"
    out_md = TABLE_DIR / "ours_v1_three_suite_formal.md"
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "blockers": blockers,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Ours v1 Three-Suite Formal Results",
        "",
        f"Updated: {payload['updated_at']}",
        "",
        "| Suite | Official base | Ours v1 | Delta | +1/3 target | Pass/Fail | Manifest |",
        "|-------|---------------|---------|-------|---------------|-----------|----------|",
    ]
    for r in rows:
        md.append(
            f"| {r['suite']} | {r.get('official_base','—')} | {r.get('ours_v1','—')} | "
            f"{r.get('delta','—')} | {r.get('target_plus_one_third','—')} | {r['pass_fail']} | "
            f"`{r['manifest']}` |"
        )
    if blockers:
        md.extend(["", "## Blockers", ""])
        for b in blockers:
            md.append(f"- **{b['suite']}**: {b.get('blockers')}")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out_json), "md": str(out_md), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
