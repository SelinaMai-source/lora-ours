#!/usr/bin/env python3
"""Publish ours-v1 three-suite formal summary table with blockers and +1/3 targets."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "results" / "manifests" / "ours_v1_three_suite_preflight.json"
OUT_MD = REPO / "results" / "tables" / "ours_v1_three_suite_formal.md"
OUT_JSON = REPO / "results" / "tables" / "ours_v1_three_suite_formal.json"

# Published reference baselines (operational bases per plan)
PUBLISHED = {
    "CITB InstrDialog": {"metric": "AR", "base": 39.98, "direction": "higher_bounded", "bound": 100.0},
    "CITB InstrDialog++": {"metric": "AR", "base": None, "direction": "higher_bounded", "bound": 100.0, "note": "split not paper-exact"},
    "Standard": {"metric": "avg_EM", "base": 55.2, "direction": "higher_bounded", "bound": 100.0},
    "ARPER": {"metric": "BLEU4", "base": 0.5989, "direction": "higher_bounded", "bound": 1.0},
    "ToDCL BLEU": {"metric": "BLEU", "base": None, "direction": "higher_bounded", "bound": 1.0},
    "ToDCL EER": {"metric": "EER", "base": None, "direction": "lower"},
}


def target_plus_third(base: Optional[float], direction: str, bound: float = 100.0) -> Optional[float]:
    if base is None:
        return None
    if direction == "lower":
        return base * (2.0 / 3.0)
    if direction == "higher_bounded":
        return base + (bound - base) / 3.0
    return base + (100.0 - base) / 3.0 if base <= 100 else base * (4.0 / 3.0)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def _queue_state() -> List[str]:
    p = REPO / "results/logs/ours_v1_formal_serial_queue.json"
    if not p.is_file():
        return []
    return _read_json(p).get("results", [])


def main() -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    preflight = _read_json(MANIFEST)
    blockers = preflight.get("blockers", [])
    queue = _queue_state()

    rows: List[Dict[str, Any]] = []
    for suite, ref in PUBLISHED.items():
        tgt = target_plus_third(ref.get("base"), ref["direction"], ref.get("bound", 100.0))
        suite_key = suite.lower().replace(" ", "_").replace("++", "pp")
        blocker = None
        if "instrdialog++" in suite.lower() or "instrdialogpp" in suite_key:
            blocker = next((b for b in blockers if "instrdialogpp" in b), None)
        elif "instrdialog" in suite.lower():
            blocker = None  # disclosed split only; formal allowed
        elif "todcl" in suite.lower():
            blocker = next((b for b in blockers if "todcl" in b), None)
        else:
            blocker = next((b for b in blockers if suite.split()[0].lower() in b.lower()), None)
        rows.append({
            "suite": suite,
            "official_base": ref.get("base"),
            "target_plus_one_third": tgt,
            "ours_v1": None,
            "delta": None,
            "pass": None,
            "blocker": blocker,
            "queue": [q for q in queue if suite.split()[0].lower() in q.lower() or ("todcl" in q.lower() and "ToDCL" in suite)],
        })

    OUT_JSON.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(),
        "ours_v1_sha": preflight.get("meta", {}).get("ours_v1_sha"),
        "rows": rows,
        "blockers": blockers,
        "queue_state": queue,
        "manifest": str(MANIFEST),
    }, indent=2) + "\n")

    lines = [
        "# Ours v1 Three-Suite Formal Results",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Blockers",
        "",
    ]
    if blockers:
        for b in blockers:
            lines.append(f"- {b}")
    else:
        lines.append("- (none at preflight)")
    lines.extend(["", "## Summary Table", "", "| Suite | Official Base | Ours v1 | Delta | +1/3 Target | Status |", "|---|---:|---:|---:|---:|---|"])
    for r in rows:
        base = r["official_base"]
        tgt = r["target_plus_one_third"]
        status = r["blocker"] or ("queued: " + ", ".join(r["queue"]) if r["queue"] else "pending")
        lines.append(
            f"| {r['suite']} | {base if base is not None else 'N/A'} | — | — | "
            f"{tgt if tgt is not None else 'N/A'} | {status} |"
        )
    lines.extend([
        "",
        "## Manifests",
        "",
        f"- Matrix: `results/manifests/ours_v1_formal_matrix_frozen.json`",
        f"- Preflight: `results/manifests/ours_v1_three_suite_preflight.json`",
        f"- Standard: `results/manifests/standard_ours_v1_formal_manifest.json`",
        f"- Phase0 seal: `results/archives/phase0_stop_seal_20260712_044615/PHASE0_STOP_SEAL_MANIFEST.json`",
        "",
        "## Disclosures",
        "",
        "- CITB InstrDialog: `official_script_500_50_50` (not paper test=100)",
        "- CITB InstrDialog++: `public_script_100_25_25_not_paper_exact`",
        "- Standard: single-GPU official-equivalent",
        "- ARPER: no `ARPER_EWC_FISHER_CPU`; v89 SCLSTM anchor",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(OUT_MD)


if __name__ == "__main__":
    main()
