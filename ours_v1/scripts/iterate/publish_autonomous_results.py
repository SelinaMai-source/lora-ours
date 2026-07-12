#!/usr/bin/env python3
"""Publish autonomous iteration leaderboard + refresh main formal table."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STATE = REPO / "ours_v1" / "scripts" / "iterate" / "suite_state.json"
TABLE = REPO / "results" / "tables"


def main() -> int:
    TABLE.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    rows = []
    for suite, st in state["suites"].items():
        for mname, m in (st.get("metrics") or {}).items():
            base = m.get("base")
            ours = m.get("ours")
            rows.append({
                "suite": suite,
                "metric": mname,
                "published_base": base,
                "official_or_local_anchor": base,
                "ours": ours,
                "delta": None if ours is None or base is None else ours - base,
                "target": m.get("target"),
                "setting_label": st.get("overlay_active"),
                "version": st.get("current_version"),
                "status": st.get("status"),
                "blocker": st.get("blocker"),
                "last_failure": st.get("last_failure"),
                "seed_order": "order1/seed1" if "citb" in suite or suite == "standard" else "suite-default",
            })
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "campaign": state.get("campaign"),
        "policy": state.get("policy"),
        "rows": rows,
        "suites": state["suites"],
    }
    (TABLE / "ours_iteration_leaderboard.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md = [
        "# Ours Autonomous Iteration Leaderboard",
        "",
        f"Updated: {payload['updated_at']}",
        "",
        "| Suite | Metric | Base | Ours | Δ | Target | Version | Status |",
        "|-------|--------|------|------|---|--------|---------|--------|",
    ]
    for r in rows:
        md.append(
            f"| {r['suite']} | {r['metric']} | {r['published_base']} | {r['ours']} | "
            f"{r['delta']} | {r['target']} | {r['version']} | {r['status']} |"
        )
    # Blockers section
    md.extend(["", "## Blockers / Failures", ""])
    for suite, st in state["suites"].items():
        if st.get("blocker"):
            md.append(f"- **{suite} BLOCKED:** {st['blocker']}")
        if st.get("last_failure"):
            md.append(f"- **{suite} last failure:** `{st['last_failure'].get('path')}`")
    (TABLE / "ours_iteration_leaderboard.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Refresh legacy three-suite table if publisher exists
    pub = REPO / "ours_v1" / "scripts" / "publish_formal_results.py"
    if pub.is_file():
        subprocess.run(["python3", str(pub)], cwd=REPO, check=False)

    # Main paper-facing table snapshot
    main_md = TABLE / "ours_autonomous_main_table.md"
    main_md.write_text(
        "\n".join(
            [
                "# Ours Autonomous Main Table",
                "",
                f"Updated: {payload['updated_at']}",
                "",
                "Sources: published base / local official anchor from frozen matrix; ours from official scorers only.",
                "",
                *md[4:],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"leaderboard": str(TABLE / "ours_iteration_leaderboard.json"), "main": str(main_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
