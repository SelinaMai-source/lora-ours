#!/usr/bin/env python3
"""Generate gap manifest: only missing/blocked/queued cells from full matrix audit."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "results/tables/full_matrix_gap_audit_s123.csv"
OUT = REPO / "results/tables/published_setting_run_v2_gap_manifest.csv"
WANDB = "lora-run_v10"


def main() -> None:
    if not AUDIT.is_file():
        raise SystemExit(f"Run gen_full_matrix_gap_audit_s123.py first: missing {AUDIT}")

    rows_in = list(csv.DictReader(AUDIT.open("r", encoding="utf-8")))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_rows: list[dict] = []
    for row in rows_in:
        if row.get("status") == "completed":
            continue
        status = row["status"]
        if status not in {"queued", "blocked", "missing"}:
            continue
        bench = row["benchmark_key"]
        method = row["method_key"]
        out_rows.append({
            "run_name": row["run_name"],
            "benchmark": bench if bench != "multiwoz" else "multiwoz_nlg",
            "method": "ours_published" if method == "ours" else method,
            "seed": "123",
            "config_path": row.get("config_path", ""),
            "wandb_project": WANDB,
            "wandb_group": WANDB,
            "status": status,
            "notes": row.get("notes", ""),
            "runner": "lfpt5_external" if method == "lfpt5" else "core_train",
            "exit_code": "",
            "updated_at": now,
            "log_file": f"results/logs/published_setting_run_v2_gap/{row['run_name']}.log",
        })

    fields = [
        "run_name", "benchmark", "method", "seed", "config_path",
        "wandb_project", "wandb_group", "status", "runner", "notes",
        "exit_code", "updated_at", "log_file",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    counts: dict[str, int] = {}
    for r in out_rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"Wrote {len(out_rows)} gap rows -> {OUT}")
    print("status:", counts)


if __name__ == "__main__":
    main()
