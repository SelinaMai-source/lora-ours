#!/usr/bin/env python3
"""
Compute ToDCL ADAPTER BLEU/EER for paper-alignment watch.

Important:
- Do NOT retrain ToDCL. This script only runs scorer on existing generated_responses.
- Parse BLEU/EER from scorer's GitHub-flavored Markdown table *data rows* (ADAPTER),
  not from header rows.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _parse_float(s: str) -> float:
    m = re.search(r"[-+]?(?:\d+\.\d+|\d+)", s.strip())
    if not m:
        raise ValueError(f"no numeric value in: {s!r}")
    return float(m.group(0))


def _parse_bleu_eer_from_table_github(stdout: str) -> tuple[float, float]:
    """
    Expected (tabulate tablefmt="github"):
      | Name    |   BLEU |   EER |
      |---------|--------|--------|
      | ADAPTER | 21.77  | 0.164  |
    """
    bleu = None
    eer = None
    for line in stdout.splitlines():
        if "|" not in line:
            continue
        # Strip leading/trailing '|' then split by '|'
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 3:
            continue
        if parts[0].upper() != "ADAPTER":
            continue

        # parts: [Name, BLEU, EER]
        bleu = _parse_float(parts[1])
        eer = _parse_float(parts[2])
        break

    if bleu is None or eer is None:
        raise RuntimeError("Failed to find ADAPTER row BLEU/EER in scorer output table")
    return bleu, eer


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    todcl_root = Path(
        os.environ.get(
            "TODCL_ROOT",
            "/root/autodl-tmp/lora-baselines-run_v1/external_sources/todcl",
        )
    ).resolve()

    # Folder containing immediate run candidates for scorer.py.
    # This path is where train.py writes ADAPTER/REPLAY checkpoints.
    todcl_runs_nlg_root = Path(
        os.environ.get(
            "TODCL_RUNS_NLG_ROOT",
            str(todcl_root / "runs_NLG/SGD,TM19,TM20,MWOZ"),
        )
    ).resolve()

    run_id = os.environ.get("TODCL_RUN_ID", "todcl_adapter_nlg_official_anchor_20260706")
    python_bin = os.environ.get(
        "PYTHON_BIN",
        "/root/autodl-tmp/conda_envs/todcl_legacy_py37/bin/python",
    )

    metrics_json_path = Path(
        os.environ.get(
            "TODCL_METRICS_JSON",
            str(repo_root / f"results/logs/todcl_adapter_nlg_official_anchor_20260706_metrics.json"),
        )
    ).resolve()
    wrapper_base_dir = Path(
        os.environ.get(
            "TODCL_SCORER_WRAPPER_DIR",
            str(repo_root / "results/logs/.todcl_scorer_wrappers"),
        )
    ).resolve()

    if not todcl_root.exists():
        raise FileNotFoundError(f"Missing TODCL_ROOT: {todcl_root}")
    if not todcl_runs_nlg_root.exists():
        raise FileNotFoundError(f"Missing TODCL_RUNS_NLG_ROOT: {todcl_runs_nlg_root}")

    # Locate an existing generated_responses.json under ADAPTER runs.
    # In this repo the directory depth can include the model_checkpoint path,
    # so we search recursively for .../FINAL/generated_responses.json.
    gen_resp_glob = str(todcl_runs_nlg_root / "ADAPTER*" / "**" / "FINAL" / "generated_responses.json")
    gen_resp_paths = glob.glob(gen_resp_glob, recursive=True)
    if not gen_resp_paths:
        raise FileNotFoundError(
            "No generated_responses.json found for ToDCL ADAPTER under:\n"
            f"  {todcl_runs_nlg_root}\n"
            f"  glob={gen_resp_glob}"
        )

    # Use the newest one (most recent attempt) to avoid stale metrics.
    gen_resp_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    gen_resp_path = Path(gen_resp_paths[0]).resolve()
    final_dir = gen_resp_path.parent  # .../FINAL
    folder_with_final = final_dir.parent  # .../gpt2 (contains FINAL/)

    if not (folder_with_final / "FINAL" / "generated_responses.json").exists():
        raise RuntimeError(
            "Internal error: resolved folder_with_final does not contain FINAL/generated_responses.json:\n"
            f"  folder_with_final={folder_with_final}"
        )

    # scorer.py's table row name is derived from the immediate folder name.
    # Here the directory containing FINAL is a nested model_checkpoint folder name (often 'gpt2'),
    # so to get an 'ADAPTER' row we create a small wrapper dir with a symlink named ADAPTER.
    wrapper_dir = wrapper_base_dir / run_id
    shutil.rmtree(wrapper_dir, ignore_errors=True)
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    adapter_link = wrapper_dir / "ADAPTER"
    os.symlink(str(folder_with_final), str(adapter_link))

    cmd = [
        python_bin,
        str(todcl_root / "scorer.py"),
        "--model_checkpoint",
        str(wrapper_dir),
        "--task_type",
        "NLG",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(todcl_root),  # ensure multi-bleu.perl path resolution inside scorer
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        raise RuntimeError(
            "ToDCL scorer failed.\n"
            f"cmd={' '.join(cmd)}\n"
            f"returncode={proc.returncode}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}\n"
        )

    bleu, eer = _parse_bleu_eer_from_table_github(stdout)

    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "suite": "ToDCL_ADAPTER_NLG_official_anchor",
        "paper_metric": {"bleu": 21.7719, "eer": 0.163975},
        "local_metric": {"bleu": bleu, "eer": eer},
        "parsed_from": {
            "scorer_output_table": "tabulate(github)",
            "selected_row": "ADAPTER (data row, not header)",
        },
        "source": {
            "generated_responses_json": str(gen_resp_path),
            "wrapper_dir": str(wrapper_dir),
            "folder_with_final": str(folder_with_final),
        },
    }
    metrics_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Print a compact line for shell consumers.
    print(f"ToDCL_ADAPTER_METRICS bleu={bleu:.6f} eer={eer:.6f} json={metrics_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

