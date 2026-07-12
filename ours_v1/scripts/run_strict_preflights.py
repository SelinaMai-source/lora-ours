#!/usr/bin/env python3
"""Triple strict preflight: code, data/setting, eval/env gates for ours-v1 formals."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "manifests"
LOG_DIR = REPO / "results" / "logs"
AUTODL = Path("/root/autodl-tmp/lora-baselines-run_v1/external_sources")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(path: Path) -> str:
    if not (path / ".git").exists():
        return "missing"
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd or REPO, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def gate_code(suite: str) -> dict[str, Any]:
    ours_sha = git_head(REPO)
    overlays = {
        "citb_instrdialog": ["ours_v1/suites/citb", "core/methods/ours_spectral_replay.py"],
        "citb_instrdialogpp": ["ours_v1/suites/citb", "core/methods/ours_spectral_replay.py"],
        "standard": ["ours_v1/suites/standard/olora_overlay_ssrg.py", "ours_v1/suites/standard/olora_overlay_assess_retention_gate.py"],
        "arper": ["ours_v1/suites/arper/arper_ssrg_exemplar_selection.py", "ours_v1/suites/arper/arper_ssrg_overlay_train_wrapper.py"],
        "todcl": ["ours_v1/suites/todcl/todcl_assess_orthogonal_overlay.py"],
    }
    official = {
        "citb_instrdialog": AUTODL / "citb",
        "citb_instrdialogpp": AUTODL / "citb",
        "standard": AUTODL / "o_lora",
        "arper": AUTODL / "arper",
        "todcl": AUTODL / "todcl",
    }
    files = overlays.get(suite, [])
    return {
        "pass": all((REPO / f).exists() for f in files),
        "ours_v1_sha": ours_sha,
        "official_sha": git_head(official[suite]),
        "overlay_files": [{ "path": f, "exists": (REPO / f).exists() } for f in files],
    }


def gate_data(suite: str) -> dict[str, Any]:
    if suite == "citb_instrdialog":
        rc, _ = run_cmd([
            sys.executable, "scripts/preflight_citb_official_split_counts.py",
            "--citb-root", str(AUTODL / "citb"),
            "--task-order-dir", "data/CIT_data/task_orders/stream=cl_dialogue_tasks",
            "--order", "1", "--train", "500", "--dev", "50", "--test", "50",
            "--output", str(LOG_DIR / "citb_instrdialog_v1_split_preflight.json"),
        ])
        return {"pass": rc == 0 or rc == 3, "exit_code": rc, "policy": "500/50/50", "note": "exit 3 = short tasks expected"}
    if suite == "citb_instrdialogpp":
        rc, _ = run_cmd([
            sys.executable, "scripts/preflight_citb_official_split_counts.py",
            "--citb-root", str(AUTODL / "citb"),
            "--task-order-dir", "data/CIT_data/task_orders/stream=cl_dialogue_long_tasks",
            "--order", "1", "--train", "100", "--dev", "50", "--test", "100",
            "--output", str(LOG_DIR / "citb_instrdialogpp_v1_split_preflight.json"),
        ])
        return {"pass": rc in (0, 3), "exit_code": rc, "policy": "100/50/100 public actual"}
    if suite == "standard":
        v58 = REPO / "scripts/run_olora_standard_order1_official_base_ours_overlay_v58.sh"
        text = v58.read_text(encoding="utf-8") if v58.is_file() else ""
        math_ok = "import math" in text
        return {"pass": math_ok and (AUTODL / "o_lora").is_dir(), "math_import_patch": math_ok, "hardware_label": "official-equivalent single-GPU grad_accum=8"}
    if suite == "arper":
        cfg = REPO / "results/logs/arper_woz3_paper_aligned_exemplar500_batch128_formal_v89.cfg"
        return {"pass": cfg.is_file(), "base_cfg": str(cfg), "exists": cfg.is_file()}
    if suite == "todcl":
        ckpts = list((AUTODL / "todcl/runs_NLG").glob("**/*.ckpt"))
        return {"pass": len(ckpts) > 0, "checkpoint_count": len(ckpts), "sample_ckpt": str(ckpts[-1]) if ckpts else None}
    return {"pass": False, "error": "unknown suite"}


def gate_eval_env(suite: str) -> dict[str, Any]:
    gpu_rc, gpu_out = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    py_rc, py_out = run_cmd([sys.executable, "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"])
    env = {
        "gpu": gpu_out.split("\n")[0] if gpu_rc == 0 else "unknown",
        "pytorch": py_out if py_rc == 0 else "unknown",
        "gpu_count": 1,
        "hardware_label": "official-equivalent" if suite in ("standard",) else "official",
    }
    scorer = {
        "citb_instrdialog": "official max-over-reference ROUGE-L task-aware",
        "citb_instrdialogpp": "official max-over-reference ROUGE-L task-aware",
        "standard": "O-LoRA official EM scorer",
        "arper": "ARPER run_woz3.py BLEU4/SER",
        "todcl": "ToDCL official NLG scorer BLEU/EER",
    }
    return {"pass": gpu_rc == 0 and py_rc == 0, "scorer": scorer.get(suite), "environment": env}


def build_manifest(suite: str) -> dict[str, Any]:
    code = gate_code(suite)
    data = gate_data(suite)
    eval_env = gate_eval_env(suite)
    blockers = []
    if not code["pass"]:
        blockers.append("code_gate_failed")
    if not data.get("pass"):
        blockers.append("data_gate_failed")
    if not eval_env["pass"]:
        blockers.append("eval_env_gate_failed")
    if suite == "todcl" and not data.get("checkpoint_count"):
        blockers.append("todcl_anchor_ckpt_missing")
    status = "ready" if not blockers else "blocked"
    return {
        "suite": suite,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ours_v1_sha": code["ours_v1_sha"],
        "official_sha": code["official_sha"],
        "preflight_rounds": {"code": code, "data_setting": data, "eval_environment": eval_env},
        "status": status,
        "blockers": blockers,
        "ready_for_formal": status == "ready",
    }


def main() -> int:
    suites = ["standard", "citb_instrdialog", "citb_instrdialogpp", "arper", "todcl"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "suites": {}}
    any_blocked = False
    for suite in suites:
        manifest = build_manifest(suite)
        out_json = OUT_DIR / f"{suite}_ours_v1_formal_manifest.json"
        out_md = OUT_DIR / f"{suite}_ours_v1_formal_manifest.md"
        out_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        md_lines = [
            f"# {suite} ours-v1 formal preflight",
            "",
            f"- Status: **{manifest['status']}**",
            f"- Blockers: {manifest['blockers'] or 'none'}",
            f"- Ready: {manifest['ready_for_formal']}",
            "",
            "## Code gate",
            json.dumps(manifest["preflight_rounds"]["code"], indent=2),
            "",
            "## Data/setting gate",
            json.dumps(manifest["preflight_rounds"]["data_setting"], indent=2),
            "",
            "## Eval/environment gate",
            json.dumps(manifest["preflight_rounds"]["eval_environment"], indent=2),
        ]
        out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        summary["suites"][suite] = {"status": manifest["status"], "blockers": manifest["blockers"]}
        if manifest["status"] == "blocked":
            any_blocked = True
        print(f"{suite}: {manifest['status']}")
    summary_path = OUT_DIR / "ours_v1_strict_preflight_summary_20260712.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 1 if any_blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
