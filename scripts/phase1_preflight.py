#!/usr/bin/env python3
"""Phase-1 preflight for official continual-learning experiments.

The script is intentionally conservative: it gathers evidence and writes a JSON
report, but it never starts training and never upgrades scaffold results to
benchmark claims.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "configs/phase1/official_target_manifest.yaml"
DEFAULT_OUT = REPO / "results/preflight/phase1_preflight_latest.json"


def run_cmd(args: list[str], timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": "timeout",
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip() if isinstance(exc.stderr, str) else "",
            "timeout_sec": timeout,
        }
    except FileNotFoundError as exc:
        return {"ok": False, "returncode": "missing", "stderr": str(exc)}


def repo_head(url: str, timeout: int) -> dict[str, Any]:
    result = run_cmd(["git", "ls-remote", url, "HEAD"], timeout=timeout)
    head = ""
    if result["ok"] and result["stdout"]:
        head = result["stdout"].split()[0]
    result["head"] = head
    return result


def collect_repos(manifest: dict[str, Any]) -> dict[str, str]:
    repos: dict[str, str] = {}
    for suite_name, suite in (manifest.get("suites") or {}).items():
        repo = suite.get("official_repo")
        if repo:
            repos[f"{suite_name}.official_repo"] = str(repo)
        for key, value in (suite.get("official_repos") or {}).items():
            repos[f"{suite_name}.{key}"] = str(value)
    return repos


def check_paths(manifest: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for suite_name, suite in (manifest.get("suites") or {}).items():
        required = suite.get("local_required_paths") or {}
        out[suite_name] = {
            key: {
                "path": value,
                "exists": (REPO / str(value)).exists(),
                "is_dir": (REPO / str(value)).is_dir(),
            }
            for key, value in required.items()
        }
    return out


def check_wandb(timeout: int) -> dict[str, Any]:
    cli = shutil.which("wandb")
    out: dict[str, Any] = {"cli": cli or "", "status": None}
    if cli:
        out["status"] = run_cmd([cli, "status"], timeout=timeout)
    return out


def check_torch() -> dict[str, Any]:
    try:
        import torch

        payload = {
            "import_ok": True,
            "version": getattr(torch, "__version__", ""),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_devices": [],
        }
        if torch.cuda.is_available():
            payload["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
        return payload
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"import_ok": False, "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run phase-1 environment and alignment preflight.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--remote-timeout-sec", type=int, default=20)
    parser.add_argument("--wandb-timeout-sec", type=int, default=10)
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else REPO / args.manifest
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    repos = collect_repos(manifest)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(REPO),
        "git": {
            "status": run_cmd(["git", "status", "--short", "--branch"], timeout=10),
            "head": run_cmd(["git", "rev-parse", "HEAD"], timeout=10),
            "branch": run_cmd(["git", "branch", "--show-current"], timeout=10),
            "remote": run_cmd(["git", "remote", "-v"], timeout=10),
        },
        "tools": {
            "git": run_cmd(["git", "--version"], timeout=10),
            "tmux": run_cmd(["tmux", "-V"], timeout=10),
            "python": {"executable": sys.executable, "version": sys.version},
            "torch": check_torch(),
            "wandb": check_wandb(timeout=args.wandb_timeout_sec),
        },
        "official_repos": {
            name: {"url": url, **repo_head(url, timeout=args.remote_timeout_sec)}
            for name, url in repos.items()
        },
        "local_required_paths": check_paths(manifest),
        "guardrail": {
            "training_started": False,
            "toy_smoke_counts_as_benchmark": False,
            "strict_results_available": False,
        },
    }

    out_path = args.out if args.out.is_absolute() else REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    missing_paths = [
        f"{suite}.{key}"
        for suite, paths in report["local_required_paths"].items()
        for key, info in paths.items()
        if not info["exists"]
    ]
    blocked_remotes = [
        name for name, info in report["official_repos"].items() if not info.get("ok")
    ]
    if missing_paths or blocked_remotes:
        print("Preflight status: blocked")
        if missing_paths:
            print("Missing local paths:", ", ".join(missing_paths))
        if blocked_remotes:
            print("Remote checks failed:", ", ".join(blocked_remotes))
        return 2
    print("Preflight status: ready for explicit smoke/full-run decision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
