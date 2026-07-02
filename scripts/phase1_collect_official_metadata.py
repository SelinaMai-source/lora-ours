#!/usr/bin/env python3
"""Collect lightweight official-repo metadata for alignment audits.

This avoids large dataset/model downloads. It records HEAD commits, README files,
and candidate script/config paths through git ls-remote and the GitHub tree API.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "docs/official_alignment"
PREFLIGHT = REPO / "results/preflight/phase1_preflight_latest.json"

OFFICIAL_REPOS = {
    "citb": "https://github.com/hyintell/CITB",
    "o_lora": "https://github.com/cmnfriend/O-LoRA",
    "lfpt5": "https://github.com/qcwthu/Lifelong-Fewshot-Language-Learning",
    "progressive_prompts": "https://github.com/arazd/ProgressivePrompts",
    "arper_nlg": "https://github.com/MiFei/Continual-Learning-for-NLG",
    "todcl": "https://github.com/andreamad8/ToDCL",
}


def run_git(args: list[str], timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["git", *args],
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
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": "timeout", "stdout": "", "stderr": ""}


def github_owner_repo(url: str) -> tuple[str, str]:
    suffix = url.rstrip("/").removeprefix("https://github.com/")
    owner, repo_name = suffix.split("/", 1)
    return owner, repo_name.removesuffix(".git")


def normalized_url(url: str) -> str:
    return url.rstrip("/").removesuffix(".git")


def fetch_json(url: str, timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_text(url: str, timeout: int) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def load_preflight_heads(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    heads: dict[str, str] = {}
    for info in (payload.get("official_repos") or {}).values():
        url = str(info.get("url", ""))
        head = str(info.get("head", ""))
        if url and head:
            heads[normalized_url(url)] = head
    return heads


def remote_head(url: str, timeout: int) -> tuple[str, str, dict[str, Any]]:
    result = run_git(["ls-remote", "--symref", url, "HEAD"], timeout=timeout)
    branch = ""
    head = ""
    for line in result.get("stdout", "").splitlines():
        if line.startswith("ref:"):
            parts = line.split()
            if len(parts) >= 2:
                branch = parts[1].removeprefix("refs/heads/")
        elif line.strip().endswith("HEAD"):
            head = line.split()[0]
    return branch, head, result


def candidate_paths(tree: list[dict[str, Any]]) -> dict[str, list[str]]:
    paths = [item["path"] for item in tree if item.get("type") == "blob" and item.get("path")]
    readmes = [p for p in paths if Path(p).name.lower().startswith("readme")]
    scripts = [
        p
        for p in paths
        if (
            p.endswith((".py", ".sh", ".yaml", ".yml", ".json"))
            and any(token in p.lower() for token in ["run", "train", "script", "config", "baseline", "order"])
        )
    ]
    return {"readmes": sorted(readmes)[:20], "script_config_candidates": sorted(scripts)[:200]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect official repo metadata.")
    parser.add_argument("--timeout-sec", type=int, default=15)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--preflight-json", type=Path, default=PREFLIGHT)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    readme_dir = out_dir / "readmes"
    readme_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repos": {},
    }
    preflight_path = args.preflight_json if args.preflight_json.is_absolute() else REPO / args.preflight_json
    preflight_heads = load_preflight_heads(preflight_path)

    for name, url in OFFICIAL_REPOS.items():
        branch, head, git_result = remote_head(url, timeout=args.timeout_sec)
        entry: dict[str, Any] = {
            "url": url,
            "default_branch": branch,
            "head": head,
            "git_ls_remote": git_result,
            "github_tree_ok": False,
            "readme_files_saved": [],
            "script_config_candidates": [],
        }
        if not head:
            fallback_head = preflight_heads.get(normalized_url(url), "")
            if fallback_head:
                head = fallback_head
                entry["head"] = head
                entry["head_source"] = str(preflight_path.relative_to(REPO))
        if head:
            owner, repo_name = github_owner_repo(url)
            if not branch:
                try:
                    repo_payload = fetch_json(
                        f"https://api.github.com/repos/{owner}/{repo_name}",
                        timeout=args.timeout_sec,
                    )
                    branch = str(repo_payload.get("default_branch") or "")
                    entry["default_branch"] = branch
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                    entry["default_branch_error"] = repr(exc)
            tree_url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/{head}?recursive=1"
            try:
                tree_payload = fetch_json(tree_url, timeout=args.timeout_sec)
                paths = candidate_paths(tree_payload.get("tree", []))
                entry["github_tree_ok"] = True
                entry["readme_paths"] = paths["readmes"]
                entry["script_config_candidates"] = paths["script_config_candidates"]
                ref = branch or head
                for readme_path in paths["readmes"][:5]:
                    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/{ref}/{readme_path}"
                    try:
                        text = fetch_text(raw_url, timeout=args.timeout_sec)
                    except (urllib.error.URLError, TimeoutError) as exc:
                        entry.setdefault("readme_fetch_errors", []).append(
                            {"path": readme_path, "error": repr(exc)}
                        )
                        continue
                    safe_path = readme_path.replace("/", "__")
                    target = readme_dir / f"{name}__{safe_path}"
                    target.write_text(text, encoding="utf-8")
                    entry["readme_files_saved"].append(str(target.relative_to(REPO)))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                entry["github_tree_error"] = repr(exc)
        report["repos"][name] = entry

    out_path = out_dir / "official_repo_metadata_v0.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
