#!/usr/bin/env python3
"""Download TRACE benchmark raw JSON (HF mirror first, Google Drive fallback).

Expected layout after success:
  data/raw/trace/{C-STANCE,FOMC,...}/train.json|eval.json|test.json
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = REPO / "data/raw/trace"
DEFAULT_DOWNLOAD_DIR = DEFAULT_RAW_ROOT / "_downloads"
GDRIVE_FILE_ID = "1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV"
ZIP_NAME = "trace_benchmark.zip"

TRACE_TASKS = (
    "C-STANCE",
    "FOMC",
    "MeetingBank",
    "Py150",
    "ScienceQA",
    "NumGLUE-cm",
    "NumGLUE-ds",
    "20Minuten",
)

# Community mirrors / repacks (best-effort; official source is Google Drive).
HF_CANDIDATES = (
    "BeyonderXX/TRACE-benchmark",
    "wangzx/TRACE-benchmark",
    "trace-benchmark/TRACE",
)


def _task_ready(raw_root: Path, task: str) -> bool:
    task_dir = raw_root / task
    return all((task_dir / split).is_file() for split in ("train.json", "eval.json", "test.json"))


def is_raw_complete(raw_root: Path) -> bool:
    return all(_task_ready(raw_root, task) for task in TRACE_TASKS)


def _log(msg: str) -> None:
    print(msg, flush=True)


def try_hf_download(raw_root: Path, download_dir: Path) -> bool:
    """Try HuggingFace Hub mirrors (uses HF_ENDPOINT if set, e.g. hf-mirror.com)."""
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        _log("[hf] huggingface_hub not installed; skip")
        return False

    api = HfApi()
    search_terms = ["TRACE continual learning", "trace benchmark BeyonderXX", "C-STANCE FOMC MeetingBank"]
    repo_ids: list[str] = list(HF_CANDIDATES)
    for term in search_terms:
        try:
            for ds in api.list_datasets(search=term, limit=10):
                if ds.id not in repo_ids:
                    repo_ids.append(ds.id)
        except Exception as exc:
            _log(f"[hf] search failed for {term!r}: {exc}")

    for repo_id in repo_ids:
        _log(f"[hf] trying snapshot_download: {repo_id}")
        try:
            local_dir = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(download_dir / f"hf_{repo_id.replace('/', '_')}"),
            )
        except Exception as exc:
            _log(f"[hf] {repo_id}: {exc}")
            continue

        root = Path(local_dir)
        # Flat zip at repo root
        for zpath in root.rglob("*.zip"):
            if _unpack_zip(zpath, raw_root):
                if is_raw_complete(raw_root):
                    _log(f"[hf] success via {repo_id} ({zpath.name})")
                    return True

        # Task folders already extracted
        if is_raw_complete(root):
            _copy_task_tree(root, raw_root)
            if is_raw_complete(raw_root):
                _log(f"[hf] success via {repo_id} (task tree)")
                return True

        # Nested task dirs
        _copy_task_tree(root, raw_root)
        if is_raw_complete(raw_root):
            _log(f"[hf] success via {repo_id} (nested copy)")
            return True

    return False


def _copy_task_tree(src: Path, dst: Path) -> None:
    for task in TRACE_TASKS:
        src_task = src / task
        if not src_task.is_dir():
            continue
        dst_task = dst / task
        dst_task.mkdir(parents=True, exist_ok=True)
        for split in ("train.json", "eval.json", "test.json"):
            s = src_task / split
            if s.is_file():
                shutil.copy2(s, dst_task / split)


def _unpack_zip(zip_path: Path, raw_root: Path) -> bool:
    if not zip_path.is_file() or zip_path.stat().st_size < 1024:
        return False
    _log(f"[unzip] {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_root)
        return True
    except zipfile.BadZipFile as exc:
        _log(f"[unzip] bad zip: {exc}")
        return False


def try_gdown(download_dir: Path, raw_root: Path) -> bool:
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / ZIP_NAME
    log_path = download_dir / "gdown.log"

    cmd = [
        sys.executable, "-m", "gdown",
        f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
        "-O", str(zip_path),
    ]
    _log(f"[gdown] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    if proc.returncode != 0:
        _log(f"[gdown] failed (exit={proc.returncode}); see {log_path}")
        return False
    if not zip_path.is_file() or zip_path.stat().st_size < 1024:
        _log(f"[gdown] zip missing or too small: {zip_path}")
        return False
    if not _unpack_zip(zip_path, raw_root):
        return False
    return is_raw_complete(raw_root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull TRACE raw benchmark (HF first, gdown fallback).")
    p.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--download-dir", default=str(DEFAULT_DOWNLOAD_DIR))
    p.add_argument("--skip-hf", action="store_true")
    p.add_argument("--skip-gdown", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root)
    if not raw_root.is_absolute():
        raw_root = REPO / raw_root
    download_dir = Path(args.download_dir)
    if not download_dir.is_absolute():
        download_dir = REPO / download_dir
    raw_root.mkdir(parents=True, exist_ok=True)

    if is_raw_complete(raw_root):
        _log(f"[ok] TRACE raw already complete under {raw_root}")
        return 0

    ready = [t for t in TRACE_TASKS if _task_ready(raw_root, t)]
    if ready:
        _log(f"[partial] {len(ready)}/{len(TRACE_TASKS)} tasks present: {ready}")

    if not args.skip_hf:
        if try_hf_download(raw_root, download_dir):
            return 0

    if not args.skip_gdown:
        if try_gdown(download_dir, raw_root):
            _log("[ok] TRACE raw downloaded via Google Drive")
            return 0

    _log("[fail] TRACE raw download failed (HF + gdown). Network may block Google Drive.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
