#!/usr/bin/env python3
"""
LFPT5 published_setting external runner wrapper.

LFPT5 requires T5 prompt tuning in an isolated environment (see external_baselines/lfpt5/).
This wrapper validates prerequisites, exports our processed stream to LFPT5-compatible JSON,
and launches the continual bridge in conda env ``lfll_1`` when the LM-adapted checkpoint exists.

Usage:
  python scripts/run_lfpt5_published_setting.py --config configs/paper/published_setting/instrdialog__lfpt5__s123.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _load_stream(stream_file: str) -> dict:
    path = REPO / "data/processed" / stream_file
    if not path.is_file():
        raise FileNotFoundError(f"Processed stream missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _export_for_lfpt5(stream: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for seg in stream.get("stream", []):
        sid = seg["segment_id"]
        name = seg.get("segment_name", f"task_{sid}")
        task_dir = out_dir / f"segment_{sid:03d}_{name}"
        task_dir.mkdir(parents=True, exist_ok=True)
        with (task_dir / "train.json").open("w", encoding="utf-8") as f:
            json.dump(seg.get("train", []), f, ensure_ascii=False, indent=2)
        with (task_dir / "eval.json").open("w", encoding="utf-8") as f:
            json.dump(seg.get("eval", []), f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="LFPT5 published_setting wrapper")
    parser.add_argument("--config", required=True, help="published_setting LFPT5 yaml")
    parser.add_argument("--dry-run", action="store_true", help="Only validate and export stream")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path
    if not cfg_path.is_file():
        print(f"[blocked] config missing: {cfg_path}", file=sys.stderr)
        return 2

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    run_name = (cfg.get("output") or {}).get("run_name", "lfpt5_run")
    lfpt5_cfg = cfg.get("lfpt5") or {}
    ckpt = REPO / lfpt5_cfg.get(
        "lm_adapted_torch_ckpt",
        "assets/pretrained/lfpt5/lm_adapted_t5_large_torch/pytorch_model.bin",
    )
    stream_file = (cfg.get("paths") or {}).get("processed_stream_file", "")
    lfpt5_repo = REPO / lfpt5_cfg.get("repo_subdir", "external_baselines/lfpt5")

    run_dir = REPO / "results/runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    export_dir = run_dir / "lfpt5_export"
    manifest = {
        "run_name": run_name,
        "runner": "lfpt5_external",
        "config": str(cfg_path.relative_to(REPO)),
        "lfpt5_repo": str(lfpt5_repo.relative_to(REPO)),
        "checkpoint": str(ckpt),
        "checkpoint_ready": ckpt.is_file(),
        "official_entry": str(lfpt5_repo / "Classification/Classification.py"),
        "status": "blocked",
        "block_reason": "",
    }

    if not lfpt5_repo.is_dir():
        manifest["block_reason"] = f"LFPT5 repo missing: {lfpt5_repo}"
        _write_manifest(run_dir, manifest)
        print(f"[blocked] {manifest['block_reason']}", file=sys.stderr)
        return 3

    try:
        stream = _load_stream(stream_file)
        _export_for_lfpt5(stream, export_dir)
        manifest["export_dir"] = str(export_dir.relative_to(REPO))
        manifest["n_segments"] = len(stream.get("stream", []))
    except FileNotFoundError as e:
        manifest["block_reason"] = str(e)
        _write_manifest(run_dir, manifest)
        print(f"[blocked] {e}", file=sys.stderr)
        return 4

    if not ckpt.is_file():
        manifest["block_reason"] = (
            "LM-adapted T5-large PyTorch checkpoint missing. "
            "Download TF ckpt via gsutil and run external_baselines/lfpt5/convertmodel.py. "
            "See docs/lfpt5_published_setting.md"
        )
        _write_manifest(run_dir, manifest)
        print(f"[blocked] {manifest['block_reason']}", file=sys.stderr)
        return 5

    manifest["status"] = "ready"
    manifest["block_reason"] = ""
    _write_manifest(run_dir, manifest)

    if args.dry_run:
        print(f"[dry-run] LFPT5 prerequisites OK for {run_name}; export at {export_dir}")
        return 0

    bridge = REPO / "external_baselines/lfpt5/citb_bridge/run_continual.py"
    bridge_max_epoch = int(lfpt5_cfg.get("bridge_max_epoch", 4))
    max_segments = int((cfg.get("data") or {}).get("max_segments", -1))
    cuda = str(lfpt5_cfg.get("cuda", "0"))

    cmd = [
        "conda",
        "run",
        "-n",
        "lfll_1",
        "--no-capture-output",
        "python",
        str(bridge),
        "--config",
        str(cfg_path),
        "--export-dir",
        str(export_dir),
        "--run-dir",
        str(run_dir),
        "--max-epoch",
        str(bridge_max_epoch),
        "--cuda",
        cuda,
    ]
    if max_segments > 0:
        cmd.extend(["--max-segments", str(max_segments)])

    manifest["status"] = "running"
    manifest["bridge_cmd"] = cmd
    _write_manifest(run_dir, manifest)

    print(f"[lfpt5] launching bridge: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO))
    if result.returncode == 0:
        manifest["status"] = "completed"
    else:
        manifest["status"] = "failed"
        manifest["block_reason"] = f"bridge exit code {result.returncode}"
    _write_manifest(run_dir, manifest)
    return int(result.returncode)


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
