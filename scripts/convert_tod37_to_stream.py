#!/usr/bin/env python3
"""TOD37 preprocessing skeleton for strict paper-aligned runs.

This intentionally does not download data or launch training. It checks whether
the raw TM19/TM20/SGD/MultiWOZ files required by AdapterCL/ToDCL are present and
records an actionable blocker manifest when they are not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_RAW_ROOT = Path("external_baselines/adaptercl_dialogue/data")
DEFAULT_OUT = Path("data/processed/tod37_nlg_cl_domains_train50_eval10.json")
DEFAULT_MANIFEST = Path("results/tables/tod37_preprocess_blocker_manifest.json")

EXPECTED_RAW_FILES: Dict[str, List[str]] = {
    "TM19": ["Taskmaster/TM-1-2019/woz-dialogs.json"],
    "TM20": ["Taskmaster/TM-2-2020/data"],
    "SGD": ["dstc8-schema-guided-dialogue/train", "dstc8-schema-guided-dialogue/dev", "dstc8-schema-guided-dialogue/test"],
    "MultiWOZ": ["multiwoz/data/MultiWOZ_2.2/data.json"],
}


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def check_raw_files(raw_root: Path) -> Dict[str, object]:
    datasets = []
    missing = []
    for dataset, rel_paths in EXPECTED_RAW_FILES.items():
        entries = []
        dataset_ok = True
        for rel in rel_paths:
            path = raw_root / rel
            ok = path.exists()
            entries.append({"path": str(path), "present": bool(ok)})
            if not ok:
                dataset_ok = False
                missing.append(str(path))
        datasets.append({"dataset": dataset, "ready": bool(dataset_ok), "required": entries})
    return {
        "benchmark": "TOD37",
        "source": "AdapterCL/ToDCL EMNLP 2021",
        "raw_root": str(raw_root),
        "ready": not missing,
        "missing": missing,
        "datasets": datasets,
        "download_commands": [
            "cd external_baselines/adaptercl_dialogue/data && bash download.sh",
        ],
        "preprocess_commands": [
            "python external_baselines/adaptercl_dialogue/utils/preprocess.py",
            "python scripts/convert_tod37_to_stream.py --check-only",
        ],
        "next_action": (
            "Run external_baselines/adaptercl_dialogue/data/download.sh in an isolated AdapterCL environment, "
            "then run AdapterCL preprocessing and rerun this script before exporting the 37-domain NLG stream."
        ),
        "strict_blocker": (
            "Raw TM19/TM20/SGD/MultiWOZ files are external and absent locally; no Method x TOD37 cell may be strict "
            "until these files exist and the processed 37-domain stream is exported and audited."
        ),
    }


def write_manifest(manifest: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check TOD37 raw data readiness for strict preprocessing.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check-only", action="store_true", help="Only check raw data and write the blocker manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    raw_root = _resolve(repo_root, args.raw_root)
    out_path = _resolve(repo_root, args.out)
    manifest_path = _resolve(repo_root, args.manifest)

    manifest = check_raw_files(raw_root)
    manifest["planned_output"] = str(out_path)
    write_manifest(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    if not manifest["ready"]:
        raise SystemExit(2)
    if args.check_only:
        return
    raise SystemExit(
        "TOD37 raw files are present, but export is still a skeleton. "
        "Next implementation step: adapt AdapterCL utils/dataloader.py NLG extraction "
        "to write the unified 37-domain stream JSON."
    )


if __name__ == "__main__":
    main()
