#!/usr/bin/env python3
"""Prepare or verify phase-1 data inputs.

By default this script only verifies required official inputs. Use
`--write-toy-smoke` to materialize the tiny pipeline-health stream; that stream
is explicitly excluded from benchmark reporting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TOY_STREAM = REPO / "data/sample/ours_v0_smoke_stream.json"


def validate_stream(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    stream = raw.get("stream")
    if not isinstance(stream, list) or not stream:
        raise ValueError(f"Invalid stream in {path}: missing non-empty stream list")
    for seg in stream:
        if "segment_id" not in seg or "segment_name" not in seg:
            raise ValueError(f"Invalid segment metadata in {path}: {seg}")
        for split in ("train", "eval"):
            rows = seg.get(split)
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"Segment {seg.get('segment_id')} missing non-empty {split}")
            for row in rows:
                for key in ("instruction", "input", "output"):
                    if key not in row:
                        raise ValueError(f"Example missing {key}: {row}")
    return {
        "path": str(path.relative_to(REPO)),
        "benchmark": raw.get("benchmark"),
        "version": raw.get("version"),
        "segments": len(stream),
        "strict_benchmark_result": False,
    }


def write_toy_stream(path: Path) -> None:
    if path.exists():
        validate_stream(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "toy_smoke_do_not_report",
        "version": "v0_pipeline_healthcheck",
        "stream": [
            {
                "segment_id": 0,
                "segment_name": "toy_weather",
                "train": [
                    {
                        "instruction": "Answer with the requested weather label.",
                        "input": "city=alpha",
                        "output": "sunny",
                    }
                ],
                "eval": [
                    {
                        "instruction": "Answer with the requested weather label.",
                        "input": "city=alpha",
                        "output": "sunny",
                    }
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify phase-1 data readiness.")
    parser.add_argument("--write-toy-smoke", action="store_true")
    parser.add_argument("--toy-stream", type=Path, default=TOY_STREAM)
    parser.add_argument("--check-official", action="store_true")
    args = parser.parse_args()

    toy_path = args.toy_stream if args.toy_stream.is_absolute() else REPO / args.toy_stream
    if args.write_toy_smoke:
        write_toy_stream(toy_path)
    toy_info = validate_stream(toy_path)
    print(json.dumps({"toy_smoke": toy_info}, ensure_ascii=False, indent=2))

    if args.check_official:
        required = [
            REPO / "data/raw/citb",
            REPO / "data/processed/citb_cl_dialogue_tasks_train50_eval10.json",
            REPO / "data/processed/citb_cl_38_random_tasks_train50_eval10.json",
            REPO / "data/processed/multiwoz_nlg_cl_domains_train50_eval10.json",
            REPO / "assets/pretrained/citb/lm_adapted_t5_small",
            REPO / "assets/pretrained/t5-large",
        ]
        missing = [str(path.relative_to(REPO)) for path in required if not path.exists()]
        if missing:
            print("Official-data status: blocked")
            print("Missing:", ", ".join(missing))
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
