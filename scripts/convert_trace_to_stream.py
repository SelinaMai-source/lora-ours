#!/usr/bin/env python3
"""
Convert TRACE benchmark raw JSON (prompt/answer) into the unified continual stream format.

Raw layout (per TRACE README / external_baselines/trace_rcl):
  {raw_root}/{task_name}/train.json
  {raw_root}/{task_name}/eval.json
  {raw_root}/{task_name}/test.json

Each JSON file is a list of {"prompt": "...", "answer": "..."}.

Output layout matches CITB processed streams:
  {"benchmark": "TRACE", "version": "...", "stream": [{segment_id, segment_name, train, eval}, ...]}

Download (when raw data is missing):
  https://drive.google.com/file/d/1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV/view?usp=drive_link
  Unpack so that task folders (e.g. C-STANCE/) sit under data/raw/trace/.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Canonical continual-learning task order from external_baselines/trace_rcl/scripts/train_lora.sh
TRACE_TASK_ORDER: Tuple[str, ...] = (
    "C-STANCE",
    "FOMC",
    "MeetingBank",
    "Py150",
    "ScienceQA",
    "NumGLUE-cm",
    "NumGLUE-ds",
    "20Minuten",
)

# Task-level instruction prefixes (from external_baselines/trace_rcl/inference/ICL.py TASK_PROMT)
TRACE_TASK_INSTRUCTIONS: Dict[str, str] = {
    "FOMC": (
        "What is the monetary policy stance for the following text? "
        "A. dovish, B. hawkish, C. neutral. Choose one from A, B and C."
    ),
    "C-STANCE": "判断以下文本对指定对象的态度，选择一项：A.支持，B.反对，C.中立。输出A，B或者C。",
    "ScienceQA": "Choose an answer for the following question and give your reasons.",
    "NumGLUE-cm": "Solve the following math problem.",
    "NumGLUE-ds": "Solve the following math problem.",
    "MeetingBank": "Write a summary of the following meeting transcripts.",
    "Py150": "Continue writing the code.",
    "20Minuten": "Provide a simplified version of the following paragraph in German.",
}

DEFAULT_OUT = "data/processed/trace_cl_tasks_train50_eval10.json"
DEFAULT_RAW_ROOT = "data/raw/trace"
DEFAULT_VERSION = "trace_train50_eval10_v1"


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(obj).__name__}")
    return obj


def _trace_sample_to_example(sample: Dict[str, Any], task_name: str) -> Dict[str, str]:
    if "prompt" not in sample or "answer" not in sample:
        raise ValueError(f"TRACE sample must contain prompt/answer keys: {sample}")
    prompt = str(sample["prompt"])
    answer = str(sample["answer"])
    instruction = TRACE_TASK_INSTRUCTIONS.get(task_name, task_name)

    input_text = prompt
    prefix = TRACE_TASK_INSTRUCTIONS.get(task_name, "")
    if prefix:
        if prompt.startswith(prefix):
            input_text = prompt[len(prefix) :].lstrip("\n")
        elif prompt.startswith(prefix.rstrip("\n")):
            input_text = prompt[len(prefix.rstrip("\n")) :].lstrip("\n")

    return {
        "instruction": instruction,
        "input": input_text,
        "output": answer,
    }


def _select_split(
    samples: Sequence[Dict[str, Any]],
    *,
    max_count: int,
    rng: random.Random,
    shuffle: bool,
) -> List[Dict[str, Any]]:
    items = list(samples)
    if shuffle:
        rng.shuffle(items)
    if max_count > 0:
        items = items[: max_count]
    return items


def convert_task_segment(
    *,
    task_name: str,
    segment_id: int,
    raw_root: Path,
    max_train: int,
    max_eval: int,
    seed: int,
    eval_split: str,
) -> Dict[str, Any]:
    task_dir = raw_root / task_name
    train_path = task_dir / "train.json"
    eval_path = task_dir / eval_split

    if not train_path.exists():
        raise FileNotFoundError(f"Missing TRACE train split: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing TRACE eval split ({eval_split}): {eval_path}")

    rng = random.Random(int(seed) + segment_id)
    train_raw = _load_json_list(train_path)
    eval_raw = _load_json_list(eval_path)

    train_selected = _select_split(train_raw, max_count=max_train, rng=rng, shuffle=True)
    # TRACE inference uses test.json; keep eval order deterministic (first N).
    eval_selected = _select_split(eval_raw, max_count=max_eval, rng=rng, shuffle=False)

    train_examples = [_trace_sample_to_example(x, task_name) for x in train_selected]
    eval_examples = [_trace_sample_to_example(x, task_name) for x in eval_selected]

    if not train_examples or not eval_examples:
        raise ValueError(
            f"Task {task_name} produced empty train/eval after selection "
            f"(train={len(train_examples)}, eval={len(eval_examples)})"
        )

    print(
        f"[{segment_id:03d}] {task_name}: "
        f"train={len(train_examples)} eval={len(eval_examples)} "
        f"(raw train={len(train_raw)} raw {eval_split}={len(eval_raw)})"
    )

    return {
        "segment_id": segment_id,
        "segment_name": task_name,
        "train": train_examples,
        "eval": eval_examples,
    }


def build_toy_segments(*, max_train: int, max_eval: int) -> List[Dict[str, Any]]:
    """Minimal synthetic TRACE-like segments for pipeline smoke tests."""

    toy_specs = [
        (
            "C-STANCE",
            [
                (
                    "判断以下文本对指定对象的态度，选择一项：A.支持，B.反对，C.中立。输出A，B或者C。\n"
                    "文本：该政策有助于稳定就业市场。对象：央行降息。",
                    "A",
                ),
                (
                    "判断以下文本对指定对象的态度，选择一项：A.支持，B.反对，C.中立。输出A，B或者C。\n"
                    "文本：企业利润受到挤压。对象：加息。",
                    "B",
                ),
                (
                    "判断以下文本对指定对象的态度，选择一项：A.支持，B.反对，C.中立。输出A，B或者C。\n"
                    "文本：市场反应平淡。对象：维持利率不变。",
                    "C",
                ),
            ],
        ),
        (
            "FOMC",
            [
                (
                    "What is the monetary policy stance for the following text? "
                    "A. dovish, B. hawkish, C. neutral. Choose one from A, B and C.\n"
                    "The committee expects further gradual increases in the target range.",
                    "B",
                ),
                (
                    "What is the monetary policy stance for the following text? "
                    "A. dovish, B. hawkish, C. neutral. Choose one from A, B and C.\n"
                    "Inflation has eased but remains elevated.",
                    "C",
                ),
            ],
        ),
    ]

    segments: List[Dict[str, Any]] = []
    for seg_id, (task_name, pairs) in enumerate(toy_specs):
        examples = [_trace_sample_to_example({"prompt": p, "answer": a}, task_name) for p, a in pairs]
        train_n = min(max_train, len(examples))
        eval_n = min(max_eval, len(examples))
        segments.append(
            {
                "segment_id": seg_id,
                "segment_name": task_name,
                "train": examples[:train_n],
                "eval": examples[:eval_n],
            }
        )
        print(f"[{seg_id:03d}] {task_name} (toy): train={train_n} eval={eval_n}")
    return segments


def convert_trace_to_stream(
    *,
    raw_root: Path,
    out_path: Path,
    max_train: int,
    max_eval: int,
    seed: int,
    limit_tasks: int,
    task_order: Sequence[str],
    eval_split: str,
    mode: str,
    version: str,
) -> Dict[str, Any]:
    if mode == "toy":
        segments = build_toy_segments(max_train=max_train, max_eval=max_eval)
    else:
        if not raw_root.exists():
            raise FileNotFoundError(
                f"TRACE raw root not found: {raw_root}. "
                "Download from Google Drive (see scripts/convert_trace_to_stream.py header "
                "or data/raw/trace/README.md) and re-run with --mode full."
            )
        tasks = list(task_order)
        if limit_tasks > 0:
            tasks = tasks[: limit_tasks]

        segments = []
        for seg_id, task_name in enumerate(tasks):
            segments.append(
                convert_task_segment(
                    task_name=task_name,
                    segment_id=seg_id,
                    raw_root=raw_root,
                    max_train=max_train,
                    max_eval=max_eval,
                    seed=seed,
                    eval_split=eval_split,
                )
            )

    payload = {
        "benchmark": "TRACE",
        "version": version if mode == "full" else f"{version}_toy",
        "stream": segments,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path} ({len(segments)} segments)")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TRACE raw JSON to unified continual stream JSON.")
    parser.add_argument("--mode", choices=("full", "toy"), default="full", help="full=read raw TRACE; toy=smoke data")
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT, help="Root dir containing TRACE task folders")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output processed JSON path")
    parser.add_argument("--seed", type=int, default=123, help="Shuffle seed for train split subsampling")
    parser.add_argument("--max-train", type=int, default=50, help="Max train examples per segment")
    parser.add_argument("--max-eval", type=int, default=10, help="Max eval examples per segment")
    parser.add_argument("--limit-tasks", type=int, default=-1, help="Limit number of tasks (-1 = all 8)")
    parser.add_argument(
        "--eval-split",
        choices=("test.json", "eval.json"),
        default="test.json",
        help="Which split to use as eval (TRACE inference uses test.json)",
    )
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Version string stored in output JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    raw_root = Path(args.raw_root)
    if not raw_root.is_absolute():
        raw_root = repo_root / raw_root
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    convert_trace_to_stream(
        raw_root=raw_root,
        out_path=out_path,
        max_train=int(args.max_train),
        max_eval=int(args.max_eval),
        seed=int(args.seed),
        limit_tasks=int(args.limit_tasks),
        task_order=TRACE_TASK_ORDER,
        eval_split=str(args.eval_split),
        mode=str(args.mode),
        version=str(args.version),
    )


if __name__ == "__main__":
    main()
