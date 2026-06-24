#!/usr/bin/env python3
"""
Convert Seq-GLUE raw JSON (HF parquet exports) into unified continual stream format.

Raw layout (per task under data/raw/seqglue/):
  glue_sst2/train.json, validation.json, test.json
  super_glue_copa/train.json, ...

Output matches CITB processed streams:
  {"benchmark": "Seq-GLUE", "version": "...", "stream": [{segment_id, segment_name, train, eval}, ...]}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 8-task Seq-GLUE stream (GLUE + SuperGLUE subset; Progressive Prompts / Long-CL style)
SEQGLUE_TASK_ORDER: Tuple[Tuple[str, str], ...] = (
    ("glue", "sst2"),
    ("glue", "mrpc"),
    ("glue", "rte"),
    ("glue", "cola"),
    ("super_glue", "boolq"),
    ("super_glue", "wic"),
    ("super_glue", "cb"),
    ("super_glue", "copa"),
)

TASK_INSTRUCTIONS: Dict[str, str] = {
    "sst2": "Classify the sentiment of the sentence. Answer with positive or negative.",
    "mrpc": "Are the two sentences equivalent? Answer with equivalent or not_equivalent.",
    "rte": "Does the first sentence entail the second? Answer with entailment or not_entailment.",
    "cola": "Is the sentence grammatically acceptable? Answer with acceptable or not_acceptable.",
    "boolq": "Answer the question based on the passage with true or false.",
    "wic": "Does the word have the same meaning in both sentences? Answer with true or false.",
    "cb": "Classify the relationship between premise and hypothesis: entailment, contradiction, or neutral.",
    "copa": "Choose the more plausible cause or effect. Answer with choice1 or choice2.",
}

LABEL_TEXT: Dict[str, Dict[int, str]] = {
    "sst2": {0: "negative", 1: "positive"},
    "mrpc": {0: "not_equivalent", 1: "equivalent"},
    "rte": {0: "not_entailment", 1: "entailment"},
    "cola": {0: "not_acceptable", 1: "acceptable"},
    "boolq": {0: "false", 1: "true"},
    "wic": {0: "false", 1: "true"},
    "cb": {0: "entailment", 1: "contradiction", 2: "neutral"},
    "copa": {0: "choice1", 1: "choice2"},
}

DEFAULT_OUT = "data/processed/seqglue_cl_tasks_train50_eval10.json"
DEFAULT_RAW_ROOT = "data/raw/seqglue"
DEFAULT_VERSION = "seqglue_train50_eval10_v1"


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON list in {path}, got {type(obj).__name__}")
    return obj


def _label_to_text(task: str, label: Any) -> str:
    if isinstance(label, str):
        return label
    mapping = LABEL_TEXT.get(task, {})
    return mapping.get(int(label), str(label))


def _sample_to_example(task: str, row: Dict[str, Any]) -> Dict[str, str]:
    instruction = TASK_INSTRUCTIONS[task]
    if task == "sst2":
        input_text = str(row.get("sentence") or row.get("text", ""))
        label = row.get("label", row.get("label_text"))
    elif task in {"mrpc", "rte"}:
        input_text = f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}"
        label = row["label"]
    elif task == "cola":
        input_text = str(row["sentence"])
        label = row["label"]
    elif task == "boolq":
        input_text = f"Passage: {row['passage']}\nQuestion: {row['question']}"
        label = row["label"]
    elif task == "wic":
        input_text = (
            f"Word: {row['word']}\n"
            f"Sentence 1: {row['sentence1']}\n"
            f"Sentence 2: {row['sentence2']}"
        )
        label = row["label"]
    elif task == "cb":
        input_text = f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}"
        label = row["label"]
    elif task == "copa":
        q = row.get("question", "cause")
        input_text = (
            f"Premise: {row['premise']}\n"
            f"Choice 1: {row['choice1']}\n"
            f"Choice 2: {row['choice2']}\n"
            f"Question type: {q}"
        )
        label = row["label"]
    else:
        raise ValueError(f"Unsupported task: {task}")

    if label is None or (isinstance(label, int) and label < 0):
        raise ValueError(f"Missing label in {task} sample: {row}")
    output = _label_to_text(task, label)
    return {"instruction": instruction, "input": input_text, "output": output}


def _select_split(
    samples: Sequence[Dict[str, Any]],
    *,
    max_count: int,
    rng: random.Random,
    shuffle: bool,
) -> List[Dict[str, str]]:
    items = list(samples)
    if shuffle:
        rng.shuffle(items)
    if max_count > 0:
        items = items[:max_count]
    return items  # type: ignore[return-value]


def _task_raw_dir(raw_root: Path, bench: str, task: str) -> Path:
    return raw_root / f"{bench}_{task}"


def _collect_train_eval(
    raw_root: Path,
    bench: str,
    task: str,
    *,
    train_n: int,
    eval_n: int,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    tdir = _task_raw_dir(raw_root, bench, task)
    if not tdir.is_dir():
        raise FileNotFoundError(f"Missing raw task dir: {tdir}")

    train_path = tdir / "train.json"
    val_path = tdir / "validation.json"
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing {train_path}")
    if not val_path.is_file():
        val_path = tdir / "dev.json"
    if not val_path.is_file():
        raise FileNotFoundError(f"Missing validation split for {task}")

    rng = random.Random(seed + hash(task) % 10000)
    train_raw = _load_json_list(train_path)
    val_raw = _load_json_list(val_path)

    train_rows = [_sample_to_example(task, r) for r in train_raw]
    val_rows = [_sample_to_example(task, r) for r in val_raw]

    train_sel = _select_split(train_rows, max_count=train_n, rng=rng, shuffle=True)
    eval_sel = _select_split(val_rows, max_count=eval_n, rng=rng, shuffle=True)
    return train_sel, eval_sel


def build_stream(
    raw_root: Path,
    *,
    train_per_task: int = 50,
    eval_per_task: int = 10,
    seed: int = 123,
    version: str = DEFAULT_VERSION,
) -> Dict[str, Any]:
    stream: List[Dict[str, Any]] = []
    for seg_id, (bench, task) in enumerate(SEQGLUE_TASK_ORDER):
        train, eval_ = _collect_train_eval(
            raw_root, bench, task, train_n=train_per_task, eval_n=eval_per_task, seed=seed
        )
        stream.append(
            {
                "segment_id": seg_id,
                "segment_name": f"{bench}_{task}",
                "train": train,
                "eval": eval_,
            }
        )
    return {"benchmark": "Seq-GLUE", "version": version, "stream": stream}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Seq-GLUE raw JSON to processed CL stream.")
    parser.add_argument("--raw-root", type=Path, default=Path(DEFAULT_RAW_ROOT))
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--train-per-task", type=int, default=50)
    parser.add_argument("--eval-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--version", type=str, default=DEFAULT_VERSION)
    args = parser.parse_args()

    payload = build_stream(
        args.raw_root,
        train_per_task=args.train_per_task,
        eval_per_task=args.eval_per_task,
        seed=args.seed,
        version=args.version,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    n_seg = len(payload["stream"])
    print(f"Wrote {args.out} ({n_seg} segments, train{args.train_per_task}/eval{args.eval_per_task})")


if __name__ == "__main__":
    main()
