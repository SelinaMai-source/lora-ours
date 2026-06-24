#!/usr/bin/env python3
"""Export the local Seq-GLUE continual stream to O-LoRA UIE JSON layout.

This is a no-training bridge-prep step. It writes:
  - data_dir/<TaskType>/<dataset_name>/{train,dev,test}.json
  - data_dir/<TaskType>/<dataset_name>/labels.json
  - task_config_dir/{train,dev,test}_tasks.json
  - instruction_config_cl.json

The official O-LoRA runner can then be pointed at these paths via
`--data_dir`, `--task_config_dir`, and `--instruction_file`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_STREAM = "data/processed/seqglue_cl_tasks_train50_eval10.json"
DEFAULT_OUT = "data/olora/seqglue_s123"

TASK_TYPE_BY_SEGMENT = {
    "glue_sst2": "SC",
    "glue_mrpc": "QQP",
    "glue_rte": "NLI",
    "glue_cola": "SC",
    "super_glue_boolq": "BoolQA",
    "super_glue_wic": "WiC",
    "super_glue_cb": "NLI",
    "super_glue_copa": "COPA",
    "super_glue_multirc": "MultiRC",
}

INSTRUCTIONS = {
    "NLI": [{"instruction_type": "zero-shot", "instruction": 'What is the logical relationship between the "sentence 1" and the "sentence 2"? Choose one from the option.\n'}],
    "QQP": [{"instruction_type": "zero-shot", "instruction": 'Whether the "first sentence" and the "second sentence" have the same meaning? Choose one from the option.\n'}],
    "SC": [{"instruction_type": "zero-shot", "instruction": "What is the sentiment or acceptability of the following paragraph? Choose one from the option.\n"}],
    "BoolQA": [{"instruction_type": "zero-shot", "instruction": "According to the following passage, is the question true or false? Choose one from the option.\n"}],
    "MultiRC": [{"instruction_type": "zero-shot", "instruction": "According to the following passage and question, is the candidate answer true or false? Choose one from the option.\n"}],
    "WiC": [{"instruction_type": "zero-shot", "instruction": "Given a word and two sentences, whether the word is used with the same sense in both sentence? Choose one from the option.\n"}],
    "COPA": [{"instruction_type": "zero-shot", "instruction": ""}],
}


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _labels(examples: Iterable[Dict[str, Any]]) -> List[str]:
    labels: List[str] = []
    seen = set()
    for ex in examples:
        label = str(ex.get("output", "")).strip()
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def _format_sentence(ex: Dict[str, Any]) -> str:
    instruction = str(ex.get("instruction", "")).strip()
    input_text = str(ex.get("input", "")).strip()
    if instruction and input_text:
        return f"{instruction}\n{input_text}"
    return input_text or instruction


def _convert_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "sentence": _format_sentence(ex),
            "label": str(ex.get("output", "")).strip(),
        }
        for ex in examples
    ]


def export(stream_path: Path, out_root: Path) -> Dict[str, Any]:
    stream_data = json.loads(stream_path.read_text(encoding="utf-8"))
    data_dir = out_root / "data"
    task_config_dir = out_root / "task_configs"
    data_dir.mkdir(parents=True, exist_ok=True)
    task_config_dir.mkdir(parents=True, exist_ok=True)

    configs: Dict[str, Dict[str, List[Dict[str, str]]]] = {
        "train": {},
        "dev": {},
        "test": {},
    }
    manifest: List[Dict[str, Any]] = []

    for segment in stream_data.get("stream", []):
        seg_name = str(segment.get("segment_name", ""))
        task_type = TASK_TYPE_BY_SEGMENT.get(seg_name)
        if not task_type:
            raise ValueError(f"No O-LoRA task type mapping for segment {seg_name!r}")
        dataset_name = _safe_name(seg_name)
        ds_dir = data_dir / task_type / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        train_examples = list(segment.get("train", []))
        eval_examples = list(segment.get("eval", []))
        labels = _labels([*train_examples, *eval_examples])
        if not labels:
            raise ValueError(f"No labels for segment {seg_name}")

        (ds_dir / "train.json").write_text(json.dumps(_convert_examples(train_examples), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # O-LoRA expects dev/test config files; use the local eval split for both.
        converted_eval = _convert_examples(eval_examples)
        (ds_dir / "dev.json").write_text(json.dumps(converted_eval, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (ds_dir / "test.json").write_text(json.dumps(converted_eval, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (ds_dir / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        item = {"sampling strategy": "full", "dataset name": dataset_name}
        for split in configs:
            configs[split].setdefault(task_type, []).append(item)
        manifest.append(
            {
                "segment_id": int(segment.get("segment_id", len(manifest))),
                "segment_name": seg_name,
                "task_type": task_type,
                "dataset_name": dataset_name,
                "num_train": len(train_examples),
                "num_eval": len(eval_examples),
                "labels": labels,
            }
        )

    for split, cfg in configs.items():
        (task_config_dir / f"{split}_tasks.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    instruction_file = out_root / "instruction_config_cl.json"
    instruction_file.write_text(json.dumps(INSTRUCTIONS, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps({"stream": str(stream_path), "segments": manifest}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "out_root": str(out_root),
        "data_dir": str(data_dir),
        "task_config_dir": str(task_config_dir),
        "instruction_file": str(instruction_file),
        "manifest": str(manifest_path),
        "num_segments": len(manifest),
    }


def validate_export(out_root: Path) -> Dict[str, Any]:
    """Validate the local Seq-GLUE -> O-LoRA bridge without launching training."""
    manifest_path = out_root / "manifest.json"
    instruction_path = out_root / "instruction_config_cl.json"
    task_config_dir = out_root / "task_configs"
    data_dir = out_root / "data"
    errors: List[str] = []

    if not manifest_path.is_file():
        errors.append(f"missing manifest: {manifest_path}")
        manifest = {"segments": []}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not instruction_path.is_file():
        errors.append(f"missing instruction config: {instruction_path}")
        instructions: Dict[str, Any] = {}
    else:
        instructions = json.loads(instruction_path.read_text(encoding="utf-8"))

    task_configs: Dict[str, Dict[str, Any]] = {}
    for split in ["train", "dev", "test"]:
        path = task_config_dir / f"{split}_tasks.json"
        if not path.is_file():
            errors.append(f"missing {split} task config: {path}")
            task_configs[split] = {}
        else:
            task_configs[split] = json.loads(path.read_text(encoding="utf-8"))

    checked_segments = 0
    expected_segments = list(TASK_TYPE_BY_SEGMENT.keys())[:8]
    observed_segments = [str(segment.get("segment_name", "")) for segment in manifest.get("segments", [])]
    if observed_segments != expected_segments:
        errors.append(f"unexpected segment order/count: expected {expected_segments}, observed {observed_segments}")

    for segment in manifest.get("segments", []):
        checked_segments += 1
        task_type = str(segment.get("task_type", ""))
        dataset_name = str(segment.get("dataset_name", ""))
        labels = list(segment.get("labels", []))
        if task_type not in instructions:
            errors.append(f"{dataset_name}: missing instruction for task type {task_type}")
        if not labels:
            errors.append(f"{dataset_name}: empty manifest labels")
        if int(segment.get("num_train", -1)) != 50:
            errors.append(f"{dataset_name}: expected 50 train examples, got {segment.get('num_train')}")
        if int(segment.get("num_eval", -1)) != 10:
            errors.append(f"{dataset_name}: expected 10 eval examples, got {segment.get('num_eval')}")
        for split, cfg in task_configs.items():
            configured = cfg.get(task_type, []) if isinstance(cfg, dict) else []
            configured_names = {str(item.get("dataset name", "")) for item in configured if isinstance(item, dict)}
            if dataset_name not in configured_names:
                errors.append(f"{dataset_name}: missing from {split}_tasks.json under task type {task_type}")
        ds_dir = data_dir / task_type / dataset_name
        for split in ["train", "dev", "test"]:
            split_path = ds_dir / f"{split}.json"
            if not split_path.is_file():
                errors.append(f"{dataset_name}: missing {split_path}")
                continue
            examples = json.loads(split_path.read_text(encoding="utf-8"))
            if not examples:
                errors.append(f"{dataset_name}: empty {split} split")
            for idx, ex in enumerate(examples[:5]):
                if "sentence" not in ex or "label" not in ex:
                    errors.append(f"{dataset_name}: malformed {split}[{idx}]")
                elif str(ex.get("label", "")).strip() not in labels:
                    errors.append(f"{dataset_name}: label {ex.get('label')!r} not in labels.json/manifest")
        labels_path = ds_dir / "labels.json"
        if not labels_path.is_file():
            errors.append(f"{dataset_name}: missing labels.json")
        else:
            stored_labels = json.loads(labels_path.read_text(encoding="utf-8"))
            if stored_labels != labels:
                errors.append(f"{dataset_name}: labels.json differs from manifest labels")

    return {
        "out_root": str(out_root),
        "num_segments": checked_segments,
        "ready": not errors,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-json", type=Path, default=Path(DEFAULT_STREAM))
    parser.add_argument("--out-root", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--validate-only", action="store_true", help="Validate an existing O-LoRA export without rewriting it.")
    args = parser.parse_args()
    summary = validate_export(args.out_root) if args.validate_only else export(args.stream_json, args.out_root)
    print(json.dumps(summary, indent=2))
    if not summary.get("ready", True):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
