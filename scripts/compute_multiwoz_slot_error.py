#!/usr/bin/env python3
"""Compute MultiWOZ NLG slot error rate from semantic acts and generated text.

The metric follows the AdapterCL/ToDCL EER-style check used for dialogue NLG:
for values requested by semantic dialogue acts, count the value as erroneous
when it does not appear in the generated system response.

Expected prediction JSONL/JSON entries contain at least:
  - prediction/generated/genr/text/output: generated response
  - input/semantic_acts/dialog_act: semantic dialogue acts

If no input acts are present in the prediction file, pass --stream-json so the
script aligns predictions with the processed MultiWOZ eval examples by segment.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PRED_KEYS = ("prediction", "generated", "genr", "response", "text", "output")
ACT_KEYS = ("input", "semantic_acts", "dialog_act", "acts")
IGNORED_VALUES = {"", "true", "false", "yes", "no", "?", "none"}


def _load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("examples", "details", "predictions", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    raise ValueError(f"Unsupported prediction file structure: {path}")


def _load_stream_eval_inputs(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    inputs: List[str] = []
    for segment in data.get("stream", []):
        for example in segment.get("eval", []):
            inputs.append(str(example.get("input", "")))
    return inputs


def _first_text(record: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return str(value)
    return ""


def parse_act_values(text: str) -> List[str]:
    values: List[str] = []
    for match in re.finditer(r'([A-Za-z0-9_:-]+)\s*=\s*"([^"]*)"', text):
        _slot, value = match.groups()
        normalized = value.strip()
        if normalized.lower() not in IGNORED_VALUES:
            values.append(normalized)
    return values


def compute_slot_error(records: Sequence[Dict[str, Any]], stream_inputs: Sequence[str]) -> Dict[str, Any]:
    total = 0
    missing = 0
    details: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        prediction = _first_text(record, PRED_KEYS)
        acts = _first_text(record, ACT_KEYS)
        if not acts and idx < len(stream_inputs):
            acts = stream_inputs[idx]
        values = parse_act_values(acts)
        missing_values = [value for value in values if value.lower() not in prediction.lower()]
        total += len(values)
        missing += len(missing_values)
        details.append(
            {
                "index": idx,
                "num_slots": len(values),
                "num_missing": len(missing_values),
                "missing_values": missing_values,
            }
        )
    return {
        "slot_error_rate": float(missing / total) if total else 0.0,
        "num_missing": int(missing),
        "num_slots": int(total),
        "num_examples": len(records),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--stream-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = _load_records(args.predictions)
    stream_inputs = _load_stream_eval_inputs(args.stream_json) if args.stream_json else []
    metrics = compute_slot_error(records, stream_inputs)

    output = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
