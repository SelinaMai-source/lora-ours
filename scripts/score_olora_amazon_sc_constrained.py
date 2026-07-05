#!/usr/bin/env python3
"""Train-only constrained-label scoring diagnostic for amazon/SC.

This script scores only the label verbalizers listed in labels.json. It does not
read dev/test splits and does not use test confusion to tune rules.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


REPO = Path("/root/lora-ours")
DEFAULT_ADAPTER = (
    REPO
    / "results/runs/olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1"
    / "official_outputs/4-agnews/adapter"
)
DEFAULT_DATA_DIR = (
    REPO
    / "results/runs/olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1"
    / "official_runtime/CL_Benchmark/SC/amazon"
)
DEFAULT_INSTRUCTIONS = (
    REPO
    / "results/runs/olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1"
    / "official_runtime/configs/instruction_config.json"
)
DEFAULT_RUNTIME_SRC = (
    REPO
    / "results/runs/olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1"
    / "official_runtime/src"
)


LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]


def build_prompt(sentence: str, labels: list[str], instruction_file: Path) -> str:
    instructions = json.loads(instruction_file.read_text(encoding="utf-8"))
    instruction = instructions["SC"][0]["instruction"]
    instruction += "Option: " + ", ".join(labels) + " \n" + "{0}" + "\nAnswer:"
    instruction = "Task:SC\nDataset:amazon\n" + instruction
    return instruction.format(sentence)


def load_model(adapter: Path, runtime_src: Path, device: str):
    sys.path.insert(0, str(runtime_src))
    from peft import PeftConfig, PeftModel

    config = PeftConfig.from_pretrained(str(adapter))
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    base = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base, str(adapter))
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def score_batch(
    model,
    tokenizer,
    prompts: list[str],
    labels: list[str],
    device: str,
    max_source_length: int,
    max_target_length: int,
) -> tuple[list[str], list[dict[str, float]]]:
    encoded = tokenizer(
        prompts,
        max_length=max_source_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    batch_scores: list[dict[str, float]] = []
    for label in labels:
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                [label] * len(prompts),
                max_length=max_target_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
        target_mask = targets["attention_mask"].bool()
        target_ids = targets["input_ids"]
        model_labels = target_ids.masked_fill(~target_mask, -100)
        outputs = model(**encoded, labels=model_labels)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        token_nll = -log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_nll = token_nll.masked_fill(~target_mask, 0.0)
        lengths = target_mask.sum(dim=1).clamp(min=1)
        normalized_nll = (token_nll.sum(dim=1) / lengths).detach().cpu().tolist()
        for idx, score in enumerate(normalized_nll):
            if len(batch_scores) <= idx:
                batch_scores.append({})
            batch_scores[idx][label] = float(score)
    predictions = [min(scores.items(), key=lambda item: item[1])[0] for scores in batch_scores]
    return predictions, batch_scores


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(row["gold"] == row["prediction"] for row in rows)
    gold_counts = collections.Counter(row["gold"] for row in rows)
    pred_counts = collections.Counter(row["prediction"] for row in rows)
    confusions = collections.Counter(
        (row["gold"], row["prediction"]) for row in rows if row["gold"] != row["prediction"]
    )
    per_label = {}
    for label in LABELS:
        items = [row for row in rows if row["gold"] == label]
        right = sum(row["gold"] == row["prediction"] for row in items)
        per_label[label] = {
            "total": len(items),
            "correct": right,
            "accuracy": round(100 * right / len(items), 4) if items else None,
        }
    return {
        "total": total,
        "accuracy": round(100 * correct / total, 4) if total else None,
        "gold_counts": dict(gold_counts),
        "prediction_counts": dict(pred_counts),
        "per_label": per_label,
        "top_confusions": [
            {"gold": gold, "prediction": pred, "count": count}
            for (gold, pred), count in confusions.most_common(20)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--instruction-file", default=str(DEFAULT_INSTRUCTIONS))
    parser.add_argument("--runtime-src", default=str(DEFAULT_RUNTIME_SRC))
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=8)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    label_path = data_dir / "labels.json"
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    if labels != LABELS:
        raise ValueError(f"unexpected amazon labels: {labels}")
    rows = json.loads((data_dir / f"{args.split}.json").read_text(encoding="utf-8"))
    rows = rows[: args.max_samples] if args.max_samples >= 0 else rows

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(Path(args.adapter), Path(args.runtime_src), device)

    predictions: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [build_prompt(item["sentence"], labels, Path(args.instruction_file)) for item in batch]
        preds, scores = score_batch(
            model,
            tokenizer,
            prompts,
            labels,
            device,
            args.max_source_length,
            args.max_target_length,
        )
        for item, pred, score in zip(batch, preds, scores):
            predictions.append(
                {
                    "gold": item["label"],
                    "prediction": pred,
                    "scores": score,
                    "sentence": item["sentence"],
                }
            )

    summary = summarize(predictions)
    payload = {
        "policy": {
            "split": args.split,
            "uses_test_json": False,
            "uses_test_targets": False,
            "label_space_source": str(label_path),
            "adapter": args.adapter,
            "runtime_src": args.runtime_src,
            "method": "teacher_forced_normalized_nll_over_train_label_verbalizers",
        },
        "summary": summary,
        "predictions": predictions,
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# v74 Amazon/SC Constrained-Label Scoring Diagnostic",
        "",
        f"- Split: `{args.split}`",
        "- Leakage policy: train-only, no dev/test split, no test targets/confusion.",
        f"- Accuracy: `{summary['accuracy']}` over `{summary['total']}` examples.",
        f"- Prediction counts: `{summary['prediction_counts']}`",
        "",
        "## Per-Label Accuracy",
        "",
    ]
    for label in LABELS:
        item = summary["per_label"][label]
        lines.append(f"- `{label}`: `{item['accuracy']}` ({item['correct']}/{item['total']})")
    lines.extend(["", "## Top Confusions", ""])
    for item in summary["top_confusions"][:10]:
        lines.append(f"- `{item['gold']}` -> `{item['prediction']}`: `{item['count']}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "json": str(out_json), "md": str(out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
