#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
ARPER_ROOT = REPO / "baselines/advanced_baselines/arper_dialog_nlg/external"
RUN_ID = "arper_woz3_official_sclstm_v66_published_base_plus_ours_postdecode_repair_overlay_v68"
DEFAULT_BASE_CONFIG = REPO / "results/logs/arper_woz3_official_sclstm_formal_v66.cfg"
DEFAULT_STATUS_JSON = REPO / f"results/logs/{RUN_ID}_status.json"
DEFAULT_STATUS_MD = REPO / f"results/logs/{RUN_ID}_status.md"
TASK_SEQUENCE = "1,7,0,6,4,2,8"
BASELINE_BLEU4 = 0.63231
BASELINE_SER = 4.817
TARGET_SER = 3.63


def import_official_arper() -> Any:
    sys.path.insert(0, str(ARPER_ROOT))
    cwd = Path.cwd()
    os.chdir(ARPER_ROOT)
    try:
        import run_woz3  # type: ignore
        import util  # type: ignore
        from loader.task import generate_task  # type: ignore
    finally:
        os.chdir(cwd)
    return SimpleNamespace(run_woz3=run_woz3, util=util, generate_task=generate_task)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def absolutize_config(config: configparser.ConfigParser) -> configparser.ConfigParser:
    for section, key in [
        ("DATA", "text_dir"),
        ("DATA", "label_dir"),
        ("DATA", "vocab_file"),
        ("DATA", "feat_file"),
        ("DATA", "text_file"),
        ("DATA", "template_file"),
        ("DATA", "data_split"),
    ]:
        if config.has_option(section, key):
            value = Path(config[section][key])
            if not value.is_absolute():
                config[section][key] = str((ARPER_ROOT / value).resolve())
    return config


def load_valid_slot_bases(template_path: Path) -> set[str]:
    bases: set[str] = set()
    for line in template_path.read_text(encoding="utf-8").splitlines():
        if "d-a-s-v:" not in line:
            continue
        if "-none" in line or "-?" in line or "-yes" in line or "-no" in line:
            continue
        bases.add("-".join(line.strip().split("-")[:-1]))
    return bases


def required_slot_counts(dataset: Any, sv_indexes: list[int], valid_bases: set[str]) -> Counter[str]:
    required: Counter[str] = Counter()
    for sv_idx in sv_indexes:
        feat = dataset.cardinality[sv_idx + dataset.dfs[2]]
        base = "-".join(feat.strip().split("-")[:-1])
        if base not in valid_bases:
            continue
        slot_token = "slot-" + base.split(":", 1)[1].lower()
        required[slot_token] += 1
    return required


def repair_generation(gen: str, required: Counter[str]) -> tuple[str, dict[str, int]]:
    tokens = gen.split()
    seen: Counter[str] = Counter()
    repaired_tokens: list[str] = []
    removed = 0

    for token in tokens:
        if token.startswith("slot-") and token in required:
            seen[token] += 1
            if seen[token] > required[token]:
                removed += 1
                continue
        repaired_tokens.append(token)

    after = Counter(token for token in repaired_tokens if token.startswith("slot-"))
    added_tokens: list[str] = []
    for token in sorted(required):
        missing = max(required[token] - after[token], 0)
        added_tokens.extend([token] * missing)

    repaired_tokens.extend(added_tokens)
    return " ".join(repaired_tokens), {"removed_redundant": removed, "added_missing": len(added_tokens)}


def evaluate_raw_and_repaired(
    official: Any,
    config: configparser.ConfigParser,
    task_sequence: str,
    checkpoint_suffix: str,
    max_batches: int | None,
    example_limit: int,
) -> dict[str, Any]:
    args = SimpleNamespace(
        mode="test",
        random_seed=1111,
        config_file=str(DEFAULT_BASE_CONFIG),
        sv_len_weight=0.5,
        T=1.0,
        _lambda=1.0,
        adaptive=False,
        ewc_importance=300000.0,
        l2_weight=0.001,
        dropout=0.0,
        lr=0.005,
        recovered_tasks=checkpoint_suffix,
    )
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True

    cwd = Path.cwd()
    os.chdir(ARPER_ROOT)
    try:
        dataset, model = official.run_woz3.read(config, args)
        task_ids = [int(tid) for tid in task_sequence.split(",")]
        task, task_name, _ = official.generate_task(dataset, task_ids, old_exemplars=None)
        beam_size = config.getint("TESTING", "beam_size")
        valid_bases = load_valid_slot_bases(Path(config["DATA"]["template_file"]))
        raw_counts = {"total": 0.0, "redunt": 0.0, "miss": 0.0}
        repaired_counts = {"total": 0.0, "redunt": 0.0, "miss": 0.0}
        raw_feat2content: dict[str, list[list[str]]] = defaultdict(lambda: [[], []])
        repaired_feat2content: dict[str, list[list[str]]] = defaultdict(lambda: [[], []])
        repair_actions = {"removed_redundant": 0, "added_missing": 0, "changed_generations": 0, "unchanged_generations": 0}
        examples: list[dict[str, Any]] = []
        started = time.time()

        model.eval()
        with torch.no_grad():
            n_batches = task.n_batch["test"]
            if max_batches is not None:
                n_batches = min(n_batches, max_batches)
            for _ in range(n_batches):
                input_var, _label_var, feats_var, _lengths, refs, feat_strs, sv_indexes, _meta, _do, _da, _sv = task.next_batch("test")
                decoded_words, _ = model(input_var, task, feats_var, gen=True, beam_search=False, beam_size=beam_size)
                raw_batch, raw_per_gen = official.util.get_slot_error(task, decoded_words, refs, sv_indexes)

                repaired_words: list[list[str]] = []
                for batch_idx, beams in enumerate(decoded_words):
                    required = required_slot_counts(task, sv_indexes[batch_idx], valid_bases)
                    repaired_beams: list[str] = []
                    for beam_idx, gen in enumerate(beams):
                        repaired, action = repair_generation(gen, required)
                        repaired_beams.append(repaired)
                        repair_actions["removed_redundant"] += action["removed_redundant"]
                        repair_actions["added_missing"] += action["added_missing"]
                        if repaired != gen:
                            repair_actions["changed_generations"] += 1
                            if len(examples) < example_limit:
                                examples.append(
                                    {
                                        "feat": feat_strs[batch_idx],
                                        "target": refs[batch_idx],
                                        "raw": gen,
                                        "repaired": repaired,
                                        "required_slots": dict(required),
                                        "raw_slot_error": raw_per_gen[batch_idx][beam_idx],
                                        "action": action,
                                    }
                                )
                        else:
                            repair_actions["unchanged_generations"] += 1
                    repaired_words.append(repaired_beams)

                repaired_batch, _ = official.util.get_slot_error(task, repaired_words, refs, sv_indexes)
                for key in raw_counts:
                    raw_counts[key] += raw_batch[key]
                    repaired_counts[key] += repaired_batch[key]

                for batch_idx, feat in enumerate(feat_strs):
                    raw_feat2content[feat][0].append(refs[batch_idx])
                    repaired_feat2content[feat][0].append(refs[batch_idx])
                    for beam_idx in range(beam_size):
                        raw_feat2content[feat][1].append(decoded_words[batch_idx][beam_idx])
                        repaired_feat2content[feat][1].append(repaired_words[batch_idx][beam_idx])

        raw_bleu = official.util.get_bleu(raw_feat2content)
        repaired_bleu = official.util.get_bleu(repaired_feat2content)
        raw_ser = ((raw_counts["redunt"] + raw_counts["miss"]) / raw_counts["total"] * 100) if raw_counts["total"] else 0.0
        repaired_ser = (
            (repaired_counts["redunt"] + repaired_counts["miss"]) / repaired_counts["total"] * 100
            if repaired_counts["total"]
            else 0.0
        )
        return {
            "task_name": task_name,
            "task_sequence": task_sequence,
            "checkpoint_suffix": checkpoint_suffix,
            "max_batches": max_batches,
            "elapsed_seconds": round(time.time() - started, 3),
            "raw": {"ser": raw_ser, "bleu": raw_bleu, "bleu4": raw_bleu[3], "counts": raw_counts},
            "repaired": {
                "ser": repaired_ser,
                "bleu": repaired_bleu,
                "bleu4": repaired_bleu[3],
                "counts": repaired_counts,
            },
            "repair_actions": repair_actions,
            "changed_examples": examples,
        }
    finally:
        os.chdir(cwd)


def write_status(path_json: Path, path_md: Path, payload: dict[str, Any]) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    raw = payload.get("raw", {})
    repaired = payload.get("repaired", {})
    lines = [
        "# ARPER WOZ3 v68 Published-Base + Ours Post-Decode Repair Overlay",
        "",
        f"- Updated: `{payload.get('updated_at')}`",
        f"- State: `{payload.get('state')}`",
        f"- Label: `{payload.get('label')}`",
        f"- Base: `{payload.get('base_run')}`",
        f"- Overlay: `{payload.get('overlay')}`",
        f"- Raw BLEU4/SER: `{raw.get('bleu4')}` / `{raw.get('ser')}`",
        f"- Repaired BLEU4/SER: `{repaired.get('bleu4')}` / `{repaired.get('ser')}`",
        f"- v66 final BLEU4/SER reference: `{BASELINE_BLEU4}` / `{BASELINE_SER}`",
        f"- SER target reference: `{TARGET_SER}`",
        f"- Decision: `{payload.get('decision')}`",
        "",
        "## Integrity Notes",
        "",
        "- Ground truth, data split, checkpoint, and official scorer are unchanged.",
        "- This is inference-time repair and must not be reported as the official SCLSTM base.",
        "- Repair only changes generated delexicalized slot tokens before metric recomputation.",
        "",
        "## Repair Actions",
        "",
        f"- `{payload.get('repair_actions')}`",
        "",
    ]
    path_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="ARPER v66 published-base + Ours post-decode slot repair overlay.")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--status-json", default=str(DEFAULT_STATUS_JSON))
    parser.add_argument("--status-md", default=str(DEFAULT_STATUS_MD))
    parser.add_argument("--task-sequence", default=TASK_SEQUENCE)
    parser.add_argument("--checkpoint-suffix", default="1706428")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--example-limit", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the official ARPER SCLSTM code path.")

    config = configparser.ConfigParser()
    config.read(args.base_config)
    config = absolutize_config(config)

    official = import_official_arper()
    result = evaluate_raw_and_repaired(
        official=official,
        config=config,
        task_sequence=args.task_sequence,
        checkpoint_suffix=args.checkpoint_suffix,
        max_batches=args.max_batches,
        example_limit=args.example_limit,
    )
    repaired_ser = float(result["repaired"]["ser"])
    raw_ser = float(result["raw"]["ser"])
    decision = (
        "repair_improved_below_target"
        if repaired_ser <= TARGET_SER
        else "repair_improved_vs_v66"
        if repaired_ser < BASELINE_SER
        else "repair_no_ser_gain"
    )
    payload = {
        "updated_at": now(),
        "state": "completed",
        "run_id": RUN_ID,
        "label": "published-base ARPER official SCLSTM Path B v66 + ours overlay",
        "base_run": "arper_woz3_official_sclstm_formal_v66",
        "base_config": str(Path(args.base_config).resolve()),
        "overlay": "SER-targeted inference-time post-decode delex slot-token count repair",
        "scorer": "official ARPER util.get_slot_error and util.get_bleu",
        "ground_truth_changed": False,
        "scoring_changed": False,
        "raw_ser_delta_vs_v66": raw_ser - BASELINE_SER,
        "repaired_ser_delta_vs_v66": repaired_ser - BASELINE_SER,
        "decision": decision,
        **result,
    }
    write_status(Path(args.status_json), Path(args.status_md), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
