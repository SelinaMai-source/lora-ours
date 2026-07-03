#!/usr/bin/env python3
"""CPU-only preflight for published-method base alignment.

This script checks local official sources, processed streams, model assets, and
strict-comparability blockers for the v52 "published method first" route. It
does not import torch, touch CUDA, or start any training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover - keeps the preflight useful in minimal envs.
    yaml = None  # type: ignore[assignment]


REPO = Path(__file__).resolve().parents[1]
AUTODL = Path("/root/autodl-tmp")
BASE_RUN = AUTODL / "lora-baselines-run_v1"


def _exists(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file() or yaml is None:
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _stream_summary(path: Path) -> Dict[str, Any]:
    data = _load_json(path)
    if not data:
        return {"path": str(path), "exists": path.is_file(), "error": "missing_or_unreadable"}
    stream = data.get("stream") or data.get("segments") or []
    split_counts = (data.get("metadata") or {}).get("split_counts") or []
    names: List[str] = []
    counts: List[Dict[str, Any]] = []
    for idx, segment in enumerate(stream):
        name = str(segment.get("segment_name") or segment.get("task") or segment.get("name") or idx)
        train = segment.get("train") or []
        dev = segment.get("dev") or []
        test = segment.get("test") or segment.get("eval") or []
        names.append(name)
        counts.append({"name": name, "train": len(train), "dev": len(dev), "test": len(test)})
    if not counts and split_counts:
        for row in split_counts:
            counts.append(
                {
                    "name": row.get("task"),
                    "train": row.get("train"),
                    "dev": row.get("dev"),
                    "test": row.get("test"),
                }
            )
    zero_or_short = [
        row
        for row in counts
        if int(row.get("train") or 0) == 0 or int(row.get("dev") or 0) == 0 or int(row.get("test") or 0) == 0
    ]
    return {
        "path": str(path),
        "exists": path.is_file(),
        "benchmark": data.get("benchmark"),
        "version": data.get("version"),
        "num_segments": len(counts),
        "first_segments": counts[:8],
        "zero_or_empty_splits": zero_or_short,
        "metadata": data.get("metadata", {}),
    }


def _config_summary(path: Path) -> Dict[str, Any]:
    cfg = _load_yaml(path)
    if not cfg:
        return {"path": str(path), "exists": path.is_file(), "error": "missing_or_unreadable_yaml"}
    return {
        "path": str(path),
        "exists": path.is_file(),
        "experiment_name": cfg.get("experiment_name"),
        "strict_alignment": cfg.get("strict_alignment", {}),
        "data": cfg.get("data", {}),
        "model": cfg.get("model", {}),
        "train": cfg.get("train", {}),
        "paper": cfg.get("paper", {}),
        "official_reference": cfg.get("official_reference", {}),
    }


def _status(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    text = path.read_text(encoding="utf-8")
    fields: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            fields[key.strip()] = value.strip().strip("`")
    return {"path": str(path), "exists": True, "fields": fields}


def _suite_citb() -> Dict[str, Any]:
    citb_root = BASE_RUN / "external_sources/citb"
    stream = BASE_RUN / "data/ccfa_three_suite/citb/citb_instrdialogpp_order1_train100_dev50_test100.json"
    stage1 = AUTODL / "model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
    cfg = REPO / "configs/ccfa_three_suite/citb_instrdialogpp_order1_seed1_ours_v51_decodecal_skipempty6_smoke_strict.yaml"
    red_flags = [
        "InstrDialog++ 100/50/100 is a CCF-A aligned variant; official long-stream script evidence records eval count 25.",
        "skip_empty_segments changes the effective long-stream task set and must be reported explicitly.",
        "v36b/v50/v51 are diagnostic smokes, not published-base reproduction results.",
    ]
    return {
        "suite": "CITB",
        "selected_published_base": "CITB official repo / Tk-Instruct protocol",
        "readiness": "base-assets-present; strict result still blocked by official reproduction and split policy audit",
        "official_sources": {
            "repo": _exists(citb_root),
            "readme": _exists(citb_root / "README.md"),
            "short_order1": _exists(citb_root / "data/CIT_data/task_orders/stream=cl_dialogue_tasks/order1.txt"),
            "long_order1": _exists(citb_root / "data/CIT_data/task_orders/stream=cl_dialogue_long_tasks/order1.txt"),
            "collect_results": _exists(citb_root / "collect_results.py"),
        },
        "assets": {
            "t5_small_lm_adapt": _exists(AUTODL / "model_cache/hf_snapshots/google__t5-small-lm-adapt"),
            "superni_stage1_checkpoint": _exists(stage1),
        },
        "config": _config_summary(cfg),
        "stream": _stream_summary(stream),
        "recent_status": {
            "v49": _status(REPO / "results/logs/ours_v49_instrdialogpp_prefix3_strict_status.md"),
            "v50": _status(REPO / "results/logs/ours_v50_instrdialogpp_skipempty6_strict_status.md"),
            "v51": _status(REPO / "results/logs/ours_v51_instrdialogpp_decodecal_skipempty6_strict_status.md"),
        },
        "minimal_ours_overlay": [
            "start from official Stage-1 checkpoint and official task order",
            "verify official-equivalent score matrix before adding Ours",
            "enable Ours only through drift/bank/router/overlap config switches",
        ],
        "red_flags": red_flags,
    }


def _suite_standard() -> Dict[str, Any]:
    olora = BASE_RUN / "external_sources/o_lora"
    stream = BASE_RUN / "data/ccfa_three_suite/standard_peft/olora_standard_order1_t5large.json"
    cfg = REPO / "configs/ccfa_three_suite/standard_peft_cl_o_lora_standard_order1_seed1_ours_strict.yaml"
    red_flags = [
        "Ours runner on converted O-LoRA streams is not itself an official O-LoRA reproduction.",
        "LFPT5 and Progressive Prompts are backup/evidence bases here; they are not the selected standard-suite base.",
        "memory/runtime changes must be labeled as engineering changes, not official hyperparameters.",
    ]
    return {
        "suite": "Standard T5-large PEFT CL",
        "selected_published_base": "O-LoRA official standard T5-large setup",
        "readiness": "ready for no-GPU official/source preflight; GPU run should wait",
        "official_sources": {
            "repo": _exists(olora),
            "readme": _exists(olora / "README.md"),
            "order_1_script": _exists(olora / "scripts/order_1.sh"),
            "order1_configs": _exists(olora / "configs/order1_configs"),
            "cl_benchmark": _exists(olora / "CL_Benchmark"),
        },
        "assets": {"t5_large": _exists(AUTODL / "model_cache/hf_snapshots/t5-large")},
        "config": _config_summary(cfg),
        "stream": _stream_summary(stream),
        "minimal_ours_overlay": [
            "preserve O-LoRA order/checkpoint/metric surface",
            "first parse or reproduce official O-LoRA outputs",
            "add Ours modules in seq2seq PEFT runner with unchanged task stream",
        ],
        "red_flags": red_flags,
    }


def _suite_dialogue() -> Dict[str, Any]:
    arper = BASE_RUN / "external_sources/arper"
    todcl = BASE_RUN / "external_sources/todcl"
    stream = BASE_RUN / "data/ccfa_three_suite/arper/arper_woz3_unique_dialogue_act.json"
    cfg = REPO / "configs/ccfa_three_suite/dialogue_nlg_arper_woz3_dialogue_act_seed1_ours_strict.yaml"
    red_flags = [
        "ARPER Path A Ours T5 is not the official SCLSTM architecture.",
        "ToDCL is not runnable as strict base until official data download/preprocess/export is complete.",
        "ARPER WOZ3 and ToDCL TOD37/MultiWOZ NLG must not be treated as interchangeable baselines.",
    ]
    return {
        "suite": "Dialogue NLG / MultiWOZ",
        "selected_published_base": "ARPER official WOZ3 SCLSTM first; ToDCL AdapterCL NLG backup after data export",
        "readiness": "ARPER source/scorer/stream ready; ToDCL data preprocessing blocked",
        "official_sources": {
            "arper_repo": _exists(arper),
            "arper_config": _exists(arper / "config/config.cfg"),
            "arper_run": _exists(arper / "run.sh"),
            "arper_bleu": _exists(arper / "bleu.py"),
            "arper_run_woz3": _exists(arper / "run_woz3.py"),
            "todcl_repo": _exists(todcl),
            "todcl_readme": _exists(todcl / "README.md"),
            "todcl_scorer": _exists(todcl / "scorer.py"),
            "todcl_data_download": _exists(todcl / "data/download.sh"),
        },
        "config": _config_summary(cfg),
        "stream": _stream_summary(stream),
        "minimal_ours_overlay": [
            "preflight/run ARPER official SCLSTM Path B first",
            "validate BLEU/SER scorer on official-style .res outputs",
            "then run Ours T5 on the same stream as an adapted variant",
        ],
        "red_flags": red_flags,
    }


def build_report() -> Dict[str, Any]:
    suites = [_suite_citb(), _suite_standard(), _suite_dialogue()]
    all_red_flags = [flag for suite in suites for flag in suite["red_flags"]]
    return {
        "generated_by": "scripts/preflight_published_method_bases.py",
        "gpu_safe": True,
        "starts_training": False,
        "repo": str(REPO),
        "recommendation": {
            "first_experiment_to_run": "CITB InstrDialog order1 official-base reproduction",
            "reason": "It is the main lora-ours target and the Stage-1/strict runner assets are already closest to ready.",
            "do_not_run": "Do not repeat v51 or launch any GPU job while another worker owns the single GPU.",
        },
        "suites": suites,
        "red_flags": all_red_flags,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight published-method bases without training or GPU use.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    report = build_report()
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        out = args.out if args.out.is_absolute() else REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
