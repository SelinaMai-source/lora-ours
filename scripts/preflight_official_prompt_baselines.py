#!/usr/bin/env python3
"""CPU-only official-source preflight for Progressive Prompts and Continual-T0.

The script builds command manifests and blocker lists from the official source
trees. It does not import legacy dependencies or start training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO = Path(__file__).resolve().parents[1]

PP_TASK_MAP = {
    "glue_sst2": "sst2",
    "glue_mrpc": "mrpc",
    "glue_rte": "rte",
    "glue_cola": "cola",
    "super_glue_boolq": "boolq",
    "super_glue_wic": "wic",
    "super_glue_cb": "cb",
    "super_glue_copa": "copa",
}

PP_PAPER_LONG_ORDERS = {
    "order_8": ["mnli", "cb", "wic", "copa", "qqp", "boolq", "rte", "imdb", "yelp", "amazon", "sst2", "dbpedia", "ag", "multirc", "yahoo"],
    "order_9": ["multirc", "boolq", "wic", "mnli", "cb", "copa", "qqp", "rte", "imdb", "sst2", "dbpedia", "ag", "yelp", "amazon", "yahoo"],
    "order_10": ["yelp", "amazon", "mnli", "cb", "copa", "qqp", "rte", "imdb", "sst2", "dbpedia", "ag", "yahoo", "multirc", "boolq", "wic"],
}

PP_PAPER_HPARAMS = {
    "model_name_or_path": "t5-large",
    "prefix_len": 10,
    "learning_rate": 0.3,
    "num_epochs": 10,
    "select_k_per_class": 1000,
    "early_stopping": True,
    "freeze_weights": True,
}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _load_stream(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _progressive_prompts(cfg: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    source = REPO / "external_baselines/progressive_prompts"
    entry = source / "T5_codebase/train_t5_cl.py"
    env_yaml = source / "environment.yaml"
    stream_path = REPO / cfg["data"]["processed_stream_file"]
    errors: List[str] = []
    blockers: List[str] = []
    for label, path in {"source": source, "entry": entry, "environment": env_yaml, "stream": stream_path}.items():
        if not path.exists():
            errors.append(f"{label} missing: {path}")

    task_list: List[str] = []
    unmapped: List[str] = []
    if stream_path.is_file():
        for segment in _load_stream(stream_path).get("stream", []):
            name = str(segment.get("segment_name", ""))
            mapped = PP_TASK_MAP.get(name)
            if mapped:
                task_list.append(mapped)
            else:
                unmapped.append(name)
        if unmapped:
            blockers.append("unmapped Seq-GLUE tasks for official PP T5_codebase: " + ", ".join(unmapped))

    local_hparams = {
        "model_name_or_path": cfg["model"]["model_name_or_path"],
        "prefix_len": cfg["training"]["prefix_len"],
        "learning_rate": cfg["training"]["learning_rate"],
        "num_epochs": cfg["training"]["num_epochs"],
        "select_k_per_class": cfg["training"]["select_k_per_class"],
        "early_stopping": bool(cfg["training"].get("early_stopping", True)),
        "freeze_weights": True,
    }
    hparam_mismatches = {
        key: {"expected": expected, "observed": local_hparams.get(key)}
        for key, expected in PP_PAPER_HPARAMS.items()
        if local_hparams.get(key) != expected
    }
    paper_order_matches = {
        order_name: task_list == order
        for order_name, order in PP_PAPER_LONG_ORDERS.items()
    }

    run_name = cfg["output"]["run_name"]
    save_dir = REPO / "results/runs" / run_name / "progressive_prompts_output"
    command = [
        "conda",
        "run",
        "-p",
        str(Path(cfg["method"].get("env_prefix", "/root/autodl-tmp/conda_envs/lora_v10_progressive_prompts"))),
        "python",
        str(entry),
        "--task_list",
        *task_list,
        "--select_k_per_class",
        str(cfg["training"]["select_k_per_class"]),
        "--lr",
        str(cfg["training"]["learning_rate"]),
        "--num_epochs",
        str(cfg["training"]["num_epochs"]),
        "--freeze_weights",
        "1",
        "--prefix_len",
        str(cfg["training"]["prefix_len"]),
        "--model_name",
        cfg["model"]["model_name_or_path"],
        "--early_stopping",
        str(int(bool(cfg["training"].get("early_stopping", True)))),
        "--save_name",
        run_name,
        "--save_dir",
        str(save_dir),
    ]
    blockers.extend(
        [
            "official PP runner uses HuggingFace task loaders, not the local train50/eval10 exported stream",
            "local 8-task Seq-GLUE bridge does not match the published 15-task long-sequence orders",
        ]
    )
    if hparam_mismatches:
        blockers.append("local config hparams differ from the published T5 example: " + json.dumps(hparam_mismatches, sort_keys=True))
    return {
        "cell": "Progressive Prompts / Seq-GLUE",
        "status": "strict-runner-skeleton-needs-paper-audit-and-metric-parser",
        "strict_allowed": False,
        "preflight_passed": not errors,
        "config": _rel(config_path),
        "official_entry": _rel(entry),
        "task_list": task_list,
        "command": command,
        "results_parser": {
            "implemented": True,
            "command_template": "python scripts/parse_progressive_prompts_results.py --results <save_dir>/<save_name>/results_dict.npy",
            "expected_artifact": str(save_dir / run_name / "results_dict.npy"),
        },
        "paper_audit": {
            "source": "Progressive Prompts, ICLR 2023; README and paper Appendix A.2/A.3",
            "published_t5_example_hparams": PP_PAPER_HPARAMS,
            "local_hparams": local_hparams,
            "hparam_mismatches": hparam_mismatches,
            "published_long_orders": PP_PAPER_LONG_ORDERS,
            "local_task_list": task_list,
            "paper_order_matches": paper_order_matches,
            "strict_protocol_match": not hparam_mismatches and any(paper_order_matches.values()),
        },
        "errors": errors,
        "blockers": blockers,
    }


def _continual_t0(cfg: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    source = REPO / "external_baselines/continual_t0"
    readme = source / "README.md"
    setup_py = source / "setup.py"
    requirements = source / "requirements.txt"
    checkpoint_dir = REPO / "assets/pretrained/continual_t0/CT0-11B"
    checkpoint_bin = checkpoint_dir / "pytorch_model.bin"
    drive_attempt = REPO / "assets/download_attempts/continual_t0/gdown_attempt.json"
    drive_probe = REPO / "assets/download_attempts/continual_t0/http_probe.tsv"
    errors: List[str] = []
    blockers: List[str] = []
    for label, path in {"source": source, "README": readme, "setup.py": setup_py, "requirements": requirements}.items():
        if not path.exists():
            errors.append(f"{label} missing: {path}")
    blockers.extend(
        [
            "official README points to Colab/Google Drive material; local cleaned training script is not present",
            "CT0 full checkpoint pytorch_model.bin is not present locally",
            "v10 benchmark templates and evaluation parser are not mapped",
            "no safe full-run command can be produced until the official Drive/Colab assets are materialized",
        ]
    )
    if checkpoint_bin.is_file():
        blockers = [item for item in blockers if "checkpoint" not in item]
    if drive_attempt.is_file():
        blockers.append("Google Drive folder was probed with gdown; first protected file is not downloadable by non-owner/editor permissions")
    return {
        "cell": f"Continual-T0 / {cfg['data']['benchmark']}",
        "status": "strict-runner-skeleton-needs-official-assets",
        "strict_allowed": False,
        "preflight_passed": not errors,
        "config": _rel(config_path),
        "official_source": _rel(source),
        "asset_audit": {
            "hf_checkpoint_repo": cfg.get("model", {}).get("required_checkpoint", "ThomasNLG/CT0-11B"),
            "local_checkpoint_dir": _rel(checkpoint_dir),
            "tokenizer_config_ready": (checkpoint_dir / "tokenizer_config.json").is_file(),
            "model_config_ready": (checkpoint_dir / "config.json").is_file(),
            "full_pytorch_model_ready": checkpoint_bin.is_file(),
            "expected_pytorch_model_bytes": 44540671113,
            "drive_probe": _rel(drive_probe) if drive_probe.is_file() else None,
            "gdown_attempt": _rel(drive_attempt) if drive_attempt.is_file() else None,
        },
        "safe_smoke_commands": [
            "ACTION=check bash scripts/prepare_official_baseline_envs.sh continual_t0",
            "python -m py_compile external_baselines/continual_t0/setup.py",
        ],
        "errors": errors,
        "blockers": blockers,
    }


def preflight(config_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    method = str((cfg.get("method") or {}).get("name", ""))
    if method == "progressive_prompts":
        return _progressive_prompts(cfg, config_path)
    if method == "continual_t0":
        return _continual_t0(cfg, config_path)
    raise ValueError(f"Unsupported method in {config_path}: {method}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight official prompt baselines without training.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else REPO / args.config
    result = preflight(config_path)
    if args.out is not None:
        out_path = args.out if args.out.is_absolute() else REPO / args.out
    else:
        safe_name = config_path.stem.replace("__", "_")
        out_path = REPO / "results/tables" / f"{safe_name}_official_preflight.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["preflight_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
