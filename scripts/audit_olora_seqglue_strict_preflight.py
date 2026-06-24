#!/usr/bin/env python3
"""CPU-only strict preflight for O-LoRA / Seq-GLUE.

This validates local bridge/runner readiness and writes an explicit audit record.
It never launches the official O-LoRA trainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO = Path(__file__).resolve().parents[1]

EXPECTED_SEGMENTS = [
    "glue_sst2",
    "glue_mrpc",
    "glue_rte",
    "glue_cola",
    "super_glue_boolq",
    "super_glue_wic",
    "super_glue_cb",
    "super_glue_copa",
]

# These are the canonical text labels expected by the local Seq-GLUE stream and
# the official Progressive Prompts T5 dataset mapping. O-LoRA accepts string
# labels via labels.json; this check catches accidental numeric/ID labels.
EXPECTED_LABELS = {
    "glue_sst2": {"negative", "positive"},
    "glue_mrpc": {"not_equivalent", "equivalent"},
    "glue_rte": {"entailment", "not_entailment"},
    "glue_cola": {"not_acceptable", "acceptable"},
    "super_glue_boolq": {"false", "true"},
    "super_glue_wic": {"false", "true"},
    "super_glue_cb": {"entailment", "contradiction", "neutral"},
    "super_glue_copa": {"choice1", "choice2"},
}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_from_cfg(cfg: Dict[str, Any], key: str) -> Path:
    return REPO / cfg["data"][key]


def audit(config_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = cfg["data"]
    method = cfg["method"]
    training = cfg["training"]
    output = cfg["output"]

    source_dir = REPO / method["source_dir"]
    entry = source_dir / "src/run_uie_lora.py"
    env_prefix = Path(method["env_prefix"])
    olora_root = REPO / data["olora_root"]
    bridge_manifest = olora_root / "manifest.json"
    runner_manifest = REPO / "results/runs" / output["run_name"] / "run_manifest.json"

    errors: List[str] = []
    warnings: List[str] = []
    blockers: List[str] = []

    required_paths = {
        "official_source": source_dir,
        "official_entry": entry,
        "conda_env": env_prefix,
        "stream_json": REPO / data["stream_json"],
        "olora_manifest": bridge_manifest,
        "runner_manifest": runner_manifest,
    }
    for label, path in required_paths.items():
        if not path.exists():
            errors.append(f"{label} missing: {path}")

    segment_audit: List[Dict[str, Any]] = []
    if bridge_manifest.is_file():
        manifest = _read_json(bridge_manifest)
        observed = [str(seg.get("segment_name", "")) for seg in manifest.get("segments", [])]
        if observed != EXPECTED_SEGMENTS:
            errors.append(f"segment order mismatch: expected {EXPECTED_SEGMENTS}, observed {observed}")
        for seg in manifest.get("segments", []):
            name = str(seg.get("segment_name", ""))
            labels = {str(x).strip() for x in seg.get("labels", [])}
            expected = EXPECTED_LABELS.get(name, set())
            label_ok = bool(expected) and labels == expected
            if not label_ok:
                errors.append(f"{name}: label set mismatch expected={sorted(expected)} observed={sorted(labels)}")
            if int(seg.get("num_train", -1)) != 50 or int(seg.get("num_eval", -1)) != 10:
                warnings.append(f"{name}: local split is not a published full-data split, got train={seg.get('num_train')} eval={seg.get('num_eval')}")
            segment_audit.append(
                {
                    "segment_name": name,
                    "task_type": seg.get("task_type"),
                    "dataset_name": seg.get("dataset_name"),
                    "labels": sorted(labels),
                    "expected_labels": sorted(expected),
                    "label_mapping_ok": label_ok,
                    "num_train": seg.get("num_train"),
                    "num_eval": seg.get("num_eval"),
                }
            )

    runner_status = "missing"
    command_ready = False
    if runner_manifest.is_file():
        runner = _read_json(runner_manifest)
        runner_status = str(runner.get("status", "missing"))
        command_ready = isinstance(runner.get("command"), list) and str(entry) in " ".join(runner.get("command", []))
        if runner_status not in {"ready", "running", "completed"}:
            errors.append(f"runner manifest status is {runner_status!r}, expected ready/running/completed")
        if not command_ready:
            errors.append("runner manifest does not contain official O-LoRA command")

    hparam_audit = {
        "model_name_or_path": cfg["model"]["model_name_or_path"],
        "lora_dim": training["lora_dim"],
        "learning_rate": training["learning_rate"],
        "num_train_epochs": training["num_train_epochs"],
        "lamda_1": training["lamda_1"],
        "lamda_2": training["lamda_2"],
        "per_device_train_batch_size": training["per_device_train_batch_size"],
        "per_device_eval_batch_size": training["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "status": "config-recorded-needs-paper-citation",
    }

    if runner_status != "completed":
        blockers.append("no_completed_official_full_run")
    blockers.extend(
        [
            "completed local run is diagnostic because published O-LoRA target is T5-large standard/long CL or later Seq-GLUE-7 rather than this local T5-small 8-segment bridge",
            "paper hparams/backbone/LoRA target modules still need citation beyond README/config evidence",
            "local Seq-GLUE train50/eval10 bridge is validated but not proven equivalent to published full-data protocol",
        ]
    )

    preflight_passed = not errors
    status = "strict-preflight-ready-needs-full-run-and-paper-audit"
    if runner_status == "completed" and preflight_passed:
        status = "completed-diagnostic-paper-benchmark-mismatch"
    elif not preflight_passed:
        status = "blocked-preflight-failed"
    return {
        "cell": "O-LoRA / Seq-GLUE",
        "status": status,
        "strict_allowed": False,
        "preflight_passed": preflight_passed,
        "ready_for_safe_full_run_when_gpu_free": preflight_passed and runner_status != "completed",
        "config": _rel(config_path),
        "paths": {k: _rel(v) for k, v in required_paths.items()},
        "runner_manifest_status": runner_status,
        "official_command_ready": command_ready,
        "hparam_audit": hparam_audit,
        "segment_label_audit": segment_audit,
        "warnings": warnings,
        "errors": errors,
        "strict_blockers": blockers,
        "safe_full_run_command": (
            "cd /root/autodl-tmp/Lora-code && "
            "python scripts/run_olora_seqglue_official.py "
            "--config configs/paper/lora_run_v10/seqglue__o_lora_official__s123.yaml"
        ),
        "safe_preflight_commands": [
            "python scripts/export_seqglue_to_olora.py --out-root data/olora/seqglue_s123 --validate-only",
            "python scripts/run_olora_seqglue_official.py --dry-run",
            "python scripts/audit_olora_seqglue_strict_preflight.py",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit O-LoRA Seq-GLUE strict preflight without training.")
    parser.add_argument("--config", type=Path, default=Path("configs/paper/lora_run_v10/seqglue__o_lora_official__s123.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/tables/olora_seqglue_strict_preflight_audit.json"))
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else REPO / args.config
    out_path = args.out if args.out.is_absolute() else REPO / args.out
    result = audit(config_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["preflight_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
