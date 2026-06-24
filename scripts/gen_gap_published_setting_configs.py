#!/usr/bin/env python3
"""Generate published_setting configs for Seq-GLUE (8 methods) and LFPT5 (5 benchmarks)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs/paper/published_setting"
WANDB = "lora-published-setting-run_v2"

METHODS_CORE = [
    ("ours", {"mode": "ours", "method_variant": "ours", "category": "ours_single_seed", "family": "ours"}),
    ("o_lora", {"mode": "baseline", "baseline_name": "o_lora", "method_variant": "o_lora", "category": "advanced_baseline_single_seed", "family": "advanced_baseline"}),
    ("lb_cl", {"mode": "baseline", "baseline_name": "lb_cl", "method_variant": "lb_cl", "category": "advanced_baseline_single_seed", "family": "advanced_baseline"}),
    ("progressive_prompts", {"mode": "baseline", "baseline_name": "progressive_prompts", "method_variant": "progressive_prompts", "category": "advanced_baseline_single_seed", "family": "advanced_baseline"}),
    ("continual_t0", {"mode": "baseline", "baseline_name": "continual_t0", "method_variant": "continual_t0", "category": "advanced_baseline_single_seed", "family": "advanced_baseline"}),
    ("sequential_lora", {"mode": "baseline", "baseline_name": "sequential_lora", "method_variant": "sequential_lora", "category": "basic_baseline_single_seed", "family": "baseline"}),
    ("replay_lora", {"mode": "baseline", "baseline_name": "replay_lora", "method_variant": "replay_lora", "category": "basic_baseline_single_seed", "family": "baseline"}),
]

BENCHMARKS = {
    "instrdialog": {
        "stream_file": "citb_cl_dialogue_tasks_train50_eval10.json",
        "stream_name": "instrdialog",
        "run_prefix": "published_instrdialog",
        "max_branches": 15,
        "segments_note": "InstrDialog train50/eval10",
    },
    "instrdialogpp": {
        "stream_file": "citb_cl_38_random_tasks_train50_eval10.json",
        "stream_name": "instrdialogpp",
        "run_prefix": "published_instrdialogpp",
        "max_branches": 15,
        "segments_note": "InstrDialog++ train50/eval10",
    },
    "multiwoz": {
        "stream_file": "multiwoz_nlg_cl_domains_train50_eval10.json",
        "stream_name": "multiwoz_nlg",
        "run_prefix": "full_multiwoz",
        "max_branches": 8,
        "segments_note": "MultiWOZ NLG train50/eval10",
    },
    "trace": {
        "stream_file": "trace_cl_tasks_train50_eval10.json",
        "stream_name": "trace",
        "stream_format": "trace_processed",
        "run_prefix": "trace_full",
        "max_branches": 8,
        "segments_note": "TRACE full 8-segment train50/eval10",
    },
    "seqglue": {
        "stream_file": "seqglue_cl_tasks_train50_eval10.json",
        "stream_name": "seqglue",
        "run_prefix": "published_seqglue",
        "max_branches": 8,
        "segments_note": "Seq-GLUE 8-task train50/eval10",
    },
}


def _base_cfg(bench_key: str, method_key: str, meta: dict) -> dict:
    b = BENCHMARKS[bench_key]
    run_name = f"{b['run_prefix']}_{method_key}_s123"
    if method_key == "ours":
        run_name = run_name.replace("_ours_", "_ours_")  # published_seqglue_ours_s123
    cfg: dict = {
        "experiment_name": f"published_setting_{bench_key}_{method_key}",
        "mode": meta["mode"],
        "seed": 123,
        "paths": {
            "project_root": ".",
            "processed_stream_dir": "data/processed",
            "processed_stream_file": b["stream_file"],
            "assets_dir": "assets",
            "results_dir": "results",
        },
        "data": {
            "stream_format": b.get("stream_format", "citb_processed"),
            "stream_name": b["stream_name"],
            "auto_prepare_processed": False,
            "processed_stream_train_instances_per_task": 50,
            "processed_stream_eval_instances_per_task": 10,
            "processed_stream_limit_tasks": -1,
            "max_segments": -1,
            "max_train_examples_per_segment": -1,
            "max_eval_examples_per_segment": -1,
            "split": "default",
        },
        "model": {
            "backbone_name": "llama31-8b-instruct",
            "hf_model_name_or_path": "assets/pretrained/meta-llama/Llama-3.1-8B-Instruct",
            "torch_dtype": "bfloat16",
            "device": "auto",
            "max_seq_len": 512,
            "gen_max_new_tokens": 64,
            "gen_do_sample": False,
            "gen_num_beams": 1,
            "mask_eos_token_in_labels": True,
        },
        "lora": {
            "enabled": True,
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
        "train": {
            "epochs_per_segment": 1,
            "batch_size": 2,
            "lr": 0.0002,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "log_every": 25,
        },
        "output": {
            "run_name": run_name,
            "save_every_segment": True,
            "write_tables": True,
            "write_logs": True,
            "tracking": {
                "use_wandb": True,
                "wandb_project": WANDB,
                "wandb_group": f"v2_{b['stream_name']}",
                "wandb_mode": "online",
                "wandb_tags": ["published_setting", b["stream_name"], method_key, "seed_123", "train50_eval10", "gap_v2"],
                "wandb_notes": f"Published-setting gap run: {b['segments_note']}, method={method_key}, seed=123",
                "log_artifacts": True,
            },
        },
        "paper": {
            "track": "published_setting",
            "category": meta["category"],
            "benchmark_alias": b["stream_name"],
            "method_variant": meta["method_variant"],
            "memory_budget": 0,
            "family": meta["family"],
            "published_setting_note": b["segments_note"],
        },
    }
    if meta["mode"] == "baseline" and "baseline_name" in meta:
        cfg["baseline_name"] = meta["baseline_name"]
    if meta["mode"] == "ours":
        cfg["train"]["trajectory_early_stop"] = False
        cfg["train"]["hard_early_stop_enabled"] = False
        cfg["modules"] = {
            "use_drift_detector": True,
            "use_lora_bank": True,
            "use_router": True,
            "use_overlap_loss": True,
        }
        cfg["bank"] = {"max_branches": b["max_branches"], "spawn_on_drift": True, "freeze_old_branches": True}
    if method_key == "replay_lora":
        cfg["replay"] = {"enabled": False, "buffer_size": 10, "replay_ratio": 0.3, "strategy": "uniform"}
    if method_key == "continual_t0":
        cfg["replay"] = {"enabled": True, "buffer_size": 256, "replay_ratio": 0.01, "strategy": "uniform"}
        cfg["advanced_baseline"] = {"continual_t0": {"buffer_size": 256, "replay_ratio": 0.01, "strategy": "uniform"}}
    if method_key == "o_lora":
        cfg["advanced_baseline"] = {
            "o_lora": {
                "adapter_prefix": "olora_s",
                "freeze_previous_adapters": True,
                "orthogonal_penalty_weight": 0.1,
                "orthogonal_projection_strength": 1.0,
            }
        }
    return cfg


def _lfpt5_cfg(bench_key: str) -> dict:
    b = BENCHMARKS[bench_key]
    method_key = "lfpt5"
    run_name = f"{b['run_prefix']}_lfpt5_s123"
    return {
        "experiment_name": f"published_setting_{bench_key}_lfpt5",
        "runner": "lfpt5_external",
        "seed": 123,
        "lfpt5": {
            "repo_subdir": "external_baselines/lfpt5",
            "lm_adapted_torch_ckpt": "assets/pretrained/lfpt5/lm_adapted_t5_large_torch/pytorch_model.bin",
            "prompt_number": 300,
            "batch_size_per_gpu": 2,
            "max_length": 128,
        },
        "paths": {
            "project_root": ".",
            "processed_stream_dir": "data/processed",
            "processed_stream_file": b["stream_file"],
            "results_dir": "results",
        },
        "data": {
            "stream_name": b["stream_name"],
            "stream_format": b.get("stream_format", "citb_processed"),
            "max_segments": -1,
        },
        "output": {
            "run_name": run_name,
            "save_every_segment": True,
            "write_tables": True,
            "write_logs": True,
            "tracking": {
                "use_wandb": True,
                "wandb_project": WANDB,
                "wandb_group": f"v2_{b['stream_name']}",
                "wandb_mode": "online",
                "wandb_tags": ["published_setting", b["stream_name"], "lfpt5", "seed_123", "external_runner", "gap_v2"],
                "wandb_notes": f"LFPT5 external runner: {b['segments_note']}, seed=123",
                "log_artifacts": True,
            },
        },
        "paper": {
            "track": "published_setting",
            "category": "advanced_baseline_single_seed",
            "benchmark_alias": b["stream_name"],
            "method_variant": "lfpt5",
            "family": "advanced_baseline",
            "published_setting_note": f"LFPT5 T5 prompt tuning; {b['segments_note']}",
        },
    }


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for method_key, meta in METHODS_CORE:
        cfg = _base_cfg("seqglue", method_key, meta)
        path = CONFIG_DIR / f"seqglue__{method_key}__s123.yaml"
        path.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written += 1
    for bench_key in BENCHMARKS:
        cfg = _lfpt5_cfg(bench_key)
        path = CONFIG_DIR / f"{bench_key}__lfpt5__s123.yaml"
        path.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written += 1
    print(f"Wrote {written} gap configs under {CONFIG_DIR}")


if __name__ == "__main__":
    main()
