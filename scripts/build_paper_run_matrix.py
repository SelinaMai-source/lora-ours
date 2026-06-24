from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


DEFAULT_BENCHMARKS = ["instrdialog", "instrdialog++"]
DEFAULT_SEEDS = [123]
DEFAULT_OURS_MAIN_VARIANT = "ours_full"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_seed_list(text: str) -> List[int]:
    seeds = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _slug(text: str) -> str:
    out = text.strip().lower()
    out = out.replace("++", "pp")
    out = out.replace("+", "p")
    out = out.replace("/", "_")
    out = out.replace(" ", "_")
    return out


def _ensure_mapping(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = parent.setdefault(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping at key '{key}'")
    return value


def _set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = cfg
    for part in parts[:-1]:
        cur = _ensure_mapping(cur, part)
    cur[parts[-1]] = value


def _set_common_stream(cfg: Dict[str, Any], benchmark: str) -> None:
    paths = _ensure_mapping(cfg, "paths")
    paths["processed_stream_dir"] = str(paths.get("processed_stream_dir", "data/processed"))
    paths["processed_stream_file"] = ""
    paths["raw_citb_root"] = str(paths.get("raw_citb_root", "data/raw/citb"))

    data_cfg = _ensure_mapping(cfg, "data")
    data_cfg["stream_name"] = benchmark
    data_cfg["auto_prepare_processed"] = True
    data_cfg["processed_stream_train_instances_per_task"] = int(
        data_cfg.get("processed_stream_train_instances_per_task", 50)
    )
    data_cfg["processed_stream_eval_instances_per_task"] = int(
        data_cfg.get("processed_stream_eval_instances_per_task", 10)
    )
    data_cfg["processed_stream_limit_tasks"] = int(data_cfg.get("processed_stream_limit_tasks", -1))


def _apply_common_metadata(
    cfg: Dict[str, Any],
    *,
    benchmark: str,
    category: str,
    family: str,
    method_variant: str,
    seed: int,
    run_name: str,
    memory_budget: int,
) -> None:
    cfg["seed"] = int(seed)
    output_cfg = _ensure_mapping(cfg, "output")
    output_cfg["run_name"] = run_name

    paper_cfg = _ensure_mapping(cfg, "paper")
    paper_cfg["track"] = "paper"
    paper_cfg["category"] = category
    paper_cfg["family"] = family
    paper_cfg["benchmark_alias"] = benchmark
    paper_cfg["method_variant"] = method_variant
    paper_cfg["memory_budget"] = int(memory_budget)


def _baseline_variants() -> List[Dict[str, Any]]:
    return [
        {
            "variant_id": "seq",
            "category": "main",
            "family": "baseline",
            "method_variant": "sequential_lora",
            "memory_budget": 0,
            "overrides": {
                "baseline_name": "sequential_lora",
                "replay.enabled": False,
                "periodic.enabled": False,
                "router.enabled": False,
            },
        },
        {
            "variant_id": "replay_b10",
            "category": "main",
            "family": "baseline",
            "method_variant": "replay_lora",
            "memory_budget": 10,
            "overrides": {
                "baseline_name": "replay_lora",
                "replay.enabled": True,
                "replay.buffer_size": 10,
                "periodic.enabled": False,
                "router.enabled": False,
            },
        },
        {
            "variant_id": "replay_b50",
            "category": "main",
            "family": "baseline",
            "method_variant": "replay_lora",
            "memory_budget": 50,
            "overrides": {
                "baseline_name": "replay_lora",
                "replay.enabled": True,
                "replay.buffer_size": 50,
                "periodic.enabled": False,
                "router.enabled": False,
            },
        },
        {
            "variant_id": "periodic_latest",
            "category": "main",
            "family": "baseline",
            "method_variant": "periodic_multilora",
            "memory_budget": 0,
            "overrides": {
                "baseline_name": "periodic_multilora",
                "periodic.enabled": True,
                "replay.enabled": False,
                "router.enabled": False,
            },
        },
        {
            "variant_id": "bank_no_router",
            "category": "main",
            "family": "baseline",
            "method_variant": "bank_no_router",
            "memory_budget": 0,
            "overrides": {
                "baseline_name": "bank_no_router",
                "replay.enabled": False,
                "periodic.enabled": False,
                "router.enabled": False,
            },
        },
        {
            "variant_id": "router_only",
            "category": "main",
            "family": "baseline",
            "method_variant": "router_only",
            "memory_budget": 0,
            "overrides": {
                "baseline_name": "router_only",
                "replay.enabled": False,
                "periodic.enabled": False,
                "router.enabled": True,
                "router.num_initial_branches": 8,
            },
        },
    ]


def _ours_base_variants() -> List[Dict[str, Any]]:
    return [
        {
            "variant_id": "ours_no_overlap",
            "category": "ablation",
            "family": "ours",
            "method_variant": "ours_no_overlap",
            "memory_budget": 0,
            "overrides": {"modules.use_overlap_loss": False},
        },
        {
            "variant_id": "ours_full",
            "category": "ablation",
            "family": "ours",
            "method_variant": "ours_full",
            "memory_budget": 0,
            "overrides": {},
        },
        {
            "variant_id": "ours_no_drift",
            "category": "ablation",
            "family": "ours",
            "method_variant": "ours_no_drift",
            "memory_budget": 0,
            "overrides": {"modules.use_drift_detector": False},
        },
        {
            "variant_id": "ours_no_router",
            "category": "ablation",
            "family": "ours",
            "method_variant": "ours_no_router",
            "memory_budget": 0,
            "overrides": {"modules.use_router": False},
        },
        {
            "variant_id": "ours_no_bank",
            "category": "ablation",
            "family": "ours",
            "method_variant": "ours_no_bank",
            "memory_budget": 0,
            "overrides": {
                "modules.use_drift_detector": False,
                "modules.use_lora_bank": False,
                "modules.use_router": False,
                "modules.use_overlap_loss": False,
            },
        },
    ]


def _winner_method_variant(variant_id: str) -> str:
    if variant_id.startswith("ours_"):
        return f"winner_{variant_id}"
    return f"winner_{_slug(variant_id)}"


def _ours_variants(include_sweeps: bool, *, ours_main_variant: str) -> List[Dict[str, Any]]:
    variants = []
    winner_found = False
    for variant in _ours_base_variants():
        row = copy.deepcopy(variant)
        if row["variant_id"] == ours_main_variant:
            row["category"] = "main"
            row["method_variant"] = _winner_method_variant(str(row["variant_id"]))
            winner_found = True
        variants.append(row)
    if not winner_found:
        raise ValueError(
            f"Unsupported ours main variant: {ours_main_variant}. "
            f"Choose one of {[x['variant_id'] for x in _ours_base_variants()]}"
        )
    if include_sweeps:
        variants.extend(
            [
                {
                    "variant_id": "ours_beta001",
                    "category": "sweep",
                    "family": "ours",
                    "method_variant": "ours_overlap_beta",
                    "memory_budget": 0,
                    "overrides": {"overlap.beta": 0.01},
                },
                {
                    "variant_id": "ours_beta003",
                    "category": "sweep",
                    "family": "ours",
                    "method_variant": "ours_overlap_beta",
                    "memory_budget": 0,
                    "overrides": {"overlap.beta": 0.03},
                },
                {
                    "variant_id": "ours_beta006",
                    "category": "sweep",
                    "family": "ours",
                    "method_variant": "ours_overlap_beta",
                    "memory_budget": 0,
                    "overrides": {"overlap.beta": 0.06},
                },
                {
                    "variant_id": "ours_no_meta_threshold",
                    "category": "sweep",
                    "family": "ours",
                    "method_variant": "ours_change_point_meta",
                    "memory_budget": 0,
                    "overrides": {"drift.meta_threshold_enabled": False},
                },
                {
                    "variant_id": "ours_reverse_curriculum",
                    "category": "sweep",
                    "family": "ours",
                    "method_variant": "ours_anchor_curriculum",
                    "memory_budget": 0,
                    "overrides": {"drift.curriculum_strategy": "hard_core_easy_probe"},
                },
                {
                    "variant_id": "ours_M64",
                    "category": "sweep",
                    "family": "ours",
                    "method_variant": "ours_anchor_size",
                    "memory_budget": 0,
                    "overrides": {"drift.anchor_size": 64},
                },
                {
                    "variant_id": "ours_M128",
                    "category": "sweep",
                    "family": "ours",
                    "method_variant": "ours_anchor_size",
                    "memory_budget": 0,
                    "overrides": {"drift.anchor_size": 128},
                },
                {
                    "variant_id": "ours_M256",
                    "category": "sweep",
                    "family": "ours",
                    "method_variant": "ours_anchor_size",
                    "memory_budget": 0,
                    "overrides": {"drift.anchor_size": 256},
                },
                {
                    "variant_id": "ours_K50",
                    "category": "sweep",
                    "family": "ours",
                    "method_variant": "ours_monitor_interval",
                    "memory_budget": 0,
                    "overrides": {"drift.monitor_interval": 50},
                },
                {
                    "variant_id": "ours_K100",
                    "category": "sweep",
                    "family": "ours",
                    "method_variant": "ours_monitor_interval",
                    "memory_budget": 0,
                    "overrides": {"drift.monitor_interval": 100},
                },
                {
                    "variant_id": "ours_r8",
                    "category": "sweep",
                    "family": "ours",
                    "method_variant": "ours_lora_rank",
                    "memory_budget": 0,
                    "overrides": {"lora.r": 8, "lora.alpha": 16},
                },
                {
                    "variant_id": "ours_r16",
                    "category": "sweep",
                    "family": "ours",
                    "method_variant": "ours_lora_rank",
                    "memory_budget": 0,
                    "overrides": {"lora.r": 16, "lora.alpha": 32},
                },
                {
                    "variant_id": "ours_r32",
                    "category": "sweep",
                    "family": "ours",
                    "method_variant": "ours_lora_rank",
                    "memory_budget": 0,
                    "overrides": {"lora.r": 32, "lora.alpha": 64},
                },
                {
                    "variant_id": "ours_br4",
                    "category": "sweep",
                    "family": "ours",
                    "method_variant": "ours_max_branches",
                    "memory_budget": 0,
                    "overrides": {"bank.max_branches": 4},
                },
                {
                    "variant_id": "ours_br8",
                    "category": "sweep",
                    "family": "ours",
                    "method_variant": "ours_max_branches",
                    "memory_budget": 0,
                    "overrides": {"bank.max_branches": 8},
                },
            ]
        )
    return variants


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_configs(
    *,
    template: Dict[str, Any],
    variants: Iterable[Dict[str, Any]],
    mode: str,
    benchmarks: List[str],
    seeds: List[int],
    output_dir: Path,
    repo_root: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for benchmark in benchmarks:
        benchmark_slug = _slug(benchmark)
        for seed in seeds:
            for variant in variants:
                cfg = copy.deepcopy(template)
                _set_common_stream(cfg, benchmark)
                variant_id = str(variant["variant_id"])
                run_name = f"paper_{benchmark_slug}_{variant_id}_s{seed}"

                cfg["experiment_name"] = f"paper_{mode}_{benchmark_slug}"
                _apply_common_metadata(
                    cfg,
                    benchmark=benchmark,
                    category=str(variant["category"]),
                    family=str(variant["family"]),
                    method_variant=str(variant["method_variant"]),
                    seed=seed,
                    run_name=run_name,
                    memory_budget=int(variant["memory_budget"]),
                )

                for dotted_key, value in dict(variant.get("overrides", {})).items():
                    _set_nested(cfg, dotted_key, value)

                out_path = output_dir / f"{benchmark_slug}__{variant_id}__s{seed}.yaml"
                _write_yaml(out_path, cfg)

                rows.append(
                    {
                        "mode": mode,
                        "benchmark": benchmark,
                        "benchmark_slug": benchmark_slug,
                        "category": str(variant["category"]),
                        "family": str(variant["family"]),
                        "method_variant": str(variant["method_variant"]),
                        "variant_id": variant_id,
                        "seed": int(seed),
                        "run_name": run_name,
                        "memory_budget": int(variant["memory_budget"]),
                        "config_path": str(out_path.relative_to(repo_root)),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-grade benchmark/method config matrix.")
    parser.add_argument(
        "--baseline-template",
        type=Path,
        default=Path("configs/paper_baseline.yaml"),
        help="Baseline template YAML.",
    )
    parser.add_argument(
        "--ours-template",
        type=Path,
        default=Path("configs/paper_ours.yaml"),
        help="Ours template YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/paper"),
        help="Directory where generated YAML configs will be written.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/tables/paper_run_matrix.csv"),
        help="CSV manifest for generated configs.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/tables/paper_run_matrix.json"),
        help="JSON manifest for generated configs.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=",".join(DEFAULT_BENCHMARKS),
        help="Comma-separated benchmark aliases, e.g. instrdialog,instrdialog++",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SEEDS),
        help="Comma-separated seeds.",
    )
    parser.add_argument(
        "--include-sweeps",
        action="store_true",
        help="Also emit anchor/K/r/max-branches sweep configs for ours.",
    )
    parser.add_argument(
        "--ours-main-variant",
        type=str,
        default=DEFAULT_OURS_MAIN_VARIANT,
        help="Which ours variant should be promoted into category=main.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    baseline_template = _load_yaml((repo_root / args.baseline_template).resolve())
    ours_template = _load_yaml((repo_root / args.ours_template).resolve())
    output_dir = (repo_root / args.output_dir).resolve()
    benchmarks = _parse_csv_list(args.benchmarks)
    seeds = _parse_seed_list(args.seeds)

    rows: List[Dict[str, Any]] = []
    rows.extend(
        _build_configs(
            template=baseline_template,
            variants=_baseline_variants(),
            mode="baseline",
            benchmarks=benchmarks,
            seeds=seeds,
            output_dir=output_dir,
            repo_root=repo_root,
        )
    )
    rows.extend(
        _build_configs(
            template=ours_template,
            variants=_ours_variants(include_sweeps=args.include_sweeps, ours_main_variant=str(args.ours_main_variant)),
            mode="ours",
            benchmarks=benchmarks,
            seeds=seeds,
            output_dir=output_dir,
            repo_root=repo_root,
        )
    )

    rows.sort(key=lambda row: (row["benchmark_slug"], row["mode"], row["category"], row["variant_id"], row["seed"]))
    _write_csv((repo_root / args.summary_csv).resolve(), rows)

    summary_json_path = (repo_root / args.summary_json).resolve()
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
