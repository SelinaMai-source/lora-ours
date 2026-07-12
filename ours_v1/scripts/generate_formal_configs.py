#!/usr/bin/env python3
"""Generate ours-v1 formal YAML configs from smoke templates."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKTREE = str(REPO)


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {path}")


def fix_paths(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    paths = cfg.setdefault("paths", {})
    paths["project_root"] = WORKTREE
    paths["results_dir"] = f"{WORKTREE}/results"
    return cfg


def citb_instrdialog_formal() -> None:
    smoke = load(REPO / "configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v1_20260708_smoke_strict.yaml")
    formal = fix_paths(smoke)
    formal["experiment_name"] = "citb_instrdialog_order1_seed1_ours_v1_20260708_formal_strict"
    formal.setdefault("data", {}).pop("max_segments", None)
    formal["strict_alignment"]["status"] = "formal_replay50_overlay_v1"
    formal["output"]["run_name"] = formal["experiment_name"]
    formal["output"]["tracking"]["wandb_group"] = "ccfa_citb_ours_v1_20260708_formal"
    formal["output"]["tracking"]["wandb_tags"] = [
        "ccfa_v2", "citb", "strict-formal", "ours_v1_20260708",
        "replay50-ratio-align", "official_script_500_50_50",
    ]
    formal["paper"]["status"] = "formal_pending"
    formal["paper"]["strict_ready"] = True
    dump(formal, REPO / "configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v1_20260708_formal_strict.yaml")


def citb_instrdialogpp_v1() -> None:
    base = load(REPO / "configs/ccfa_three_suite/citb_instrdialogpp_order1_seed1_ours_v50_skipempty6_smoke_strict.yaml")
    for kind, max_seg in [("smoke", 6), ("formal", None)]:
        cfg = fix_paths(base)
        suffix = f"{kind}_strict"
        name = f"citb_instrdialogpp_order1_seed1_ours_v1_20260708_{suffix}"
        cfg["experiment_name"] = name
        cfg["strict_alignment"] = {
            "status": f"instrdialogpp_v1_{kind}_skip_empty_ssrg",
            "blocker": "none" if kind == "smoke" else "none_if_preflight_passes",
            "official_evidence": "Public long stream; skip_empty_segments; SSRG replay_ratio 0.5 aligned to Replay50 base",
            "split_disclosure": "Not paper-exact 100/50/100; short tasks below target counts",
        }
        cfg.setdefault("data", {})["skip_empty_segments"] = True
        cfg["data"]["split_policy"] = "public_actual_100_50_100_with_shortfall"
        if max_seg is not None:
            cfg["data"]["max_segments"] = max_seg
        else:
            cfg["data"].pop("max_segments", None)
        sr = cfg.setdefault("spectral_replay", {})
        sr["replay_ratio"] = 0.5
        sr["min_replay_per_segment"] = 64
        cfg["output"]["run_name"] = name
        cfg["output"]["tracking"]["wandb_project"] = "lora-ours-v1"
        cfg["output"]["tracking"]["wandb_group"] = f"ccfa_citb_instrdialogpp_ours_v1_20260708_{kind}"
        cfg["output"]["tracking"]["wandb_tags"] = [
            "ccfa_v2", "citb", "instrdialogpp", f"strict-{kind}", "ours_v1_20260708", "skip-empty",
        ]
        cfg["paper"]["method_variant"] = "ours_v1_replay50_ratio_ssrg_skip_empty"
        cfg["paper"]["status"] = f"{kind}_pending"
        dump(cfg, REPO / f"configs/ccfa_three_suite/{name}.yaml")


def fix_smoke_paths() -> None:
    for name in [
        "citb_instrdialog_order1_seed1_ours_v1_20260708_smoke_strict.yaml",
    ]:
        path = REPO / "configs/ccfa_three_suite" / name
        if path.is_file():
            cfg = fix_paths(load(path))
            dump(cfg, path)


def main() -> int:
    fix_smoke_paths()
    citb_instrdialog_formal()
    citb_instrdialogpp_v1()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
