# Ours v1

This directory is the clean GitHub entry point for the `ours-v1` branch. It
groups the v1 contribution code, suite-specific overlays, launchers, configs,
and status notes without changing the official experiment settings.

**完整目录树与每个文件夹用途**：见 [`STRUCTURE.md`](STRUCTURE.md)。

## What To Read First

- `core/`: contribution map for the shared Ours mechanisms.
- `suites/`: suite-by-suite overlays for CITB, Standard, ARPER, and ToDCL.
- `scripts/`: v1 queue and launcher entry points.
- `configs/`: index of the v1 configs that remain at their runnable legacy
  locations.
- `docs/`: branch compatibility, result status, and first-principles notes.

## Core Contribution Points

| Contribution | Primary file |
| --- | --- |
| SSRG spectral sparse replay | `core/methods/ours_spectral_replay.py` |
| Assess-then-Update gate | `core/methods/assess_update.py` |
| Router / branch selection | `core/methods/router.py` |
| Drift detection | `core/methods/drift_detector.py` |
| Anti-overlap regularization | `core/methods/overlap_loss.py` |
| Standard O-LoRA SSRG proxy | `ours_v1/suites/standard/olora_overlay_ssrg.py` |
| Standard assess-retention gate | `ours_v1/suites/standard/olora_overlay_assess_retention_gate.py` |
| ARPER SSRG exemplar overlay | `ours_v1/suites/arper/arper_ssrg_exemplar_selection.py` |
| ToDCL assess orthogonal overlay | `ours_v1/suites/todcl/todcl_assess_orthogonal_overlay.py` |

## Suite Summary

| Suite | Published base | v1 overlay | Launcher |
| --- | --- | --- | --- |
| CITB | Replay(50), InstrDialog official setting | SSRG-aligned replay budget, `replay_ratio=0.5` | `ours_v1/scripts/launchers/run_citb_v1.sh` |
| Standard | O-LoRA official-equivalent anchor | Class-coverage SSRG + assess-retention gate | `ours_v1/scripts/launchers/run_standard_v1.sh` |
| ARPER | ARPER SCLSTM Path B | SSRG exemplar selection | `ours_v1/scripts/launchers/run_arper_v1.sh` |
| ToDCL | ToDCL ADAPTER NLG anchor | Assess-then-Update orthogonal penalty | `ours_v1/scripts/launchers/run_todcl_v1.sh` |

LB-CL is kept as a paper-only reference for Standard and is not used as the
operational v1 code base.

## Run Entry Points

```bash
bash ours_v1/scripts/queue/run_v1_iterate.sh
bash ours_v1/scripts/launchers/run_citb_v1.sh
bash ours_v1/scripts/launchers/run_standard_v1.sh
bash ours_v1/scripts/launchers/run_arper_v1.sh
bash ours_v1/scripts/launchers/run_todcl_v1.sh
```

Legacy paths under `scripts/` are kept as compatibility wrappers and delegate to
the new paths above. Logs, checkpoints, W&B directories, and `results/runs/` are
not part of this presentation package.

## Current Coverage Notes

- CITB currently has a checked smoke config for
  `citb_instrdialog_order1_seed1_ours_v1_20260708_smoke_strict.yaml`; the
  previously referenced formal config is not present in this branch.
- InstrDialog++ is documented as a coverage gap and is not reported as complete.
- ToDCL overlay remains blocked until the official ADAPTER anchor checkpoint is
  available.
