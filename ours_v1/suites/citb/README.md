# CITB Ours v1

## Base

Replay(50) under the InstrDialog official setting.

## Overlay

The v1 smoke config keeps the official task order and metric path while using an
SSRG-aligned replay budget:

- `replay_ratio=0.5`
- minimum replay budget aligned to Replay(50)
- official scorer through `core.train`

## Entry Points

- Launcher: `ours_v1/scripts/launchers/run_citb_v1.sh`
- Legacy wrapper: `scripts/run_citb_ours_v1_20260708_smoke.sh`
- Smoke config: `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v1_20260708_smoke_strict.yaml`
- First-principles note: `docs/experiments/citb_ours_v1_20260708_first_principles.md`

## Coverage Gap

`InstrDialog++` is not marked complete on this branch. The old launcher
referenced a formal config name that is not present, so this package exposes the
checked smoke config and documents the gap instead of fabricating a config.
