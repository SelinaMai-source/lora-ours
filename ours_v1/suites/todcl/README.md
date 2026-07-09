# ToDCL Ours v1

## Base

ToDCL 37-domain ADAPTER NLG official anchor.

## Overlay

The overlay installs an Assess-then-Update orthogonal penalty after loading the
anchor checkpoint. It does not change the official dataset list, task order,
metric path, or ADAPTER setting.

## Entry Points

- Launcher: `ours_v1/scripts/launchers/run_todcl_v1.sh`
- Legacy wrapper: `scripts/run_todcl_adapter_nlg_ours_assess_overlay_v1_20260708.sh`
- Overlay implementation: `ours_v1/suites/todcl/todcl_assess_orthogonal_overlay.py`
- Metadata config: `configs/ccfa_three_suite/todcl_ours_v1_20260708_assess_overlay.yaml`
- First-principles note: `docs/experiments/todcl_ours_v1_20260708_first_principles.md`

The launcher still blocks if the official anchor checkpoint is unavailable.
