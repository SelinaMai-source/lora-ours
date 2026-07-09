# Standard Ours v1

## Base

O-LoRA official-equivalent anchor. LB-CL is retained only as a paper reference
and is not the operational code base for v1.

## Overlay

The v1 launcher exports a single-mechanism delta over the O-LoRA launcher:

- class-coverage SSRG ordering
- assess-retention gate with threshold `0.25`
- SSRG spectral parameters `top_k=8`, `energy_threshold=0.85`

## Entry Points

- Launcher: `ours_v1/scripts/launchers/run_standard_v1.sh`
- Legacy wrapper: `scripts/run_olora_standard_order1_official_base_ours_overlay_v1_20260708.sh`
- SSRG proxy: `ours_v1/suites/standard/olora_overlay_ssrg.py`
- Assess gate: `ours_v1/suites/standard/olora_overlay_assess_retention_gate.py`
- First-principles note: `docs/experiments/standard_olora_overlay_v1_20260708_first_principles.md`
