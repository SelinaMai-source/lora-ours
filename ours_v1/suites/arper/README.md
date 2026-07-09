# ARPER Ours v1

## Base

ARPER MultiWOZ-2.0 SCLSTM Path B, aligned to the v89 published-base run.

## Overlay

The overlay injects SSRG exemplar selection into the ARPER exemplar construction
path while keeping the official training command, data split, and monitor path.

## Entry Points

- Launcher: `ours_v1/scripts/launchers/run_arper_v1.sh`
- Legacy wrapper: `scripts/run_arper_woz3_sclstm_plus_ssrg_overlay_v1_20260708.sh`
- Exemplar selector: `ours_v1/suites/arper/arper_ssrg_exemplar_selection.py`
- Train wrapper: `ours_v1/suites/arper/arper_ssrg_overlay_train_wrapper.py`
- Metadata config: `configs/ccfa_three_suite/arper_woz3_sclstm_v89_ssrg_exemplar_overlay_v1_20260708.yaml`
- First-principles note: `docs/experiments/arper_ours_v1_20260708_first_principles.md`
