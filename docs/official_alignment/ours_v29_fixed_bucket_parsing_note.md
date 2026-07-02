# Ours v29-Fixed Bucket Parsing Note

Generated: 2026-07-02

This is an implementation and experiment-control note. It does not claim SOTA.

## Run

- Config: `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v29_fixed_smoke_strict.yaml`.
- Run: `citb_instrdialog_order1_seed1_ours_v29_fixed_smoke_strict`.
- W&B: project `lora-ours-v29-fixed`, run `89farydc`.
- Tmux:
  - train window: `lora-ours:v29fix-smoke`.
  - monitor window: `lora-ours:v29fix-mon`.

## Result

- Status: completed smoke; do not launch full strict.
- Segment0/1 remained comparable to prior smoke: task-aware `0.46`, then `0.29`.
- Segment2 remained healthy:
  - current task-aware score: `0.29`.
  - seen task-aware score: `0.3433333333333333`.
- The bucket parsing fix worked for task1714:
  - original buckets: `{"i": 80, "no": 250, "other": 57, "yes": 113}`.
  - balanced buckets: `{"i": 188, "no": 188, "other": 188, "yes": 188}`.
- Segment3 still failed:
  - current score: `0.06`.
  - current task-aware score: `0.08`.
  - current debug predictions: `25/25` raw `no`.
  - current debug route distribution: `b2=20`, `b3=5`.

## Decision

The fix is effective only for bucket parsing and balanced sampling activation. It is not sufficient for segment3 generation health. The next step is v30: keep balanced sampling and test forced task-aware assigned-branch routing so task1714 eval uses the branch that received balanced supervision.
