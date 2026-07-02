# Ours v20 Task-Aware Forced Fallback Note

Generated: 2026-07-02

This is an implementation and experiment-control note. It does not claim SOTA.

## Failure Basis

- v19 smoke `citb_instrdialog_order1_seed1_ours_v19_smoke_strict` reached segment 2 but repeated the early failure: `current_score=0.0`, `current_task_aware_score=0.05`.
- Eval routing collapsed to `b0=300/300` even though segment 2 training assignment favored `b1`: `{"b1": 536, "b2": 28}`.
- v19 therefore proved that recording a segment-to-training-branch mapping is not enough if eval routing can still override it with low-confidence prototype/NLL arbitration.

## V20 Change

- `core/methods/router.py`
  - Add `task_aware_fallback_force_assigned`, default `false`.
  - When enabled and a segment has a recorded branch assignment, task-aware fallback chooses that branch instead of only using it under confidence/margin thresholds.
- `core/evaluate.py`
  - Do not run NLL arbitration after a forced task-aware fallback decision.
- `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v20_smoke_strict.yaml`
  - Same official CITB stream/model/training/generation settings as v19 smoke.
  - Uses W&B project `lora-ours-v20`.
  - Enables `router.task_aware_fallback_force_assigned: true`.
- `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v20_strict.yaml`
  - Full strict config is staged but blocked until smoke clears.

## Gate

- Start only v20 smoke first.
- Required early signal: segment 2 must not repeat `current_score=0.0` with `b0=300/300` eval routing collapse.
- Full strict CITB can be queued only after smoke clears the early gate; smoke results must not be reported as SOTA.
