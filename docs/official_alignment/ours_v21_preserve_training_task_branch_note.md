# Ours v21 Preserve Training Task Branch Note

Generated: 2026-07-02

This is an implementation and experiment-control note. It does not claim SOTA.

## Failure Basis

- v20 smoke `citb_instrdialog_order1_seed1_ours_v20_smoke_strict` reached segment 2 and repeated the routing collapse: `current_score=0.0`, `current_task_aware_score=0.05`, `seen_avg_score=0.19000000000000003`.
- Eval routing remained `b0=300/300`, with oracle-best counts `b0=199`, `b1=52`, `b2=49`.
- The same segment's routed training assignment favored `b1`: `task_aware_fallback_training_branch=b1`, `routed_train_branch_counts_json={"b1": 536, "b2": 28}`.
- Code inspection showed a plausible overwrite path: `_train_with_routed_assignments` records the routed training majority branch, then `_update_router_with_segment_pseudo_labels` can record a later noisy pseudo-label majority for the same segment.

## V21 Change

- `core/train.py`
  - After prototype pseudo-label refresh, re-record the segment assignment to the branch that actually received the segment's routed training updates when `task_aware_fallback_force_assigned` is enabled.
  - Log `task_aware_fallback_assignment_after_proto` and `task_aware_fallback_assignment_preserved` in train metrics for smoke audit.
- `core/evaluate.py`
  - Save eval debug examples stratified by source segment so current-task routing examples are visible at segment 2 instead of only the earliest seen segment.
- `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v21_smoke_strict.yaml`
  - Same official CITB stream/model/training/generation settings as v20 smoke.
  - Uses W&B project `lora-ours-v21`.
- `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v21_strict.yaml`
  - Full strict config is staged but blocked until smoke clears.

## Gate

- Start only v21 smoke first.
- Required early signal: segment 2 must not repeat `current_score=0.0` with `b0=300/300` eval routing collapse.
- Segment 2 train metrics should show `task_aware_fallback_training_branch=b1` and `task_aware_fallback_assignment_after_proto=b1`; eval routing should route the current segment away from the old all-`b0` collapse.
- Full strict CITB can be queued only after smoke clears the early gate; smoke results must not be reported as SOTA.

## Smoke Result

- 2026-07-02 local: v21 smoke `citb_instrdialog_order1_seed1_ours_v21_smoke_strict` reached segment 2 and was early-stopped.
- Segment 2 preserved the intended mapping: `task_aware_fallback_training_branch=b1`, `task_aware_fallback_assignment_after_proto=b1`, `task_aware_fallback_assignment_preserved=true`.
- Eval routing no longer collapsed to all `b0`: aggregate branch counts were `b0=200`, `b1=100`; debug examples were stratified by source segment and showed source segment 2 routed to `b1=20/20`.
- The score gate still failed: `current_score=0.0`, `current_task_aware_score=0.04`, `seen_avg_score=0.19000000000000003`, `seen_avg_task_aware_score=0.20333333333333334`.
- Diagnosis: v21 fixed the eval-routing collapse, but the branch that received/routed the current task (`b1`) still generated zero exact matches on segment 2. Do not launch v21 full strict. The next iteration should focus on segment 2 training/generation effectiveness for the routed branch, not on task-aware routing.
