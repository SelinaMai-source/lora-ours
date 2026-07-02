# Ours v22 Generation Effective Note

Generated: 2026-07-02

This is an implementation and experiment-control note. It does not claim SOTA.

## V21 Failure Evidence

- v21 smoke `citb_instrdialog_order1_seed1_ours_v21_smoke_strict` reached segment 2 and early-stopped with `current_score=0.0`, `current_task_aware_score=0.04`, `seen_avg_score=0.19000000000000003`.
- Eval routing collapse was already fixed relative to v20: aggregate branch counts were `b0=200`, `b1=100`, and current-task debug examples routed to `b1=20/20`.
- Segment 2 data was not empty or label-reversed. The processed stream has question inputs and answer outputs for `task565_circa_answer_generation`.
- Current-task predictions were non-empty but question-like: examples include `What is the stress that you are feeling?`, `Will you be around for a while?`, and `You are given a question`, while gold outputs were answers such as `it is very rarely stressful`.
- Training metrics showed the fresh active branch was not the dominant current-task training target: segment 2 had active adapter `b2`, but routed train counts were `{"b1": 536, "b2": 28}` and task-aware eval mapped the current task to `b1`.

## V22 Change

- `core/train.py`
  - Adds a config-gated routed-training override: if the active LoRA branch was created for the current segment and is trainable, assign that segment's training examples to the active branch before router/NLL assignment can pull them back to an older branch.
  - Logs `routed_train_force_active_on_spawn_segment` and `routed_train_forced_active_branch` for audit.
- `core/methods/router.py`
  - Adds `force_active_branch_on_spawn_segment` to router config and `router_state.json`.
- `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v22_smoke_strict.yaml`
  - Keeps the v21 official stream/model/training/generation settings.
  - Enables `router.force_active_branch_on_spawn_segment: true`.
  - Uses W&B project `lora-ours-v22`.

## Gate

- Run only v22 strict smoke first.
- Required early evidence:
  - Segment 2 train metrics should show `routed_train_force_active_on_spawn_segment=true` and training branch `b2`.
  - Current-task predictions should stop predominantly copying or restating the input question.
  - Segment 2 `current_task_aware_score` should clear the configured early gate before any full strict run is considered.
- Full strict CITB remains blocked unless smoke proves generation effectiveness. Smoke results must not be reported as SOTA.
