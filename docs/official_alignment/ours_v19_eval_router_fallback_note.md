# Ours v19 Eval Router Fallback Note

Generated: 2026-07-02

This is an implementation and experiment-control note. It does not claim SOTA.

## Failure Basis

- v18 smoke run: `citb_instrdialog_order1_seed1_ours_v18_smoke_strict`, W&B run `4arqimqs`, project `lora-ours-v18`.
- Segment 0 and 1 were healthy: `seen_avg_score=0.28` then `0.285`.
- Segment 2 reproduced the old failure trajectory: `current_score=0.0`, `current_task_aware_score=0.05`, with eval routing collapsed to `b0=300/300`.
- The same segment's training assignment was not collapsed to `b0`: train metrics showed `routed_train_branch_counts_json={"b1": 536, "b2": 28}`. This means v18 learned/used trainable branches during training but did not carry that task assignment into eval routing.

## V19 Changes

- `core/train.py`
  - After routed training assignment is resolved to trainable branches, record the majority actual training branch with `router.record_segment_assignment`.
  - Add `task_aware_fallback_training_branch` to train metrics so the mapping is auditable.
- `core/evaluate.py`
  - Record `task_aware_fallback_count` in routing metrics.
  - Add per-example `routing_reason` to debug examples so fallback use can be audited.
- `scripts/monitor_ours_v18_strict.py`
  - Parameterized run id/status filenames by environment variables and shortened refresh to `60s`; v18 defaults are preserved.
- `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v19_smoke_strict.yaml`
  - Same official CITB stream/model/training/generation settings as v18 smoke.
  - Uses W&B project `lora-ours-v19`.
- `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v19_strict.yaml`
  - Full strict config is staged but blocked until smoke clears.

## Gate

- Start only v19 smoke first.
- Required early signal: segment 2 must not repeat `current_score=0.0` with `b0=100%` eval routing.
- Full strict CITB can be queued only after smoke clears the early gate; smoke results must not be reported as SOTA.
