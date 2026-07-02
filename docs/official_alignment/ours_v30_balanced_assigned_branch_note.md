# Ours v30 Balanced Assigned Branch Note

Generated: 2026-07-02

This is an implementation and experiment-control note. It does not claim SOTA.

## Evidence From V29-Fixed

- Run: `citb_instrdialog_order1_seed1_ours_v29_fixed_smoke_strict`.
- W&B: project `lora-ours-v29-fixed`, run `89farydc`.
- The YAML bucket parsing fix worked:
  - original task1714 buckets: `{"i": 80, "no": 250, "other": 57, "yes": 113}`.
  - balanced task1714 buckets: `{"i": 188, "no": 188, "other": 188, "yes": 188}`.
- Segment2 stayed healthy:
  - `task565_circa_answer_generation` current task-aware score: `0.29`.
  - seen task-aware after segment2: `0.3433333333333333`.
- Segment3 still failed:
  - current score: `0.06`.
  - current task-aware score: `0.08`.
  - final seen task-aware: `0.27749999999999997`.
  - current debug predictions: `25/25` raw `no`.
  - current debug routes: `b2=20`, `b3=5`.

## V30 Change

- Keep v29-fixed official data, prompt protocol, target protocol, scoring, and balanced first-token sampling.
- Restore `router.task_aware_fallback_force_assigned=true`.
- Rationale: v29-fixed confirms the newly spawned segment3 branch `b3` received balanced current-task training, but eval still routed most current debug examples to old `b2` via prompt-NLL arbitration. V30 tests whether evaluating task1714 on the branch that actually received balanced supervision improves current generation.

## Smoke Gate

Do not launch full strict unless v30 smoke verifies:

- segment2 task-aware remains near `0.29`.
- segment3 current debug is no longer `25/25` pure `no`.
- segment3 current task-aware improves over `0.08`.
- routing/debug evidence shows task1714 current examples use the balanced segment branch.
