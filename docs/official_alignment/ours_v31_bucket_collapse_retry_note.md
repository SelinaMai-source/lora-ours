# Ours v31 Bucket Collapse Retry Note

Generated: 2026-07-02

This is an implementation and experiment-control note. It does not claim SOTA.

## Evidence From V30

- Run: `citb_instrdialog_order1_seed1_ours_v30_smoke_strict`.
- W&B: project `lora-ours-v30`, run `y0hcae3k`.
- Segment2 stayed healthy:
  - `task565_circa_answer_generation` current task-aware score: `0.29`.
  - seen task-aware after segment2: `0.3433333333333333`.
- Segment3 training and routing alignment worked:
  - training branch: `b3`.
  - balanced buckets: `{"i": 188, "no": 188, "other": 188, "yes": 188}`.
  - `task_aware_fallback_assignment_preserved=true`.
  - final `segment_branch_map`: `{"0": "b0", "1": "b1", "2": "b2", "3": "b3"}`.
  - current task debug routes: `b3=24`, `b2=1`.
- Segment3 still failed:
  - current score: `0.06`.
  - current task-aware score: `0.08`.
  - final seen task-aware: `0.275`.
  - current debug predictions: `25/25` raw `no`.

## V31 Change

- Keep v30 official data, prompt protocol, target protocol, scoring, balanced sampling, and forced assigned-branch evaluation.
- Add `eval_normalization.enable_bucket_collapse_retry=true`, gated to `task1714` / `sentence_generation`.
- If greedy decoding returns exactly one bucket token (`no`, `yes`, or `i`), retry generation once with `min_new_tokens=6`.
- Accept the retry only when it still starts with the same bucket token, has at least two normalized tokens, and does not look like a prompt-template continuation.

## Smoke Gate

Do not launch full strict unless v31 smoke verifies:

- segment2 task-aware remains near `0.29`.
- segment3 current debug is no longer `25/25` raw `no`.
- segment3 current task-aware improves over `0.08`.
- retry audit counters show whether the change was actually used and accepted.

## V31 Smoke Result

- Run: `citb_instrdialog_order1_seed1_ours_v31_smoke_strict`.
- W&B: project `lora-ours-v31`, run `nz0uhi5b`.
- Segment2 stayed healthy:
  - current task-aware score: `0.29`.
  - seen task-aware after segment2: `0.3433333333333333`.
- Segment3 retry was active:
  - retry count: `99`.
  - accepted retry count: `94`.
  - current task-aware score: `0.12` (up from v30 `0.08`).
  - final task-aware AR: `0.285` (up from v30 `0.275`).
- Segment3 is still not full-strict ready:
  - current exact score: `0.0`.
  - current debug first token: `25/25` still `no`.
  - many accepted outputs are prompt-template continuations, e.g. `no Now complete the following example - Input...`.
  - other accepted outputs include diagnostic-looking continuations such as `no if so. This is a concatenated string...`.

Conclusion: v31 proves that the single-token `no` collapse can be nudged into longer generations, but the current retry acceptance is too permissive. Do not launch full strict from v31.
