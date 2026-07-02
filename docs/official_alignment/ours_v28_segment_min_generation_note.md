# Ours v28 Segment Min Generation Note

Generated: 2026-07-02

This is an implementation and experiment-control note. It does not claim SOTA.

## V27 Smoke Result

- Run: `citb_instrdialog_order1_seed1_ours_v27_smoke_strict`.
- Branch: `ours-v27-seq2seq-prompt-nll-arbitration`.
- Final smoke metrics:
  - `seen_avg_score=0.20`
  - `seen_avg_task_aware_score=0.2725`
  - segment 2 task-aware remained `0.29`
- Segment 3 still failed the current-task generation gate:
  - current task debug slice: 25/25 predictions were `no`
  - current task routing: `b2=20`, `b3=5`
  - even the `b3` examples generated `no`
- Conclusion: v27 improved routing observability but did not fix the real generation collapse.

## V28 Change

- Keep v27 label-free prompt-reconstruction NLL arbitration.
- Add eval debug fields for NLL arbitration candidates, original branch, and changed flag.
- Add config-gated minimum generation length:
  - default remains `0`
  - v28 enables `segment_min_new_tokens=4`
  - activation is by segment name only: `task1714` / `sentence_generation`

## Smoke Gate

Do not launch full strict unless v28 smoke verifies:

- segment 2 remains comparable to v26/v27: task-aware score near `0.29`
- segment 3 current debug examples are not 25/25 `no`
- segment 3 current-task task-aware score improves over `0.06`
- NLL arbitration debug shows candidate scores and whether routing changed
