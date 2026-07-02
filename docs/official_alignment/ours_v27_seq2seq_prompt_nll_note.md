# Ours v27 Seq2Seq Prompt NLL Arbitration Note

Generated: 2026-07-02

This is an implementation and experiment-control note. It does not claim SOTA.

## V26 Failure Audit

- Source run: `citb_instrdialog_order1_seed1_ours_v26_smoke_strict`.
- Branch: `ours-v26-router-arbitration`.
- V26 change under test: `router.task_aware_fallback_force_assigned=false`.
- Smoke completed 4 segments without low-score early failure:
  - final `seen_avg_score=0.20`
  - final `seen_avg_task_aware_score=0.2725`
  - segment 2 remained healthy: `current_task_aware_score=0.29`, `seen_avg_task_aware_score=0.3433`
- Segment 3 did not repeat the exact v25 forced-`b3` routing failure, but still failed the current-task generation gate:
  - eval routing totals: `b0=200`, `b1=100`, `b2=100`, `b3=0`
  - current segment 3 debug examples: 25/25 routed to `b2`
  - current segment 3 debug predictions: 25/25 generated `no`
  - current segment 3 task-aware score over debug slice: about `0.164`
- Root-cause hypothesis:
  - `HFSeq2SeqLMBackbone.score_prompt_nlls()` returned constant `0.0`.
  - Therefore T5 low-margin NLL arbitration had no branch-specific evidence and could deterministically fall through to a stale branch.
  - In v26 segment 3, that stale branch was `b2`, producing the same all-`no` current-task collapse even though forced `b3` fallback was disabled.

## Official Alignment

- V27 keeps the CITB/Tk-Instruct seq2seq protocol:
  - `add_task_definition=True`
  - `num_pos_examples=2`
  - `max_source_length=1024`
  - `max_target_length=128`
  - `generation_max_length=128`
  - label padding masked to `-100`
- V27 does not score or inspect gold answers during routing arbitration.
- The new prompt score reconstructs the formatted prompt only, so it remains label-free for routing.

## V27 Change

- `core/models/seq2seq_lora_wrapper.py`
  - Replaces constant seq2seq prompt NLL with prompt-reconstruction NLL.
  - Uses the formatted prompt as both encoder input and decoder target.
  - Masks padding and returns mean token NLL per prompt.
- `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v27_smoke_strict.yaml`
  - Same data, training, routing, and evaluation settings as v26.
  - W&B project: `lora-ours-v27`.
- `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v27_strict.yaml`
  - Full strict counterpart for launch only after smoke passes.

## Smoke Gate

- Run v27 strict smoke first: `citb_instrdialog_order1_seed1_ours_v27_smoke_strict`.
- Do not launch full strict unless:
  - segment 2 stays comparable to v26/v25: `seen_avg_task_aware_score` near `0.3433`
  - segment 3 current debug examples are not all `no`
  - segment 3 current debug examples are not all routed to one stale branch with constant-tie arbitration
  - final `seen_avg_task_aware_score` remains above the configured early gate
- If segment 3 still all-`no` collapses, stop and diagnose generation/training behavior next rather than adding another router shim.
