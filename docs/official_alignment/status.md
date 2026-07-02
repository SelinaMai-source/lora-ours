# Official Alignment Status

Updated: 2026-07-02

## Current Gate

- Active branch: `ours-v25-supervision-metrics`.
- Latest pushed base before this branch: v24 `bcc5370` on `ours-v24-anti-copy-leakage`.
- Latest completed smoke reviewed: `citb_instrdialog_order1_seed1_ours_v24_smoke_strict`.
- Latest completed smoke: `citb_instrdialog_order1_seed1_ours_v25_smoke_strict`.
- W&B: project `lora-ours-v25`, run `ntv3qpgq`.
- Decision: v25 smoke completed without early stop. Segment2 gate passed (`current_task_aware_score=0.29`); full strict is now a launch candidate after a full segment2 leakage audit, but not yet SOTA-ready.

## Evidence

- Official CITB/Tk-Instruct positive examples are allowed and used in this setting: `add_task_definition=True`, `num_pos_examples=2`, `num_neg_examples=0`, `add_explanation=False`, `tk_instruct=False`.
- v24 preserved that prompt/data protocol and kept evaluation scoring unchanged.
- v24 only added generation-time anti-copy controls:
  - `gen_no_repeat_ngram_size=3`
  - `gen_encoder_no_repeat_ngram_size=3`
  - `gen_repetition_penalty=1.05`
- Segment 0/1 smoke remained healthy: `0.46` and `0.29`.
- Segment 2 failed: `current_score=0.0`, `current_task_aware_score=0.09608540925266904`, below v23b `0.12811387900355872`.
- Segment2 is `task565_circa_answer_generation`. Its official task JSON contains multiple valid outputs per input. The previous processed stream flattened those references into separate single-reference examples, while Tk-Instruct evaluation keeps `Instance.output` as a list and scores max over references.
- Official Tk-Instruct computes `exact_match`, `rouge1`, and `rougeL`; CL collection commonly reads `rougeL`, while category/task reporting uses exact match for classification-style categories. A strict exact `current_score=0` is therefore not a sufficient health signal for answer generation; task-aware ROUGE-L is the relevant early gate for segment2.
- v25 smoke uses `auto_official`: classification/option tasks use exact-match task-aware health; generation tasks use max-over-reference ROUGE-L. Segment2 improved to `current_task_aware_score=0.29` and `seen_avg_task_aware_score=0.3433`, with `task_score_type_counts={"exact_match": 200, "rouge_l": 100}`.
- Light segment2 debug audit over the saved current-task 25 examples found `0` exact/contained positive-example target leaks, `3` input-copy/contains cases, and `5` question-like template outputs. This is a clear improvement over v23b/v24 but still needs a full saved-output audit before any SOTA claim.
- Final smoke summary (4 segments): `seen_avg_task_aware_score=0.27`, `ROUGE-L AR=0.27`, `BWT=-0.0067`, no `stop_and_diagnose`. Segment matrix task-aware: `[0.46, 0.29, 0.28, 0.06]`. Segment3 current-task drop to `0.06` is expected for a fresh generation segment and is not an early-gate failure.

## Next Step

Run a full segment2 leakage audit on saved `eval_debug/eval_segment_002.json`. If copied-output counts remain acceptable, launch `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v25_strict.yaml` with monitor/W&B; otherwise iterate on supervision/routing before full strict.
