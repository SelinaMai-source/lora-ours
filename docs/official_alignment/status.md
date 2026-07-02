# Official Alignment Status

Updated: 2026-07-02

## Current Gate

- Active branch: `ours-v29-segment3-calibration`.
- Latest pushed base before this branch: v28 `537d16c` on `ours-v28-segment-min-generation-debug-nll`.
- Latest completed smoke reviewed: `citb_instrdialog_order1_seed1_ours_v28_smoke_strict`.
- Latest completed smoke: `citb_instrdialog_order1_seed1_ours_v28_smoke_strict`.
- W&B: project `lora-ours-v28`, run `fvdftnlw`.
- Running smoke: `citb_instrdialog_order1_seed1_ours_v29_smoke_strict` after v29 commit/push.
- W&B: project `lora-ours-v29`.
- Monitor: status file `results/logs/ours_v29_strict_status.md`.
- Decision: v28 confirmed NLL debug works but did not fix segment3. Do not launch full strict unless v29 smoke keeps segment2 healthy and improves task1714 current-task ROUGE-L without prompt-template continuations.

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
- Final v25 smoke summary (4 segments): `seen_avg_task_aware_score=0.27`, `ROUGE-L AR=0.27`, `BWT=-0.0067`, no `stop_and_diagnose`. Segment matrix task-aware: `[0.46, 0.29, 0.28, 0.06]`.
- Segment3 (`task1714_convai3_sentence_generation`) is the blocker: current task-aware score dropped to `0.06`; saved current-task debug examples routed `25/25` examples to `b3` via `task_aware_fallback_forced` and generated `no` for all `25/25` current-task debug examples. This is a current generation collapse, so v25 full strict was not launched.
- v26 small-step change: keep v25 official multi-reference/task-type-aware metrics, but set `router.task_aware_fallback_force_assigned=false` so low-margin generation routing can use NLL arbitration instead of being unconditionally forced to the newly spawned branch.
- v27/v28 result: NLL arbitration debug is effective and v28 changed 5/25 task1714 current debug routes, but segment3 remained blocked with `current_score=0.0` and `current_task_aware_score=0.08`. V28 minimum generation length changed pure `no` into longer `no ...` / `no Now complete the following` outputs, so the issue is no longer primarily route observability.
- v28 segment3 audit: `task1714_convai3_sentence_generation` is a `Dialogue Generation` task and should be treated as ROUGE-L generation, not classification. Its processed train split has a real first-token prior (`no=250`, `yes=113`, `i=80` among 500 examples), positive examples include `yes` and `no ...`, and v28 training supervision was weak (`train.loss=3.4283`, `train.answer_token_acc=0.3953`).
- v29 small-step change: remove task1714 minimum generation length, fix `auto_official` generation-vs-intent metric inference, and enable config-scoped first-token balanced sampling only for `task1714` / `sentence_generation`.

## Next Step

Launch `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v29_smoke_strict.yaml` with W&B online and monitor. If segment2 remains near `0.29` task-aware and segment3 current-task task-aware improves over v28 `0.08` without prompt-template continuations, then consider `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v29_strict.yaml` for full strict; otherwise stop and diagnose the next training-supervision issue.
