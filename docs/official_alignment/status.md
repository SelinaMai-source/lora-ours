# Official Alignment Status

Updated: 2026-07-02

## Current Gate

- Active branch: `ours-v33-target-supervision-guard`.
- Latest pushed base before this branch: v28 `537d16c` on `ours-v28-segment-min-generation-debug-nll`.
- Latest completed smoke reviewed: `citb_instrdialog_order1_seed1_ours_v31_smoke_strict`.
- Latest completed smoke: `citb_instrdialog_order1_seed1_ours_v28_smoke_strict`.
- W&B: project `lora-ours-v28`, run `fvdftnlw`.
- Latest completed smoke: `citb_instrdialog_order1_seed1_ours_v29_smoke_strict`.
- W&B: project `lora-ours-v29`, run `t67uf51q`.
- Monitor: status file `results/logs/ours_v29_strict_status.md`.
- Latest completed smoke: `citb_instrdialog_order1_seed1_ours_v29_fixed_smoke_strict`.
- W&B: project `lora-ours-v29-fixed`, run `89farydc`.
- Monitor: status file `results/logs/ours_v29_fixed_strict_status.md`.
- Latest completed smoke: `citb_instrdialog_order1_seed1_ours_v30_smoke_strict`.
- W&B: project `lora-ours-v30`, run `y0hcae3k`.
- Monitor: status file `results/logs/ours_v30_strict_status.md`.
- Latest completed smoke: `citb_instrdialog_order1_seed1_ours_v31_smoke_strict`.
- W&B: project `lora-ours-v31`, run `nz0uhi5b`.
- Monitor: status file `results/logs/ours_v31_strict_status.md`.
- Latest completed smoke: `citb_instrdialog_order1_seed1_ours_v32_smoke_strict`.
- W&B: project `lora-ours-v32`, run `fqjw2qzb`.
- Monitor: status file `results/logs/ours_v32_strict_status.md`.
- Decision: v32 keeps segment2 healthy but still improves task1714 by accepting prompt-template variants (`Now finish...`, `This is a concatenated...`). Do not launch full strict. Next step should move away from denylist-only retry acceptance toward a training-time generation supervision fix or a more principled official-compatible decode objective.
- Current v33 smoke candidate: `citb_instrdialog_order1_seed1_ours_v33_smoke_strict`.
- W&B: project `lora-ours-v33`.
- Monitor: status file `results/logs/ours_v33_strict_status.md`.

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
- v29 smoke result: segment2 stayed healthy at `0.29` task-aware, but segment3 stayed at `0.08` task-aware and current debug remained `25/25` pure `no`. The metric mapping fix worked and prompt-template continuations disappeared, but balanced sampling was misconfigured because YAML parsed unquoted `no` / `yes` buckets as booleans; a follow-up fix quotes the values and maps YAML booleans defensively.
- v29-fixed smoke result: bucket parsing and balanced sampling are now active for `task1714_convai3_sentence_generation`; train metrics recorded original buckets `{"i": 80, "no": 250, "other": 57, "yes": 113}` and balanced buckets `{"i": 188, "no": 188, "other": 188, "yes": 188}`. Segment2 remained healthy at `0.29` task-aware, but segment3 stayed blocked at `0.08` task-aware with `25/25` current debug predictions still raw `no`. Current debug routed mostly to old `b2` (`b2=20`, `b3=5`) even though `b3` received balanced segment3 training.
- v30 small-step change: keep v29-fixed scoring/data/sampling and restore `router.task_aware_fallback_force_assigned=true`, so task1714 eval is aligned to the freshly balanced segment branch instead of low-margin prompt-NLL routing back to older branches.
- v30 smoke result: segment2 remained healthy (`current_task_aware_score=0.29`, seen task-aware after segment2 `0.3433333333333333`). Segment3 training used balanced sampling on `b3` and preserved assignment after prototype refresh; `segment_branch_map` ended as `{"0": "b0", "1": "b1", "2": "b2", "3": "b3"}`. Current task debug routed `24/25` task1714 examples to `b3`, proving eval no longer fell back to old `b2`; however, current debug still generated `25/25` raw `no`, and final segment3 task-aware stayed `0.08`.
- v31 small-step change: keep v30 routing/sampling and add a label-free, config-gated eval retry only for task1714/sentence_generation when greedy decoding collapses to a single bucket token (`no`/`yes`/`i`). The retry is accepted only if it begins with the same bucket token and produces a longer non-template continuation.
- v31 smoke result: segment2 remained healthy (`current_task_aware_score=0.29`, seen task-aware after segment2 `0.3433333333333333`). Segment3 retry triggered `99` times and accepted `94`, raising segment3 task-aware from `0.08` to `0.12` and final task-aware AR from `0.275` to `0.285`. However, current debug still started `25/25` with `no`; many accepted continuations were prompt-template artifacts such as `no Now complete the following example - Input...` or diagnostic-looking text such as `no if so. This is a concatenated string...`. This is not healthy enough for full strict.
- v32 audit and small-step change: official Tk-Instruct decodes predictions with `skip_special_tokens=True` and scores the decoded prediction directly; there is no answer-only truncation that would turn `no Now complete...` into `no`. `task1714_convai3_sentence_generation` is open `Dialogue Generation` (`2295` outputs, `1959` unique normalized outputs, only `233` exact bare `yes`/`no`/`i`), so it is not a closed classification label set and no classification verbalizer/constrained label decoding was used. V32 kept scoring unchanged and made the retry prompt-copy resistant with retry-only bad-word blocking plus stricter template rejection.
- v32 smoke result: segment2 remained healthy (`current_task_aware_score=0.29`, seen task-aware after segment2 `0.37`). Segment3 final task-aware stayed at `0.12` and final task-aware AR stayed `0.28500000000000003`; retry accepted count fell from v31 `94` to v32 `82`, but current debug still started `25/25` with `no`. Broad debug audit found `13/25` accepted current outputs were still template/definition variants such as `no Now finish the following sentence`, `no Now finish the following form`, and `no if so. This is a concatenated`. Do not launch full strict.
- v33 audit before smoke: core seq2seq labels are tokenized from target only via `text_target`, pad is masked to `-100`, EOS is present in T5 target labels, and task1714 targets are not truncated (`max target tokens=31` vs `max_target_len=128`). Raw task1714 has no multi-reference instances; processed train has `500` examples, `434` unique normalized targets, no prompt-template targets, and only `53/500` exact bare `yes/no/i`. V33 disables the brittle bucket-collapse retry and adds training-time target supervision guard metrics plus continuation-token loss weighting for `no/yes/i` open-generation targets (`369/500` task1714 train rows affected).

## Next Step

Run v33 smoke only. Do not launch full strict unless segment2 stays healthy near `0.29` task-aware and segment3 improves without bucket-collapse retry acceptance or prompt-template continuations.
