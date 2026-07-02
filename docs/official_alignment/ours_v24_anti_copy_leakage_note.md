# Ours v24 Anti-Copy Leakage Note

Generated: 2026-07-02

This is an implementation and experiment-control note. It does not claim SOTA.

## V23b Failure Audit

- Source run: `citb_instrdialog_order1_seed1_ours_v23b_smoke_strict`.
- Segment 2 current task: `task565_circa_answer_generation`.
- Debug sample size: 25 current-task examples from `eval_debug/eval_segment_002.json`.
- Manual/scripted categorization:
  - 3 input-copying cases, including exact repetition of `Is it stressful all the time?`.
  - 10 positive-example leakage cases, dominated by generated `I have a lot of assignments to do.` from the task positive example answer.
  - 9 label/reference mismatch or semantic-mismatch cases where the output was answer-shaped but not the held-out reference.
  - 0 empty generation/decoding failures.
  - 3 task-aware partial matches.
- Conclusion: the next fix should target answer-generation copying/leakage behavior, not routing.

## Official Alignment

- Local official CITB/Tk-Instruct code allows positive examples and the published scripts use them for this setting:
  - `Tk-Instruct/src/ni_collator.py` renders `Definition: ...`, up to `num_pos_examples=2`, then `Now complete the following example -\nInput: ...\nOutput: `.
  - Official arguments remain `add_task_definition=True`, `num_pos_examples=2`, `num_neg_examples=0`, `add_explanation=False`, `tk_instruct=False`.
- V24 keeps the v23b processed stream and official prompt format. It does not remove positive examples and does not change evaluation normalization or scoring.

## V24 Change

- `core/models/seq2seq_lora_wrapper.py`
  - Adds optional generation parameters:
    - `gen_no_repeat_ngram_size`
    - `gen_encoder_no_repeat_ngram_size`
    - `gen_repetition_penalty`
  - Defaults preserve old behavior.
- `core/evaluate.py`
  - Records these generation parameters in `eval_debug/*/generation_cfg` for auditability.
- `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v24_smoke_strict.yaml`
  - Uses the same v23b strict-smoke data, training, and routing settings.
  - Sets deterministic greedy decoding with:
    - `gen_no_repeat_ngram_size=3`
    - `gen_encoder_no_repeat_ngram_size=3`
    - `gen_repetition_penalty=1.05`
  - Keeps W&B online under project `lora-ours-v24`.

## Smoke Gate

- Run only v24 strict smoke first: `citb_instrdialog_order1_seed1_ours_v24_smoke_strict`.
- Inspect segment 2 before any full strict run.
- Healthy segment 2 gate requires both:
  - current-task predictions are not dominated by input copying or positive-example answer copying.
  - segment 2 task-aware score is at least competitive with v23b without collapsing segments 0/1.
- If segment 2 still has `current_score=0.0` with copy/leakage-dominated predictions, stop and do not launch full strict.

## Smoke Result

- 2026-07-02 local: v24 strict smoke started with W&B project `lora-ours-v24`, run `f0ychza9`, then was manually early-stopped after segment 2 gate inspection.
- Segment 0 remained healthy:
  - `current_score=0.46`, `current_task_aware_score=0.46`.
- Segment 1 remained comparable to v23b:
  - `current_score=0.29`, `current_task_aware_score=0.29`, `seen_avg_score=0.37`.
- Segment 2 failed the answer-generation gate:
  - `current_score=0.0`, `current_task_aware_score=0.09608540925266904`, `seen_avg_score=0.25`, `seen_avg_task_aware_score=0.2820284697508897`.
  - This is below v23b segment 2 `current_task_aware_score=0.12811387900355872`.
- V24 `eval_debug/eval_segment_002.json` confirms the generation parameters were active:
  - `no_repeat_ngram_size=3`
  - `encoder_no_repeat_ngram_size=3`
  - `repetition_penalty=1.05`
- Qualitative audit of the same 25 current-task debug examples:
  - 0 exact input-copying cases under the previous detector.
  - 0 exact positive-example answer leakage cases under the previous detector.
  - 23 answer-shaped but wrong/template-neighbor outputs.
  - 2 task-aware partial matches.
- Gate decision:
  - Do not launch full strict from v24.
  - Do not claim SOTA.
  - The hard decoding constraint reduced exact copying but pushed the model toward nearby stale templates such as `I have a lot of homework to do` and `I have a lot of tasks to do`.
  - Next step should not add stronger decoding penalties. It should inspect task565 target/reference multiplicity and training construction: multiple acceptable answers per same input, whether official random-reference training is being collapsed to one target in the processed stream, and whether leave-current-task-out or higher-diversity positive examples can be selected without changing the official evaluation metric.
