# Official Alignment Status

Updated: 2026-07-02

## Current Gate

- Active branch: `ours-v25-supervision-metrics`.
- Latest pushed base before this branch: v24 `bcc5370` on `ours-v24-anti-copy-leakage`.
- Latest completed smoke reviewed: `citb_instrdialog_order1_seed1_ours_v24_smoke_strict`.
- Next smoke: `citb_instrdialog_order1_seed1_ours_v25_smoke_strict`.
- W&B: project `lora-ours-v25`.
- Decision: v25 is a supervision/metric-alignment smoke only; do not launch full strict until segment2 health is reviewed.

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

## Next Step

Run v25 strict smoke from `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v25_smoke_strict.yaml`. Gate segment2 on official max-over-reference ROUGE-L task-aware score and qualitative leakage audit; only prepare full strict if segment2 clearly improves without positive-example target leakage.
