# Official Alignment Status

Updated: 2026-07-02

## Current Gate

- Active branch: `ours-v24-anti-copy-leakage`.
- Latest pushed base before this branch: v23 `b125935` on `ours-v23-seq2seq-protocol`.
- Latest smoke: `citb_instrdialog_order1_seed1_ours_v24_smoke_strict`.
- W&B: project `lora-ours-v24`, run `f0ychza9`.
- Decision: v24 is not full-strict ready and must not be reported as SOTA.

## Evidence

- Official CITB/Tk-Instruct positive examples are allowed and used in this setting: `add_task_definition=True`, `num_pos_examples=2`, `num_neg_examples=0`, `add_explanation=False`, `tk_instruct=False`.
- v24 preserves that prompt/data protocol and keeps evaluation scoring unchanged.
- v24 only adds generation-time anti-copy controls:
  - `gen_no_repeat_ngram_size=3`
  - `gen_encoder_no_repeat_ngram_size=3`
  - `gen_repetition_penalty=1.05`
- Segment 0/1 smoke remained healthy: `0.46` and `0.29`.
- Segment 2 failed: `current_score=0.0`, `current_task_aware_score=0.09608540925266904`, below v23b `0.12811387900355872`.

## Next Step

Do not launch full strict from v24. The next iteration should inspect task565 target/reference multiplicity and processed-stream training target construction rather than adding stronger decoding penalties.
