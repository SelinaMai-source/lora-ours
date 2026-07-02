# Ours v23 Seq2Seq Protocol Note

Generated: 2026-07-02

This is an implementation and experiment-control note. It does not claim SOTA.

## V22 Evidence

- v22 smoke `citb_instrdialog_order1_seed1_ours_v22_smoke_strict` completed 4 strict-smoke segments and synced to W&B project `lora-ours-v22`, run `k9wnkwza`.
- v22 ruled out the main routing/training-control failures:
  - Segment 2 spawned and trained active branch `b2`: `routed_train_branch_counts_json={"b2": 564}`.
  - Task-aware fallback preserved segment-to-branch mapping through eval, including current segment 2 -> `b2`.
  - Predictions were non-empty and labels were not reversed.
- Generation still failed on answer-generation tasks:
  - Segment 2 `current_score=0.0`, `current_task_aware_score=0.05`, `seen_avg_score=0.19333333333333333`.
  - Final segment 3 `current_score=0.0`, `current_task_aware_score=0.05`, `seen_avg_score=0.145`, `seen_avg_task_aware_score=0.16749999999999998`.
  - Segment 2 outputs remained question-like/input-copying, e.g. `Is it stressful all the time?`, `Will you be around for a while?`, and `You are given a question`, while gold outputs were answer utterances.

## Official Seq2Seq Protocol Evidence

- Local official CITB/Tk-Instruct code:
  - `Tk-Instruct/src/ni_collator.py` builds source as `Definition: ...` plus `Positive Example {i}` blocks plus `Now complete the following example -\nInput: ...\nOutput: `.
  - Official scripts for initial tuning, continual tuning, and eval use `--add_task_definition True --num_pos_examples 2 --num_neg_examples 0 --add_explanation False --tk_instruct False`.
  - `max_source_length=1024`, `max_target_length=128`, and `generation_max_length=128`.
  - Labels are tokenized separately from source and pad positions are masked to `-100`.
  - Eval metric uses ROUGE-L F-measure with stemming and reports percentage scale; our CCFA matrices store the same score on `[0, 1]` scale and export AR as percent.
- Existing ours seq2seq wrapper already matched the core label/generation contract:
  - `text_target` tokenization for labels.
  - Pad labels masked to `-100`.
  - T5 generation decodes generated decoder output without causal prompt slicing.
  - `decoder_start_token_id` is read from model config.
- The material mismatch was prompt construction: v22 processed stream exposed only `instruction/input/output`, and `format_citb_t5` used definition-only prompts, while official CITB uses two positive examples for this setting.

## V23 Change

- `core/data.py`
  - Adds official order-file support for processed stream regeneration through `processed_task_order_file`.
  - Adds explicit dev split count support so InstrDialog order1 can preserve `train=500/dev=50/test=100`; this prevents low-resource tasks such as `task1590` from becoming empty train segments.
  - Preserves the existing `(instruction, input) -> output` training API by rendering official two positive examples into the instruction string when a processed example carries `positive_examples`.
- `core/formatting.py` and `core/models/seq2seq_lora_wrapper.py`
  - Apply official collator-style length-aware fitting: each positive example is included only if `definition + selected examples + task input` fits `max_source_len=1024`.
  - This prevents long examples from truncating the current input/output boundary off the right edge.
- `core/train.py`
  - Threads `paths.processed_task_order_file` and `data.processed_stream_dev_instances_per_task` into stream loading.
- `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v23_smoke_strict.yaml`
  - Keeps v22 routing/training controls.
  - Points to an external generated processed stream `citb_instrdialog_order1_train500_dev50_test100_pos2.json`.
  - Uses W&B project `lora-ours-v23`.

## Local Verification

- Rebuilt external v23 processed stream from official local CITB sources with:
  - order file `data/CIT_data/task_orders/stream=cl_dialogue_tasks/order1.txt`.
  - `train=500`, `dev=50`, `test=100`.
- Verified first four segments match v22 order:
  - `task848_pubmedqa_classification`
  - `task611_mutual_multi_turn_dialogue`
  - `task565_circa_answer_generation`
  - `task1714_convai3_sentence_generation`
- Verified loaded segment 2 prompt contains both `Positive Example 1 -` and `Positive Example 2 -`, and target remains answer text such as `Baseball is fun.`
- Verified length-aware prompt fitting:
  - Long PubMed segment 0 example: positives skipped, current input retained, source length 692 tokens.
  - Short Circa segment 2 example: positives retained, current input retained, source length 100 tokens.
- Python compile and IDE lint checks passed for `core/data.py` and `core/train.py`.

## Smoke Gate

- Invalid attempt `citb_instrdialog_order1_seed1_ours_v23_smoke_strict` was stopped after segment 0 because static two-shot prompts truncated the current input for long PubMed examples, producing `current_score=0.0` and fixed outputs such as `synthesis`.
- Run only corrected v23b strict smoke first: `citb_instrdialog_order1_seed1_ours_v23b_smoke_strict`.
- Full strict CITB remains blocked unless smoke shows early segment improvement and current-task predictions stop predominantly copying or restating the input.

## Smoke Result

- 2026-07-02 local: corrected v23b smoke started with W&B project `lora-ours-v23`, run `dur0o490`, and was manually early-stopped after the segment 2 gate inspection.
- Segment 0 recovered from the invalid static-pos2 attempt and improved over v22:
  - v23b segment 0: `current_score=0.46`, `current_task_aware_score=0.46`.
  - The formatted long PubMed eval prompts skipped positive examples when needed, preserving the current input and producing `0/1` classification outputs rather than the invalid fixed `synthesis` output.
- Segment 1 remained comparable to v22:
  - v23b segment 1: `current_score=0.29`, `current_task_aware_score=0.29`, `seen_avg_score=0.37`.
- Segment 2 showed partial metric improvement but failed the qualitative generation gate:
  - v23b segment 2: `current_score=0.0`, `current_task_aware_score=0.12811387900355872`, `seen_avg_score=0.25`, `seen_avg_task_aware_score=0.2927046263345196`.
  - Training and eval branch control remained correct: `routed_train_branch_counts_json={"b2": 1155}` and router state preserved `{"0": "b0", "1": "b1", "2": "b2"}`.
  - Current-task predictions were still not acceptable. Early examples copied the input question, e.g. `Is it stressful all the time?`; later examples copied a positive-example answer, e.g. `I have a lot of assignments to do.`
- Gate decision:
  - Do not launch full strict from v23b.
  - Do not claim SOTA.
  - Next step should target generation behavior under few-shot prompts: prevent positive-example leakage and input-copying, likely by auditing target/reference multiplicity, training sample expansion, prompt/example selection, and generation constraints for answer-generation tasks.
