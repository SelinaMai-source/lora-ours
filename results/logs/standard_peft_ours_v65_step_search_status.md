# Standard PEFT Ours v65 Step Search Status

- Updated: `2026-07-04T06:55:15`
- State: `completed_all_zero`
- Decision: `no SIGTERM, but all tested dbpedia/amazon eval scores remained zero`
- Best safe candidate: `t20_e16_accum4`
- Best safe train-cap candidate: `t28_e16_accum8`
- First non-zero candidate: `n/a`

## Candidate Results

- `t16_e16_accum8` / `standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v65_step_search_t16_e16_accum8`
  - state: return `0`, sigterm `False`, all_zero `True`, nonzero `False`
  - train/eval/accum: `16` / `16` / `8`
  - batches/optimizer_steps: `4` / `1`
  - elapsed: `35.0` sec
  - max cuda allocated/reserved: `5180.12744140625` / `6476.0` MB
  - score_matrix: `[[0.0, None], [0.0, 0.0]]`
- `t20_e16_accum8` / `standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v65_step_search_t20_e16_accum8`
  - state: return `0`, sigterm `False`, all_zero `True`, nonzero `False`
  - train/eval/accum: `20` / `16` / `8`
  - batches/optimizer_steps: `5` / `1`
  - elapsed: `34.0` sec
  - max cuda allocated/reserved: `5180.12744140625` / `6550.0` MB
  - score_matrix: `[[0.0, None], [0.0, 0.0]]`
- `t24_e16_accum8` / `standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v65_step_search_t24_e16_accum8`
  - state: return `0`, sigterm `False`, all_zero `True`, nonzero `False`
  - train/eval/accum: `24` / `16` / `8`
  - batches/optimizer_steps: `6` / `1`
  - elapsed: `36.0` sec
  - max cuda allocated/reserved: `5180.12744140625` / `6518.0` MB
  - score_matrix: `[[0.0, None], [0.0, 0.0]]`
- `t28_e16_accum8` / `standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v65_step_search_t28_e16_accum8`
  - state: return `0`, sigterm `False`, all_zero `True`, nonzero `False`
  - train/eval/accum: `28` / `16` / `8`
  - batches/optimizer_steps: `7` / `1`
  - elapsed: `38.0` sec
  - max cuda allocated/reserved: `5890.6884765625` / `7740.0` MB
  - score_matrix: `[[0.0, None], [0.0, 0.0]]`
- `t20_e16_accum4` / `standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v65_step_search_t20_e16_accum4`
  - state: return `0`, sigterm `False`, all_zero `True`, nonzero `False`
  - train/eval/accum: `20` / `16` / `4`
  - batches/optimizer_steps: `5` / `2`
  - elapsed: `34.0` sec
  - max cuda allocated/reserved: `5180.12744140625` / `6550.0` MB
  - score_matrix: `[[0.0, None], [0.0, 0.0]]`

## CPU-Only Audit

- ARPER official v66 remains the current GPU owner and appears healthy; do not kill it and do not launch Standard v66 until `nvidia-smi --query-compute-apps` is empty.
- v65 all-zero is not primarily a label verbalizer or target mapping failure. The same dbpedia labels and scoring path are used in v61, where eval_debug generates exact labels such as `Office Holder` and `Album`.
- v65 predictions are untrained-format outputs (`True`, `False`, or copied title/text fragments), with `bad_prefix_mismatch` on all audited examples.
- v65 training was too short for learning signal: only `3-7` batches and `1-2` optimizer steps, mean batch acc `0.0`, answer-token acc about `0.31-0.36`, and loss about `5.7`. By contrast, v61 earlygate used full caps with `1750/782` batches, `219/98` optimizer steps, answer-token acc `0.99/0.84`, and strong eval scores.
- Proposed Standard v66 direction: GPU-exclusive tiny learning-signal probe with more effective optimizer steps under the safe memory envelope, likely via smaller accumulation and/or repeated micro-epochs on a small train cap; keep eval cap small, enable teacher-forced eval/debug on a tiny sample, and consider temporarily reducing replay/overlap pressure for the first segment-learning probe.
