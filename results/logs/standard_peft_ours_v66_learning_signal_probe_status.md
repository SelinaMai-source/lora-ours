# Standard PEFT Ours v66 Learning-Signal Probe Status

- Updated: `2026-07-04T09:29:44`
- State: `completed`
- Decision: `found non-zero learning signal without SIGTERM`
- First non-zero candidate: `t20_e16_ep3_accum2_noreplay`

## Candidate Results

- `t20_e16_ep3_accum2_noreplay` / `standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v66_learning_signal_t20_e16_ep3_accum2_noreplay`
  - state: return `0`, sigterm `False`, all_zero `False`, nonzero `True`
  - train/eval/epochs/accum: `20` / `16` / `3` / `2`
  - elapsed: `35.4` sec
  - max cuda allocated/reserved: `5205.015625` / `5810.0` MB
  - debug examples / teacher-forced acc mean: `48` / `0.0`
  - score_matrix: `[[0.5625, None], [0.5, 0.375]]`
  - train: `{'segment_000': {'loss': 2.6822916666666665, 'mean_batch_acc': 0.4444444444444444, 'answer_token_acc': 0.6453423334492577, 'optimizer_steps': 6, 'batches': 9}, 'segment_001': {'loss': 2.9296875, 'mean_batch_acc': 0.1111111111111111, 'answer_token_acc': 0.5387426912784576, 'optimizer_steps': 6, 'batches': 9}}`
