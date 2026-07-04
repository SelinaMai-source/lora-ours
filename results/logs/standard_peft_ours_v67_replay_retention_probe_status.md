# Standard PEFT Ours v67 Replay-Retention Probe Status

- Updated: `2026-07-04T09:33:37`
- State: `completed_no_gain`
- Decision: `no candidate improved over the v66 no-replay baseline`
- Baseline final avg/task-aware acc: `0.4375`
- Best candidate: `t20_e16_ep3_accum2_noreplay_repeat`

## Candidate Results

- `t20_e16_ep3_accum2_noreplay_repeat` / `standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v67_replay_retention_t20_e16_ep3_accum2_noreplay_repeat`
  - state: return `0`, sigterm `False`, nonzero `True`
  - replay ratio/min: `0.0` / `0`
  - final avg/task-aware: `0.4375` / `0.4375`
  - dbpedia retention / amazon current / seen avg: `0.5` / `0.375` / `0.4375`
  - elapsed: `34.4` sec; max cuda allocated/reserved: `5205.015625` / `5808.0` MB
  - score_matrix: `[[0.5625, None], [0.5, 0.375]]`
- `t20_e16_ep3_accum2_v61_label_balanced_replay` / `standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v67_replay_retention_t20_e16_ep3_accum2_v61_label_balanced_replay`
  - state: return `0`, sigterm `False`, nonzero `True`
  - replay ratio/min: `0.25` / `64`
  - final avg/task-aware: `0.4375` / `0.4375`
  - dbpedia retention / amazon current / seen avg: `0.4375` / `0.4375` / `0.4375`
  - elapsed: `37.6` sec; max cuda allocated/reserved: `5205.015625` / `5810.0` MB
  - score_matrix: `[[0.5625, None], [0.4375, 0.4375]]`

## Next Gate

- Record replay/current-task trade-off; avoid 4-task extension until the two-task average improves.
