# CITB InstrDialog Split Determination

Updated: 2026-07-03

## Decision

For strict, reproducible CITB InstrDialog comparisons in this repository, use the
released official short-stream code setting:

- Stream: `cl_dialogue_tasks` (19 tasks)
- Stage-2 cap: `max_num_instances_per_task=500`
- Stage-2 eval cap: `max_num_instances_per_eval_task=50`
- Effective per-task split: up to `500/50/50` train/dev/test, with short tasks using all remaining training instances.

Do not report a local `500/50/100` run as an official-comparable CITB
InstrDialog result unless upstream provides a different data release or split
file that makes the four short tasks satisfy that setting.

## Evidence

The paper text says InstrDialog uses a fixed `500/50/100` train/dev/test split.
However, the released official code and released score metadata point to
`500/50/50` for the short stream:

- `scripts/short_stream_scripts/run_cit_ft_instr.sh` sets
  `max_num_instances_per_task=500` and `max_num_instances_per_eval_task=50`.
- `scripts/data_scripts/prepare_cl_dialogue_data.sh` uses the same values.
- `continual_learning/utils.py` states dev and test are both set by
  `max_num_instances_per_eval_task`.
- Official `scores/continual_instruction_tuning/stream=cl_dialogue_tasks/CL=FT_INSTR/scores.json`
  reports `average_train_samples=411.5`, matching the released `500/50/50`
  split, not the paper-text `500/50/100` split.

Local task counts in the official `data/tasks` release make strict
`500/50/100` impossible for four order1 tasks:

| Task | Total instances | Train after 50/50 | Train after 50/100 |
| --- | ---: | ---: | ---: |
| `task1590_diplomacy_text_generation` | 158 | 58 | 8 |
| `task639_multi_woz_user_utterance_generation` | 178 | 78 | 28 |
| `task1713_convai3_sentence_generation` | 182 | 82 | 32 |
| `task766_craigslist_bargains_classification` | 200 | 100 | 50 |

The released short-stream average training sample count is:

- `500/50/50`: `411.47`, matching official scores rounded to `411.5`.
- `500/50/100`: `400.95`, not matching official scores.

## Reporting Rule

Use these labels:

- `CITB official-script reproduction`: released code/scores setting
  `500/50/50`.
- `CITB paper-text diagnostic`: patched `500/50/100`; not official-comparable
  because the released data cannot provide 500 train instances for all tasks.

This is a setting blocker, not a Hugging Face token or network blocker.
