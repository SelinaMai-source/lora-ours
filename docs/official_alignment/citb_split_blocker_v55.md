# CITB InstrDialog Split Determination

Updated: 2026-07-05

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

For CCFA-facing claims, CITB must stay a setting blocker / appendix audit item,
not a main SOTA claim, until an upstream official `500/50/100` path is found.
The completed local run may be cited only as an official-script `500/50/50`
reproduction smoke.

## Evidence

The paper text says InstrDialog uses a fixed `500/50/100` train/dev/test split
and InstrDialog++ uses `100/50/100`. However, the released official code and
released score metadata point to smaller public script settings:

- `scripts/short_stream_scripts/run_cit_ft_instr.sh` sets
  `max_num_instances_per_task=500` and `max_num_instances_per_eval_task=50`.
- Other short-stream runners (`run_cit_l2.sh`, `run_cit_ewc.sh`,
  `run_cit_agem*.sh`, `run_cit_replay*.sh`, `run_cit_ft_no_instr.sh`, and
  `run_cit_adaptercl_200.sh`) also pass `max_num_instances_per_eval_task=50`.
- `scripts/data_scripts/prepare_cl_dialogue_data.sh` uses the same values.
- `continual_learning/utils.py` states dev and test are both set by
  `max_num_instances_per_eval_task`.
- Official `scores/continual_instruction_tuning/stream=cl_dialogue_tasks/CL=FT_INSTR/scores.json`
  reports `average_train_samples=411.5`, matching the released `500/50/50`
  split, not the paper-text `500/50/100` split.
- The public long-stream scripts use `max_num_instances_per_task=100` and
  `max_num_instances_per_eval_task=25`, so the released runnable path is
  effectively `100/25/25`, not paper-text `100/50/100`.
- The archived submodule path `official_repos/citb/CITB` is a pinned gitlink
  archive pointer; the clean full checkout audited here is
  `/root/autodl-tmp/CITB` at `bf50533b5bced4c388691ecc75e26773da96b3fd`.

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

Local task counts in the official `data/tasks` release also make strict
InstrDialog++ `100/50/100` impossible for nine order1 tasks:

| Task | Total instances | Train after 50/100 | Train after 25/25 |
| --- | ---: | ---: | ---: |
| `task1549_wiqa_answer_generation_missing_step` | 92 | 0 | 42 |
| `task639_multi_woz_user_utterance_generation` | 178 | 28 | 100 |
| `task766_craigslist_bargains_classification` | 200 | 50 | 100 |
| `task459_matres_static_classification` | 98 | 0 | 48 |
| `task1590_diplomacy_text_generation` | 158 | 8 | 100 |
| `task1427_country_region_in_world` | 237 | 87 | 100 |
| `task1713_convai3_sentence_generation` | 182 | 32 | 100 |
| `task1151_swap_max_min` | 200 | 50 | 100 |
| `task1607_ethos_text_classification` | 171 | 21 | 100 |

The long-stream average training sample count is:

- Public script `100/25/25`: `91.82`.
- Paper-text `100/50/100`: `83.58`.

## Public Upstream Search

README and public web checks on 2026-07-05 found no hidden release, separate
download, issue note, or alternate script that provides an official
`500/50/100` InstrDialog path or `100/50/100` InstrDialog++ path. The README
points to the repository `data/` folder and `scripts/data_scripts/`; the paper
and arXiv/ACL pages continue to state the larger split.

## Reporting Rule

Use these labels:

- `CITB official-script reproduction`: released code/scores setting
  `500/50/50`.
- `CITB paper-text diagnostic`: patched `500/50/100`; not official-comparable
  because the released data cannot provide 500 train instances for all tasks.
- `CITB appendix/blocker`: preferred CCFA treatment until upstream publishes an
  official paper-setting split/data artifact.

This is a setting blocker, not a Hugging Face token or network blocker.
