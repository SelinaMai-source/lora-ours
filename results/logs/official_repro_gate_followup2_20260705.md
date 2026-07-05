# Official Repro Gate Follow-up 2 - 2026-07-05

Scope: documentation and gate audit only. No new Ours candidate and no GPU training job were launched.

## Branch Context

- Official code archive branch: `official-method-code-archive-20260705`
- Archive commit: `b060357`
- Follow-up documentation branch: `official-repro-gate-followup-20260705`
- Archive manifest: `official_repos/OFFICIAL_REPOS.md`

## LB-CL Code Source

- Rechecked with:
  - `"Learn More, but Bother Less" "code" "Fuli Qiao"`
  - `"LB-CL" "github" "Fuli Qiao" "Mahdavi"`
- Results again point to paper/index/slides pages, not an official repository.
- Status remains `paper_only_baseline`; no LB-CL submodule is added.

## CITB Split Blocker

New code-level evidence:

- `continual_learning/utils.py::train_dev_test_split_by_task` states that dev and test counts are both controlled by `max_num_instances_per_eval_task`.
- The function assigns `test=instances[:N]` and `dev=instances[N:2N]`.
- Public scripts pass `max_num_instances_per_eval_task=50`:
  - `scripts/run_initial_multitask_tuning_with_CL.sh`
  - `scripts/eval_model.sh`
  - `scripts/short_stream_scripts/run_cit_ft_instr.sh`
  - `scripts/short_stream_scripts/run_cit_l2.sh`
  - `scripts/short_stream_scripts/run_cit_ewc.sh`
  - `scripts/short_stream_scripts/run_cit_agem*.sh`
  - `scripts/short_stream_scripts/run_cit_replay*.sh`
  - `scripts/data_scripts/prepare_cl_dialogue_data.sh`

Conclusion: public runnable path remains script-strict `500/50/50`; paper `500/50/100` is not yet reproduced from public scripts.

## ToDCL Data And Env

- GitHub repo API approximate sizes:
  - SGD: `51095 KB`
  - Taskmaster: `111002 KB`
  - MultiWOZ: `125780 KB`
- `/root/autodl-tmp` has enough capacity.
- Data download attempts:
  - `git clone --depth 1` for SGD failed with `GnuTLS recv error` / `early EOF`.
  - codeload SGD archive attempt produced a `23068672` byte file, but `zipfile` reports `BadZipFile`.
  - codeload HEAD returns `application/zip` without `Content-Length`, so downloads need post-download zip validation.
- Legacy env audit:
  - Entry smoke venv still passes `train.py --help`.
  - `pip install --dry-run torch==1.4.0 transformers==3.5.1` cannot resolve `torch==1.4.0` from the configured Python 3.9 package index.

Conclusion: ToDCL is beyond import blocker but still not a data-loader/training smoke. Remaining blockers are stable verified data download/layout and faithful legacy runtime.

## Gate Decision

- No new Ours candidates are allowed from this follow-up.
- Next safe actions:
  - LB-CL: continue only official/author/supplement code search; do not use third-party code.
  - CITB: either locate a public official `500/50/100` path or explicitly limit claims to script-strict `500/50/50`.
  - ToDCL: implement resumable verified data download into `/root/autodl-tmp`, then run data-loader smoke only.
