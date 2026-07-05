# CITB Official Repo / Code Status

- Updated: `2026-07-05T21:25+08:00`
- Official repo: `https://github.com/hyintell/CITB`
- Local checkouts:
  - `/root/autodl-tmp/lora-baselines-run_v1/external_sources/citb`
  - `/root/autodl-tmp/CITB`
- Commit: `bf50533b5bced4c388691ecc75e26773da96b3fd`
- Repo cleanliness:
  - `/root/autodl-tmp/CITB`: clean
  - `/root/autodl-tmp/lora-baselines-run_v1/external_sources/citb`: dirty due to `continual_learning/run_initial_multitask_tuning.py` and untracked `Tk-Instruct/`; do not treat as pristine without diff audit.

## Official Assets

- Environment used locally: `/root/autodl-tmp/conda_envs/lora_v10_citb` for official stack compatibility.
- LM-adapted base: `/root/autodl-tmp/model_cache/hf_snapshots/google__t5-small-lm-adapt`.
- Stage-1 SuperNI checkpoint: `/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469`.
- Official scripts:
  - Stage 1: `scripts/run_initial_multitask_tuning.sh`
  - Stage 2 short stream: `scripts/short_stream_scripts/meta_job.sh`
  - Stage 2 long stream: `scripts/long_stream_scripts/meta_job.sh`
  - Results collection: `collect_results.py`, `scripts/score_scripts/*.sh`
- Official data/order files:
  - InstrDialog: `data/CIT_data/task_orders/stream=cl_dialogue_tasks/order1.txt`
  - InstrDialog++: `data/CIT_data/task_orders/stream=cl_dialogue_long_tasks/order1.txt`

## Paper vs Runnable Setting

- Paper setting to verify: InstrDialog 19 tasks with `500/50/100`; InstrDialog++ 38 tasks with `100/50/100`; 3 seeds; Replay/AGEM memory variants.
- Local official-script reproduction currently validated only for InstrDialog short stream under script-strict `500/50/50`.
- Blocker: official Stage-2 code uses one `max_num_instances_per_eval_task` for both dev and per-task test, so unpatched official script yields `500/50/50`, not paper `500/50/100`.
- Additional blocker: some InstrDialog++ tasks are short or empty under requested split counts, e.g. observed `train=0/dev=0/test<100` tasks in the long stream audit.
- Official released scores provide partial explanation for the mismatch:
  - `scores/continual_instruction_tuning/stream=cl_dialogue_tasks/CL=FT_INSTR/scores.json` reports order1 ROUGE-L `average_train_samples=411.5`, not 500, which supports the short-task / available-instance constraint.
  - `scores/continual_instruction_tuning/stream=cl_dialogue_tasks/CL=REPLAY/scores.json` reports order1 ROUGE-L `average_train_samples=5855.5`, consistent with replay-expanded training rather than simple 500 examples per current task.
  - These released scores help explain why the actual official results are not a literal `500` current-task-only count, but they do not by themselves resolve the dev/test `50/100` script ambiguity.

## Existing Reproduction Evidence

- Completed official-script run: `citb_instrdialog_order1_seed1_official_script_500_50_50_tie_fixed_ft_instr_stage1_v54`.
- Status file: `results/logs/citb_official_script_500_50_50_tie_fixed_status.md`.
- Output dir: `/root/autodl-tmp/citb_official_base_repro/citb_instrdialog_order1_seed1_official_script_500_50_50_tie_fixed_ft_instr_stage1_v54`.
- Result dirs: `19/19`.
- Latest aggregate metrics from status:
  - `predict_official_exact_match=21.0756`
  - `predict_official_rougeL=33.109`
  - `predict_official_samples=2975`
  - `predict_initial_multi_exact_match=27.84`
  - `predict_initial_multi_rougeL=38.0874`

## Gate Decision

- CITB is not cleared for new ours increments until the official paper-comparable split path is resolved or explicitly downgraded to script-strict `500/50/50`.
- Any future ours result must state whether it is paper `500/50/100` comparable, official-script `500/50/50` comparable, or only diagnostic.
- Current status: keep blocker open; prepare only non-expensive audits/smokes until a released official `500/50/100` path is located or the paper claim is explicitly treated as non-reproducible from public scripts.
