# CITB Official Base Repro Status

- Updated: `2026-07-03T16:43:34`
- Run: `citb_instrdialog_order1_seed1_paper_target_500_50_100_tie_fixed_ft_instr_stage1_v54`
- State: `stopped_incomplete`
- Output dir: `/root/autodl-tmp/citb_official_base_repro/citb_instrdialog_order1_seed1_paper_target_500_50_100_tie_fixed_ft_instr_stage1_v54`
- Result dirs: `1`
- Expected tasks: `19`
- Train processes: `0`
- Latest result dir: `/root/autodl-tmp/citb_official_base_repro/citb_instrdialog_order1_seed1_paper_target_500_50_100_tie_fixed_ft_instr_stage1_v54/results/0_task848_pubmedqa_classification`
- Latest activity: `/root/autodl-tmp/citb_official_base_repro/citb_instrdialog_order1_seed1_paper_target_500_50_100_tie_fixed_ft_instr_stage1_v54/checkpoint-63/rng_state.pth`
- Latest activity age seconds: `73.03696131706238`

## Latest Metrics
- No per-task metrics yet.

## RED FLAG
- CITB paper states InstrDialog uses 500/50/100 train/dev/test instances, but the official short-stream FT_INSTR script sets max_num_instances_per_eval_task=50.
- Official CITB Stage-2 code uses one max_num_instances_per_eval_task for both dev and per-task test split size, yielding script-strict 500/50/50 unless split code is patched.
- Default launcher keeps the official-script 500/50/50 smoke policy; requested 500/50/100 remains blocked unless split code is patched and audited.
- Official dry-run currently requires the lora_v10_citb Python 3.9 environment; base Python lacks datasets.load_metric.
- Stage-1 tokenizer files are incompatible with the old official stack, so the launcher uses the local google__t5-small-lm-adapt tokenizer as a compatibility override.
- Tk-Instruct metric code imports AutoTokenizer.from_pretrained('gpt2'); launcher redirects only that tokenizer lookup to a local GPT-2 tokenizer cache via project-local sitecustomize because this environment cannot reach Hugging Face.
- The hyintell/CITB checkout has an untracked Tk-Instruct copy whose collator lacks add_task_id; launcher now defaults to the tracked local citb_official tree whose collator matches the CL entrypoint.
