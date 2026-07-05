# Official Reproducibility Gate Follow-up

- Updated: `2026-07-05T21:29+08:00`
- Scope: LB-CL code source search, CITB released-score audit, ToDCL minimal smoke readiness.

## LB-CL

- Search terms used:
  - `"Learn more, but bother less" LB-CL code GitHub`
  - `"LB-CL" "Learn more, but bother less" GitHub`
  - `"ZxtaNh5UYB" "LB-CL" code`
- Result: found paper/OpenReview/NeurIPS pages and downstream citations, but no official or author code repository.
- Local GitHub CLI blocker: `gh` is not installed, so local GitHub API search could not run.
- OpenReview direct fetch blocker: browser verification page.
- Decision: keep LB-CL as paper-only published baseline until official/author code is found.

## CITB

- Official released scores checked:
  - `scores/continual_instruction_tuning/stream=cl_dialogue_tasks/CL=FT_INSTR/scores.json`
  - `scores/continual_instruction_tuning/stream=cl_dialogue_tasks/CL=REPLAY/scores.json`
- Finding:
  - FT_INSTR order1 ROUGE-L reports `average_train_samples=411.5`, supporting the short-task/available-instance explanation.
  - REPLAY order1 ROUGE-L reports `average_train_samples=5855.5`, consistent with replay-expanded training.
- Decision: this explains why public official scores are not literal 500 current-task samples, but it does not resolve the public script `500/50/50` vs paper `500/50/100` ambiguity. Keep blocker open.

## ToDCL

- Data paths checked under `/root/autodl-tmp/lora-baselines-run_v1/external_sources/todcl/data`.
- Current missing data: `dstc8-schema-guided-dialogue`, `Taskmaster`, `multiwoz`.
- Minimal no-training smoke attempted: `python train.py --help`.
- Smoke result: failed before argument parsing with `ModuleNotFoundError: No module named 'pytorch_lightning'`.
- Decision: ToDCL is downloaded but not runnable; requires isolated legacy environment and official data download/preprocess before ours can use it as a base.
