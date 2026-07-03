# CCFA Experiment Gate For CITB Official-Base Runs

Updated: 2026-07-03

## Source And Scope

This gate adapts the already reviewed external skill candidates for the local
`lora-ours` experiment workflow. It does not install or enable external skills,
hooks, MCP servers, npm packages, dashboards, or autonomous loops.

- Reviewed source: `docs/skills_review.md`.
- CCFA references used as checklist ideas only: `ccf-common`,
  `ccf-experiment-designer`, `ccf-integrity-auditor`, and
  `ccf-submission-checker` from the pinned `mikubaka88/CCFA-Skills` candidate.
- Auto-research reference used as a negative control: its loop, auto-commit, and
  auto-revert behavior are not allowed to manage live GPU, W&B, or git state.

## Required Gate Before Launch

- Official path: use the official CITB Stage-2 FT_INSTR entrypoint
  `continual_learning/run_continual_instruct_tuning.py` through the tracked local
  `citb_official` tree whose Tk-Instruct collator supports `add_task_id`.
- Dataset/split disclosure: record whether the run is script-strict
  `500/50/50` or paper-target `500/50/100`. Do not mix the two in tables.
- Comparability boundary: a script-strict `500/50/50` smoke can validate the
  official script path, environment, collator, W&B, and GPU flow, but it is not
  the final paper-comparable InstrDialog `500/50/100` result.
- Early stop and monitoring: every GPU run must have a tmux session, a status
  monitor, and a clear stop criterion or post-launch audit point.
- W&B: online mode is required for real smoke/full runs unless explicitly
  documented as a dry-run. Run names must encode the split policy.
- Logs and artifacts: keep large outputs under `/root/autodl-tmp`; commit only
  small launchers, configs, status docs, and parsers.
- RED FLAG tracking: any tokenizer, collator, split-policy, local patch, or
  official-script mismatch must stay visible in status docs and W&B notes.
- Runtime shims: project-local shims may only bridge cache/runtime gaps, must be
  narrowly scoped, and must be documented as compatibility RED FLAGs. They must
  not change training data, split policy, prompts, metrics, or model updates.
- Integrity rule: never fabricate metrics, W&B IDs, result files, convergence
  status, or comparisons. Mark missing values as pending.

## Current CITB Official-Base Decision

- Launch allowed now if GPU is idle: official-script smoke using
  `500` train, `50` dev, and `50` per-task test via the unmodified short-stream
  Stage-2 split parameter.
- Required run-name marker: `official_script_500_50_50`.
- Claim boundary: report it only as an official-script reproduction smoke, not
  as the final paper-aligned InstrDialog result.

## Strict 500/50/100 Follow-Up

The paper-target strict run remains blocked until the split policy is patched
and audited. The follow-up should:

1. Separate dev and test caps in the official Stage-2 data loading path instead
   of overloading `max_num_instances_per_eval_task`.
2. Preserve official task order, SuperNI Stage-1 checkpoint, T5-small LM-adapted
   tokenizer/model family, prompt construction, metrics, and seeds.
3. Add a no-GPU preflight that proves train/dev/test counts are `500/50/100` for
   InstrDialog where data is available, and records short-task exceptions.
4. Run a 1-step dry-run before any full strict launch.
5. Label any local code patch as `paper_target_500_50_100_strict` and keep it
   separate from `official_script_500_50_50` results.
