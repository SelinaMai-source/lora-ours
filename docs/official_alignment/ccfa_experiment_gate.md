# CCFA Experiment Gate For Published-Base Runs

Updated: 2026-07-04

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

## Published-Base + Ours Overlay Gate

- Base first: every Ours overlay must name the completed published-method base
  it is layered on. For v58 this is `O-LoRA official T5-large Standard CL
  order1 seed1`, using the same official task order and metric surface as v57.
- Separate labels: keep `official-base`, `official-equivalent single-GPU port`,
  and `ours-overlay` as different result types. Do not move an overlay score into
  the official-base row.
- Minimal overlay: change only one auditable mechanism per round. For v58 the
  only overlay is limited prior-task replay in training `train_tasks.json`;
  current-round `dev_tasks.json` and `test_tasks.json` remain copied unchanged
  from the official O-LoRA order1 configs.
- Run naming: overlay run names must include the base method, Ours overlay, task
  setting, and version. Example:
  `standard_olora_official_base_plus_ours_replay_overlay_order1_seed1_smoke_v58`.
- Comparability boundary: smoke caps, single-GPU runtime copies, W&B shims, or
  overlay training data changes are RED FLAGs for paper comparability. Record
  them in `docs/official_alignment/status.md`, run manifests, and W&B notes.
- Commit hygiene: commit only small scripts/config/docs. Do not commit generated
  `results/runs/`, large logs, `wandb/`, `__pycache__/`, or model artifacts.

## Standard PEFT Ours Earlygate

- v61 scope: run only the `dbpedia -> amazon` earlygate before any full Standard
  PEFT Ours formal retry. Preserve the official order, converted-stream mapping,
  T5-large checkpoint, LR `1e-3`, one epoch, batch/effective batch, 512/50/50
  length settings, and final average accuracy/forgetting/BWT metrics.
- v61 allowed change: an Ours-runner retention diagnostic only. Disable the
  prompt-NLL arbitration path that v60 debug showed was branch-inverting both
  tasks, and enable label-balanced prior-task replay inside the existing replay
  buffer. Do not change eval prompts, label verbalizers, task order, scorer, or
  postprocess metric export.
- v61 gate: after segment1, dbpedia retention and amazon current must clearly
  beat v60 (`dbpedia=0.1517`, `amazon=0.0233`, seen average `0.0875`,
  BWT `-0.8332`) before launching any full four-task Standard PEFT Ours run.
- v61 decision: `standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v61_earlygate`
  cleared this gate with dbpedia retention `0.9825`, amazon current `0.4517`,
  seen average `0.7171`, forgetting `0.0024`, and BWT `-0.0024`. A full
  four-task follow-up is allowed as a diagnostic only; keep the adapted-Ours
  label and do not mix it into official O-LoRA base rows.

## Current CITB Official-Base Decision

- Previous official-script smoke `citb_instrdialog_order1_seed1_official_script_500_50_50_ft_instr_stage1_v53`
  is failed, not a baseline: it was stopped at 11/19 result dirs after saved
  predictions showed repeated garbage tokens caused by old-transformers loading
  the Stage-1 T5 v1.1 checkpoint with `tie_word_embeddings=true`.
- Relaunch is allowed only with a fresh run name after the launcher runtime-config
  override (`tie_word_embeddings=false`) passes dry-run/preflight. Keep the
  failed v53 W&B run visible as a reproduction failure, not a method score.
- Launch scope remains official-script smoke using `500` train, `50` dev, and
  `50` per-task test via the unmodified short-stream Stage-2 split parameter.
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
6. Keep `AUTO_LAUNCH_FORMAL=0` unless a human explicitly enables the strict run;
   the monitor must not auto-promote a failed or stopped-incomplete smoke.

Implementation note: `scripts/preflight_citb_official_split_counts.py` now
provides the no-GPU count audit for the official task order and raw task JSONs.
It should be run before strict launch and its JSON output kept under
`results/logs/`; if any task is short, report the exception rather than
synthesizing examples.
