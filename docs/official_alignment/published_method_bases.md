# Published-Method Base Roadmap

Updated: 2026-07-03

This note records the v52 decision for requirement 2.2: every experiment must first stand on a published-method base, then add the RP/Ours mechanisms only through the smallest auditable code path. Do not claim comparability when the official setting is not matched.

## Current Local State

- Active branch for this note: `ours-v52-published-base-roadmap`.
- Ours v49/v50/v51 status:
  - v49 InstrDialog++ prefix-3 smoke completed. It proved the converted stream and task-aware scoring are not globally broken, but ASDiv and CNN/DailyMail were still weak.
  - v50 InstrDialog++ skip-empty6 smoke loaded the full order1 stream with explicit zero-data segment skipping and stopped by `hard_early_stop_min_seen_avg`; segment1 stayed healthy, ASDiv/CNN-DailyMail/MultiWOZ remained blockers.
  - v51 decode-calibration smoke is already recorded as stopped by the same early-stop gate. Do not repeat it or start another GPU run while parent/v51 workers are active.
- RP(Lora) material state:
  - RP v2 plan requires paper-level artifacts, drift/router/bank/overlap metrics, full matrix, and benchmark alignment.
  - RP v3 artifact builder exists in `scripts/build_rp_lora_v3.py`, but it is an artifact/report path, not an official published-method reproduction base.
  - Implemented Ours modules are available in `core/methods/` and the seq2seq LoRA wrapper: drift detector, LoRA bank, learned/prototype router, spectral replay, overlap loss, task-aware evaluation, and seq2seq PEFT support.

## Suite A: CITB

Published base: **CITB official repo / Tk-Instruct protocol**.

Official setting to reproduce first:

- Source: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/citb`, official CITB repository.
- Model: `google/t5-small-lm-adapt`, then CITB Stage-1 100-SuperNI initial multitask checkpoint.
- Streams: official `cl_dialogue_tasks` for InstrDialog and `cl_dialogue_long_tasks` for InstrDialog++.
- Metrics: CITB score matrix fields through official/equivalent AR, FWT, BWT, plus Tinit/Tunseen probes when Stage-1 is finalized.

Current readiness:

- Local LM-adapted T5-small and a Stage-1 checkpoint path exist: `/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469`.
- Ours seq2seq config path is already using the Stage-1 checkpoint for v49-v51 CITB smokes.
- Strict runner lineage exists through v36b/v50/v51 configs and status logs.

Blocking comparability notes:

- **RED FLAG: InstrDialog++ `100/50/100` is not script-strict to official long-stream eval, whose script records eval count 25.** It can be a CCF-A aligned variant, but not an official CITB long-stream reproduction unless the paper text/code evidence supports that split.
- **RED FLAG: skip-empty policy is explicit and non-synthetic, but changes the effective long-stream task count.** Report skipped official zero-data tasks and do not compare as if all 38 tasks were trained/evaluated.
- **RED FLAG: v36b full, v50, and v51 are not SOTA-ready results.** They are diagnostic smokes with early-stop/low-score gates.

Minimum code path for Ours on top of published base:

1. Reproduce/verify the CITB official Stage-1 checkpoint and baseline score collection on InstrDialog order1.
2. Run the official-equivalent sequential/Tk-Instruct base with unchanged split and metric parser.
3. Add only Ours modules through the existing config switches: `use_drift_detector`, `use_lora_bank`, `use_router`, `use_overlap_loss`, and LoRA PEFT settings.
4. Keep generation calibration as an ablation unless it is also applied to the reproduced published base.

First runnable no-GPU action:

```bash
python scripts/preflight_published_method_bases.py --out results/logs/published_method_bases_preflight.json
```

Next GPU action when clear:

- First priority is **CITB InstrDialog order1 official-base reproduction** with the Stage-1 checkpoint and official split, not another v51 InstrDialog++ decode-calibration rerun.

## Suite B: Standard T5-Large PEFT CL

Published base: **O-LoRA official standard T5-large continual-learning setup**.

Official setting to reproduce first:

- Source: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/o_lora`.
- Model: `initial_model/t5-large`, locally mapped to `/root/autodl-tmp/model_cache/hf_snapshots/t5-large`.
- Order1: `dbpedia -> amazon -> yahoo -> agnews`; order2/order3 follow official scripts.
- Main hyperparameters from official `scripts/order_1.sh`: one epoch, train batch 8, eval batch 128, LR `1e-3`, max source 512, target/generation length 50, `lamda_1=0.5`, `lamda_2=0`.
- Metrics: final average accuracy / forgetting / BWT from per-task prediction matrices.

Current readiness:

- Local O-LoRA source, CL_Benchmark conversion, order1/2/3 streams, and T5-large snapshot exist.
- Ours configs for order1/2/3 are present under `configs/ccfa_three_suite/`.

Blocking comparability notes:

- **RED FLAG: an Ours runner on converted O-LoRA streams is not the same as running official O-LoRA.** First reproduce or at least preflight the official O-LoRA script/output parser, then layer Ours.
- **RED FLAG: LFPT5 and Progressive Prompts are evidence/backup bases here, not the selected primary base.** LFPT5 needs LM-adapted T5-large conversion; Progressive Prompts uses a different prompt-tuning protocol and task regime.
- **RED FLAG: any local memory knobs such as `activation_batch_size=1` must be recorded as engineering changes, not paper hyperparameters.**

Minimum code path for Ours on top of published base:

1. Freeze the official O-LoRA standard stream/task order and T5-large checkpoint.
2. Build a preflight manifest from official `scripts/order_{1,2,3}.sh`.
3. Run the official O-LoRA base or an explicitly named official-equivalent runner.
4. Add Ours modules in the existing seq2seq PEFT runner while keeping task order, checkpoint, and metric parser unchanged.

Next GPU action:

- Run after CITB base reproduction or when a separate GPU is free. Do not run alongside CITB/v51 workers on the single GPU.

## Suite C: Dialogue NLG / MultiWOZ

Published base: **ARPER official WOZ3 continual NLG** as primary; **ToDCL AdapterCL NLG** as backup once processed streams are available.

Official setting to reproduce first:

- ARPER source: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/arper`.
- ARPER data: `resource/woz3` unique-domain / unique-dialogue-act splits.
- ARPER official model: SCLSTM (`model_type=lm`, `dec_type=sclstm`) with exemplar size 250.
- Metrics: official/equivalent BLEU-4 and SER (`redunt`, `miss`, `total`), plus forgetting over the continual matrix.
- ToDCL source: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/todcl`; task type `NLG`; metrics BLEU/EER.

Current readiness:

- ARPER WOZ3 dialogue-act stream and official-equivalent scorer adapter are present.
- Path B wrapper exists for official SCLSTM reproduction.
- Path A Ours T5 seq2seq config is ready but is not architecture-strict to ARPER.
- ToDCL source exists, and its README/results document AdapterCL NLG BLEU/EER, but local processed TOD37/MultiWOZ stream is not ready.

Blocking comparability notes:

- **RED FLAG: ARPER Path A Ours T5 is not the official ARPER SCLSTM architecture.** It is useful only after Path B establishes the published base or if reported as an architecture-adapted variant.
- **RED FLAG: ToDCL cannot be claimed ready until `data/download.sh` and preprocessing/export produce the 37-domain stream locally.**
- **RED FLAG: MultiWOZ-only NLG and ARPER WOZ3 are related dialogue NLG settings, but they are not interchangeable baselines.**

Minimum code path for Ours on top of published base:

1. Run/preflight ARPER official SCLSTM Path B on WOZ3 DA or domain setting.
2. Validate official BLEU/SER scorer on generated `.res` outputs.
3. Run Ours T5/LoRA Path A only as an adaptation layer over the same stream and scorer.
4. Prepare ToDCL only after full data download/preprocess/export exists; then compare AdapterCL before adding Ours.

Next GPU/CPU action:

- If GPU is busy, ARPER official SCLSTM/CPU preflight and scorer validation are the safest next actions.
- If a training slot opens after CITB, prioritize ARPER Path B before reporting any Ours T5 Dialogue result.

## Global Guardrails

- Every table row must name its base as either `official reproduction`, `official-equivalent port`, or `adapted variant`.
- A result is not comparable if model backbone, stream split, task order, metric, or architecture differs without a separate ablation label.
- Do not fill missing official results with Ours-runner numbers.
- Do not launch more GPU work until the parent worker confirms no v51/CITB process needs the single GPU.
