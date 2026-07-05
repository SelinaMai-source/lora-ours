# ToDCL ADAPTER / REPLAY Official Anchor Plan - 2026-07-05

Scope: official reproducibility gate only. No Ours candidate was launched.

## Published Base Selection

Source: ToDCL official README tables and commands.

NLG modularized:

- `ADAPTER`: BLEU `21.7719`, EER `0.163975`
- `REPLAY`: BLEU `21.4832`, EER `0.0559855`
- `MULTI`: BLEU `26.1462`, EER `0.0341823`, upper bound rather than continual deployed base

E2E:

- `ADAPTER`: INTENT `0.906857`, JGA `0.35059`, BLEU `16.5768`, EER `0.331949`
- `REPLAY`: INTENT `0.785325`, JGA `0.297534`, BLEU `16.2668`, EER `0.190309`
- `MULTI`: upper bound

Decision:

- Primary NLG base for BLEU: `ADAPTER`.
- Primary NLG base for EER: `REPLAY`.
- Both must be represented in the official gate before any ToDCL Ours overlay.

## Official Commands

ADAPTER NLG:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --task_type NLG \
  --CL ADAPTER \
  --bottleneck_size 50 \
  --lr 6.25e-3 \
  --n_epochs 10 \
  --train_batch_size 10 \
  --gradient_accumulation_steps 8
```

REPLAY NLG:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --task_type NLG \
  --CL REPLAY \
  --episodic_mem_size 50 \
  --lr 6.25e-5 \
  --n_epochs 10 \
  --train_batch_size 8 \
  --gradient_accumulation_steps 8
```

Official scoring path:

```bash
python scorer.py --model_checkpoint <runs_NLG_root> --task_type NLG
```

## Bounded Smoke Results

Shared smoke setup:

- Env: `/root/autodl-tmp/conda_envs/todcl_legacy_py37`
- Data: `SGD`, ToDCL `--debug` subset
- Model: local tiny GPT-2 symlink `tiny-gpt2-local`
- Wrapper: `results/logs/official_todcl_method_smoke_bounded_adapter_replay_nlg_sgd_debug_20260705.py`
- Bounds: one train batch, one val batch, bounded final generation, clean scorer root

ADAPTER bounded NLG smoke:

- Status: passed
- Log: `results/logs/official_todcl_method_smoke_bounded_adapter_nlg_sgd_debug_20260705.log`
- Scorer log: `results/logs/official_todcl_method_smoke_bounded_adapter_nlg_sgd_debug_scorer_20260705.log`
- Evidence:
  - initialized `GPT2Adapter`, `123K` parameters
  - selected task `['sgd_alarm']`
  - checkpoint written and reloaded
  - final `generated_responses.json` written with `1` row
  - official scorer entry returned `ADAPTER BLEU=0, EER=1`

REPLAY bounded NLG smoke:

- Status: passed
- Log: `results/logs/official_todcl_method_smoke_bounded_replay_nlg_sgd_debug_20260705.log`
- Scorer log: `results/logs/official_todcl_method_smoke_bounded_replay_nlg_sgd_debug_scorer_20260705.log`
- Evidence:
  - initialized `GPT2LMHeadModel`, `102K` parameters
  - selected tasks `['sgd_alarm']`, `['sgd_buses']`
  - first task starts with `Memory Size 0`
  - second task starts with `Memory Size 1`, so replay memory is exercised
  - two task checkpoints written and reloaded
  - final `generated_responses.json` written with `1` row
  - official scorer entry returned `REPLAY BLEU=0, EER=1`

The bounded scores are not metrics. They only validate method and scorer paths.

## Official-Equivalent Reproduction Plan

Data:

- Full ToDCL data layout under `/root/autodl-tmp/todcl_official_data_20260705`
- `dataset_list=SGD,TM19,TM20,MWOZ`
- `setting=single`
- full 37-domain loader already passes: train/dev/test `31425/4035/4742`, BYDOMAIN `37/37/37`

Model:

- Official default `model_checkpoint=gpt2`, unless exact paper checkpoint evidence indicates otherwise.
- Cache/model files must live under `/root/autodl-tmp`, not `/root`.

Method priority:

1. ADAPTER NLG full official-equivalent run, because it is the best non-MULTI NLG BLEU row.
2. REPLAY NLG full official-equivalent run, because it is the best non-MULTI NLG EER row.
3. MULTI only as upper-bound/context, not deployed continual base.

Metrics:

- `scorer.py --task_type NLG`
- report BLEU and EER
- compare against README modularized references:
  - ADAPTER BLEU `21.7719`, EER `0.163975`
  - REPLAY BLEU `21.4832`, EER `0.0559855`

Resource estimate and controls:

- GPU is available: vGPU 48GB observed idle during smoke.
- Full ADAPTER/REPLAY use official `n_epochs=10`; expected to be substantially longer than smoke because they cover 37 domains and full GPT-2.
- Run under `tmux` with stdout log, pid/sentinel, and W&B project `lora-ours` only if instrumentation is available without changing official behavior.
- Store checkpoints/runs under `/root/autodl-tmp` or symlinked external storage to avoid `/root` pressure.
- Early blocker detection:
  - fail if GPU OOM, missing full `gpt2`, or scorer cannot process official run folder
  - do not tune method hyperparameters before a faithful official-equivalent anchor is recorded

## Current Gate

- `ADAPTER` and `REPLAY` bounded method smokes passed.
- Official-scale/paper-number reproduction is still pending.
- No ToDCL Ours overlay may start until ADAPTER/REPLAY official reproduction or official-equivalent anchor is documented.
