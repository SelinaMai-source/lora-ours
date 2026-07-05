# ToDCL Bounded Method-Level Smoke - 2026-07-05

Scope: official reproducibility gate only. No Ours candidate was launched.

## Official Anchor

- Official repo: `https://github.com/andreamad8/ToDCL`
- Commit: `e70c1edf937f6eb570296ea2897dbc8d6815bc6d`
- Official README example used as anchor:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --CL VANILLA --task_type NLG --n_epochs 1
```

## Bounded Smoke Configuration

The smoke keeps the official `train.py` / `train()` / `Seq2SeqToD` / Lightning training path, but bounds compute for gate validation.

- Environment: `/root/autodl-tmp/conda_envs/todcl_legacy_py37`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- CL method: `VANILLA`
- task type: `NLG`
- dataset list: `SGD`
- data mode: `--debug`, ToDCL develop subset
- selected bounded task: `['sgd_alarm']`
- model checkpoint: local symlink `tiny-gpt2-local` -> `/root/autodl-tmp/todcl_official_model_cache/tiny-gpt2-local`
- model source: `sshleifer/tiny-gpt2` files downloaded through proxy into `/root/autodl-tmp`
- batch sizes: train/valid/test `1`
- epochs: `1`
- smoke wrapper: `results/logs/official_todcl_method_smoke_bounded_vanilla_nlg_sgd_debug_20260705.py`
- stdout:
  - `results/logs/official_todcl_method_smoke_bounded_vanilla_nlg_sgd_debug_20260705.log`
  - `results/logs/official_todcl_method_smoke_bounded_vanilla_nlg_sgd_debug_scorable_20260705.log`
  - `results/logs/official_todcl_method_smoke_bounded_vanilla_nlg_sgd_debug_scorer_20260705.log`

Engineering bounds:

- `limit_train_batches=1`
- `limit_val_batches=1`
- one selected continual task
- final generation bounded to one test batch and `max_length=8`
- clean scorer root contains only the successful smoke run, avoiding a stale interrupted folder

## Result

Passed.

Observed evidence:

- ToDCL loaded SGD debug data: train/dev/test `532/78/158`, `10` domains, `17` intents.
- Official model class initialized: `GPT2LMHeadModel`, `102K` parameters.
- Lightning/GPU training path ran one train step and one validation step.
- Checkpoint written and loaded:
  - `runs_NLG/SGD/VANILLA_EM_1_LAMOL_0.2_REG_0.01_PERM_1_tiny-gpt2-local/0_['sgd_alarm']/lightning_logs/version_0/checkpoints/epoch=0-step=0.ckpt`
- Final model/tokenizer save path was created under:
  - `runs_NLG/SGD/VANILLA_EM_1_LAMOL_0.2_REG_0.01_PERM_1_tiny-gpt2-local`
- Bounded final generation wrote:
  - `runs_NLG/SGD/VANILLA_EM_1_LAMOL_0.2_REG_0.01_PERM_1_tiny-gpt2-local/FINAL/generated_responses.json`
  - rows: `1`
- Official `scorer.py` entry ran on a clean smoke root and produced:

```text
| Name    |   BLEU |   EER |
|---------|--------|-------|
| VANILLA |      0 |     1 |
```

The score is not meaningful because this is a one-batch tiny-model smoke. It only verifies the scorer path.

## Resolved Issues During Smoke

- Legacy `transformers==3.5.1` failed against HuggingFace's current resolve-cache URL for direct model loading. Workaround: download tiny GPT-2 files explicitly into `/root/autodl-tmp` and load from a local path.
- An unbounded official `train.py` smoke was interrupted before completion. It created a stale incomplete run folder; this was not used for scoring.
- Lightning `fast_dev_run` did not provide a usable `best_model_path` for ToDCL's checkpoint reload code. Workaround: use explicit batch limits instead.

## Gate Status

- ToDCL is upgraded to `bounded_method_smoke_passed`.
- This is still not an official paper reproduction and not a paper-comparable result.
- Remaining blocker: run an official-scale method reproduction or an official-equivalent anchor for the best ToDCL NLG method, likely `ADAPTER` or `REPLAY`, under documented full settings.
