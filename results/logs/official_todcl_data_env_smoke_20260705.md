# ToDCL Official Data / Env Smoke - 2026-07-05

Scope: official reproducibility gate only. No Ours candidate and no GPU training job were launched.

## Data Download

- Data root: `/root/autodl-tmp/todcl_official_data_20260705`
- Proxy: temporary mihomo proxy on `127.0.0.1:7890/7891`.
- Prior `git clone --depth 1` failed with `GnuTLS recv error` / `early EOF`.
- Prior partial SGD codeload archive:
  - path: `/root/autodl-tmp/todcl_official_data_20260705/archives/dstc8-schema-guided-dialogue.zip.bad-23068672`
  - size: `23068672` bytes
  - validation: invalid zip
- Successful SGD archive:
  - path: `/root/autodl-tmp/todcl_official_data_20260705/archives/dstc8-schema-guided-dialogue.zip`
  - size: `36962546` bytes
  - validation: Python `zipfile.testzip()` passed
  - entries: `234`
- SGD extraction:
  - extracted path: `/root/autodl-tmp/todcl_official_data_20260705/dstc8-schema-guided-dialogue`
  - ToDCL expected path linked at `data/dstc8-schema-guided-dialogue`

- Taskmaster archive:
  - path: `/root/autodl-tmp/todcl_official_data_20260705/archives/Taskmaster.zip`
  - size: `138699973` bytes
  - validation: Python `zipfile.testzip()` passed
  - entries: `161`
  - note: first attempt hit `IncompleteRead` after the 20MB progress point; automatic retry completed and validated.
- MultiWOZ archive:
  - path: `/root/autodl-tmp/todcl_official_data_20260705/archives/multiwoz.zip`
  - size: `60601152` bytes
  - validation: Python `zipfile.testzip()` passed
  - entries: `70`

Taskmaster and MultiWOZ source archives are now downloaded and zip-validated.

## Controlled Extraction And Layout

- Taskmaster extracted to `/root/autodl-tmp/todcl_official_data_20260705/Taskmaster`.
  - extracted size: `978M`
  - ToDCL expected path linked at `data/Taskmaster`
- MultiWOZ extracted to `/root/autodl-tmp/todcl_official_data_20260705/multiwoz`.
  - extracted size: `311M` before generated conversion output
  - ToDCL expected path linked at `data/multiwoz`
- Archives are preserved under `/root/autodl-tmp/todcl_official_data_20260705/archives`.
- MultiWOZ official conversion:
  - extracted `MultiWOZ_2.1.zip` to `data/multiwoz/data/MultiWOZ_2.1`
  - ran `data/multiwoz/data/MultiWOZ_2.2/convert_to_multiwoz_format.py`
  - output: `data/multiwoz/data/MultiWOZ_2.2/data.json`
  - output size: `263592313` bytes
  - converted dialogues: `10437`
  - conversion note: official script logged that `SNG01862.json` does not exist in MultiWOZ 2.2, then completed.

## System Disk Cleanup

- Trigger: `/root` overlay was critically low while Taskmaster download was in progress.
- Before cleanup:
  - `/root` overlay: `30G` size, `30G` used, `561M` available, `99%` used.
  - `/root/autodl-tmp`: `150G` size, `101G` used, `50G` available, `67%` used.
- Safe cleanup performed:
  - purged pip cache with `python -m pip cache purge`
  - removed `/root/.cache/pip`
  - ran `conda clean -a -y`
  - cleared `/root/miniconda3/pkgs/*` package cache contents
- Preserved:
  - git repositories, official results, logs, checkpoints, submodule/gitlink metadata, and traceable artifacts
  - `/root/.cache/huggingface`, because it may contain model assets/checkpoints rather than disposable package cache
- After cleanup:
  - `/root` overlay: `30G` size, `27G` used, `3.3G` available, `90%` used.
  - `/root/autodl-tmp`: `150G` size, `100G` used, `51G` available, `67%` used.

## Legacy Environment

- Env path: `/root/autodl-tmp/conda_envs/todcl_legacy_py37`
- Conda package cache override: `/root/autodl-tmp/conda_pkgs`
- Python: `3.7.16`
- Pytorch: `1.4.0`
- CUDA runtime: `10.1`
- `torch.cuda.is_available()`: `True`
- Key ToDCL deps installed:
  - `transformers==3.5.1`
  - `pytorch-lightning==1.2.5`
  - `torchmetrics==0.3.2`
  - `sentencepiece==0.1.91`
  - `dictdiffer==0.8.1`

`train.py --help` passes in this env.

## SGD Preprocess Smoke

Command shape:

```python
from utils.preprocess import get_datasets
out = get_datasets(dataset_list=['SGD'], setting='single', verbose=False, develop=True)
```

Result:

- train/dev/test totals: `532/78/158`
- domains: `10`
- intents: `17`
- sample domains include `sgd_restaurants`, `sgd_media`, `sgd_flights`, `sgd_ridesharing`, `sgd_rentalcars`

Gate status:

- ToDCL was previously upgraded from entrypoint-only smoke to `partial_data_smoke`.

## Full Data-Loader Smoke

Scope: no training, no Ours candidate. This only validates that the official ToDCL preprocessing path can read the full 37-domain data under the faithful legacy environment.

Command shape:

```python
from utils.preprocess import get_datasets
out = get_datasets(dataset_list=['TM19', 'TM20', 'MWOZ', 'SGD'], setting='single', verbose=False, develop=False)
```

Result:

- smoke summary: `results/logs/official_todcl_full_dataloader_smoke_20260705.md`
- train/dev/test totals after service filtering: `31425/4035/4742`
- BYDOMAIN counts: train `37`, dev `37`, test `37`
- aggregate train domains/intents: `37` domains, `79` intents
- dataset contributions after filtering:
  - train: MWOZ `7905`, SGD `5278`, TMA `4403`, TMB `13839`
  - dev: MWOZ `1000`, SGD `753`, TMA `551`, TMB `1731`
  - test: MWOZ `1000`, SGD `1455`, TMA `553`, TMB `1734`

Gate status:

- ToDCL is upgraded to `full_data_loader_smoke_passed`.
- This is still not an official method reproduction: method-level training/eval smoke and paper-number reproduction are pending.
