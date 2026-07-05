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

Taskmaster and MultiWOZ are still not downloaded. Apply the same `curl --http1.1 --retry --retry-all-errors` plus zip validation pattern next.

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

- ToDCL is upgraded from entrypoint-only smoke to `partial_data_smoke`.
- This is not a full ToDCL data-loader or method reproduction because Taskmaster and MultiWOZ are missing.
