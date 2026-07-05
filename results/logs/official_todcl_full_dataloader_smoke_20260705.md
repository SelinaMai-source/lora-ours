# ToDCL Full Data-Loader Smoke - 2026-07-05

Scope: official reproducibility gate only. No training job and no Ours candidate were launched.

## Data Layout

- Data root: `/root/autodl-tmp/todcl_official_data_20260705`
- Archives preserved under `/root/autodl-tmp/todcl_official_data_20260705/archives`
- ToDCL repo: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/todcl`
- Symlinks in ToDCL `data/`:
  - `data/dstc8-schema-guided-dialogue` -> `/root/autodl-tmp/todcl_official_data_20260705/dstc8-schema-guided-dialogue`
  - `data/Taskmaster` -> `/root/autodl-tmp/todcl_official_data_20260705/Taskmaster`
  - `data/multiwoz` -> `/root/autodl-tmp/todcl_official_data_20260705/multiwoz`

## MultiWOZ Conversion

- Extracted `MultiWOZ_2.1.zip` to `data/multiwoz/data/MultiWOZ_2.1`.
- Ran official converter:

```bash
python data/multiwoz/data/MultiWOZ_2.2/convert_to_multiwoz_format.py \
  --multiwoz21_data_dir=../MultiWOZ_2.1 \
  --output_file=data.json
```

- Output: `data/multiwoz/data/MultiWOZ_2.2/data.json`
- Output size: `263592313` bytes
- Converted dialogues: `10437`
- Note: converter logged `SNG01862.json` missing in MultiWOZ 2.2, then completed successfully.

## Full Data-Loader Smoke

Environment: `/root/autodl-tmp/conda_envs/todcl_legacy_py37`.

Command shape:

```python
from utils.preprocess import get_datasets
out = get_datasets(
    dataset_list=['TM19', 'TM20', 'MWOZ', 'SGD'],
    setting='single',
    verbose=False,
    develop=False,
)
```

Result:

- Status: passed
- BYDOMAIN counts: train `37`, dev `37`, test `37`
- Final train/dev/test totals after service filtering: `31425/4035/4742`
- Aggregate train domains/intents: `37` domains, `79` intents

Dataset contributions after filtering:

- train: MWOZ `7905`, SGD `5278`, TMA `4403`, TMB `13839`
- dev: MWOZ `1000`, SGD `753`, TMA `551`, TMB `1731`
- test: MWOZ `1000`, SGD `1455`, TMA `553`, TMB `1734`

## Gate Status

- ToDCL is now at `full_data_loader_smoke_passed`.
- Remaining blocker: method-level smoke and paper-number reproduction are still pending.
