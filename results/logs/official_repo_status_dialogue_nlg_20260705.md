# Dialogue NLG Official Repo / Code Status

- Updated: `2026-07-05T23:02+08:00`
- Current selected runnable base: ARPER official WOZ3 SCLSTM.
- Current ours overlay anchor: v81 post-decode slot repair overlay on ARPER/SCLSTM output path.
- Official archive branch: `official-method-code-archive-20260705` commit `b060357`.

## ARPER / Continual-Learning-for-NLG

- Official repo: `https://github.com/MiFei/Continual-Learning-for-NLG`
- Local path: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/arper`
- Commit: `99019defe6bf35e8459ca6abd6f25882724bc956`
- Official archive submodule: `official_repos/dialogue/Continual-Learning-for-NLG`.
- Dirty status: untracked `__pycache__` directories only.
- Official run script: `run.sh`.
- Official config: `config/config.cfg`.
- Official model setting:
  - `model_type=lm`
  - `dec_type=sclstm`
  - `experiment=exemplar_ewc_loss_250`
  - `task_seq=0,5,2,1,3,4`
  - `random_seed=1111`
  - `sv_len_weight=0.5`
  - `adaptive=True`
  - `ewc_importance=300000`
  - `lr=0.005`, `dropout=0`, `_lambda=2.0`
  - `exemplar_size=250`
- Data/config assets:
  - text: `data/woz3/text/`
  - acts: `data/woz3/acts/`
  - vocab: `resource/woz3/vocab.txt`
  - features: `resource/woz3/feat_unique_do.json`
  - split: `resource/woz3/data_split/all_unique_do_datasplit.json`
- Existing official reproduction:
  - Run: `arper_woz3_official_sclstm_formal_v66`
  - Status: `results/logs/arper_woz3_official_sclstm_formal_v66_status.md`
  - Result: final logged BLEU4 `0.63231`, slot error/SER `4.817`
  - Error count: `0`
- Current ours overlay evidence:
  - v81 post-decode slot repair overlay keeps ARPER/SCLSTM as the base path.
  - Result from prior status: repaired BLEU4 `0.63080`, SER `0.3942`; SER improves strongly but BLEU remains below ARPER target `0.701`.

## ToDCL

- Official repo: `https://github.com/andreamad8/ToDCL`
- Local path: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/todcl`
- Commit: `e70c1edf937f6eb570296ea2897dbc8d6815bc6d`
- Official archive submodule: `official_repos/dialogue/ToDCL`.
- Dirty status: clean.
- Official data setup: `data/download.sh`.
- Data requirements from `data/download.sh`:
  - clone `https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git`
  - clone `https://github.com/google-research-datasets/Taskmaster.git`
  - clone `https://github.com/budzianowski/multiwoz.git`
  - unzip/convert MultiWOZ 2.1/2.2
- Current data availability: `SGD_AVAILABLE_FOR_SMOKE`, `TM_ARCHIVE_VALIDATED`, `MWOZ_ARCHIVE_VALIDATED`; `data/download.sh`, `data/TM.py`, `data/SGD.py`, and `data/MWOZ.py` are present.
- Data size/download audit:
  - GitHub repo API reports approximate repository sizes: SGD `51095 KB`, Taskmaster `111002 KB`, MultiWOZ `125780 KB`.
  - `/root/autodl-tmp` has enough capacity for these source repos.
  - `git clone --depth 1` for SGD failed with `GnuTLS recv error` / `early EOF`.
  - A codeload SGD zip attempt left `/root/autodl-tmp/todcl_official_data_20260705/archives/dstc8-schema-guided-dialogue.zip` at `23068672` bytes, but Python `zipfile` reports `BadZipFile`, so it is not a valid completed archive.
  - GitHub codeload HEAD returns `application/zip` but no `Content-Length`, so the current download path cannot reliably pre-estimate or verify size without a checksum/complete archive test.
  - Retrying codeload with retry plus Python `zipfile.testzip()` validation produced a valid SGD archive at `/root/autodl-tmp/todcl_official_data_20260705/archives/dstc8-schema-guided-dialogue.zip`, size `36962546` bytes, `234` zip entries.
  - The same validated archive path now also has Taskmaster at `/root/autodl-tmp/todcl_official_data_20260705/archives/Taskmaster.zip`, size `138699973` bytes, `161` zip entries. The first Taskmaster attempt hit `IncompleteRead`; the automatic retry completed and validated.
  - MultiWOZ official archive is at `/root/autodl-tmp/todcl_official_data_20260705/archives/multiwoz.zip`, size `60601152` bytes, `70` zip entries.
  - The invalid earlier `23068672` byte partial is retained as `dstc8-schema-guided-dialogue.zip.bad-23068672` for traceability.
  - SGD is extracted under `/root/autodl-tmp/todcl_official_data_20260705/dstc8-schema-guided-dialogue` and linked into the ToDCL expected path `data/dstc8-schema-guided-dialogue`.
  - Taskmaster/MultiWOZ are downloaded and validated as source archives, but controlled extraction/layout and MultiWOZ conversion are still pending.
- Current network/proxy availability:
  - Temporary mihomo proxy is running in tmux `official-gate-mihomo-proxy`.
  - GitHub and HuggingFace are reachable; OpenReview still redirects to browser verification.
  - `git ls-remote` succeeds for the three upstream data repos: SGD, Taskmaster, and MultiWOZ.
- Current environment availability:
  - Isolated smoke venv: `/root/autodl-tmp/venvs/todcl_official_smoke_py39`.
  - The venv inherits the existing O-LoRA Python 3.9 environment with system-site packages and installs only minimal ToDCL smoke deps inside the venv.
  - Added inside venv: `pytorch-lightning==1.2.5`, `torchmetrics==0.3.2`, `dictdiffer==0.8.1`, `termcolor==1.1.0`, `tabulate==0.8.9`, `sentencepiece==0.1.99`, `sacremoses==0.0.45`.
  - `python train.py --help` now passes in the isolated venv. This verifies the official entrypoint can parse args, but it is not a training/data smoke.
  - `requirements.txt` still pins a legacy full stack including `torch==1.4.0`, `pytorch-lightning==1.2.5`, and `transformers==3.5.1`; a fully faithful runtime remains unresolved because local inherited torch/transformers versions are newer.
  - `pip install --dry-run torch==1.4.0 transformers==3.5.1` under the Python 3.9 smoke venv cannot find `torch==1.4.0` from the configured package index, so a faithful env likely needs an older Python/CUDA-compatible channel or container.
  - Faithful legacy env now exists at `/root/autodl-tmp/conda_envs/todcl_legacy_py37`, created with Python `3.7.16`, pytorch `1.4.0`, CUDA `10.1`.
  - Creation required `CONDA_PKGS_DIRS=/root/autodl-tmp/conda_pkgs` because the default conda package cache under `/root/miniconda3/pkgs` hit `No space left on device`.
  - `train.py --help` passes in the faithful legacy env after installing ToDCL minimal dependencies including `transformers==3.5.1`, `pytorch-lightning==1.2.5`, and `sentencepiece==0.1.91`.
- Current data/preprocess smoke:
  - Command: `get_datasets(dataset_list=['SGD'], setting='single', develop=True)`.
  - Result: train/dev/test totals `532/78/158`, `10` domains, `17` intents.
  - This validates SGD path and ToDCL preprocess code only; it is not full 37-domain ToDCL reproduction.
- Official examples:
  - `python train.py --CL VANILLA --task_type NLG`
  - `python train.py --task_type NLG --CL REPLAY --episodic_mem_size 10`
  - `python train.py --task_type NLG --CL ADAPTER --bottleneck_size 50 --lr 6.25e-3 --n_epochs 10 --train_batch_size 10 --gradient_accumulation_steps 8`
- Official evaluation: `python scorer.py --model_checkpoint runs_NLG/BEST/ --task_type NLG`.
- Paper/README NLG modularized reference:
  - REPLAY BLEU `21.4832`, EER `0.0559855`
  - ADAPTER BLEU `21.7719`, EER `0.163975`
  - MULTI BLEU `26.1462`, EER `0.0341823`
- Status: repo downloaded and clean; faithful legacy help smoke and SGD-only preprocess smoke pass. Official Taskmaster/MultiWOZ archives are downloaded and validated. Controlled extraction/layout, MultiWOZ conversion, full data-loader smoke, and method-level training smoke are still required before ToDCL reproduction.

## Gate Decision

- Dialogue ours increments may continue only on the documented ARPER SCLSTM base until ToDCL official data and smoke reproduction are completed.
- Do not compare ARPER WOZ3 BLEU/SER and ToDCL TOD37/MultiWOZ BLEU/EER as interchangeable SOTA claims.
- Next safe ToDCL step: extract/layout Taskmaster and MultiWOZ under `/root/autodl-tmp`, run the official MultiWOZ conversion, then run full ToDCL data-loader smoke before any method-level training. Do not run full ToDCL training until data layout and method smoke are documented.
