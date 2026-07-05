# Dialogue NLG Official Repo / Code Status

- Updated: `2026-07-05T21:25+08:00`
- Current selected runnable base: ARPER official WOZ3 SCLSTM.
- Current ours overlay anchor: v81 post-decode slot repair overlay on ARPER/SCLSTM output path.

## ARPER / Continual-Learning-for-NLG

- Official repo: `https://github.com/MiFei/Continual-Learning-for-NLG`
- Local path: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/arper`
- Commit: `99019defe6bf35e8459ca6abd6f25882724bc956`
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
- Dirty status: clean.
- Official data setup: `data/download.sh`.
- Official examples:
  - `python train.py --CL VANILLA --task_type NLG`
  - `python train.py --task_type NLG --CL REPLAY --episodic_mem_size 10`
  - `python train.py --task_type NLG --CL ADAPTER --bottleneck_size 50 --lr 6.25e-3 --n_epochs 10 --train_batch_size 10 --gradient_accumulation_steps 8`
- Official evaluation: `python scorer.py --model_checkpoint runs_NLG/BEST/ --task_type NLG`.
- Paper/README NLG modularized reference:
  - REPLAY BLEU `21.4832`, EER `0.0559855`
  - ADAPTER BLEU `21.7719`, EER `0.163975`
  - MULTI BLEU `26.1462`, EER `0.0341823`
- Status: repo downloaded and clean; official data download/preprocess/export has not been run in this turn due disk/time constraints. ToDCL is not yet cleared as a runnable base for ours.

## Gate Decision

- Dialogue ours increments may continue only on the documented ARPER SCLSTM base until ToDCL official data and smoke reproduction are completed.
- Do not compare ARPER WOZ3 BLEU/SER and ToDCL TOD37/MultiWOZ BLEU/EER as interchangeable SOTA claims.
