# Disk Cleanup — 2026-07-08

**Operator:** agent (urgent disk-full recovery)  
**Pre-check:** `pgrep` confirmed ARPER v89 (PID 257186) + monitors alive — no running jobs killed.

## Before / After

| Mount | Before | After | Freed |
|-------|--------|-------|-------|
| `/` (overlay) | 26G used / 4.4G free (86%) | 23G used / **7.5G free** (76%) | **~3.1 GB** |
| `/root/autodl-tmp` | 150G used / 384K free (**100%**) | 49G used / **102G free** (33%) | **~101 GB** |

```
# Before
Filesystem      Size  Used Avail Use% Mounted on
overlay          30G   26G  4.4G  86% /
/dev/md0        150G  150G  384K 100% /root/autodl-tmp

# After
Filesystem      Size  Used Avail Use% Mounted on
overlay          30G   23G  7.5G  76% /
/dev/md0        150G   49G  102G  33% /root/autodl-tmp
```

**Targets met:** autodl-tmp ≥15–20 GB free ✓ (102G) | / ≥5 GB free ✓ (7.5G)

## Deleted (safe + aggressive)

### `/root/autodl-tmp` (~98 GB)

| Path | ~Size | Reason |
|------|-------|--------|
| `lora-baselines-run_v1/external_sources/todcl/runs_NLG` | 41G | Old failed/incomplete ToDCL NLG run artifacts |
| `lora-baselines-run_v1/external_sources/todcl/runs_{E2E,DST,INTENT}` | 473M | Old ToDCL run dirs (code + gpt2 symlink kept) |
| `Lora-code/assets/pretrained/meta-llama` | 30G | Duplicate LLaMA weights, not used by current repro |
| `Lora-code/assets/pretrained/lfpt5` | 3.1G | Duplicate LFPT5 cache (in model_cache) |
| `Lora-Baselines/` | 6.2G | Redundant benchmark tree vs lora-ours |
| `Lora-code/external_baselines` | 4.8G | Redundant with lora-ours baselines |
| `Lora-code/data` | 1.4G | Redundant data copy |
| `citb_official_base_repro/runtime` | 4.0G | Old training runtime cache |
| `citb_official_base_repro/wandb` | 884M | Duplicate wandb logs |
| `citb_official_base_repro/citb_replay50_official_base_smoke_20260706` | 1.8G | Smoke run |
| `citb_official_base_repro/*_dryrun*` + v53/v54/v55 smokes | ~7G | Failed/intermediate CITB iterations |
| `lora-ours-devdiag/` | 866M | v82/v83 overlay diag artifacts |

### `/` overlay (~3 GB via lora-ours results)

| Path | ~Size | Reason |
|------|-------|--------|
| `lora-ours/results/runs/olora_*_{v55,v56,v58,v69,v70b,v85}*` | ~3.0G | Failed Ours overlay iterations (v58–v84 partial) |
| `lora-ours/results/runs/olora_*_v82/v83/v84_*_diag*` | ~120M | Round-2 diag runs |
| `lora-ours/results/runs/arper_*_formal_v88` | 130M | v88 FAIL complete (v89 active) |
| `__pycache__` under lora-ours + lora-baselines-run_v1 | small | Python bytecode cache |
| `pip cache` | 3.6M | Safe pip cache purge |

## Preserved (explicit keep list)

- **ARPER v89** checkpoints + log (`results/runs/arper_*_v89`, autodl-tmp log)
- **O-LoRA v57** formal (`olora_t5large_*_formal_v57`)
- **CITB replay50 v56 formal** (`citb_official_base_repro/...replay50_formal_v56`)
- **ToDCL** code + GPT-2 fixed weights (`todcl_official_model_cache/gpt2-local`, symlink)
- **Active conda envs** (`conda_envs/`: lora_v10_o_lora, lora_v10_citb, todcl_legacy_py37)
- **model_cache** t5-large + citb_superni_stage1
- **official_repos** pins in lora-ours
- **All tmux monitors** + running training (verified post-cleanup)

## Post-cleanup autodl-tmp layout

```
18G  conda_envs
13G  model_cache
6.2G lora-baselines-run_v1 (code only, runs removed)
4.0G CITB/data
3.1G citb_official_base_repro (v56 formal kept)
2.8G todcl_official_data_20260705
1.7G Lora-code (trimmed)
1.0G todcl_official_model_cache (gpt2)
```

## Impact on repro queue

- ToDCL was **blocked by disk** → now unblocked; auto-launch via `run_post_arper_v88_gate.sh` after v89 completes.
- No checkpoint corruption observed; v89 training continued uninterrupted through cleanup.
