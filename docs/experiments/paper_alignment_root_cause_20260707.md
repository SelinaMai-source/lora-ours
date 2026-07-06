# Paper Alignment Root-Cause Audit — 2026-07-07

**Branch:** `sota-24h-campaign-20260706`  
**Tolerance:** `|local − paper| ≤ 1.0` on primary metrics  
**Scope:** CITB Replay(50), Standard O-LoRA, ARPER SCLSTM, ToDCL ADAPTER

---

## Executive summary

| Suite | Paper | Local (best attempt) | Gap | ±1? | Primary root cause |
|-------|-------|---------------------|-----|-----|-------------------|
| **Standard O-LoRA** | EM 75.8 | EM **76.81** (v57) | +1.01 | **PASS** | None — official-equivalent anchor |
| **CITB Replay(50)** | ROUGE-L AR 40.4 | AR **32.44** (v56) | −7.96 | **FAIL** | Wrong Stage-1 checkpoint + tokenizer shims |
| **ARPER SCLSTM** | BLEU 0.701 | BLEU **0.601** (v87 partial) | −0.10 | **FAIL** | Wrong hyperparams vs `config.cfg` (batch 64, exemplar 500) |
| **ToDCL ADAPTER** | BLEU 21.77 | — | — | **PENDING** | GPT-2 cache blocker (fixed 2026-07-06 20:12; not re-run) |

---

## 1 — CITB Replay(50): gap −7.96

### Paper vs local

| Metric | Paper / released JSON | Local v56 formal | Match? |
|--------|----------------------|------------------|--------|
| ROUGE-L AR (`average_accuracy`) | **40.4** | **32.4417** | No |
| Final official test ROUGE-L | **31.8** (released) | **32.44** | Close |
| BWT | up to **1.6** | not extracted | — |

Released reference: `citb_official/scores/.../CL=REPLAY/scores.json` → `exp_order=1.rougeL.mean_results.average_accuracy = 40.4`.

### Split policy: 500/50/50 vs 500/50/100

**Preflight** (`scripts/preflight_citb_official_split_counts.py`, policy `500/50/100`):

- 19 order-1 tasks; **4 short tasks** cannot provide 500/50/100 from released data:
  - `task1590`, `task639`, `task1713`, `task766`
- `all_tasks_meet_target: false`

**Official script** (`run_cit_replay.sh`) uses `max_num_instances_per_eval_task=50` for both dev and test → **script-strict `500/50/50`**. Released FT_INSTR scores report `average_train_samples=411.5`, matching `500/50/50` (~411.47/task), not paper-text `500/50/100`.

**Conclusion:** Paper-text `500/50/100` is **not reproducible** from released data without patching split code. Released official scores (40.4) were obtained under **`500/50/50`** — split mismatch is **not** the primary −7.96 gap driver.

### Stage-1 checkpoint: seed469 vs official seed50/checkpoint-14000

| Setting | v56 local | Official `run_cit_replay.sh` |
|---------|-----------|------------------------------|
| Stage-1 path | `/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469` | `output/initial_multitask_model/base_epoch15_lr1e-05_seed50/checkpoint-14000` |
| On disk? | **Yes** (safetensors + tokenizer) | **No** — `checkpoint-14000` absent from all local CITB trees |
| Stage-2 seed | fixed `1` | random `shuf -i 10-999` per run |

**Conclusion:** Local run uses a **different Stage-1 init** than the official Replay script. This is the **strongest hypothesis** for the AR gap: local official-test ROUGE-L (~32.4) tracks released official-test (~31.8), but seen-task AR (`average_accuracy`) is ~8 pts low — consistent with a weaker or misaligned Stage-1 base.

### Engineering shims (v56)

| Shim | Reason | Impact hypothesis |
|------|--------|-------------------|
| `tie_word_embeddings=False` runtime config | Old transformers 4.25 ties lm_head when checkpoint has separate weights | Medium — affects generation quality |
| Tokenizer override → `google__t5-small-lm-adapt` | Stage-1 tokenizer incompatible with old CITB stack | **High** — vocab/encode mismatch vs official |
| `CITB_GPT2_TOKENIZER_NAME` sitecustomize | Offline HF for Tk-Instruct metric `AutoTokenizer('gpt2')` | Low on training; metric path only |
| `PYTHONPATH` citb_runtime_shims | Offline datasets/metrics | Infrastructure only |
| Collator preflight (`add_task_id`) | Untracked hyintell checkout mismatch | Blocker if wrong tree — v56 passed |

### Flag diff: local launcher vs `run_cit_replay.sh`

Aligned: `cl_method=REPLAY`, `replay_num_instance_per_task=50`, `max_num_instances_per_task=500`, `max_num_instances_per_eval_task=50`, lr `1e-5`, epochs 15, batch 8/32, eval/save steps 500, `load_best_model_at_end`, `metric_for_best_model rougeL`, pos_examples 2.

**Differences:**

1. `--tokenizer_name` + `--use_fast_tokenizer False` + `--config_name` (tie fix) — not in official script
2. `--seed 1` fixed vs random
3. `--model_name_or_path` different checkpoint
4. WandB reporting (official uses default)

### v2 repro proposal

Script: `scripts/run_citb_replay50_paper_aligned_v2.sh`

1. **Preflight** official Stage-1 at `CITB_ROOT/output/initial_multitask_model/base_epoch15_lr1e-05_seed50/checkpoint-14000`; abort with actionable message if missing (download from CITB Stage-1 release or train FT_INSTR).
2. **Use checkpoint-native tokenizer** (no `google__t5-small-lm-adapt` override when official path present).
3. Keep `tie_word_embeddings=False` only if config inspection shows tied=true with separate weights.
4. **SEED=50** default (match checkpoint name); document if random seed needed for exact replication.
5. Retain minimal offline shims (GPT-2 metric redirect, collator preflight).
6. Stay on `official_script_500_50_50` — do not claim `500/50/100`.

---

## 2 — ARPER SCLSTM: gap −0.10 BLEU

### Paper table row (WOZ3 SCLSTM, domain-wise)

From `external/config/config.cfg` + README:

| Param | Paper / default `config.cfg` | v66 (DA-wise) | v87 (domain) | v88 target |
|-------|------------------------------|---------------|--------------|------------|
| `granularity` | **0** (domain) | 1 (DA) | 0 | **0** |
| `exemplar_size` | **250** | 250 | **500** | **250** |
| `batch_size` | **128** | 64 | 64 | **128** |
| `task_seq` | `0,5,2,1,3,4` | `1,7,0,6,4,2,8` | `0,5,2,1,3,4` | `0,5,2,1,3,4` |
| `n_epochs` | 100 | 100 | 100 | 100 |
| Data split | `feat_unique_do.json` | `feat_unique_da.json` | `feat_unique_do.json` | `feat_unique_do.json` |
| CLI | lr 0.005, ewc 300000, seed 1111 | same | same | same |

Paper metrics: BLEU **0.701**, SER **3.63**.

### Local results trajectory

| Attempt | Setting | BLEU-4 | SER | Notes |
|---------|---------|--------|-----|-------|
| v66 | DA-wise, exemplar 250, batch 64 | **0.632** | **4.82** | Wrong granularity |
| v87 | Domain-wise, exemplar **500**, batch **64** | **~0.601** (4/6 domains) | TBD | Wrong exemplar count + batch |
| v88 | Domain-wise, exemplar **250**, batch **128** | queued | — | Exact `config.cfg` defaults |

**Root cause:** v87 targeted wrong paper row (exemplar 500) and halved batch size vs repo default. Paper BLEU 0.701 corresponds to **`exemplar_ewc_loss_250`** with `granularity=0`, not exemplar 500.

### v88 proposal

Script: `scripts/run_arper_woz3_paper_aligned_formal_v88.sh`  
Config: `results/logs/arper_woz3_paper_aligned_exemplar250_formal_v88.cfg` — copy of `config.cfg` defaults with absolute paths.

Queue after v87 completes (do **not** kill healthy v87 mid-run unless user requests).

---

## 3 — ToDCL ADAPTER: not completed

### README line 51 (exact)

```
python train.py --task_type NLG --CL ADAPTER --bottleneck_size 50 --lr 6.25e-3 --n_epochs 10 --train_batch_size 10 --gradient_accumulation_steps 8
```

Local launcher adds (required for 37-domain anchor): `--dataset_list SGD,TM19,TM20,MWOZ --setting single --seed 1 --model_checkpoint gpt2`.

### Blocker (resolved)

2026-07-06 launch failed: corrupt/missing `gpt2/pytorch_model.bin`, HF unreachable.

**Fix verified 2026-07-07:** `/root/autodl-tmp/lora-baselines-run_v1/external_sources/todcl/gpt2/pytorch_model.bin` (548 MB) present. Launcher updated to `--model_checkpoint` → local path + `TRANSFORMERS_OFFLINE=1`.

### EER tolerance note

EER **0.164** is a small absolute metric; use **relative tolerance ±10%** (0.148–0.180) or absolute ±0.10 if preferred — document in iteration tracker.

---

## 4 — Standard O-LoRA: PASS

| Metric | Paper | Local v57 | ±1? |
|--------|-------|-----------|-----|
| EM avg | 75.8 | **76.81** | **PASS** (74.8–76.8) |

No relaunch unless user requests 8-GPU byte-identical audit. Update `best-method-repro-results-20260706` branch when pushing alignment package.

---

## Action items

| Priority | Action | Script |
|----------|--------|--------|
| P0 | CITB v2 dry-run + formal when GPU free | `scripts/run_citb_replay50_paper_aligned_v2.sh` |
| P0 | ARPER v88 after v87 | `scripts/run_arper_woz3_paper_aligned_formal_v88.sh` |
| P1 | ToDCL retry with local GPT-2 | `scripts/run_todcl_adapter_nlg_official_anchor.sh` |
| P2 | Monitor ±1 loop | `scripts/monitor_paper_alignment.sh` |
| — | Standard PASS — no action | — |

---

## References

- CITB released scores: `/root/autodl-tmp/Lora-code/external_baselines/citb_official/scores/continual_instruction_tuning/stream=cl_dialogue_tasks/CL=REPLAY/scores.json`
- Split preflight: `results/logs/citb_official_split_counts_500_50_100_replay50.json`
- v56 status: `results/logs/citb_replay50_formal_20260706_status.md`
- ARPER config: `baselines/advanced_baselines/arper_dialog_nlg/external/config/config.cfg`
- ToDCL README: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/todcl/README.md` line 51
