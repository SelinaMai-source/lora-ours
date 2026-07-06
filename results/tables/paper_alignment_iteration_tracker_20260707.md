# Paper Alignment Iteration Tracker — 2026-07-07

**Tolerance:** primary metrics `|local − paper| ≤ 1.0`  
**Branch:** `sota-24h-campaign-20260706`  
**Audit:** `docs/experiments/paper_alignment_root_cause_20260707.md`

| Suite | Attempt | Paper | Local | Gap | Fix applied | Status |
|-------|---------|-------|-------|-----|-------------|--------|
| **Standard O-LoRA** | v57 formal | EM 75.8 | **76.81** | +1.01 | Official-equivalent 1×GPU + grad_accum=8 | **PASS ±1** |
| **CITB Replay(50)** | v56 formal | ROUGE-L AR 40.4 | **32.44** | −7.96 | Official script 500/50/50; seed469 + tokenizer shim | **FAIL** |
| **CITB Stage-1** | seed50 train (queued) | — (prerequisite) | — | — | Official `run_initial_multitask_tuning.sh` parity; `lora_v10_citb`; tie patch on ckpt-14000 | **queued** (after ToDCL, before v2) |
| **CITB Replay(50)** | v2 (queued) | ROUGE-L AR 40.4 | — | — | Official checkpoint-14000 path; native tokenizer; seed50; minimal shims | **blocked** — awaits Stage-1 |
| **ARPER SCLSTM** | v66 formal | BLEU 0.701 | **0.632** | −0.069 | DA-wise granularity=1 (wrong row) | **FAIL** |
| **ARPER SCLSTM** | v87 formal | BLEU 0.701 | **~0.601** (4/6 dom) | −0.10 | Domain-wise but exemplar 500 + batch 64 | **FAIL / running** |
| **ARPER SCLSTM** | v88 (queued) | BLEU 0.701 | — | — | Exact config.cfg: exemplar 250, batch 128, granularity=0 | **queued** (after v2) |
| **ARPER SCLSTM** | v87 formal | SER 3.63 | TBD | — | Same as v87 BLEU row | **running** |
| **ToDCL ADAPTER** | anchor 20260706 | BLEU 21.77 | — | — | GPT-2 corrupt / HF offline | **blocked → fixed** |
| **ToDCL ADAPTER** | anchor retry (queued) | BLEU 21.77 | — | — | Local GPT-2 path + offline transformers | **queued** (after ARPER v87) |
| **ToDCL ADAPTER** | anchor retry | EER 0.164 | — | — | Same run; use ±10% relative for small metric | **queued** |

## Next actions (GPU serial — 1×48GB)

1. **Do not interrupt** ARPER v87 (`lora-ours-arper-v87-formal`) — healthy, ~Restaurant epoch 11/100.
2. After v87: **ToDCL ADAPTER** anchor retry → **CITB Stage-1 seed50** (`scripts/run_citb_stage1_seed50_train.sh`) → **CITB Replay(50) v2** (`scripts/run_citb_replay50_paper_aligned_v2.sh`) → **ARPER v88**.
3. **No parallel Stage-1** on this host while v87 holds the only GPU; full paper queue is multi-day (~30–40 GPU-h stacked). Optional diagnostic only: `ALLOW_FALLBACK_STAGE1=1` on v2 (not paper-comparable).
4. Monitor: `scripts/monitor_paper_alignment.sh` (tmux `lora-ours-paper-alignment-watch`).

## CITB v2 Stage-1 blocker — resolution plan (2026-07-07)

- **Expected:** `citb_official/output/initial_multitask_model/base_epoch15_lr1e-05_seed50/checkpoint-14000` (hardcoded in all official Stage-2 scripts).
- **Searched:** CITB README, all scripts, `hyintell/CITB` GitHub releases (0), GitHub issues, HuggingFace (`hyintell/CITB` and search — 0 models), Google Drive links in repo — **no public download**.
- **On disk:** absent from `/root/autodl-tmp/CITB`, `citb_official`, and full filesystem scan; only fallback `seed469` at `model_cache/citb_superni_stage1/` (wrong seed; ~10k steps in cache, not ckpt-14000).
- **Unblock (chosen):** local Stage-1 via **`scripts/run_citb_stage1_seed50_train.sh`** — mirrors official `scripts/run_initial_multitask_tuning.sh` (`save_steps=500`, 15 epochs, full SuperNI, no 500-cap), `SEED=50`, env `lora_v10_citb`, writes to official `output/.../seed50/`; optional `FIX_STAGE1_TIE_WORD_EMBEDDINGS=1` patches `checkpoint-14000/config.json` for Stage-2 load. Legacy helper `run_citb_stage1_superni_repro.sh` uses capped instances / epoch saves — **not** for paper ckpt-14000.
- **Launch:** `cd /root/lora-ours && bash scripts/run_citb_stage1_seed50_train.sh` (preflight: `DRY_RUN=1` first).
- **Released JSON as validation target:** **yes, partial** — `scores/.../CL=REPLAY/scores.json` order1 ROUGE-L `average_accuracy=40.4`, `final_official_test_score=31.8`. Use to validate metric parsing and Stage-2 pipeline; **cannot** replace end-to-end repro without official Stage-1 weights (v56 official-test ~33.1 ≈ released 31.8 but AR ~32.4 vs 40.4).
- **Preflight log:** `results/logs/citb_stage1_checkpoint14000_preflight_20260707.json`

## ETA (rough)

| Job | Estimate |
|-----|----------|
| ARPER v87 remainder | ~8–12 h |
| ToDCL ADAPTER 37-domain | ~6–10 h |
| **CITB Stage-1 → checkpoint-14000** | **~2–4 GPU-h** (extrap. seed469 ~5.2 steps/s; ~14k steps + step eval) |
| CITB v2 formal 19-task | ~8–12 h |
| ARPER v88 | ~8–12 h |

*Updated: 2026-07-07T07:25+08:00*
