# Baseline Reproduction Tracker — 2026-07-06

**Branch:** `sota-24h-campaign-20260706`  
**Policy:** Official code + official-script settings; disclose `setting_blocker` when paper text ≠ released scripts.  
**CITB split disclosure:** All local CITB runs use **official_script_500_50_50** (user confirmed). Paper text says **500/50/100** — not reproducible from released data/scripts for 4 order1 tasks.

**Queue script:** `scripts/run_baseline_reproduction_queue.sh` (tmux: `lora-ours-baseline-repro-queue`)  
**GPU policy:** Serial via `scripts/lora_ours_gpu_queue.sh` / baseline queue; W&B project `lora-ours`

**Handoff:** GPU freed from v85 smoke → baseline queue restarted → CITB Replay(50) formal running (`baseline_repro_gpu_handoff_20260706.md`).

---

## Suite 1 — CITB InstrDialog order1 seed1

| Method | Paper ROUGE-L AR | Paper BWT | Local ROUGE-L AR | Local BWT | Status | Setting notes |
|--------|------------------|-----------|------------------|-----------|--------|---------------|
| FT-init | **35.7** | **−4.6** | **33.109** | — | **partial** (`v54` formal 19/19 done) | `official_script_500_50_50`; gap −2.6 vs paper even under script-strict |
| L2 | **~35.6** | **~−3.8** | — | — | **queued** (priority 3) | Same split disclosure as FT-init |
| EWC | **~34.5** | **~−6.8** | — | — | **queued** (priority 3) | Same split disclosure |
| AGEM(10) | **~33.2** | — | — | — | **queued** (priority 3) | Official `run_cit_agem_10.sh` memory=10 |
| AGEM(50) | **~34.9** | — | — | — | **queued** (priority 3) | Official `run_cit_agem.sh` memory=50 |
| Replay(10) | **~38.4** | — | — | — | **queued** (priority 3) | — |
| Replay(50) | **40.4** | up to **1.6** | — | — | **running** (priority **1**) | Formal 19-task launched `lora-ours-citb-replay50-formal` 2026-07-06 16:53; smoke `replay50_v55_smoke` only |
| Multi | **~42.1** | — | — | — | **blocked** | `MULTI_TASK` uses different entrypoint; launcher pending |

**CITB launcher:** `scripts/run_citb_instrdialog_all_baselines_repro.sh`  
**Env:** `lora_v10_citb`, `tie_word_embeddings` fix, `CITB_ROOT=/root/autodl-tmp/Lora-code/external_baselines/citb_official`

---

## Suite 2 — Standard T5-large PEFT CL order1 seed1

| Method | Paper EM (avg) | Local EM | Status | Setting notes |
|--------|----------------|----------|--------|---------------|
| O-LoRA | **75.8** (avg); order1 **~75.4** | **76.8059** | **verified** (`v57` formal) | Official-equivalent single-GPU `grad_accum=8`; ROUGE-L **79.9715** |
| SeqLoRA | **~43.7** / LB-CL **39.3** | — | **blocked** | No isolated script in O-LoRA repo |
| IncLoRA | **~66.4** / LB-CL **63.6** | — | **blocked** | No isolated script in O-LoRA repo |
| Replay | **~57.8** | — | **blocked** | No dedicated official implementation found |
| LFPT5 | **~72.7** / **71.3** | — | **blocked** | Repo downloaded; T5-large CL run script not verified |
| Progressive Prompts | **~75.1–76.1** | — | **blocked** | BERT codebase only in repo; T5 protocol differs |
| LB-CL | **76.7** | — | **paper_only** | NeurIPS checklist: code not attached; no author repo |
| MTL upper bound | **~80.0** | — | **paper_only** | Context ceiling only |

**Launcher:** `scripts/run_standard_all_baselines_repro.sh`

---

## Suite 3 — Dialogue NLG

| Method | Paper metric | Local | Status | Setting notes |
|--------|--------------|-------|--------|---------------|
| ARPER SCLSTM | BLEU **0.701**, SER **3.63** | BLEU **0.632**, SER **4.817** | **done** (`v66`/`v86` formal) | Below paper; config audit pending (epochs, exemplar 250/500) |
| ARPER Replay/ER baselines | paper tables | — | **not run** | Same ARPER repo; script isolation pending |
| ToDCL ADAPTER (=AdapterCL) | BLEU **21.77**, EER **0.164** | — | **queued** (priority **2**) | Bounded smoke passed; full 37-domain anchor script exists |
| ToDCL REPLAY | BLEU **21.48**, EER **0.056** | — | **not run** | After ADAPTER anchor |
| LAMOL | BLEU **3.50**, EER **0.357** | — | **not run** | ToDCL bounded VANILLA smoke only |
| Multi upper bound | BLEU **26.15** | — | **not run** | Ceiling reference |

---

## GPU / launch state (live)

| Item | State |
|------|-------|
| GPU owner | **CITB Replay(50) formal** (`run_continual_instruct_tuning.py`, tmux `lora-ours-citb-replay50-formal`) |
| **Handoff (16:53)** | Stopped stuck **v85 Ours overlay smoke** (0% GPU ~3h); see `results/logs/baseline_repro_gpu_handoff_20260706.md` |
| **Launched now** | Baseline queue **restarted**; CITB Replay(50) formal **running** |
| **Queued next** | ToDCL ADAPTER → CITB L2/EWC/AGEM/Replay(10) (serial) |
| Ours overlay queue | `lora-ours-sota-gpu-queue` / `run_ours_overlay_iterate_queue.sh` — deprioritized vs baselines |

---

## Honest blockers (summary)

1. **CITB split:** Paper `500/50/100` vs official script `500/50/50` — disclose on every CITB row; never mix labels in main table.
2. **LB-CL:** No official code — `paper_only_baseline`.
3. **Standard SeqLoRA/IncLoRA/Replay:** Not exposed as runnable scripts in O-LoRA repo.
4. **LFPT5 / Progressive Prompts:** Downloaded but T5-large CL protocol not verified runnable.
5. **CITB Multi:** Different training entrypoint — not yet wired in master launcher.
6. **ARPER:** Runnable but **below paper**; needs config audit before claiming reproduction.

---

*Updated: 2026-07-06. Local numbers from completed artifacts only; queued rows have no invented metrics.*
