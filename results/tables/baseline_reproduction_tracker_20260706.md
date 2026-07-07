# Baseline Reproduction Tracker — 2026-07-06

**Branch:** `sota-24h-campaign-20260706`  
**Policy:** Official code + official-script settings; disclose `setting_blocker` when paper text ≠ released scripts.  
**±1 gate:** EM/AR = **±1.0 percentage points**; BLEU-4 = **±0.01** absolute; SER = **±1.0** absolute; ToDCL BLEU = **±1.0** percentage points.  
**CITB split disclosure:** All local CITB runs use **official_script_500_50_50** (user confirmed). Paper text says **500/50/100** — not reproducible from released data/scripts for 4 order1 tasks.  
**CITB metric note:** AR = `average_accuracy` from official score matrix (`parse_citb_official_results.py`), **not** `predict_official_rougeL` (FR column).

**Queue script:** `scripts/run_baseline_reproduction_queue.sh` (tmux: `lora-ours-baseline-repro-queue`)  
**GPU policy:** Serial via `scripts/lora_ours_gpu_queue.sh` / baseline queue; W&B project `lora-ours`

**Handoff:** GPU freed from v85 smoke → baseline queue restarted → CITB Replay(50) formal running (`baseline_repro_gpu_handoff_20260706.md`).

---

## Best-method-only scope (2026-07-06 17:15)

**Policy:** Run only the **best non-Multi** method per suite; skip secondary baselines unless needed for paper footnotes.

| Suite | Target method | Paper ref | Local | ±1 gate | Status |
|-------|---------------|-----------|-------|---------|--------|
| **1 — CITB** | **Replay(50)** | ROUGE-L AR **40.4** | AR **39.98** (`formal_v56` matrix) | **PASS** (−0.42) | **done** 19/19; prior 32.44 was FR misread |
| **2 — Standard** | **O-LoRA** | EM **75.8** avg | **76.81** (v57) | **PASS** (+1.01 borderline) | **done** |
| **3a — ARPER** | **SCLSTM domain exemplar** | BLEU **0.701**, SER **3.63** | BLEU **0.601**, SER **7.83** (v87) | **FAIL** | v87 done; **queued** v88 (exemplar 250, batch 128) |
| **3b — ToDCL** | **ADAPTER** NLG | BLEU **21.77** | — | **pending** | **running** anchor (GPU); GPT-2 fixed |

**Skipped from queue:** CITB FT-init/L2/EWC/AGEM/Replay(10); Standard SeqLoRA/IncLoRA/Replay/LFPT5/ProgPrompts; ToDCL REPLAY/LAMOL; ARPER ER baselines.

---

## Suite 1 — CITB InstrDialog order1 seed1

| Method | Paper ROUGE-L AR | Paper BWT | Local ROUGE-L AR | Local BWT | Status | Setting notes |
|--------|------------------|-----------|------------------|-----------|--------|---------------|
| FT-init | **35.7** | **−4.6** | **33.109** | — | **skipped** (not best-method) | `v54` partial exists; superseded by Replay(50) target |
| L2 | **~35.6** | **~−3.8** | — | — | **skipped** | Out of best-method scope |
| EWC | **~34.5** | **~−6.8** | — | — | **skipped** | Out of best-method scope |
| AGEM(10) | **~33.2** | — | — | — | **skipped** | Out of best-method scope |
| AGEM(50) | **~34.9** | — | — | — | **skipped** | Out of best-method scope |
| Replay(10) | **~38.4** | — | — | — | **skipped** | Out of best-method scope |
| Replay(50) | **40.4** | up to **1.6** | **39.98** (AR matrix) / FR **32.44** | — | **done** (`formal_v56`, 19/19) | **±1 PASS** on AR; W&B `4r1vg0x9` |
| Multi | **~42.1** | — | — | — | **blocked** | `MULTI_TASK` uses different entrypoint; launcher pending |

**CITB launcher:** `scripts/run_citb_instrdialog_all_baselines_repro.sh`  
**Env:** `lora_v10_citb`, `tie_word_embeddings` fix, `CITB_ROOT=/root/autodl-tmp/Lora-code/external_baselines/citb_official`

---

## Suite 2 — Standard T5-large PEFT CL order1 seed1

| Method | Paper EM (avg) | Local EM | Status | Setting notes |
|--------|----------------|----------|--------|---------------|
| O-LoRA | **75.8** (avg); order1 **~75.4** | **76.8059** | **±1 PASS** (`v57` formal) | Official-equivalent single-GPU `grad_accum=8`; ROUGE-L **79.9715** |
| SeqLoRA | **~43.7** / LB-CL **39.3** | — | **skipped** | Not best-method; no isolated script |
| IncLoRA | **~66.4** / LB-CL **63.6** | — | **skipped** | Not best-method; no isolated script |
| Replay | **~57.8** | — | **skipped** | Not best-method |
| LFPT5 | **~72.7** / **71.3** | — | **skipped** | Not best-method; blocked on LM-adapted T5-large |
| Progressive Prompts | **~75.1–76.1** | — | **skipped** | Not best-method; blocked on env/protocol |
| LB-CL | **76.7** | — | **paper_only** | NeurIPS checklist: code not attached; no author repo — **O-LoRA v57 used as best-available official anchor** |
| MTL upper bound | **~80.0** | — | **paper_only** | Context ceiling only |

**Launcher:** `scripts/run_standard_all_baselines_repro.sh`

---

## Suite 3 — Dialogue NLG

| Method | Paper metric | Local | Status | Setting notes |
|--------|--------------|-------|--------|---------------|
| ARPER SCLSTM | BLEU **0.701**, SER **3.63** | BLEU **0.601**, SER **7.831** (v87 final) | **±1 FAIL** | v87 done (wrong row: exemplar 500, batch 64). **Next:** v88 exemplar **250**, batch **128** |
| ARPER Replay/ER baselines | paper tables | — | **skipped** | Not best-method |
| ToDCL ADAPTER (=AdapterCL) | BLEU **21.77**, EER **0.164** | — | **pending** | **running** `lora-ours-todcl-adapter-anchor`; README args verified |
| ToDCL REPLAY | BLEU **21.48**, EER **0.056** | — | **skipped** | Not best-method (ADAPTER wins BLEU) |
| LAMOL | BLEU **3.50**, EER **0.357** | — | **skipped** | Not best-method |
| Multi upper bound | BLEU **26.15** | — | **not run** | Ceiling reference |

---

## GPU / launch state (live)

| Item | State |
|------|-------|
| GPU owner | **ARPER v88** (`lora-ours-arper-v88-formal`, Restaurant ~E27) |
| **Just completed** | ARPER v87 — BLEU **0.601**, SER **7.831** (FAIL ±1) |
| **Queued next** | ToDCL ADAPTER anchor (GPT-2 fixed) → optional CITB Stage-1 if `FORCE_CITB_STAGE1=1` |
| **Orchestrator** | `scripts/run_strict_paper_repro_iteration.sh` (tmux `lora-ours-strict-paper-repro`) |
| Ours overlay queue | `lora-ours-sota-gpu-queue` — **paused** until baseline queue empty |

---

## Honest blockers (summary)

1. **CITB split:** Paper `500/50/100` vs official script `500/50/50` — disclose on every CITB row; never mix labels in main table.
2. **LB-CL:** No official code — `paper_only_baseline`.
3. **Standard SeqLoRA/IncLoRA/Replay:** Not exposed as runnable scripts in O-LoRA repo.
4. **LFPT5:** `Classification.sh` runnable in principle; **blocked** on LM-adapted T5-large weights + few-shot protocol ≠ O-LoRA order1.
5. **Progressive Prompts:** T5 CL entrypoint exists; **blocked** on conda env `nlp` + order1 task-list launcher vs paper table protocol.
6. **CITB Multi:** Different training entrypoint — not yet wired in master launcher.
7. **ARPER:** v88 running with paper-exact `config.cfg` (exemplar 250, batch 128, domain-wise); v87 failed due to exemplar 500 / batch 64 drift.

---

*Updated: 2026-07-07T16:40+08:00. Gap analysis: `docs/experiments/paper_repro_gap_analysis_20260707.md`; iteration tracker: `results/tables/paper_repro_iteration_tracker_20260707.md`.*
