# Official Strictness Audit — Best-Method Reproduction (2026-07-06)

**Updated:** 2026-07-07T07:05+08:00  
**Scope:** Four best-method anchors across CITB / Standard / Dialogue (ToDCL + ARPER)  
**Overall verdict:** **Not uniformly paper-strict.** Each suite uses the strongest *runnable* official or official-equivalent path; deviations are documented per row below.

## Executive summary

| Suite | Best method | Official strictness verdict | Runnable? | Local result |
|-------|-------------|----------------------------|-----------|--------------|
| CITB | Replay(50) formal | **Official entry, non-paper setting** | Yes | ROUGE-L AR 32.44 |
| Standard | O-LoRA v57 | **Official-equivalent single-GPU port** | Yes | EM 76.81 |
| Dialogue ToDCL | ADAPTER 37-domain | **README-aligned `train.py` path** | Blocked (GPT-2 env) | — |
| Dialogue ARPER | v87 paper-aligned | **Domain-wise paper-aligned attempt** | Yes | BLEU 0.601 (TBD SER) |

**Honest answer to “严格按照官方代码来跑的吗?”**

- **CITB & Standard:** Yes for *released* official scripts/code; no for *paper-text* settings where public artifacts differ (CITB split) or hardware is ported (Standard 1×GPU + grad_accum).
- **ARPER:** Yes for official `run_woz3.py` / SCLSTM repo; hyperparams aligned to paper table (domain-wise `granularity=0`, exemplar 500) — not a byte-identical re-run of every `run.sh` default.
- **ToDCL:** Intended yes (`train.py --CL ADAPTER` per README); blocked before training by GPT-2 tokenizer/weights cache — not yet a completed repro.

---

## Per-suite audit

### 1. CITB — Replay(50) formal

| Dimension | Verdict |
|-----------|---------|
| **Code source** | Official CITB repo (`hyintell/CITB`, commit `bf50533…`); launcher `scripts/run_citb_instrdialog_replay50_official_base_repro.sh` mirrors `scripts/short_stream_scripts/run_cit_replay.sh`. |
| **Method** | Official Stage-2 REPLAY with `replay_num_instance_per_task=50` — matches released Replay(50) scripts. |
| **Setting vs paper** | **Not paper-strict.** Paper InstrDialog = `500/50/100`; public short-stream scripts use one `max_num_instances_per_eval_task=50` for both dev and test → **script-strict `500/50/50`**. Four order-1 tasks cannot satisfy paper counts from released data. |
| **Engineering shims** | `tie_word_embeddings=False` fix (v54, documented); collator preflight; single-GPU runtime copy — labeled engineering, not official hyperparams. |
| **Paper metric** | ROUGE-L AR **40.4** (Replay 50, paper table) |
| **Local metric** | ROUGE-L AR **32.4417** (19/19 tasks, W&B `4r1vg0x9`) |
| **Gate label** | `official_script_500_50_50` — disclose on every CITB row; never mix with `500/50/100`. |

**Strictness grade:** Official-script strict ✅ · Paper-setting strict ❌

---

### 2. Standard — O-LoRA v57 formal

| Dimension | Verdict |
|-----------|---------|
| **Code source** | Official O-LoRA repo (`cmnfriend/O-LoRA`, commit `07117e1…`); training surface from `scripts/order_1.sh` / `src/run_uie_lora.py`. |
| **Method** | O-LoRA on T5-large, Standard order1 (dbpedia → amazon → yahoo → agnews). |
| **Setting vs paper** | **Official-equivalent, not literal 8-GPU.** Paper uses 8 GPUs; local port = **1 GPU + `grad_accum=8`** with documented eval-batch/runtime-copy deltas. Task order, lr, epochs, metric protocol match official launcher intent. |
| **Engineering shims** | Single-GPU grad accumulation, eval batch sizing, runtime env paths — all labeled `official-equivalent`. |
| **Paper metric** | EM avg **75.8** (O-LoRA paper); LB-CL ref **76.7** |
| **Local metric** | EM **76.8059**, ROUGE-L **79.9715** (v57 formal manifest completed) |
| **LB-CL note** | `paper_only` — no author code; O-LoRA v57 is best-available anchor. |

**Strictness grade:** Official-equivalent anchor ✅ · Literal 8-GPU byte-identical ❌

---

### 3. Dialogue — ToDCL ADAPTER 37-domain NLG

| Dimension | Verdict |
|-----------|---------|
| **Code source** | Official ToDCL repo (`andreamad8/ToDCL`, commit `e70c1ed…`); entry `train.py` per README. |
| **Method** | `--task_type NLG --CL ADAPTER --bottleneck_size 50 --dataset_list SGD,TM19,TM20,MWOZ --setting single` — README Modularized NLG anchor (BLEU **21.77**, EER **0.164**). |
| **Setting vs paper** | README-aligned command line; legacy **Python 3.7 / torch 1.4** faithful env (`todcl_legacy_py37`). 37-domain loader preflight passed (`31425/4035/4742` train/dev/test). |
| **Engineering shims** | Data archives downloaded via proxy; MultiWOZ conversion layout — infrastructure only, not method changes. |
| **Blocker** | GPT-2 `pytorch_model.bin` was corrupt; tokenizer cache incomplete → `EXIT=1`. Weights re-downloaded 2026-07-06 20:12; tokenizer online fetch verified 2026-07-07 — **retry pending GPU after ARPER**. |
| **Local metric** | — (not started) |

**Strictness grade:** README/`train.py` aligned ✅ (intent) · Completed official repro ❌ (blocked)

---

### 4. Dialogue — ARPER v87 paper-aligned

| Dimension | Verdict |
|-----------|---------|
| **Code source** | Official ARPER repo (`MiFei/Continual-Learning-for-NLG`, commit `99019de…`); `run_woz3.py` SCLSTM Path B. |
| **Method** | Paper-aligned domain-wise continual NLG: `granularity=0`, exemplar **500**, `task_seq=0,5,2,1,3,4` (6 WOZ3 domains). Launcher `scripts/run_arper_woz3_paper_aligned_formal_v87.sh`. |
| **Setting vs paper** | **Domain-wise paper-aligned attempt** — targets paper table BLEU **0.701** / SER **3.63** (exemplar 500 row). Uses official SCLSTM stack, not adapted T5/Ours Path A. |
| **Engineering shims** | Monitor/status wrappers, tmux launch, log symlinks — no training-algorithm changes. |
| **Paper metric** | BLEU **0.701**, SER **3.63** |
| **Local metric** | BLEU **0.60059** (last-domain test aggregate from formal run); SER extraction TBD |
| **Note** | First formal run completed before 2026-07-07 04:02; queue relaunch may re-run ARPER — branch results reflect first completion. |

**Strictness grade:** Official SCLSTM + paper-aligned hyperparams ✅ · Exact paper match TBD (BLEU gap ~0.10)

---

## What is *not* official-strict (disclosure checklist)

1. **CITB:** Never claim `500/50/100` without a separately audited `paper_target` run; default is `official_script_500_50_50`.
2. **Standard:** Always label single-GPU `grad_accum=8` as official-equivalent; do not equate to literal `order_1.sh` 8-GPU launch.
3. **LB-CL:** Paper-only reference; no code reproduction.
4. **ToDCL:** Bounded VANILLA/ADAPTER/REPLAY smokes passed; full 37-domain ADAPTER anchor incomplete.
5. **Ours overlays:** Any RP(LoRA)/Ours module runs only *after* these anchors — not part of this best-method repro package.

---

## References

- Launchers: `scripts/run_citb_instrdialog_replay50_official_base_repro.sh`, `scripts/run_standard_all_baselines_repro.sh`, `scripts/run_todcl_adapter_nlg_official_anchor.sh`, `scripts/run_arper_woz3_paper_aligned_formal_v87.sh`
- Matrix: `docs/official_alignment/official_method_matrix.md`
- Results branch: `best-method-repro-results-20260706`
- User notification: `results/logs/USER_NOTIFY_BEST_METHOD_REPRO_COMPLETE.md`
