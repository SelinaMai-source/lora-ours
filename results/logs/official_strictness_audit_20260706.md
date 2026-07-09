# Official Strictness Audit — Best-Method Reproduction (2026-07-06)

**Updated:** 2026-07-09T10:30+08:00  
**Scope:** Four best-method anchors across CITB / Standard / Dialogue (ToDCL + ARPER)  
**Overall verdict:** **Not uniformly paper-strict.** Each suite uses the strongest *runnable* official or official-equivalent path; deviations are documented per row below.

## Executive summary

| Suite | Best method | Official strictness verdict | Runnable? | Local result | ±1 gate |
|-------|-------------|----------------------------|-----------|--------------|---------|
| CITB | Replay(50) formal v56 | **Official entry, non-paper setting** | Yes | ROUGE-L AR **39.98** | **PASS** |
| Standard | O-LoRA v57 | **Official-equivalent single-GPU port** | Yes | EM **76.81** | **FAIL*** (+1.01) |
| Dialogue ToDCL | ADAPTER 37-domain | **README-aligned `train.py` path** | Yes | BLEU **22.61** / EER **0.115** | **PASS** |
| Dialogue ARPER | v89 ex500+bs128 | **Domain-wise paper-aligned attempt** | Yes | BLEU **0.599** / SER **5.94** | **FAIL** |

**Honest answer to “严格按照官方代码来跑的吗?”**

- **CITB & Standard:** Yes for *released* official scripts/code; no for *paper-text* settings where public artifacts differ (CITB split) or hardware is ported (Standard 1×GPU + grad_accum).
- **ToDCL:** Yes — `train.py --CL ADAPTER` per README; legacy py37 env; completed anchor run.
- **ARPER:** Yes for official `run_woz3.py` / SCLSTM repo; hyperparams aligned to paper table — metric match still FAIL.

---

## Per-suite audit

### 1. CITB — Replay(50) formal v56

**Strictness grade:** Official-script strict ✅ · Paper-setting strict ❌  
**Local:** AR **39.9826** — **PASS ±1**

### 2. Standard — O-LoRA v57 formal

**Strictness grade:** Official-equivalent anchor ✅ · Literal 8-GPU byte-identical ❌  
**Local:** EM **76.8059** — **FAIL ±1** (borderline +0.006 over tolerance)

### 3. Dialogue — ToDCL ADAPTER 37-domain NLG

**Strictness grade:** README/`train.py` aligned ✅ · Completed official repro ✅  
**Local:** BLEU **22.6105**, EER **0.115031** — **PASS**

### 4. Dialogue — ARPER v89 (best completed)

**Strictness grade:** Official SCLSTM + paper-aligned hyperparams ✅ · Metric match ❌  
**Local:** BLEU **0.59890**, SER **5.938** — **FAIL**; v88 re-run active, v90 queued
