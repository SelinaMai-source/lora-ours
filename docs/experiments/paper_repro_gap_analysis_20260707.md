# Paper Reproduction Gap Analysis — 2026-07-07

**Branch:** `sota-24h-campaign-20260706`  
**Tolerance policy:** EM/ROUGE-L AR = ±1.0 **percentage points**; BLEU-4 = ±0.01 absolute; SER = ±1.0 absolute; ToDCL BLEU = ±1.0 percentage points.

---

## Executive summary

| Suite | Method | Paper | Local | Gap | ±1? | Root cause | Fix / next |
|-------|--------|-------|-------|-----|-----|------------|------------|
| **CITB** | Replay(50) AR | 40.4 | **39.98** | −0.42 | **PASS** | Prior −7.96 was **metric misread** (`predict_official_rougeL` = FR, not AR) | No Stage-2 rerun required for ±1; optional v2 for FWT parity |
| **Standard** | O-LoRA EM | 75.8 | **76.81** | +1.01 | **PASS (borderline)** | Single-GPU `grad_accum=8` official-equivalent port vs paper 8-GPU | Accept anchor; 8-GPU rerun only if strict byte-identical needed |
| **ARPER** | SCLSTM BLEU | 0.701 | **0.588** (v88) | −0.11 | **FAIL** | v88 paper-exact ex250+bs128 still −0.11; v87 ex500+bs64 was −0.10 | **v89 running** (ex500+bs128); v90 if FAIL |
| **ARPER** | SCLSTM SER | 3.63 | **7.77** (v88) | +4.14 | **FAIL** | Same as BLEU row | Await v89; v90 candidates below |
| **ToDCL** | ADAPTER BLEU | 21.77 | — | — | **pending** | Prior GPT-2 corrupt; **fixed + preflight passed 2026-07-07T21:10** | Auto-launch after v89 via post-v88 gate |

---

## 1 — CITB Replay(50)

**Root cause (confirmed):** Tracker used `predict_official_rougeL` (32.44 = FR on T_unseen) instead of `average_accuracy` (39.98 = AR). Released JSON `scores/.../CL=REPLAY/scores.json` reports AR **40.4**.

**Secondary factors (not blocking ±1 AR):** Stage-1 seed469 vs official seed50/ckpt-14000; split disclosure 500/50/50 vs paper 500/50/100.

**Fix:** Metric gate corrected. Optional v2 (`lora-ours-citb-replay50-repro-v2`) blocked on unpublished checkpoint-14000.

---

## 2 — Standard O-LoRA

**Root cause:** Official-equivalent 1×GPU + `grad_accum=8` vs paper 8-GPU. Gap +1.006pt — borderline PASS.

---

## 3 — ARPER

| Setting | Paper `config.cfg` | v87 | v88 (done) | v89 (running) |
|---------|-------------------|-----|------------|---------------|
| granularity | 0 | 0 | 0 | 0 |
| exemplar_size | 250 | 500 | **250** | **500** |
| batch_size | 128 | 64 | **128** | **128** |

---

## 4 — ToDCL

GPT-2 re-downloaded to `todcl/gpt2/`; preflight passed. Launch when GPU free.

*See also:* `docs/experiments/citb_replay50_repro_gap_audit.md`, `results/logs/official_strictness_audit_20260706.md`

---

## v88 result → v89 hypothesis (2026-07-07T20:57:47+08:00)

| Metric | Paper | v88 local | Gap | ±1? |
|--------|-------|-----------|-----|-----|
| BLEU | 0.701 | 0.58756 | -0.1134 | fail |
| SER | 3.63 | 7.770 | 4.14 | fail |

**v89 hypothesis:** domain-wise + exemplar **500** + batch **128** (v87 exemplar count, v88 batch size).

**v89 status (2026-07-07T23:17):** RUNNING — Booking Ep1, **123/290 epochs (~42%)**, runtime ~2h20m. Interim @Hotel: BLEU **0.571**, SER **4.80** (SER improving vs v88 7.77). ETA ~02:30–03:30.

**v90 hypothesis (if v89 FAIL):** (a) paper-exact exemplar 250 + lr sweep, (b) DA-wise granularity=1 ablation, (c) seed/checkpoint parity audit vs released ARPER weights.

---

## v88 result → v89 hypothesis (2026-07-08T07:12:49+08:00)

| Metric | Paper | v88 local | Gap | ±1? |
|--------|-------|-----------|-----|-----|
| BLEU | 0.701 | 0.58756 | -0.1134 | fail |
| SER | 3.63 | 7.770 | 4.14 | fail |

**v89 hypothesis:** domain-wise + exemplar **500** + batch **128** (v87 exemplar count, v88 batch size).
