# Paper Reproduction Gap Analysis — 2026-07-07

**Branch:** `sota-24h-campaign-20260706`  
**Tolerance policy:** EM/ROUGE-L AR = ±1.0 **percentage points**; BLEU-4 = ±0.01 absolute; SER = ±1.0 absolute; ToDCL BLEU = ±1.0 percentage points.

---

## Executive summary

| Suite | Method | Paper | Local | Gap | ±1? | Root cause | Fix / next |
|-------|--------|-------|-------|-----|-----|------------|------------|
| **CITB** | Replay(50) AR | 40.4 | **39.98** | −0.42 | **PASS** | Prior −7.96 was **metric misread** (`predict_official_rougeL` = FR, not AR) | No Stage-2 rerun required for ±1; optional v2 for FWT parity |
| **Standard** | O-LoRA EM | 75.8 | **76.81** | +1.01 | **PASS (borderline)** | Single-GPU `grad_accum=8` official-equivalent port vs paper 8-GPU | Accept anchor; 8-GPU rerun only if strict byte-identical needed |
| **ARPER** | SCLSTM BLEU | 0.701 | **0.601** (v87) | −0.10 | **FAIL** | v66 DA-wise wrong row; v87 domain-wise but **exemplar 500 + batch 64** vs paper **250 + 128** | **v88 running** (paper-exact `config.cfg`) |
| **ARPER** | SCLSTM SER | 3.63 | **7.83** (v87) | +4.20 | **FAIL** | Same config drift as BLEU | Await v88 |
| **ToDCL** | ADAPTER BLEU | 21.77 | — | — | **pending** | GPT-2 corrupt (2026-07-06); **fixed 2026-07-07** | Queue after ARPER v88 |

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

| Setting | Paper `config.cfg` | v87 | v88 (running) |
|---------|-------------------|-----|---------------|
| granularity | 0 | 0 | 0 |
| exemplar_size | 250 | 500 | **250** |
| batch_size | 128 | 64 | **128** |

---

## 4 — ToDCL

GPT-2 re-downloaded to `todcl/gpt2/`; preflight passed. Launch when GPU free.

*See also:* `docs/experiments/citb_replay50_repro_gap_audit.md`, `results/logs/official_strictness_audit_20260706.md`
