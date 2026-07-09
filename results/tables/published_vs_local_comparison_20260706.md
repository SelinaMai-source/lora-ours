# Published vs Local Comparison — 2026-07-06 (best-method-only)

**Branch:** `sota-24h-campaign-20260706`  
**Scope:** Only the **best non-Multi** method per suite (4 targets across 3 suites).  
**Policy:** Completed runs with extractable metrics; running/queued noted inline.  
**Updated:** 2026-07-09 10:30 UTC+8

---

## Target methods summary

| # | Suite | Method | Paper metric | Local result | Status | ±1 Verdict |
|---|-------|--------|--------------|--------------|--------|------------|
| 1 | **CITB** | Replay(50) | ROUGE-L AR **40.4** | AR **39.98** (`formal_v56` matrix) | **done** | **PASS** (−0.42) |
| 2 | **Standard** | O-LoRA | EM avg **75.8** | EM **76.81** (`v57`) | **done** | **FAIL*** (+1.01 borderline) |
| 2b | **Standard** | LB-CL | EM **76.7** | *not run* | **paper_only** | No official code |
| 3a | **Dialogue ARPER** | SCLSTM exemplar | BLEU **0.701**, SER **3.63** | BLEU **0.599**, SER **5.94** (v89); v88 **running** | **partial** | **FAIL** |
| 3b | **Dialogue ToDCL** | ADAPTER NLG | BLEU **21.77**, EER **0.164** | BLEU **22.61**, EER **0.115** | **done** | **PASS** |

---

## 1 — CITB: Replay(50)

| Metric | Paper | Local | Match? | Notes |
|--------|-------|-------|--------|-------|
| ROUGE-L AR | **40.4** | **39.98** (`formal_v56`, 19/19 matrix) | **Yes ±1** | AR via `average_accuracy`; prior 32.44 was FR (`predict_official_rougeL`) |
| BWT | up to **1.6** | — | — | Pending |

**Setting disclosure:** All local CITB runs use official-script `500/50/50`; paper text says `500/50/100` — disclose as RED FLAG.

---

## 2 — Standard: O-LoRA (LB-CL paper_only)

| Metric | Paper | Local (`v57` formal) | Match? | Notes |
|--------|-------|----------------------|--------|-------|
| Final avg EM | **75.8** (O-LoRA); **76.7** (LB-CL) | **76.8059** | **Borderline** | Gap +1.0059 exceeds strict ±1.0 by 0.006 |
| Final ROUGE-L | ~79–80 | **79.9715** | **Close** | Single-GPU `grad_accum=8` official-equivalent |

---

## 3a — Dialogue ARPER: SCLSTM exemplar

| Metric | Paper | Local v89 (best completed) | Match? | Notes |
|--------|-------|---------------------------|--------|-------|
| BLEU-4 | **0.701** | **0.59890** | **No** | Gap −0.10 (within BLEU tol 0.01: fail) |
| SER | **3.63** | **5.938** | **No** | Gap +2.31 |

**Active:** v88 paper-exact (ex250, bs128) running. **Queued:** v90 (ex500, bs1024 GPU-tuned) after v88/v89 gate.

---

## 3b — Dialogue ToDCL: ADAPTER modular NLG

| Metric | Paper | Local | Match? | Notes |
|--------|-------|-------|--------|-------|
| BLEU | **21.7719** | **22.6105** | **Yes ±1** | Scorer JSON ADAPTER row; GPT-2 fixed |
| EER | **0.164** | **0.115031** | **Yes ±10%** | Within 0.10 absolute tolerance |

---

## RED FLAG checklist

1. **CITB split:** Paper `500/50/100` vs local `500/50/50` — always disclose.
2. **Standard single-GPU:** v57 uses 1 GPU + `grad_accum=8` vs paper 8-GPU — documented official-equivalent.
3. **LB-CL:** No code — O-LoRA v57 is the operational anchor, not a LB-CL reproduction.
4. **ARPER:** v66/v87/v89 all FAIL; v88 re-run in progress with paper-exact config.

---

## Artifact pointers

| Run | Log / metrics |
|-----|---------------|
| CITB Replay50 formal | `results/logs/citb_replay50_formal_20260706.log` |
| Standard O-LoRA v57 | `results/logs/olora_t5large_standard_order1_seed1_official_base_formal_v57_monitor.md` |
| ARPER v89 | `results/logs/arper_woz3_paper_aligned_exemplar500_batch128_formal_v89_status.md` |
| ToDCL ADAPTER | `results/logs/todcl_adapter_nlg_official_anchor_20260706_metrics.json` |
