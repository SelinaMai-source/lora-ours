# SOTA Main Table — lora-ours 三套件 (2026-07-06)

**Branch:** `sota-24h-campaign-20260706`  
**W&B project:** `lora-ours`  
**Campaign status:** ARPER v86 **done** (no gain vs v66); CITB ours v85 smoke **done** (AR 24.6); Standard v85 smoke **RUNNING** (SSRG fix relaunch); ToDCL anchor **queued** after v85. **No numerical SOTA margin yet.**

## Campaign snapshot (2026-07-06 13:37 UTC+8)

| Job | Key metric | Value | W&B / notes |
|-----|------------|-------|-------------|
| CITB Replay(50) base smoke | EM / ROUGE-L | **50.0 / 50.0** | `a2jh0n4y` |
| CITB ours v85 smoke | ROUGE-L AR | **24.6** | 5-segment smoke; gate PASS |
| Standard v85 overlay | cumulative EM | **running** (dbpedia round1) | SSRG fix `11c8ecc` |
| ARPER v86 formal | BLEU / SER | **0.632 / 4.817** | same as v66; below paper 0.701/3.63 |
| ToDCL ADAPTER anchor | — | **queued** | preflight PASS; launch after v85 |

## Main results (order1 seed1)

| Suite | Setting label | Published base | Ours overlay (best) | Δ | SOTA target | Gap |
|-------|---------------|----------------|---------------------|---|-------------|-----|
| **CITB** | `official_script_500_50_50` | Replay(50) ROUGE-L AR **40.4** (paper) | v85 smoke AR **24.6**; v54 FT **33.1** | — | **53.87** | −29.3 vs target |
| **Standard** | `O-LoRA official-equivalent` single-GPU | O-LoRA v57 EM **76.81**; LB-CL ref **76.7** | v69 overlay EM **77.26**; v85 smoke **pending** | +0.45 vs v57 | **84.5** | −7.2 EM (v69) |
| **Dialogue ARPER** | `ARPER Path B official` SCLSTM | Paper BLEU **0.701** / SER **3.63** | v86 formal BLEU **0.632** / SER **4.82** | = v66 | BLEU **0.935** / SER **2.72** | Large |
| **Dialogue ToDCL** | `ADAPTER NLG` 37-domain | README BLEU **21.77** / EER **0.164** | Bounded smoke only | — | BLEU **29.0** / EER **0.123** | Anchor pending |

## Disclosure notes (CCFA)

1. **CITB:** Main table uses `official_script_500_50_50`; do not mix with paper `500/50/100`. Replay50 smoke is single-task capped; ours v85 smoke is 5-segment capped — not paper-comparable formal.
2. **Standard:** v57/v69/v85 use single-GPU + `grad_accum=8` official-equivalent; differs from literal 8-GPU `order_1.sh`.
3. **Dialogue:** ARPER v86 paper-epoch repro did not close gap to paper defaults; overlay iteration (v87+ SSRG) still required.
4. **ToDCL:** Legacy py37 env; full anchor queued after Standard v85 smoke.

## Honest gaps (2026-07-06)

| Item | Status |
|------|--------|
| CITB Replay50 base smoke | **PASS** |
| CITB ours v85 smoke | **DONE** AR 24.6 — early gate PASS, SOTA gap large |
| Standard v85 smoke | **RUNNING** (`lora-ours-standard-v85-smoke`) |
| ARPER v86 formal | **DONE** BLEU 0.632 / SER 4.817 — no v66 improvement |
| ToDCL ADAPTER anchor | **queued** (`lora-ours-sota-gpu-queue` continue) |
| Numerical SOTA | **not reached** on any suite |
| CITB Replay50 formal (19 tasks) | **not started** — deferred |
| 3 seeds × 3 orders | **deferred** until first margin hit |
