# SOTA Main Table — lora-ours 三套件 (2026-07-08)

**Branch:** `ours-v1-20260708`  
**W&B project:** `lora-ours-v1`  
**Campaign status:** Phase 0 anchors locked; v1 overlay smokes queued after ToDCL anchor.

## Published-base anchors (locked)

| Suite | Setting label | Paper ref | Locked local | Gate |
|-------|---------------|-----------|--------------|------|
| **CITB** | `official_script_500_50_50` Replay(50) | AR **40.4** | AR **39.98** | **PASS** |
| **Standard** | O-LoRA official-equivalent single-GPU | EM **75.8** | EM **76.81** (v57) | **PASS** |
| **ARPER** | Path B SCLSTM paper-aligned | BLEU **0.701** / SER **3.63** | BLEU **0.599** / SER **5.94** (v89) | anchor locked |
| **ToDCL** | ADAPTER NLG 37-domain | BLEU **21.77** / EER **0.164** | pending (epoch 8/10) | **RUNNING** |

## Ours v1 overlay status

| Suite | v1 mechanism | Smoke | Formal | Best local | SOTA target | Gap |
|-------|--------------|-------|--------|------------|-------------|-----|
| **CITB** | replay_ratio 0.5 SSRG | **queued** | — | v85 smoke AR **24.6** | AR **53.87** | large |
| **Standard** | class-cov SSRG + gate 0.25 | **queued** | — | v69 EM **77.26** | EM **84.5** | −7.2 |
| **ARPER** | SSRG exemplar | **queued** | — | v89 BLEU **0.599** | BLEU **0.935** | large |
| **ToDCL** | assess orthogonal | **blocked** (anchor) | — | — | BLEU **29.0** | pending |

## Disclosure notes (CCFA)

1. **CITB:** Main table uses `official_script_500_50_50`; do not mix with paper `500/50/100`.
2. **Standard:** v57/v69/v1 use single-GPU + `grad_accum=8` official-equivalent.
3. **ARPER:** v89 below paper; v1 overlay is `published-base + SSRG exemplar selection`.
4. **ToDCL:** Legacy py37; anchor must complete before v1 overlay.

## Honest gaps (2026-07-08)

| Item | Status |
|------|--------|
| Phase 0 anchors | **3/4 locked**; ToDCL running |
| v1 overlay smokes | **queued** (GPU held by ToDCL anchor) |
| Numerical SOTA | **not reached** on any suite |
| 3 seeds × 3 orders | deferred until first margin hit |

*Updated: 2026-07-08T12:10+08:00*
