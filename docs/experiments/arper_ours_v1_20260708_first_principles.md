# ARPER Ours v1 — First Principles (2026-07-08)

**Branch:** `ours-v1-20260708`  
**Base anchor:** v89 SCLSTM BLEU **0.59890** / SER **5.938**

## v1 Single-Mechanism Delta

Replace uniform/herding exemplar selection with **SSRG spectral subspace selection** during replay buffer construction. Training loop, scorer, splits unchanged.

| Item | v89 anchor | v1 overlay |
|------|------------|------------|
| Exemplar selection | herding/loss | **SSRG** (top-k=8, energy 0.85) |
| Exemplar budget | 500 | 500 (unchanged) |
| Path | SCLSTM Path B | SCLSTM Path B |

## Early gate (bounded smoke)

| Checkpoint | Criterion |
|------------|-----------|
| segment 0 (Train) | BLEU ≥ base − 0.02; SER ≤ base + 0.5 |
| stop | segment 0 BLEU < 0.55 or SER > 7.0 |

## Launch

```bash
DRY_RUN=1 bash scripts/run_arper_woz3_sclstm_plus_ssrg_overlay_v1_20260708.sh

# Bounded smoke
BOUNDED_SMOKE=1 bash scripts/run_arper_woz3_sclstm_plus_ssrg_overlay_v1_20260708.sh

# Full formal (after smoke PASS)
bash scripts/run_arper_woz3_sclstm_plus_ssrg_overlay_v1_20260708.sh
```

## SOTA target

- BLEU ≥ **0.935**; SER ≤ **2.72**

## References

- Config: `configs/ccfa_three_suite/arper_woz3_sclstm_v89_ssrg_exemplar_overlay_v1_20260708.yaml`
- SSRG module: `scripts/arper_ssrg_exemplar_selection.py`
- v89 status: `results/logs/arper_woz3_paper_aligned_exemplar500_batch128_formal_v89_status.json`
