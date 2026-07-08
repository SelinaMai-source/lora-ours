# ToDCL Ours v1 — First Principles (2026-07-08)

**Branch:** `ours-v1-20260708`  
**Base anchor:** ToDCL ADAPTER NLG 37-domain (pending completion)

## v1 Single-Mechanism Delta

**Assess-then-Update orthogonal penalty** on GPT2Adapter weights when isolated interference energy exceeds threshold.

| Parameter | Official ADAPTER | v1 overlay |
|-----------|------------------|------------|
| Training entry | `train.py` | unchanged |
| Interference detection | none | isolated energy > **0.25** |
| Penalty | none | orthogonal residual **0.1** |
| Scorer / splits | official | unchanged |

## Early gate (bounded smoke, 3 epochs)

| Checkpoint | Criterion |
|------------|-----------|
| domain 3 | BLEU ≥ anchor − 1.0 |
| stop | BLEU drop > 2.0 vs anchor mid-run |

## Launch

```bash
# Blocked until anchor completes
DRY_RUN=1 bash scripts/run_todcl_adapter_nlg_ours_assess_overlay_v1_20260708.sh

BOUNDED_SMOKE=1 bash scripts/run_todcl_adapter_nlg_ours_assess_overlay_v1_20260708.sh
```

## SOTA target

- BLEU ≥ **29.0**; EER ≤ **0.123**

## References

- Config: `configs/ccfa_three_suite/todcl_ours_v1_20260708_assess_overlay.yaml`
- Module: `scripts/todcl_assess_orthogonal_overlay.py`
- Anchor: `scripts/run_todcl_adapter_nlg_official_anchor.sh`
