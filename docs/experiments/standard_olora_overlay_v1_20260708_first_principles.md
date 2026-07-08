# Standard O-LoRA Overlay v1 — First Principles (2026-07-08)

**Branch:** `ours-v1-20260708`  
**Base anchor:** O-LoRA v57 EM **76.81**; best overlay v69 EM **77.26**

## Diagnosis

- Bottleneck: amazon/SC retention ~54% (v69 formal).
- v83 class-coverage SSRG ordering designed but not promoted to smoke.
- v85 assess gate at threshold 0.3 may be too strict for amazon round-2 learning.

## v1 Single-Mechanism Delta (from v69/v85)

| Mechanism | v85 | v1 |
|-----------|-----|-----|
| `REPLAY_MODE` | ssrg | ssrg (unchanged) |
| `SC_CLASS_COVERAGE_ORDER` | 0 | **1** |
| `ASSESS_RETENTION_THRESHOLD` | 0.3 | **0.25** |
| Official O-LoRA entry / scorer | unchanged | unchanged |

**Hypothesis:** class-coverage ordering in SSRG replay improves SC label diversity retention; slightly relaxed assess gate allows more amazon learning while blocking extreme interference.

## Early gate (smoke)

| Segment | Criterion |
|---------|-----------|
| dbpedia (round 1) | EM ≥ **90** |
| amazon (round 2) | EM ≥ **50** (above v69 ~54% baseline target) |
| stop | dbpedia EM < 90 or amazon EM < 45 |

## Launch

```bash
DRY_RUN=1 bash scripts/run_olora_standard_order1_official_base_ours_overlay_v1_20260708.sh

tmux new-session -d -s lora-ours-standard-v1-smoke \
  "cd /root/lora-ours && bash scripts/run_olora_standard_order1_official_base_ours_overlay_v1_20260708.sh"

FORMAL=1 bash scripts/run_olora_standard_order1_official_base_ours_overlay_v1_20260708.sh
```

## SOTA target

- Final avg EM ≥ **84.5** (= 76.7 + (100−76.7)/3)

## References

- Launcher: `scripts/run_olora_standard_order1_official_base_ours_overlay_v1_20260708.sh`
- v83 class-coverage: `results/logs/olora_v83_classcov_acquisition_gate.md`
- v69 metrics: `results/logs/olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1.final_metrics.md`
