# CITB Ours v1 — First Principles (2026-07-08)

**Branch:** `ours-v1-20260708`  
**Base anchor:** Replay(50) formal_v56 ROUGE-L AR **39.98** (paper 40.4)

## Diagnosis (from v85)

- v85 smoke AR **24.6**; retention collapses after segment 1.
- v85 used `replay_ratio=0.2`, `min_replay_per_segment=8` (~8% batch replay).
- Published Replay(50) base uses **50% replay** budget.

## v1 Single-Mechanism Delta

| Parameter | v85 | v1 |
|-----------|-----|-----|
| `spectral_replay.replay_ratio` | 0.2 | **0.5** |
| `spectral_replay.min_replay_per_segment` | 8 | **64** |
| SSRG `spectral_top_k` / `energy_threshold` | 8 / 0.85 | unchanged |
| Router / drift / overlap | on | unchanged |

**Hypothesis:** aligning replay mass to the published Replay(50) base lifts seen-average ROUGE-L AR without new modules.

## Early gate (smoke, 5 segments)

| Checkpoint | Criterion |
|------------|-----------|
| segment 2 | seen TA ROUGE-L AR ≥ **35** (v85 was 31.25) |
| segment 4 | seen TA ROUGE-L AR ≥ **40** |
| stop | segment 2 seen TA AR < **20** |

## Launch

```bash
# Preflight
DRY_RUN=1 bash scripts/run_citb_ours_v1_20260708_smoke.sh

# Smoke (tmux, when GPU free)
tmux new-session -d -s lora-ours-citb-ours-v1-smoke \
  "cd /root/lora-ours && bash scripts/run_citb_ours_v1_20260708_smoke.sh"

# Formal (after smoke PASS)
FORMAL=1 bash scripts/run_citb_ours_v1_20260708_smoke.sh
```

## SOTA target

- ROUGE-L AR ≥ **53.87** (= 40.4 + (53.87−40.4)/3 margin path)
- Promotion: smoke early gate → 19-task formal

## References

- Config: `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v1_20260708_smoke_strict.yaml`
- v85 result: `results/logs/citb_instrdialog_order1_seed1_ours_v85_smoke_strict_status.md`
- Base lock: `docs/experiments/published_base_anchors_locked_20260708.md`
