# CITB Ours v85 Smoke — 2026-07-06

| Field | Value |
|-------|-------|
| Run | `citb_instrdialog_order1_seed1_ours_v85_smoke_strict` |
| Config | `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v85_smoke_strict.yaml` |
| tmux | `lora-ours-citb-ours-v85-smoke` (completed) |
| State | **completed** exit 0 |

## Metrics (5-segment smoke)

| Metric | Value |
|--------|-------|
| ROUGE-L AR (task-aware) | **24.6** |
| seen_avg_score | 15.8 |
| BWT | −1.25 |
| SOTA target | **53.87** |

## Early gate

**PASS** — segment 2 seen TA AR **31.25%** (>15 threshold). Still far below Replay(50) paper **40.4** and SOTA **53.87**.

## Artifacts

- `results/runs/citb_instrdialog_order1_seed1_ours_v85_smoke_strict/final_metrics.json`
- Log: `results/logs/citb_instrdialog_order1_seed1_ours_v85_smoke_strict.log`
