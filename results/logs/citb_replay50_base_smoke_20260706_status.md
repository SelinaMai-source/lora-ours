# CITB Replay(50) Official Base Smoke — 2026-07-06

## Run

| Field | Value |
|-------|-------|
| Launcher | `scripts/run_citb_instrdialog_replay50_official_base_repro.sh` |
| Mode | `SMOKE=1` (implicit via bounded task order) |
| Run name | `citb_instrdialog_order1_seed1_official_script_500_50_50_tie_fixed_replay50_v55_smoke` |
| tmux | `lora-ours-citb-replay50-smoke` |
| Split policy | `official_script_500_50_50` |
| CL method | REPLAY(50) |
| Smoke task | `task848_pubmedqa_classification` (order1 first task) |
| CITB_ROOT | `/root/autodl-tmp/Lora-code/external_baselines/citb_official` |

## Outcome

- **Status:** completed
- **Exit code:** 0
- **Log:** `results/logs/citb_replay50_base_smoke_20260706.log`
- **W&B:** project `lora-ours-v55-citb-replay50-base`, run `a2jh0n4y`

## Smoke metrics (single-task, capped eval n=10)

| Metric | Value |
|--------|-------|
| eval/exact_match | 50.0 |
| eval/rougeL | 50.0 |
| train/global_step | 50 |
| train/train_loss | 1.508 |

## Gate

Official Replay(50) base anchor **smoke gate PASS**. Ours v85 overlay (`citb_instrdialog_order1_seed1_ours_v85_smoke_strict.yaml`) cleared to launch after serial queue reaches CITB ours job.

## Comparability

Smoke uses `max_steps=50`, `max_eval_samples=10`, single-task order — **not** paper-comparable formal. Formal repro uses full order1 (19 tasks), 15 epochs, `eval_steps=500`.
