# Ours Autonomous Campaign Status

- Updated: 2026-07-12T15:05+08:00
- Branches: **ours-v1** (controller), **ours-v2** (assess-pause; smoke FAIL amazon EM 38.5), **ours-v3** (replay_budget=96; queued)
- Running now: **CITB InstrDialog v1 smoke** (`lora-ours-citb-ours-v1-smoke`)
- Next after CITB: ARPER v1 → ToDCL v1 → Standard v3 smoke (loop round)
- InstrDialog++: **external_blocker** (public split shortfall)
- Loop: `tmux a -t lora-ours-autonomous-loop`
- Leaderboard: `results/tables/ours_iteration_leaderboard.md` and `docs/experiments/published/`

## Standard metrics so far
| Version | Delta | Smoke | Amazon EM |
|---------|-------|-------|-----------|
| ours-v1 | classcov SSRG + assess 0.25 | FAIL | 38.0 |
| ours-v2 | pause assess on amazon | FAIL | 38.5 |
| ours-v3 | REPLAY_PER_TASK=96 | pending | — |

## Targets
| Suite | Metric | Base | Target | Status |
|-------|--------|------|--------|--------|
| Standard | EM | 76.81 | 84.54 | iterating |
| CITB InstrDialog | AR | 39.98 | 53.31 | smoke running |
| CITB InstrDialog++ | — | — | — | external_blocker |
| ARPER | BLEU4/SER | 0.5989/5.938 | 0.7985/3.96 | pending |
| ToDCL | BLEU/EER | 21.77/0.164 | 29.03/0.109 | pending |
