# Ours Autonomous Main Table

Updated: 2026-07-12T06:59:56.320534+00:00

Sources: published base / local official anchor from frozen matrix; ours from official scorers only.

| Suite | Metric | Base | Ours | Δ | Target | Version | Status |
|-------|--------|------|------|---|--------|---------|--------|
| standard | EM | 76.81 | None | None | 84.54 | ours-v3 | failed_diagnosed |
| citb_instrdialog | AR | 39.98 | None | None | 53.31 | ours-v1 | pending |
| citb_instrdialog | BWT | None | None | None | None | ours-v1 | pending |
| citb_instrdialogpp | AR | None | None | None | None | ours-v1 | external_blocker |
| arper | BLEU4 | 0.5989 | None | None | 0.7985 | ours-v1 | pending |
| arper | SER | 5.938 | None | None | 3.9587 | ours-v1 | pending |
| todcl | BLEU | 21.7719 | None | None | 29.0292 | ours-v1 | pending |
| todcl | EER | 0.163975 | None | None | 0.109317 | ours-v1 | pending |

## Blockers / Failures

- **standard last failure:** `docs/experiments/standard_failure_v2.md`
- **citb_instrdialogpp BLOCKED:** {'type': 'external_blocker', 'reason': 'public split shortfall: 2 tasks under train=100 (task1549=42, task459=48); paper-exact not publicly available', 'updated_at': '2026-07-12T06:54:56.205296+00:00'}
