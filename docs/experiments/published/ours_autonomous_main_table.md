# Ours Autonomous Main Table

Updated: 2026-07-12T06:34:56.180338+00:00

Sources: published base / local official anchor from frozen matrix; ours from official scorers only.

| Suite | Metric | Base | Ours | Δ | Target | Version | Status |
|-------|--------|------|------|---|--------|---------|--------|
| standard | EM | 76.81 | None | None | 84.54 | ours-v2 | proposed |
| citb_instrdialog | AR | 39.98 | None | None | 53.31 | ours-v1 | pending |
| citb_instrdialog | BWT | None | None | None | None | ours-v1 | pending |
| citb_instrdialogpp | AR | None | None | None | None | ours-v1 | external_blocker |
| arper | BLEU4 | 0.5989 | None | None | 0.7985 | ours-v1 | pending |
| arper | SER | 5.938 | None | None | 3.9587 | ours-v1 | pending |
| todcl | BLEU | 21.7719 | None | None | 29.0292 | ours-v1 | pending |
| todcl | EER | 0.163975 | None | None | 0.109317 | ours-v1 | pending |

## Blockers / Failures

- **standard last failure:** `docs/experiments/standard_failure_v1.md`
- **citb_instrdialogpp BLOCKED:** {'type': 'external_blocker', 'reason': '2 public tasks under 100/25/25 (task1549 train=42, task459 train=48); paper-exact split not publicly reproducible', 'evidence': 'results/manifests/citb_instrdialogpp_split_gate.json'}
