# Standard SSRG runtime hang fix — 2026-07-12

## Symptom
ours-v2 smoke hung on amazon HF dataset prepare: CPU ~100%, GPU idle, `.incomplete` arrow stuck at 0B for >15 min.

## Cause
Pure-Python `_ssrg_spectral_sample` built dense covariance over full prior-task pool (~10k+) with vocab dim up to 4096 → O(n·d²) hang.

## Fix (infra only; not a new RP mechanism)
- Cap SSRG candidate pool before spectral scoring
- Cap vocab to 256
- Prefer numpy for covariance when available

## Decision
Retry **ours-v2** (amazon assess-pause) after fix. Premature `ours-v3` replay_budget proposal from killed smoke is **not** applied as the next Standard delta until v2 completes a real metric gate.
