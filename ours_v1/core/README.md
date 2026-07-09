# Core

`ours_v1/core/` is a map to the shared algorithmic contribution code. The
implementation remains in `core/methods/` so existing training imports keep
working.

## Shared Mechanisms

- `core/methods/ours_spectral_replay.py`: SSRG spectral sparse replay.
- `core/methods/assess_update.py`: Assess-then-Update decision logic.
- `core/methods/router.py`: router and branch selection.
- `core/methods/drift_detector.py`: drift detection utilities.
- `core/methods/overlap_loss.py`: anti-overlap regularization.

Suite-specific overlays live under `ours_v1/suites/` because they patch or wrap
published baselines rather than replacing the shared training core.
