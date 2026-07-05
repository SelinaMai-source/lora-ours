# v76 Train-Heldout Gate Controller

## Implementation

- Added optional `TRAIN_HELDOUT_GATE=1` controller to `scripts/run_olora_standard_order1_official_base_ours_overlay_v58.sh`.
- Gate source is only `amazon/train.json` heldout slice, default `[4500:5000]`; it does not read `dev.json`/`test.json` and does not use test targets/confusion.
- After round2 amazon and each later adapter, the launcher evaluates the current adapter on the train-heldout slice via `scripts/run_olora_amazon_dev_diagnostic.py`.
- Gate rejects a candidate if heldout EM drops by more than `3.0` from the best previous heldout EM, if final heldout EM is below the v69 stable baseline `55.2`, or if final moderate sentiment labels collapse (`negative`/`positive` prediction counts below the configured minimum).
- A gate rejection is recorded as controlled `rejected` state, not as a runtime crash.

## Validation

- Dry-run: `olora_v76_trainheldout_gate_dryrun`, `TRAIN_HELDOUT_GATE=1`, passed launcher preflight and manifest generation.
- Manifest records: source `amazon/train.json`, offset `4500`, limit `500`, drop tolerance `3.0`, final baseline `55.2`, leakage policy `training split only; no dev/test split or test targets/confusion`.

## Candidate Screening

- Candidate A: v69-style replay64 stable adapter-selection baseline. Heldout curve from v75 was `53.4 -> 53.4 -> 55.2`, so it passes stability but does not exceed the current v69 formal result; no new smoke is warranted.
- Candidate B: replay128 retention-capacity variant, used as the nearest retention regularizer proxy from v70b. Heldout curve was `51.4 -> 51.4 -> 47.2`, failing the drop and final-baseline checks; final predictions collapsed on moderate labels (`negative=11`, `positive=2`). It is rejected and must not be promoted to smoke/formal.

## Gate Decision

- v76 successfully installs the no-leakage promotion controller.
- No candidate currently beats v69 heldout stability, so no v76 smoke/formal is launched.
- Current Standard best remains v69 formal: EM `77.2566`, ROUGE-L `81.1919`.

## Next Step

- Use this gate to screen a genuinely new train-only retention module, such as a small overlap/orthogonality regularizer or adapter-selection policy that preserves the v69 heldout curve while improving other heldout signals.
- Do not revisit prompt-only calibration, replay distribution rewrites, lexical test-derived repair, or plain constrained NLL scoring without a stronger train-heldout signal.
