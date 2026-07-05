# v81 Amazon Round2 Acquisition Quality Gate

- Leakage policy: amazon train-heldout only; no dev/test predictions or targets.
- Baseline `v69_r2`: EM `53.4`, negative count `40`, positive count `13`.

## v69_r2

- Decision: `pass`
- Heldout EM: `53.4`
- Negative/positive counts: `40` / `13`
- Negative/positive accuracy: `26.4151` / `6.0606`

## v70b_r2

- Decision: `reject`
- Heldout EM: `51.4`
- Negative/positive counts: `12` / `4`
- Negative/positive accuracy: `8.4906` / `2.0202`
- Reject reasons: `heldout EM 51.4 does not exceed baseline 53.4 | negative prediction count 12 < baseline 40 | positive prediction count 4 < baseline 13 | negative per-label accuracy 8.4906 < 20.0 | positive per-label accuracy 2.0202 < 5.0`

## v72_r2

- Decision: `reject`
- Heldout EM: `29.8`
- Negative/positive counts: `1` / `11`
- Negative/positive accuracy: `0.0` / `5.0505`
- Reject reasons: `heldout EM 29.8 does not exceed baseline 53.4 | negative prediction count 1 < baseline 40 | positive prediction count 11 < baseline 13 | negative per-label accuracy 0.0 < 20.0`

## v73_r2

- Decision: `reject`
- Heldout EM: `29.0`
- Negative/positive counts: `70` / `34`
- Negative/positive accuracy: `26.4151` / `11.1111`
- Reject reasons: `heldout EM 29.0 does not exceed baseline 53.4`
