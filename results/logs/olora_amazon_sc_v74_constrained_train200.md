# v74 Amazon/SC Constrained-Label Scoring Diagnostic

- Split: `train`
- Leakage policy: train-only, no dev/test split, no test targets/confusion.
- Accuracy: `65.5` over `200` examples.
- Prediction counts: `{'very negative': 61, 'neutral': 41, 'very positive': 58, 'negative': 28, 'positive': 12}`

## Per-Label Accuracy

- `very negative`: `91.4894` (43/47)
- `negative`: `47.5` (19/40)
- `neutral`: `61.1111` (22/36)
- `positive`: `23.6842` (9/38)
- `very positive`: `97.4359` (38/39)

## Top Confusions

- `positive` -> `very positive`: `18`
- `negative` -> `very negative`: `15`
- `positive` -> `neutral`: `11`
- `neutral` -> `negative`: `8`
- `negative` -> `neutral`: `6`
- `neutral` -> `very negative`: `3`
- `very negative` -> `neutral`: `2`
- `neutral` -> `positive`: `2`
- `very positive` -> `positive`: `1`
- `very negative` -> `negative`: `1`
