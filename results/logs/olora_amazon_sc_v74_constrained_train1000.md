# v74 Amazon/SC Constrained-Label Scoring Diagnostic

- Split: `train`
- Leakage policy: train-only, no dev/test split, no test targets/confusion.
- Accuracy: `55.3` over `1000` examples.
- Prediction counts: `{'very negative': 319, 'neutral': 163, 'very positive': 320, 'negative': 161, 'positive': 37}`

## Per-Label Accuracy

- `very negative`: `82.0755` (174/212)
- `negative`: `42.9293` (85/198)
- `neutral`: `45.2736` (91/201)
- `positive`: `11.0` (22/200)
- `very positive`: `95.7672` (181/189)

## Top Confusions

- `positive` -> `very positive`: `114`
- `negative` -> `very negative`: `95`
- `positive` -> `neutral`: `46`
- `neutral` -> `negative`: `40`
- `neutral` -> `very negative`: `37`
- `very negative` -> `negative`: `28`
- `neutral` -> `very positive`: `22`
- `negative` -> `neutral`: `17`
- `neutral` -> `positive`: `11`
- `positive` -> `very negative`: `10`
