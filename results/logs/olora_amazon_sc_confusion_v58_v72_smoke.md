# Amazon/SC Prediction Confusion Diagnostics

## olora_official_base_ours_overlay_replay64_v58_smoke_order1_seed1 / 2-amazon

- Path: `results/runs/olora_official_base_ours_overlay_replay64_v58_smoke_order1_seed1/official_outputs/2-amazon/predict_eval_predictions.jsonl`
- Accuracy: `28.0` over `200` amazon/SC examples.
- Prediction counts: `{'neutral': 171, 'very positive': 15, 'very negative': 12, 'positive': 2}`

### Per-Label Accuracy

- `very negative`: `21.7391` (10/46)
- `negative`: `0.0` (0/44)
- `neutral`: `100.0` (36/36)
- `positive`: `2.2727` (1/44)
- `very positive`: `30.0` (9/30)

### Top Confusions

- `negative` -> `neutral`: `42`
- `positive` -> `neutral`: `37`
- `very negative` -> `neutral`: `36`
- `very positive` -> `neutral`: `20`
- `positive` -> `very positive`: `6`
- `negative` -> `very negative`: `2`
- `very positive` -> `positive`: `1`

## olora_official_base_ours_overlay_replay64_sc_lexical_v72_smoke_order1_seed1 / 2-amazon

- Path: `results/runs/olora_official_base_ours_overlay_replay64_sc_lexical_v72_smoke_order1_seed1/official_outputs/2-amazon/predict_eval_predictions.jsonl`
- Accuracy: `27.0` over `200` amazon/SC examples.
- Prediction counts: `{'neutral': 158, 'very positive': 13, 'positive': 16, 'negative': 2, 'very negative': 11}`

### Per-Label Accuracy

- `very negative`: `19.5652` (9/46)
- `negative`: `2.2727` (1/44)
- `neutral`: `86.1111` (31/36)
- `positive`: `13.6364` (6/44)
- `very positive`: `23.3333` (7/30)

### Top Confusions

- `negative` -> `neutral`: `41`
- `very negative` -> `neutral`: `36`
- `positive` -> `neutral`: `32`
- `very positive` -> `neutral`: `18`
- `positive` -> `very positive`: `6`
- `neutral` -> `positive`: `5`
- `very positive` -> `positive`: `5`
- `negative` -> `very negative`: `2`
- `very negative` -> `negative`: `1`
