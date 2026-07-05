# Amazon/SC Prediction Confusion Diagnostics

## olora_official_base_ours_overlay_replay64_sc_calib_only_v73_smoke_r2_order1_seed1 / 2-amazon

- Path: `results/runs/olora_official_base_ours_overlay_replay64_sc_calib_only_v73_smoke_r2_order1_seed1/official_outputs/2-amazon/predict_eval_predictions.jsonl`
- Accuracy: `31.0` over `200` amazon/SC examples.
- Prediction counts: `{'very negative': 13, 'neutral': 151, 'very positive': 19, 'negative': 12, 'positive': 5}`

### Per-Label Accuracy

- `very negative`: `21.7391` (10/46)
- `negative`: `9.0909` (4/44)
- `neutral`: `100.0` (36/36)
- `positive`: `4.5455` (2/44)
- `very positive`: `33.3333` (10/30)

### Top Confusions

- `negative` -> `neutral`: `38`
- `positive` -> `neutral`: `32`
- `very negative` -> `neutral`: `28`
- `very positive` -> `neutral`: `17`
- `positive` -> `very positive`: `9`
- `very negative` -> `negative`: `8`
- `very positive` -> `positive`: `3`
- `negative` -> `very negative`: `2`
- `positive` -> `very negative`: `1`
