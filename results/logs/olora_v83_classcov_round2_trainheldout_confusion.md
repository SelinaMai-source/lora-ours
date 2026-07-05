# Amazon/SC Prediction Confusion Diagnostics

## lora-ours-devdiag / outputs

- Path: `/root/autodl-tmp/lora-ours-devdiag/olora_v83_classcov_round2_trainheldout_diag/outputs/predict_eval_predictions.jsonl`
- Accuracy: `53.0` over `500` amazon/SC examples.
- Prediction counts: `{'neutral': 145, 'very negative': 148, 'very positive': 178, 'negative': 24, 'positive': 5}`

### Per-Label Accuracy

- `very negative`: `85.7143` (78/91)
- `negative`: `16.0377` (17/106)
- `neutral`: `69.1589` (74/107)
- `positive`: `3.0303` (3/99)
- `very positive`: `95.8763` (93/97)

### Top Confusions

- `positive` -> `very positive`: `67`
- `negative` -> `very negative`: `56`
- `negative` -> `neutral`: `30`
- `positive` -> `neutral`: `27`
- `neutral` -> `very positive`: `14`
- `neutral` -> `very negative`: `12`
- `very negative` -> `neutral`: `10`
- `neutral` -> `negative`: `5`
- `very positive` -> `neutral`: `4`
- `negative` -> `very positive`: `3`
