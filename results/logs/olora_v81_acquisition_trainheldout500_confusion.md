# Amazon/SC Prediction Confusion Diagnostics

## lora-ours-devdiag / outputs

- Path: `/root/autodl-tmp/lora-ours-devdiag/olora_v81_acquisition_trainheldout500_v72_r2/outputs/predict_eval_predictions.jsonl`
- Accuracy: `29.8` over `500` amazon/SC examples.
- Prediction counts: `{'neutral': 411, 'very positive': 54, 'positive': 11, 'very negative': 23, 'negative': 1}`

### Per-Label Accuracy

- `very negative`: `13.1868` (12/91)
- `negative`: `0.0` (0/106)
- `neutral`: `93.4579` (100/107)
- `positive`: `5.0505` (5/99)
- `very positive`: `32.9897` (32/97)

### Top Confusions

- `negative` -> `neutral`: `99`
- `very negative` -> `neutral`: `77`
- `positive` -> `neutral`: `76`
- `very positive` -> `neutral`: `59`
- `positive` -> `very positive`: `17`
- `very positive` -> `positive`: `6`
- `negative` -> `very negative`: `6`
- `neutral` -> `very negative`: `4`
- `neutral` -> `very positive`: `3`
- `negative` -> `very positive`: `1`

## lora-ours-devdiag / outputs

- Path: `/root/autodl-tmp/lora-ours-devdiag/olora_v81_acquisition_trainheldout500_v73_r2/outputs/predict_eval_predictions.jsonl`
- Accuracy: `29.0` over `500` amazon/SC examples.
- Prediction counts: `{'neutral': 380, 'negative': 70, 'positive': 34, 'very positive': 14, 'very negative': 2}`

### Per-Label Accuracy

- `very negative`: `1.0989` (1/91)
- `negative`: `26.4151` (28/106)
- `neutral`: `88.785` (95/107)
- `positive`: `11.1111` (11/99)
- `very positive`: `10.3093` (10/97)

### Top Confusions

- `positive` -> `neutral`: `82`
- `negative` -> `neutral`: `77`
- `very positive` -> `neutral`: `67`
- `very negative` -> `neutral`: `59`
- `very negative` -> `negative`: `30`
- `very positive` -> `positive`: `20`
- `neutral` -> `negative`: `10`
- `positive` -> `very positive`: `4`
- `positive` -> `negative`: `2`
- `negative` -> `positive`: `1`
