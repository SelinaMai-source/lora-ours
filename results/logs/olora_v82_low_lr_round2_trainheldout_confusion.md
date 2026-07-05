# Amazon/SC Prediction Confusion Diagnostics

## lora-ours-devdiag / outputs

- Path: `/root/autodl-tmp/lora-ours-devdiag/olora_v82_low_lr_round2_trainheldout_diag/outputs/predict_eval_predictions.jsonl`
- Accuracy: `43.4` over `500` amazon/SC examples.
- Prediction counts: `{'neutral': 166, 'very positive': 121, 'negative': 81, 'very negative': 80, 'positive': 52}`

### Per-Label Accuracy

- `very negative`: `37.3626` (34/91)
- `negative`: `32.0755` (34/106)
- `neutral`: `56.0748` (60/107)
- `positive`: `26.2626` (26/99)
- `very positive`: `64.9485` (63/97)

### Top Confusions

- `negative` -> `neutral`: `44`
- `positive` -> `very positive`: `38`
- `very negative` -> `negative`: `37`
- `positive` -> `neutral`: `31`
- `negative` -> `very negative`: `26`
- `very positive` -> `positive`: `22`
- `very negative` -> `neutral`: `19`
- `neutral` -> `very positive`: `17`
- `neutral` -> `very negative`: `16`
- `very positive` -> `neutral`: `12`
