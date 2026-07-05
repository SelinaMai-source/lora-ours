# Amazon/SC Prediction Confusion Diagnostics

## lora-ours-devdiag / outputs

- Path: `/root/autodl-tmp/lora-ours-devdiag/olora_v69_final_adapter_amazon_sc_trainonly1000_v73/outputs/predict_eval_predictions.jsonl`
- Accuracy: `59.6` over `1000` amazon/SC examples.
- Prediction counts: `{'negative': 253, 'neutral': 191, 'very negative': 208, 'very positive': 263, 'positive': 85}`

### Per-Label Accuracy

- `very negative`: `64.6226` (137/212)
- `negative`: `63.6364` (126/198)
- `neutral`: `50.2488` (101/201)
- `positive`: `30.5` (61/200)
- `very positive`: `90.4762` (171/189)

### Top Confusions

- `positive` -> `very positive`: `73`
- `very negative` -> `negative`: `61`
- `neutral` -> `negative`: `57`
- `positive` -> `neutral`: `53`
- `negative` -> `very negative`: `52`
- `negative` -> `neutral`: `19`
- `neutral` -> `very positive`: `17`
- `neutral` -> `very negative`: `13`
- `neutral` -> `positive`: `13`
- `very negative` -> `neutral`: `12`
