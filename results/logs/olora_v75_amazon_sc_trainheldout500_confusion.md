# Amazon/SC Prediction Confusion Diagnostics

## lora-ours-devdiag / outputs

- Path: `/root/autodl-tmp/lora-ours-devdiag/olora_v75_amazon_sc_trainheldout500_v69_r2/outputs/predict_eval_predictions.jsonl`
- Accuracy: `53.4` over `500` amazon/SC examples.
- Prediction counts: `{'neutral': 228, 'very negative': 84, 'very positive': 135, 'negative': 40, 'positive': 13}`

### Per-Label Accuracy

- `very negative`: `65.9341` (60/91)
- `negative`: `26.4151` (28/106)
- `neutral`: `85.9813` (92/107)
- `positive`: `6.0606` (6/99)
- `very positive`: `83.5052` (81/97)

### Top Confusions

- `negative` -> `neutral`: `56`
- `positive` -> `very positive`: `47`
- `positive` -> `neutral`: `45`
- `very negative` -> `neutral`: `22`
- `negative` -> `very negative`: `21`
- `very positive` -> `neutral`: `13`
- `very negative` -> `negative`: `8`
- `neutral` -> `very positive`: `5`
- `neutral` -> `negative`: `4`
- `neutral` -> `positive`: `4`

## lora-ours-devdiag / outputs

- Path: `/root/autodl-tmp/lora-ours-devdiag/olora_v75_amazon_sc_trainheldout500_v69_r3/outputs/predict_eval_predictions.jsonl`
- Accuracy: `53.4` over `500` amazon/SC examples.
- Prediction counts: `{'negative': 58, 'very negative': 160, 'very positive': 185, 'neutral': 87, 'positive': 10}`

### Per-Label Accuracy

- `very negative`: `92.3077` (84/91)
- `negative`: `30.1887` (32/106)
- `neutral`: `48.5981` (52/107)
- `positive`: `5.0505` (5/99)
- `very positive`: `96.9072` (94/97)

### Top Confusions

- `positive` -> `very positive`: `69`
- `negative` -> `very negative`: `62`
- `positive` -> `neutral`: `22`
- `neutral` -> `negative`: `22`
- `neutral` -> `very positive`: `18`
- `neutral` -> `very negative`: `11`
- `negative` -> `neutral`: `9`
- `neutral` -> `positive`: `4`
- `very negative` -> `negative`: `4`
- `positive` -> `very negative`: `3`

## lora-ours-devdiag / outputs

- Path: `/root/autodl-tmp/lora-ours-devdiag/olora_v75_amazon_sc_trainheldout500_v69_r4/outputs/predict_eval_predictions.jsonl`
- Accuracy: `55.2` over `500` amazon/SC examples.
- Prediction counts: `{'negative': 107, 'very negative': 128, 'very positive': 155, 'neutral': 72, 'positive': 38}`

### Per-Label Accuracy

- `very negative`: `82.4176` (75/91)
- `negative`: `50.9434` (54/106)
- `neutral`: `38.3178` (41/107)
- `positive`: `20.202` (20/99)
- `very positive`: `88.6598` (86/97)

### Top Confusions

- `positive` -> `very positive`: `54`
- `negative` -> `very negative`: `42`
- `neutral` -> `negative`: `39`
- `positive` -> `neutral`: `21`
- `very negative` -> `negative`: `14`
- `neutral` -> `very positive`: `11`
- `neutral` -> `positive`: `10`
- `very positive` -> `positive`: `7`
- `negative` -> `neutral`: `6`
- `neutral` -> `very negative`: `6`

## lora-ours-devdiag / outputs

- Path: `/root/autodl-tmp/lora-ours-devdiag/olora_v75_amazon_sc_trainheldout500_v70b_r2/outputs/predict_eval_predictions.jsonl`
- Accuracy: `51.4` over `500` amazon/SC examples.
- Prediction counts: `{'neutral': 196, 'very negative': 121, 'very positive': 167, 'negative': 12, 'positive': 4}`

### Per-Label Accuracy

- `very negative`: `78.022` (71/91)
- `negative`: `8.4906` (9/106)
- `neutral`: `79.4393` (85/107)
- `positive`: `2.0202` (2/99)
- `very positive`: `92.7835` (90/97)

### Top Confusions

- `positive` -> `very positive`: `61`
- `negative` -> `neutral`: `52`
- `negative` -> `very negative`: `43`
- `positive` -> `neutral`: `35`
- `very negative` -> `neutral`: `17`
- `neutral` -> `very positive`: `12`
- `very positive` -> `neutral`: `7`
- `neutral` -> `very negative`: `6`
- `neutral` -> `negative`: `2`
- `negative` -> `very positive`: `2`

## lora-ours-devdiag / outputs

- Path: `/root/autodl-tmp/lora-ours-devdiag/olora_v75_amazon_sc_trainheldout500_v70b_r3/outputs/predict_eval_predictions.jsonl`
- Accuracy: `51.4` over `500` amazon/SC examples.
- Prediction counts: `{'negative': 48, 'very negative': 184, 'very positive': 189, 'neutral': 72, 'positive': 7}`

### Per-Label Accuracy

- `very negative`: `95.6044` (87/91)
- `negative`: `26.4151` (28/106)
- `neutral`: `41.1215` (44/107)
- `positive`: `4.0404` (4/99)
- `very positive`: `96.9072` (94/97)

### Top Confusions

- `positive` -> `very positive`: `71`
- `negative` -> `very negative`: `68`
- `neutral` -> `very negative`: `22`
- `neutral` -> `very positive`: `20`
- `neutral` -> `negative`: `19`
- `positive` -> `neutral`: `17`
- `positive` -> `very negative`: `7`
- `negative` -> `neutral`: `7`
- `negative` -> `very positive`: `3`
- `very negative` -> `neutral`: `2`

## lora-ours-devdiag / outputs

- Path: `/root/autodl-tmp/lora-ours-devdiag/olora_v75_amazon_sc_trainheldout500_v70b_r4/outputs/predict_eval_predictions.jsonl`
- Accuracy: `47.2` over `500` amazon/SC examples.
- Prediction counts: `{'negative': 11, 'very negative': 224, 'very positive': 191, 'neutral': 72, 'positive': 2}`

### Per-Label Accuracy

- `very negative`: `97.8022` (89/91)
- `negative`: `7.5472` (8/106)
- `neutral`: `42.0561` (45/107)
- `positive`: `0.0` (0/99)
- `very positive`: `96.9072` (94/97)

### Top Confusions

- `negative` -> `very negative`: `87`
- `positive` -> `very positive`: `74`
- `neutral` -> `very negative`: `39`
- `neutral` -> `very positive`: `19`
- `positive` -> `neutral`: `16`
- `positive` -> `very negative`: `9`
- `negative` -> `neutral`: `8`
- `negative` -> `very positive`: `3`
- `neutral` -> `negative`: `3`
- `very positive` -> `neutral`: `2`
