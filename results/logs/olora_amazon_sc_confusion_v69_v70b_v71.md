# Amazon/SC Prediction Confusion Diagnostics

## olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1 / 2-amazon

- Path: `results/runs/olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1/official_outputs/2-amazon/predict_eval_predictions.jsonl`
- Accuracy: `49.1184` over `7600` amazon/SC examples.
- Prediction counts: `{'very negative': 1093, 'neutral': 3849, 'very positive': 1778, 'positive': 173, 'negative': 707}`

### Per-Label Accuracy

- `very negative`: `52.1311` (795/1525)
- `negative`: `20.8386` (328/1574)
- `neutral`: `86.9055` (1334/1535)
- `positive`: `6.2293` (94/1509)
- `very positive`: `81.1256` (1182/1457)

### Top Confusions

- `negative` -> `neutral`: `999`
- `positive` -> `neutral`: `884`
- `positive` -> `very positive`: `527`
- `very negative` -> `neutral`: `427`
- `very negative` -> `negative`: `297`
- `negative` -> `very negative`: `239`
- `very positive` -> `neutral`: `205`
- `neutral` -> `negative`: `81`
- `very positive` -> `positive`: `62`
- `neutral` -> `very positive`: `57`

## olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1 / 4-agnews

- Path: `results/runs/olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1/official_outputs/4-agnews/predict_eval_predictions.jsonl`
- Accuracy: `54.4079` over `7600` amazon/SC examples.
- Prediction counts: `{'very negative': 1572, 'negative': 2107, 'very positive': 2120, 'neutral': 1252, 'positive': 549}`

### Per-Label Accuracy

- `very negative`: `62.9508` (960/1525)
- `negative`: `60.737` (956/1574)
- `neutral`: `40.8469` (627/1535)
- `positive`: `21.67` (327/1509)
- `very positive`: `86.8222` (1265/1457)

### Top Confusions

- `positive` -> `very positive`: `714`
- `neutral` -> `negative`: `572`
- `very negative` -> `negative`: `518`
- `negative` -> `very negative`: `418`
- `positive` -> `neutral`: `360`
- `negative` -> `neutral`: `186`
- `neutral` -> `very negative`: `124`
- `very positive` -> `positive`: `122`
- `neutral` -> `very positive`: `119`
- `neutral` -> `positive`: `93`

## olora_official_base_ours_overlay_replay128_v70b_formal_order1_seed1 / 2-amazon

- Path: `results/runs/olora_official_base_ours_overlay_replay128_v70b_formal_order1_seed1/official_outputs/2-amazon/predict_eval_predictions.jsonl`
- Accuracy: `50.3421` over `7600` amazon/SC examples.
- Prediction counts: `{'very negative': 1750, 'neutral': 3269, 'very positive': 2277, 'negative': 252, 'positive': 52}`

### Per-Label Accuracy

- `very negative`: `71.4098` (1089/1525)
- `negative`: `8.0051` (126/1574)
- `neutral`: `81.6287` (1253/1535)
- `positive`: `1.6567` (25/1509)
- `very positive`: `91.4894` (1333/1457)

### Top Confusions

- `negative` -> `neutral`: `921`
- `positive` -> `very positive`: `801`
- `positive` -> `neutral`: `667`
- `negative` -> `very negative`: `517`
- `very negative` -> `neutral`: `333`
- `neutral` -> `very positive`: `122`
- `neutral` -> `very negative`: `118`
- `very positive` -> `neutral`: `95`
- `very negative` -> `negative`: `91`
- `neutral` -> `negative`: `34`

## olora_official_base_ours_overlay_replay128_v70b_formal_order1_seed1 / 4-agnews

- Path: `results/runs/olora_official_base_ours_overlay_replay128_v70b_formal_order1_seed1/official_outputs/4-agnews/predict_eval_predictions.jsonl`
- Accuracy: `46.9605` over `7600` amazon/SC examples.
- Prediction counts: `{'very negative': 3622, 'very positive': 2572, 'neutral': 1174, 'negative': 194, 'positive': 38}`

### Per-Label Accuracy

- `very negative`: `96.3279` (1469/1525)
- `negative`: `5.5909` (88/1574)
- `neutral`: `40.5212` (622/1535)
- `positive`: `1.2591` (19/1509)
- `very positive`: `94.0975` (1371/1457)

### Top Confusions

- `negative` -> `very negative`: `1283`
- `positive` -> `very positive`: `989`
- `neutral` -> `very negative`: `649`
- `positive` -> `neutral`: `327`
- `negative` -> `neutral`: `186`
- `neutral` -> `very positive`: `184`
- `positive` -> `very negative`: `168`
- `neutral` -> `negative`: `73`
- `very positive` -> `very negative`: `53`
- `very negative` -> `negative`: `27`

## olora_official_base_ours_overlay_replay64_sc_balanced_v71_smoke_r2_order1_seed1 / 2-amazon

- Path: `results/runs/olora_official_base_ours_overlay_replay64_sc_balanced_v71_smoke_r2_order1_seed1/official_outputs/2-amazon/predict_eval_predictions.jsonl`
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
