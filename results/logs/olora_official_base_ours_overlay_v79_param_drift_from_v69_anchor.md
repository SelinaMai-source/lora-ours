# v79 Adapter Parameter Drift Diagnostic

- Leakage policy: adapter weights and train-heldout logs only; no dev/test predictions or targets.
- Anchor: `results/runs/olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1/official_outputs/2-amazon/adapter`

## v69_r3

- Heldout EM: `53.4`
- Overall relative L2: `0.000000`
- Overall cosine: `1.000000`
- `lora_A` relative L2: `0.000000`, cosine: `1.000000`
- `lora_B` relative L2: `0.000000`, cosine: `1.000000`

## v69_r4

- Heldout EM: `55.2`
- Overall relative L2: `0.000000`
- Overall cosine: `1.000000`
- `lora_A` relative L2: `0.000000`, cosine: `1.000000`
- `lora_B` relative L2: `0.000000`, cosine: `1.000000`

## v70b_r3_from_v69_anchor

- Heldout EM: `51.4`
- Overall relative L2: `0.182622`
- Overall cosine: `0.983328`
- `lora_A` relative L2: `0.008484`, cosine: `0.999964`
- `lora_B` relative L2: `0.373039`, cosine: `0.930483`

## v70b_r4_from_v69_anchor

- Heldout EM: `47.2`
- Overall relative L2: `0.182622`
- Overall cosine: `0.983328`
- `lora_A` relative L2: `0.008484`, cosine: `0.999964`
- `lora_B` relative L2: `0.373039`, cosine: `0.930483`
