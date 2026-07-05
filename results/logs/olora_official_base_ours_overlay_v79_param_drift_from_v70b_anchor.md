# v79 Adapter Parameter Drift Diagnostic

- Leakage policy: adapter weights and train-heldout logs only; no dev/test predictions or targets.
- Anchor: `results/runs/olora_official_base_ours_overlay_replay128_v70b_formal_order1_seed1/official_outputs/2-amazon/adapter`

## v70b_r3

- Heldout EM: `51.4`
- Overall relative L2: `0.000000`
- Overall cosine: `1.000000`
- `lora_A` relative L2: `0.000000`, cosine: `1.000000`
- `lora_B` relative L2: `0.000000`, cosine: `1.000000`

## v70b_r4

- Heldout EM: `47.2`
- Overall relative L2: `0.000000`
- Overall cosine: `1.000000`
- `lora_A` relative L2: `0.000000`, cosine: `1.000000`
- `lora_B` relative L2: `0.000000`, cosine: `1.000000`
