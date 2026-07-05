# v80 New-Rank Interference Diagnostic

- Leakage policy: adapter weights and train-heldout logs only; no dev/test predictions or targets.
- Anchor: `results/runs/olora_official_base_ours_overlay_replay128_v70b_formal_order1_seed1/official_outputs/2-amazon/adapter`

## v70b_r3

- Heldout EM: `51.4`
- New/anchor L2 ratio: `0.377580`
- New-rank projection ratio: `0.000000`
- `q` new/anchor ratio: `0.536442`, projection: `0.000000`
- `v` new/anchor ratio: `0.305032`, projection: `0.000000`

## v70b_r4

- Heldout EM: `47.2`
- New/anchor L2 ratio: `0.425092`
- New-rank projection ratio: `0.000000`
- `q` new/anchor ratio: `0.616635`, projection: `0.000000`
- `v` new/anchor ratio: `0.335635`, projection: `0.000000`
