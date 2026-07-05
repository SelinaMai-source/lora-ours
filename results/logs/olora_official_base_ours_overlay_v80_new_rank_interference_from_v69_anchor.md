# v80 New-Rank Interference Diagnostic

- Leakage policy: adapter weights and train-heldout logs only; no dev/test predictions or targets.
- Anchor: `results/runs/olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1/official_outputs/2-amazon/adapter`

## v69_r3

- Heldout EM: `53.4`
- New/anchor L2 ratio: `0.372374`
- New-rank projection ratio: `0.000000`
- `q` new/anchor ratio: `0.529032`, projection: `0.000000`
- `v` new/anchor ratio: `0.300754`, projection: `0.000000`

## v69_r4

- Heldout EM: `55.2`
- New/anchor L2 ratio: `0.418910`
- New-rank projection ratio: `0.000000`
- `q` new/anchor ratio: `0.606028`, projection: `0.000000`
- `v` new/anchor ratio: `0.331679`, projection: `0.000000`

## v70b_r3_from_v69_anchor

- Heldout EM: `51.4`
- New/anchor L2 ratio: `0.377667`
- New-rank projection ratio: `0.000000`
- `q` new/anchor ratio: `0.536362`, projection: `0.000000`
- `v` new/anchor ratio: `0.305142`, projection: `0.000000`

## v70b_r4_from_v69_anchor

- Heldout EM: `47.2`
- New/anchor L2 ratio: `0.425190`
- New-rank projection ratio: `0.000000`
- `q` new/anchor ratio: `0.616543`, projection: `0.000000`
- `v` new/anchor ratio: `0.335756`, projection: `0.000000`
