# v84 Moderate Curriculum Pause Decision

- Run: `olora_official_base_ours_overlay_replay64_moderate_v84_round2_diag_order1_seed1`
- Candidate: O-LoRA official-base + replay64 + train-only moderate-label curriculum.
- Status: stopped before round1 completed.
- Reason: user introduced a hard official-repo reproducibility gate requiring each suite to document and verify official published method code/status before launching further ours increments.
- GPU action: `standard-v84-moderate-round2` and monitor tmux sessions were stopped; sentinel remains running.
- Result policy: no v84 metric should be reported, compared, or used for promotion because it did not complete round2 or train-heldout gate.

## Preserved Evidence

- Log: `results/logs/olora_official_base_ours_overlay_replay64_moderate_v84_round2_diag_order1_seed1.log`
- Status: `results/logs/olora_official_base_ours_overlay_replay64_moderate_v84_round2_diag_order1_seed1_status.md`
- Manifest: `results/runs/olora_official_base_ours_overlay_replay64_moderate_v84_round2_diag_order1_seed1/run_manifest.json`

## Next Gate

No additional Standard PEFT ours candidate should run until the O-LoRA official-base status remains documented and any alternative published base code claim, especially LB-CL, is resolved or explicitly marked as paper-only.
