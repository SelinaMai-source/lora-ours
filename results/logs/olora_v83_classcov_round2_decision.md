# v83 Class-Coverage Round2 Acquisition Diagnostic

- Run: `olora_official_base_ours_overlay_replay64_classcov_v83_round2_diag_order1_seed1`
- Candidate: O-LoRA official-base + ours replay64 overlay, full round1/round2 one-epoch training budget, official LR `1e-3`, `DO_PREDICT=0`, `STOP_AFTER_ROUND=2`.
- Increment: train-only class-coverage ordering for `amazon` `SC` `train/full`; no label resampling, no replay multiplier, no dev/test targets or confusion.
- W&B project/group: `lora-ours` / `published-base-standard-olora-plus-ours-overlay-v83-classcov-round2`.

## Result

- Training stopped as intended after round2; round1 and round2 both exited with code `0`.
- Train-heldout diagnostic source: `amazon/train[4500:5000]` only.
- Heldout EM: `53.0`; ROUGE-L: recorded in `olora_v83_classcov_round2_trainheldout_diag.log`.
- v81 acquisition gate decision: `reject`.
- Reject reasons:
  - heldout EM `53.0` does not exceed v69 round2 baseline `53.4`;
  - negative/positive prediction counts `24` / `5` are below v69 baseline `40` / `13`;
  - negative/positive per-label accuracy `16.0377` / `3.0303` is below gate thresholds `20.0` / `5.0`.

## Decision

Do not run v76 retention gate or formal training for this candidate. The full-budget class-coverage ordering recovered most of the v82 low-LR undertraining loss, but it still underperforms the v69 acquisition anchor and worsens moderate-label coverage.

## Next Candidate Direction

Keep O-LoRA published base and replay64. The next candidate should target moderate-label acquisition without reducing overall EM, for example a train-only moderate-label coverage curriculum limited to early batches plus full-budget training, then screen again with the same v81 acquisition gate before any retention/formal run.
