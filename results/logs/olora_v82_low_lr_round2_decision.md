# v82 Low-LR Round2 Acquisition Diagnostic

- Run: `olora_official_base_ours_overlay_replay64_low_lr_v82_round2_diag_order1_seed1`
- Candidate: O-LoRA official-base + ours replay64 overlay, `amazon` round learning rate `5e-4`, `max_steps=40`, `DO_PREDICT=0`, `STOP_AFTER_ROUND=2`.
- Leakage policy: used only `amazon/train[4500:5000]` heldout for acquisition evaluation; no dev/test targets or confusion were used.
- W&B project/group: `lora-ours` / `published-base-standard-olora-plus-ours-overlay-v82-low-lr-round2`.

## Result

- Training stopped as intended after round2; round1 and round2 both exited with code `0`.
- Train-heldout diagnostic: EM `43.4`, ROUGE-L `59.9` over 500 amazon/SC heldout examples.
- v81 acquisition gate decision: `reject`.
- Reject reason: heldout EM `43.4` does not exceed v69 round2 baseline `53.4`.
- Label coverage did not collapse: prediction counts were `negative=81`, `positive=52`, versus v69 baseline `negative=40`, `positive=13`; however the acquisition objective requires exceeding the v69 heldout EM before retention/formal promotion.

## Decision

Do not run v76 retention gate or formal training for this candidate. The candidate improves moderate-label coverage but underlearns the task relative to the v69 anchor, so promoting it would risk repeating the weak-anchor failure chain seen in v70b/v72/v73.

## Next Candidate Direction

Keep the O-LoRA published base and replay64 overlay. Prefer a train-only class-coverage sampler or anchor-quality early-stop that preserves the full round2 learning budget while avoiding the v82 undertraining effect; screen first with the same v81 acquisition gate before any retention/formal run.
