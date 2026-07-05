# O-LoRA Official-Base + Ours Overlay v70b Smoke Gate

- Updated: 2026-07-05
- Run: `olora_official_base_ours_overlay_replay128_v70b_smoke_order1_seed1`
- Base: O-LoRA official T5-large Standard CL order1 seed1.
- Overlay: ours limited replay overlay with `REPLAY_PER_TASK=128`.
- W&B project: `lora-ours`
- W&B group: `published-base-standard-olora-plus-ours-overlay-v70b-smoke`

## Smoke Metrics

- Round 1 dbpedia: EM `98.5`, exit `0`.
- Round 2 amazon/SC: EM `46.5`, exit `0`.
- Round 3 yahoo smoke surface: amazon/SC EM `45.0`, exit `0`.
- Round 4 agnews smoke surface: amazon/SC EM `44.0`, exit `0`.

## Gate Decision

- No OOM, SIGTERM, Traceback, or duplicate-key loader failure.
- v70b is much healthier than the rejected v70 SC-calibration/multiplier branch, which had amazon/SC EM `31.0` and failed with duplicate keys.
- Smoke metrics do not establish a SOTA claim because `MAX_STEPS=20` and `MAX_PREDICT_SAMPLES=200` are active.
- Proceed to no-cap formal to test whether the larger replay cap improves the final published-base overlay result.
