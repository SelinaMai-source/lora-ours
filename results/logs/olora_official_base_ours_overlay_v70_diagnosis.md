# O-LoRA Official-Base + Ours Overlay v70 Diagnosis

- Updated: 2026-07-05
- Prior best: `olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1`
- Final v69 EM / ROUGE-L: `77.2566` / `81.1919`
- Reference: v57 official O-LoRA `76.8059`, LB-CL `76.7`

## Diagnosis

- v69 has no runtime blocker: all rounds exited with `ROUND_EXIT_CODE:0`, and final pipeline exit was `0`.
- The weakest final task is amazon/SC: EM `54.4079`, ROUGE-L `70.1491`.
- v57 official O-LoRA also has amazon/SC as the weak task: final amazon/SC EM `50.9868`.
- v69 improves amazon/SC over v57, but confusion analysis shows remaining ordinal sentiment boundary errors.
- v69 final SC per-label pattern: `positive` remains weakest; common errors include `positive -> very positive` and `neutral -> negative`.

## v70 Increment

- Keep O-LoRA official-base runtime and task order unchanged.
- Add optional SC label calibration in the runtime copy only: the instruction tells the model to use exact option text and distinguishes moderate vs strong sentiment labels.
- Add optional amazon replay multiplier for later rounds: prior amazon replay entries can be repeated while still using official train data and random sampling.
- No test target or oracle is used.

## Gate

- Run v70 smoke/early-gate with `SC_LABEL_CALIBRATION=1`, `AMAZON_REPLAY_MULTIPLIER=4`, `REPLAY_PER_TASK=64`.
- If smoke is healthy and amazon/SC does not collapse, launch the no-cap formal run.
