# O-LoRA Official-Base + Ours Overlay v70 Smoke Failure

- Updated: 2026-07-05
- Run: `olora_official_base_ours_overlay_sc_calib_amazon4_v70_smoke_order1_seed1`
- Base: O-LoRA official T5-large Standard CL order1 seed1.
- Overlay attempt: SC label calibration plus amazon replay multiplier 4.
- W&B project: `lora-ours`
- W&B group: `published-base-standard-olora-plus-ours-overlay-v70-smoke`

## Result

- Round 1 dbpedia smoke completed: EM `98.5`.
- Round 2 amazon smoke completed but failed the early gate: amazon/SC EM `31.0`.
- Round 3 yahoo failed before training/eval with HuggingFace `DuplicatedKeysError`.

## Failure Chain

- The SC instruction calibration changed the current amazon prompt and degraded early amazon/SC behavior rather than improving the ordinal sentiment boundary.
- The amazon replay multiplier was implemented by repeating the same official dataset entry in the task config.
- The official dataset loader uses deterministic keys based on task, dataset path, and row id; repeated config entries therefore produced duplicate keys.

## Decision

- Do not launch this v70 variant as formal.
- Keep the failure as a traceable negative result.
- Block `AMAZON_REPLAY_MULTIPLIER != 1` in the launcher.
- Continue with v70b: no SC prompt rewrite, no duplicate dataset entries, `REPLAY_PER_TASK=128` as the conservative replay-cap boost.
