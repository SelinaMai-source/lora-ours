# O-LoRA Official-Base + Ours Overlay v71 Smoke Failure

- Updated: 2026-07-05
- Run: `olora_official_base_ours_overlay_replay64_sc_balanced_v71_smoke_r2_order1_seed1`
- Base: O-LoRA official T5-large Standard CL order1 seed1.
- Overlay attempt: v69 replay64 plus train-only balanced amazon/SC replay sampling.
- W&B project: `lora-ours`
- W&B group: `published-base-standard-olora-plus-ours-overlay-v71-smoke`

## Result

- Round 1 dbpedia smoke completed: EM `98.5`, exit `0`.
- Round 2 amazon smoke completed but failed early gate: amazon/SC EM `28.0`, ROUGE-L `31.0`, exit `0`.
- The run was stopped early after round2 because the amazon acquisition signal collapsed.

## Diagnosis

- The balanced replay sampler itself is train-only and uses no test target or oracle.
- However, balancing the amazon replay rows changes the SC label distribution seen by the O-LoRA adapter in a way that hurts early amazon acquisition badly.
- This is worse than v70b smoke (`46.5`) and the v69 replay64 line; do not launch v71 formal.

## Decision

- Current Standard best remains v69 formal: EM `77.2566`, ROUGE-L `81.1919`.
- Do not continue the label-balanced replay line as the main next step.
- Next iteration should inspect prediction/confusion differences and use a narrower retention module than replay distribution rewriting.
