# O-LoRA Official-Base + Ours Overlay v72 Smoke Failure

- Updated: 2026-07-05
- Run: `olora_official_base_ours_overlay_replay64_sc_lexical_v72_smoke_order1_seed1`
- Base: O-LoRA official T5-large Standard CL order1 seed1.
- Overlay attempt: v69 replay64 plus amazon/SC output-side lexical repair.
- W&B project: `lora-ours`
- W&B group: `published-base-standard-olora-plus-ours-overlay-v72-smoke`

## Result

- Round 1 dbpedia smoke completed: EM `98.5`, exit `0`.
- Round 2 amazon smoke completed but failed early gate: amazon/SC EM `27.0`, ROUGE-L `31.6667`, exit `0`.
- The run was stopped early after round2.

## Diagnosis

- v72 did not trigger runtime, disk, OOM, or duplicate-key failures.
- The lexical repair did not fix the replay64 20-step amazon acquisition issue.
- The earlier v58 replay64 smoke had amazon/SC EM `28.0`, so v72 is not meaningfully better than the old replay64 smoke surface.
- v70b replay128 smoke reached `46.5`, but its formal final regressed to `75.7434`, mainly due to amazon/SC retention collapse.

## Decision

- Do not launch v72 formal.
- Current Standard best remains v69 formal: EM `77.2566`, ROUGE-L `81.1919`.
- Next step should avoid test-confusion-tuned rules and instead build a dev/diagnostic-only calibration path before any new formal.
