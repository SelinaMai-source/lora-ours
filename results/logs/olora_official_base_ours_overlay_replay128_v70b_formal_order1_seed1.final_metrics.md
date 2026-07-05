# O-LoRA Official-Base + Ours Replay128 Overlay v70b Final Metrics

- Updated: 2026-07-05
- Run: `olora_official_base_ours_overlay_replay128_v70b_formal_order1_seed1`
- Pipeline state: `completed`
- Base: O-LoRA official T5-large Standard CL order1 seed1.
- Overlay: ours limited replay overlay with `REPLAY_PER_TASK=128`.
- W&B project: `lora-ours`
- W&B group: `published-base-standard-olora-plus-ours-overlay-v70b-formal`

## Final Result

- Final EM: `75.7434`
- Final ROUGE-L: `80.9145`
- Delta vs v69 published-base overlay: `-1.5132`
- Delta vs v57 official O-LoRA reference `76.8059`: `-1.0625`
- Delta vs LB-CL reference `76.7`: `-0.9566`
- Decision: not a new best; keep v69 as current Standard best.

## Round Metrics

- Round 1 dbpedia: EM `98.8026`, ROUGE-L `98.8026`.
- Round 2 amazon: EM `74.5724`, ROUGE-L `80.9035`.
- Round 3 yahoo: EM `73.4254`, ROUGE-L `79.6396`.
- Round 4 agnews/final: EM `75.7434`, ROUGE-L `80.9145`.

## Final Per-Task EM

- `SC` / amazon: `46.9605`
- `TC`: `85.3377`
- `agnews`: `86.2237`
- `dbpedia`: `98.3026`
- `yahoo`: `71.4868`

## Diagnosis

- Replay128 helped round2 amazon/SC modestly (`50.3421` vs v69 round2 `49.1184`), so the larger cap did not break amazon acquisition.
- The final failure is retention: amazon/SC dropped to `46.9605`, far below v69 final `54.4079`.
- The same run slightly improved yahoo (`71.4868` vs v69 final `70.5`) and agnews (`86.2237` vs v69 final `85.6974`), so the regression is not a global training collapse.
- Likely cause: a global replay-cap increase changes later-round task balance and hurts the ordinal SC boundary more than it helps generic retention.

## Next Step

- Do not use global `REPLAY_PER_TASK=128` as the main line.
- Return to v69 replay64 as the stable base and test a narrower train-only amazon/SC retention module that does not duplicate dataset config entries and does not change eval/test prompts.
