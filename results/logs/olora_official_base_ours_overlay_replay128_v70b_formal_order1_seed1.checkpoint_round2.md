# O-LoRA Official-Base + Ours Replay128 Overlay v70b Round2 Checkpoint

- Updated: 2026-07-05
- Run: `olora_official_base_ours_overlay_replay128_v70b_formal_order1_seed1`
- Base: O-LoRA official T5-large Standard CL order1 seed1.
- Overlay: ours limited replay overlay with `REPLAY_PER_TASK=128`.
- W&B project: `lora-ours`
- W&B group: `published-base-standard-olora-plus-ours-overlay-v70b-formal`

## Round Metrics

- Round 1 dbpedia: EM `98.8026`, ROUGE-L `98.8026`, exit `0`.
- Round 2 amazon: EM `74.5724`, ROUGE-L `80.9035`, exit `0`.
- Round 2 amazon/SC: EM `50.3421`, ROUGE-L `63.0044`.
- Round 2 dbpedia retention: EM `98.8026`, ROUGE-L `98.8026`.

## Comparison

- v69 round2 amazon/SC EM was `49.1184`; v70b round2 is `+1.2237`.
- v57 official O-LoRA round2 amazon/SC EM was `49.6316`; v70b round2 is `+0.7105`.
- No OOM, SIGTERM, Traceback, or duplicate-key failure observed through round2.

## Decision

- Continue the formal run through yahoo and agnews.
- Do not claim final improvement until round4 cumulative metrics complete.
