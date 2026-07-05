# O-LoRA Official-Base + Ours Replay Overlay v69 Final Metrics

- Updated: `2026-07-05T17:56:05+08:00`
- Run: `olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1`
- Pipeline exit: `0` at `2026-07-05T17:47:59+08:00`
- Final EM: `77.2566`; ROUGE-L: `81.1919`
- v57 official O-LoRA reference: `76.8059`; delta: `+0.4507`
- LB-CL reference: `76.7`; delta: `+0.5566`
- Decision: current best Standard published-base overlay candidate; no incomparable SOTA claim.

## Round Metrics

- Round 1 `dbpedia`: EM `98.8026`, ROUGE-L `98.8026`, train_loss `0.12077015165000334`.
- Round 2 `amazon`: EM `73.9211`, ROUGE-L `78.898`, train_loss `69.36635464052611`.
- Round 3 `yahoo`: EM `74.4781`, ROUGE-L `80.0636`, train_loss `75.86601783655867`.
- Round 4 `agnews`: EM `77.2566`, ROUGE-L `81.1919`, train_loss `149.87645768385667`.

## Final Per-Task EM

- `SC`: `54.4079`
- `TC`: `84.8728`
- `agnews`: `85.6974`
- `amazon`: `54.4079`
- `dbpedia`: `98.4211`
- `yahoo`: `70.5`

## Notes

- No OOM/SIGTERM/Traceback observed; `ROUND_EXIT_CODE:0` for all four rounds.
- Weakest final per-task EM is amazon/SC; use it as next diagnostic focus if further iteration is needed.
