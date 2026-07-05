# Standard v62 Neutral Formal Diagnostic

- Updated: `2026-07-05T16:12:20+08:00`
- Run: `standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v62_formal_neutral_20260705`
- State: `running_segment3_eval`
- Judgment: not stuck in segment0; current log has reached segment3/agnews eval.
- Comparability: diagnostic only; do not report as published-base result.

## Evidence

- Train metric at `2026-07-05 12:07:04`: examples `14000`, batches `1750`, optimizer steps `219`, branches `1`.
- Train metric at `2026-07-05 12:39:47`: examples `5000`, batches `625`, optimizer steps `79`, branches `2`.
- Train metric at `2026-07-05 14:10:26`: examples `10000`, batches `1250`, optimizer steps `157`, branches `3`.
- Train metric at `2026-07-05 16:10:39`: examples `4000`, batches `500`, optimizer steps `63`, branches `4`.

## Eval Timing

- Segment `0` `dbpedia` eval: `1151.4161801338196` seconds.
- Segment `1` `amazon` eval: `3090.6142387390137` seconds.
- Segment `2` `yahoo` eval: `6337.1121916770935` seconds.
- Segment `3` `agnews` eval started at `2026-07-05 16:10:39` and is still running.

## Action

- Continue monitoring this diagnostic run; do not stop unless eval stalls without log/heartbeat/GPU activity.
- Do not launch O-LoRA overlay formal until this releases the GPU.
