# Ours v36 Soft Adapter Dynamics Note

## Why

V34 kept segment2 healthy and lifted task1714 from bare-token collapse to non-trivial continuations, but adapter updates were effectively stale (`lora_param_delta_l2 ~= 1e-5`). V35 confirmed stale adapter updates were real by switching to AdamW, training LoRA A, increasing rank/capacity, adding `k/o` targets, and disabling rectification; update norms rose to `~1.5e-2`-`1.9e-2`, but segment2 dropped from `0.28` to `0.21` and task1714 dropped from `0.12` to `0.09`.

## Smoke Design

V36 is intentionally a single soft-combination smoke rather than a matrix:

- Keep v35's necessary direction: `optimizer: adamw` and `train_lora_a: true`.
- Revert capacity/target surface to v34: `r: 16`, `alpha: 32`, `dropout: 0.05`, and target modules `q/v`.
- Restore ODE rectification and lower LR from `5e-5` to `2e-5`.
- Keep v34 generation calibration, balanced task1714 sampling, official prompts/targets, raw decoded scoring, and task-aware ROUGE-L gates unchanged.

## Gate

Do not launch full strict unless smoke keeps segment2 near v34 health (`~0.28`) and improves task1714 beyond v34 (`>0.12`). If v36 fails, the next small probe should isolate the v35 knobs with one more smoke, preferably `AdamW + train_lora_a + q/v + r16 + no rectification` or `AdamW + train_lora_a + q/v + r16 + lr=1e-5`, depending on whether v36 under-updates or still overcorrects.
