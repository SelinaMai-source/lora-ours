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

## Result

V36 completed with segment matrix task-aware `[0.51, 0.29, 0.25, 0.16]`, final task-aware AR `0.3025`, and task-aware BWT `-0.0067`. Adapter deltas settled into a middle range (`~0.0035`-`0.0042`), above v34's stale `~1e-5` but below v35's overcorrected `~1.5e-2`-`1.9e-2`.

This is not full-strict ready because segment2 remains below the v34 gate (`0.25` final / `0.26` at segment2 vs `0.28`). It is still a useful positive direction: task1714 improved to `0.16`, current debug had no bare `yes/no/i` collapse and no template continuations, though outputs remain generic `i want...` utterances.

## Follow-Up Smoke

V36b changes only LR from `2e-5` to `1.5e-5`, leaving AdamW, trainable LoRA A, `q/v` targets, `r16/alpha32`, rectification, and generation calibration unchanged. The hypothesis is that a slightly smaller update may recover segment2 while preserving the task1714 gain.

V36b completed with segment matrix task-aware `[0.53, 0.26, 0.28, 0.18]`, final task-aware AR `0.3125`, and task-aware BWT `0.0`. Adapter deltas were still meaningful but softer (`~0.0027`-`0.0032`). This clears the local smoke gate: segment2 returned to v34 health and task1714 improved beyond v34/v36 without retry/template acceptance.

## Full Strict Candidate

`configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v36b_strict.yaml` is staged as the full strict candidate. It keeps the v36b method settings unchanged and removes the 4-segment smoke truncation.
