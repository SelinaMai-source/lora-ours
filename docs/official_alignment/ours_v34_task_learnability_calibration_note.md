# Ours v34 Task Learnability Calibration Note

Generated: 2026-07-02

This is an implementation and experiment-control note. It does not claim SOTA.

## Task1714 Audit

- Raw task `task1714_convai3_sentence_generation` has `2295` instances and no multi-reference instances.
- Processed train split has `500` examples, `434` unique normalized targets, no prompt-template target contamination, and only `53/500` exact bare `yes/no/i` targets.
- Processed eval split has `100` examples, `90` unique normalized targets, and only `11/100` exact bare `yes/no/i` targets.
- First-token prior is real but not label-only: train first tokens are `no=250`, `yes=113`, `i=80`; eval first tokens are `no=60`, `yes=23`, `i=6`.
- Target length prior is short open text: train word length median `9`, eval word length median `9`.

## Learnability Probe

- Run: `citb_task1714_overfit_probe_v34`.
- Setup: overfit the first `8` task1714 train examples for `80` steps with LoRA `r=16`, `lr=2e-4`, target supervision guard, and continuation weighting enabled.
- Result was negative under the current adapter/training path:
  - teacher-forced token accuracy stayed at `0.2812`.
  - open-loop exact match stayed `0/8`.
  - prefix-1 stayed `3/8`, while prefix-3 and prefix-5 stayed `0/8`.
  - final generated outputs were still bare `no` for all 8 examples.

This means v34 should not assume that more target-only supervision is the main fix. The observed failure is consistent with an adapter/training or autoregressive decode calibration issue.

## Baseline Context

- No current strict sequential/replay task1714 tables were found in the local `results/tables`.
- Older local strict 19-segment ours artifacts reached `task1714` current task-aware `0.1825`, above v33's `0.08`, so the task is not obviously capped at v33 levels.
- v31/v32 only reached `0.12` through retry acceptance and still produced prompt-template-like continuations, so v34 keeps retry disabled.

## V34 Change

- Keep official prompt, target labels, raw decoded prediction, and ROUGE-L scoring unchanged.
- Add a generic `eval_normalization.generation_overrides` mechanism that matches segment names and passes deterministic generation kwargs per segment.
- For v34 smoke, apply the override to `task1714` / `sentence_generation` segments:
  - `num_beams=4`
  - `no_repeat_ngram_size=3`
  - `encoder_no_repeat_ngram_size=3`
  - `repetition_penalty=1.05`
  - `min_new_tokens` from a train-target length prior: 25th percentile, floored at `4`, capped at `10`

The length prior uses only training targets from the same segment and is config-gated, so it is not benchmark answer leakage.

## Smoke Gate

- Run: `citb_instrdialog_order1_seed1_ours_v34_smoke_strict`.
- W&B: project `lora-ours-v34`, run `6bobb6qt`.
- Monitor: `results/logs/ours_v34_strict_status.md`.
- Interim through segment2: `[0.46, 0.29, 0.28]` task-aware; seen task-aware after segment2 `0.3367`. Segment3 pending.
- Do not launch full strict unless:
  - segment2 stays near healthy;
  - segment3 improves without retry acceptance;
  - task1714 debug outputs are non-bare and not prompt-template continuations.
