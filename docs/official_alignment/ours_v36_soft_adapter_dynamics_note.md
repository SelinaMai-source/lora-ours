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

## Full Strict Launch

The full strict run `citb_instrdialog_order1_seed1_ours_v36b_strict` was launched on 2026-07-02 in tmux session `ours-v36b-full` after confirming no GPU/training conflict. W&B is online in project `lora-ours-v36b-full`, run `xprhio8d`; monitor output is `results/logs/ours_v36b_full_strict_status.md`.

The first four full strict segment scores reproduce the smoke gate exactly: task-aware `[0.53, 0.26, 0.28, 0.18]`, seen task-aware AR `0.3125`, and task-aware BWT `0.0`. This clears the early stop check, so the run should continue through the remaining full strict segments unless a later collapse, stale monitor state, or `stop_and_diagnose` artifact appears.

## Full Strict Stop

The run was stopped after segment4 triggered the low-score gate. Segment4 (`task574_air_dialogue_sentence_generation`) produced current exact `0.0`, current task-aware `0.03`, seen exact AR `0.158`, and seen task-aware AR `0.236`. The segment trajectory became task-aware `[0.53, 0.26, 0.28, 0.08, 0.03]`, so the segment3 gain from the four-segment gate did not survive the next Dialogue-generation task.

This is not a completed official full-strict run and must not be used for a SOTA claim. The saved `stop_and_diagnose.json` records the immediate failure evidence: segment4 routing had oracle agreement `0.498`, branch utilization `{"b0": 0.044, "b1": 0.356, "b2": 0.2, "b4": 0.4}`, and no `b3` utilization after segment3 had previously depended on `b3`.

## Follow-Up V37

`configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v37_smoke_strict.yaml` is a five-segment smoke, not a full run. It keeps v36b adapter and generation settings unchanged, raises router PLL recalibration from `oracle_pll_min_agreement=0.5` to `0.65`, increases `oracle_pll_bonus_steps` from `1` to `2`, and raises debug coverage to `125` examples so segment4 current-task outputs are auditable. The reason is concrete: v36b segment3 oracle agreement was only `0.6025`, below the new pre-segment4 trigger, and segment4 then collapsed while routing stopped using `b3`.

Do not proceed to another full strict unless v37 keeps segment2/task1714 healthy and segment4 no longer collapses. The next SOTA plan also has to cover InstrDialog++/Standard/Dialogue explicitly rather than treating the four-segment InstrDialog smoke as sufficient.

## V37 Result

V37 completed the five-segment smoke, but did not fix the blocker. It reproduced the v36b four-segment trajectory through segment3 (`[0.53, 0.26, 0.28, 0.18]`) and then ended at task-aware `[0.53, 0.26, 0.28, 0.10, 0.03]`, seen task-aware AR `0.2400`, exact AR `0.158`, and current segment4 task-aware `0.03`.

The router change had a real but insufficient effect: segment4 PLL recalibration triggered (`router_pll_prev_oracle_agreement=0.6025`, `router_pll_bonus_steps=2`) and oracle agreement improved from v36b `0.498` to v37 `0.546`, with `b3` used for `5%` of routes. However, task574 current debug examples were already mostly routed to `b4` with oracle `b4`, and generated generic speaker-prefixed responses such as `agent: No, we are here to assist you`, `customer: No, we didn't get a confirmation`, and `agent: I am a travel agent` instead of slot/content-specific dialogue turns.

This rules out router PLL alone as the next full-strict fix. `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v38_smoke_strict.yaml` is prepared as the next small smoke: keep v37 unchanged, but add `agent` and `customer` to `generation_continuation_first_tokens` so speaker-prefixed Dialogue targets weight content tokens after the speaker label.

## V38 Result

V38 completed the five-segment smoke, but did not fix task574. The final task-aware trajectory was `[0.53, 0.26, 0.28, 0.10, 0.03]`, with seen task-aware AR `0.2400`, exact AR `0.158`, and BWT `-0.02`. Segment3/task1714 initially matched v36b/v37 at `0.18`, then fell to `0.10` after task574.

The new continuation weighting was active and strong on task574: `train.continuation_weighted_token_ratio` was about `0.87`. The task574 debug subset also showed routing was not the immediate failure inside the current task: 25/25 debug examples selected `b4` and had oracle `b4`. The remaining failure is generic speaker-prefixed content generation, e.g. `agent: No, we are here for you.` or `customer: No, we didn't get a booking from you.` instead of slot/content-specific AIR dialogue turns.

`configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v39_smoke_strict.yaml` is the next small smoke. It keeps v38 unchanged except for adding `agent` and `customer` to `balanced_generation_sampling.buckets`, because v38's sampling buckets (`no`/`yes`/`i`/`other`) made task574 speaker-prefixed targets collapse into the generic `other` bucket and therefore did not activate balanced generation sampling for the failing segment.
