# v81 Amazon Round2 Acquisition Quality Gate

## Design

- Base remains O-LoRA official T5-large Standard CL + ours overlay; no base replacement.
- Gate source is only `amazon/train[4500:5000]` heldout. It does not read `dev.json`, `test.json`, test predictions, or test targets.
- Purpose: reject weak amazon round2 anchors before any later retention regularizer, adapter selection, or smoke/formal promotion.
- Baseline: v69 round2 amazon anchor on train-heldout, EM `53.4`, prediction coverage `negative=40`, `positive=13`, per-label accuracy `negative=26.4151`, `positive=6.0606`.
- Promotion rule for new acquisition candidates: exceed v69 round2 heldout EM and preserve at least v69 moderate-label coverage before entering v76 retention gate.

## Validation

- v69 r2: passes as the control baseline.
- v70b r2: rejected, EM `51.4`, `negative=12`, `positive=4`, negative accuracy `8.4906`, positive accuracy `2.0202`. This explains why replay128 learned a weak amazon anchor that later retention could not repair.
- v72 r2: rejected, EM `29.8`, extreme neutral collapse (`neutral=411`), `negative=1`. This explains lexical repair smoke failure before any formal promotion.
- v73 r2: rejected, EM `29.0`; moderate-label coverage exists but acquisition EM is far below v69. This explains why prompt-only calibration should not proceed.

## Gate Decision

- v81 acquisition gate successfully filters weak anchors using train-heldout only.
- No new acquisition candidate exceeds v69 r2 baseline, so no smoke/formal is launched.
- Current Standard best remains v69 formal: EM `77.2566`, ROUGE-L `81.1919`.

## Next Candidate

- Future candidates must improve round2 amazon acquisition first. Safe options: keep replay64 as the default, test a tiny train-only amazon round2 acquisition adjustment such as lower LR/shorter max_steps/anchor-quality early stop, or a train-only class-coverage sampler that is validated by this gate before any later-task run.
- Do not tune on amazon dev/test because `dev.json == test.json`.
