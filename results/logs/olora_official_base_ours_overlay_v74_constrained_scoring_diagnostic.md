# v74 Train-Only Constrained-Label Scoring Diagnostic

## Design

- Base: keep O-LoRA official T5-large Standard CL + v69/v58 replay64 overlay path; no base replacement.
- Increment: constrained-label scoring for amazon/SC, using only the label verbalizers in `labels.json` (`very negative`, `negative`, `neutral`, `positive`, `very positive`).
- Scoring rule: teacher-forced normalized NLL over each candidate label; choose the minimum-NLL label.
- Leakage policy: train-only diagnostic; no `dev.json` because it is byte-identical to `test.json`; no test target/confusion is used to tune the rule.

## Results

- `train200`: constrained scoring EM `65.5`. This looked promising but was too small to trust.
- `train1000`: constrained scoring EM `55.3`, below the previous v73 free-generation train-only diagnostic EM `59.6`.
- Failure mode: scoring strongly over-predicts extreme labels (`very negative`: 319, `very positive`: 320) and under-predicts `positive` (37/1000 predictions).
- Train-heldout label-prior penalty check: using first 500 train examples to derive a prior correction and evaluating the last 500, beta `0.1` improves constrained scoring heldout EM from `53.2` to `58.6`, but still does not provide clear evidence over the free-generation diagnostic surface.

## Gate Decision

- Do not launch v74 smoke/formal from this scoring rule.
- Reason: the larger train-only diagnostic underperforms free generation and the train-only prior correction is only marginal; running smoke would likely consume GPU/disk without a strong no-leakage signal.
- Current Standard best remains v69 formal: EM `77.2566`, ROUGE-L `81.1919`.

## Next Step

- Avoid prompt-only calibration, replay distribution rewrites, lexical test-derived repair, and plain candidate NLL scoring.
- Next viable Standard direction should be retention-focused but train-only: e.g. create an internal train-heldout split for amazon/SC and test a small retention adapter/gating regularizer against that split before any smoke.
