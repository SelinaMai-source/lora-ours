# v77 RP(LoRA)-Mapped Candidate Design and Gate Rejection

## RP(LoRA) Contribution Mapping

- LoRA bank: reuse the published-base O-LoRA round adapters as an adapter bank rather than training a new weak foundation.
- Prototype/router idea: use train-heldout amazon/SC performance as a no-leakage router signal for adapter selection.
- Drift/assess-update idea: v76 gate checks heldout EM drop and moderate-label collapse to reject harmful updates before promotion.
- Spectral replay / overlap loss: not enabled as a training change in v77, because prior replay distribution/capacity changes already showed label-collapse risk and must first pass the gate.

## Candidate A: Heldout-Selected Adapter Policy

- Policy: for amazon/SC, select the round adapter with the highest train-heldout EM; this is an explainable adapter-selection overlay on top of O-LoRA.
- No leakage: selection uses only `amazon/train[4500:5000]` diagnostics from v75.
- Gate result on v69 family: selected r4/final adapter with heldout EM `55.2`, equal to the current v69 baseline, not above it.
- Decision: reject for new smoke/formal because it does not improve over current best; it remains the control policy.

## Candidate B: Replay128 Rollback / Label-Collapse Guard

- Policy: for replay128-style retention, rollback amazon/SC to the best train-heldout adapter if final adapter collapses moderate labels.
- No leakage: rollback decision uses v75 train-heldout trajectory only.
- Gate result on v70b family: best selectable adapter is r2 with heldout EM `51.4`, below v69 baseline `55.2`; final adapter drops to `47.2` and collapses moderate labels (`negative=11`, `positive=2`).
- Decision: reject; do not launch smoke/formal.

## Outcome

- v77 provides a concrete, no-leakage adapter-selection/label-collapse guard design, but both available candidates fail the v76 promotion rule.
- No test confusion/targets were used.
- Current Standard best remains v69 formal: EM `77.2566`, ROUGE-L `81.1919`.

## Next Candidate Direction

- The next viable candidate should be a genuinely new train-only retention module, not just selection among existing adapters.
- Best next target: a small O-LoRA-compatible regularizer for rounds after amazon that penalizes drift away from the amazon adapter on train-heldout prompts, then run v76 gate before any smoke/formal.
