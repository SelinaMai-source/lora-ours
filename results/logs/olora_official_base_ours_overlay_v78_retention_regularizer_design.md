# v78 Train-Only Behavior-Retention Regularizer Candidate

## Design

- Base remains O-LoRA official T5-large Standard CL + ours overlay; no base replacement.
- RP(LoRA) mapping: use the amazon round2 adapter as a LoRA-bank teacher anchor; use train-heldout prompt behavior as prototype/router signal; use v76 drift gate to reject updates that shift away from the teacher or collapse moderate labels.
- Candidate regularizer: during later yahoo/agnews training, penalize divergence from the amazon adapter on `amazon/train[4500:5000]` prompts. The lightweight prototype evaluates this by comparing teacher predictions/label distribution from v69 round2 against later adapters.
- Leakage policy: only train-heldout prompts and training labels for v76 gate; no `dev.json`, no `test.json`, no test targets/confusion.
- Full training hook was not launched because a true teacher-logit KL hook would require loading an additional T5-large adapter during training, which is high risk under the current single-GPU/disk constraints. This v78 step is therefore a lightweight diagnostic/simulation and hook design, as allowed by the task.

## Diagnostic Inputs

- Teacher anchor: v69 round2 amazon adapter on `amazon/train[4500:5000]`, heldout EM `53.4`.
- Candidate 1: v69 round4/final adapter, heldout EM `55.2`.
- Candidate 2: v70b round4/final adapter, heldout EM `47.2`.

## Gate Results

- v69 final: teacher agreement `64.2`, label distribution L1 `0.624`, heldout EM `55.2`. Rejected because it does not exceed v69 baseline `55.2` and distribution drift is above the configured tolerance.
- v70b final: teacher agreement `59.2`, label distribution L1 `0.784`, heldout EM `47.2`, moderate labels collapse (`negative=11`, `positive=2`). Rejected.

## Decision

- No v78 candidate passes v76 train-heldout gate, so no smoke/formal is launched.
- Current Standard best remains v69 formal: EM `77.2566`, ROUGE-L `81.1919`.

## Next Step

- A future v79 implementation should avoid full teacher-model KL unless memory is increased. A safer path is parameter-space retention: penalize yahoo/agnews LoRA updates from moving too far from the amazon adapter in selected LoRA matrices, then validate with v76 gate before smoke.
