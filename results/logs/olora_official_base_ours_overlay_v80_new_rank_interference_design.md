# v80 New-Rank Interference Control Diagnostic

## Design

- Base remains O-LoRA official T5-large Standard CL + ours overlay; no base replacement.
- Diagnostic reads only `adapter_model.bin` files plus train-heldout logs; no `dev.json`, no `test.json`, no test predictions/targets.
- O-LoRA expands rank over rounds (`r2=16`, `r3=24`, `r4=32`). v80 estimates the later-round new-rank output contribution `B_new @ A_new` and compares it to the amazon anchor update `B_anchor @ A_anchor`.
- Candidate control considered: penalize or gate later-round new-rank updates that dominate or project onto the amazon anchor output subspace, then promote only if v76 train-heldout gate improves over v69.

## Findings

- Projection of new-rank updates onto the amazon anchor update is effectively zero for all inspected candidates. This indicates O-LoRA's new ranks are already nearly orthogonal to the anchor update in parameter-output space.
- New/anchor L2 ratio increases from r3 to r4 in both v69 and v70b, but the magnitudes are almost the same:
  - v69 r4: overall `0.4189`, q `0.6060`, v `0.3317`, heldout EM `55.2`.
  - v70b r4: overall `0.4252`, q `0.6165`, v `0.3358`, heldout EM `47.2`.
- Because v69 tolerates nearly the same new-rank energy without heldout collapse, raw new-rank energy is not a sufficient failure predictor.
- The v70b failure remains better explained by the weak amazon anchor learned under replay128 (shown in v79 cross-run `lora_B` drift) rather than by later new-rank projection interference alone.

## Gate Decision

- No v80 new-rank interference control candidate passes v76 gate.
- Do not launch smoke/formal from a naive new-rank energy/projection penalty: it would likely suppress useful v69-style capacity while not fixing the weak-anchor failure mode.
- Current Standard best remains v69 formal: EM `77.2566`, ROUGE-L `81.1919`.

## Next Step

- Future work should target amazon acquisition quality before retention, not only later-round interference. A safe train-only candidate is an amazon-anchor quality gate: require round2 amazon train-heldout EM and moderate-label coverage to exceed v69 before allowing any later replay/regularizer candidate.
- If training resources allow, a more precise interference hook would need activation/logit-level train-heldout consistency, but v78 showed a full teacher-model KL hook is risky under current GPU constraints.
