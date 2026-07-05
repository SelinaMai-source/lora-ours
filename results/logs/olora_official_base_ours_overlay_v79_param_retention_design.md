# v79 Parameter-Space Retention Diagnostic

## Design

- Base remains O-LoRA official T5-large Standard CL + ours overlay; no base replacement.
- Diagnostic reads only `adapter_model.bin` files and train-heldout diagnostic logs. It does not read `dev.json`, `test.json`, test predictions, or test targets.
- Anchor: amazon round2 adapter. Because O-LoRA expands adapter rank each round (`r2=16`, `r3=24`, `r4=32`), the diagnostic compares the shared anchor-rank slice and separately measures new-rank energy.
- Candidate regularizer considered: parameter-space retention, i.e. penalize yahoo/agnews updates that move amazon-sensitive LoRA parameters away from the amazon round2 anchor.

## Findings

- Within the same run, the shared amazon anchor-rank slice is exactly preserved: relative L2 `0.0`, cosine `1.0` for v69 r3/r4 relative to v69 r2, and also for v70b r3/r4 relative to v70b r2.
- Therefore a naive L2/EWC regularizer on existing amazon LoRA parameters would be redundant: O-LoRA already freezes the previous subspace.
- New-rank energy grows after later rounds, but it does not by itself explain failure: v69 r4 has new-rank energy ratio `0.4216` with heldout EM `55.2`, while v70b r4 has similar new-rank ratio `0.4231` but heldout EM `47.2`.
- Cross-run comparison against v69 anchor shows v70b has nonzero shared-subspace drift, especially in `lora_B` (relative L2 `0.3730`, cosine `0.9305`), because v70b learned a different amazon adapter under replay128. This correlates with weaker heldout (`51.4` at r2/r3) and final collapse (`47.2`), but it is not a later-round drift that a post-amazon retention penalty can fix.

## Gate Decision

- No v79 parameter-space candidate passes v76 gate.
- Do not launch smoke/formal from naive L2/EWC retention: it would constrain parameters that are already frozen and would not address the observed failure.
- Current Standard best remains v69 formal: EM `77.2566`, ROUGE-L `81.1919`.

## Next Candidate

- The next plausible train-only candidate is not parameter-distance retention, but new-rank interference control: penalize the later-round new-rank LoRA contribution when it changes amazon train-heldout predictions or overlaps the amazon anchor output subspace.
- This should be implemented only as a small hook/prototype first, then evaluated by v76 train-heldout gate before any smoke/formal.
