# Dialogue ARPER v77 First-Principles Note

## Failure Chain

v76 achieved the best smoke BLEU so far, but task-aware score fell back because train-support templates were still allowed too broadly. The key failure is not SER: slot tokens are generally preserved. The failure is conditional language quality: for NoBook/Inform, the generated sentence must express the dialogue act naturally while matching the slot signature; replacing with a train prototype is only justified when the training support examples agree on a stable phrasing.

## v77 Hypothesis

High-frequency NoBook/Inform signatures can benefit from train-support prototypes, but sample count alone is not evidence of reliability. `booking:nobook:none` is frequent but semantically diffuse in the smoke support set, so replacing every risky output with its central prototype hurts task-aware similarity. v77 therefore requires train-only prototype centrality and restricts replacement to NoBook/Inform signatures, avoiding Book regressions.

## Implementation Boundary

- No dev/test targets are used for routing or template selection.
- Support templates are still built only from the segment training support set.
- Replacement is gated by signature whitelist plus support centrality (`support_confidence_mode: score`).
- Book is excluded from signature-template replacement; it keeps the v75/v76 repair behavior.

