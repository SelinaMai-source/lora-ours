# Dialogue ARPER v78 First-Principles Note

## Failure Chain

v77 reduced replacement scope too aggressively. It filtered low-cohesion NoBook support, which was directionally sound, but it also removed Book support-template repairs that were train-supported and had been helping slot-conditioned utterances. The result was lower BLEU and lower task-aware score than v76.

## v78 Hypothesis

Book replacement is safe when it remains margin-gated and support-centrality-gated, because Book signatures include explicit slot structure and high support consistency. NoBook should still avoid `booking:nobook:none`, whose training support is frequent but semantically diffuse. v78 therefore restores `booking:book:*` in the signature whitelist while keeping the v77 centrality confidence mode.

## Implementation Boundary

- No dev/test targets are used for choosing templates.
- Support evidence remains segment-train-only.
- Book is restored only through the existing margin/risk gate, not forced replacement.
- NoBook-none remains excluded; NoBook single/multi-slot signatures still require train prototype centrality.

