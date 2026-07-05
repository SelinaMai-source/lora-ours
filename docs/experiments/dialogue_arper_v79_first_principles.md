# Dialogue ARPER v79 First-Principles Note

## Failure Chain

v78 restored Book, but Inform and NoBook still underperform because the train-support prototype is selected by a single centrality score. For Dialogue NLG, the correct output is not merely the nearest sentence in support space: it must preserve all slots, express the dialogue act, avoid prompt-like fragments, and use a stable natural-language pattern for that act/slot signature.

## v79 Hypothesis

Within each train-only DA/slot signature, candidates can be ranked more reliably by combining support centrality with semantic evidence: slot coverage, language pattern cluster size, length stability, and act-compatible wording. This should help NoBook single-slot cases such as `people`, `stay`, and `time`, where ROUGE centrality alone is low but the support candidates still share a clear rejection pattern.

## Implementation Boundary

- No dev/test targets are used for template selection.
- Candidate prototypes are selected only from segment training support outputs.
- Book remains protected by the same margin/risk gate; v79 changes only how the support prototype is selected.
- `booking:nobook:none` remains excluded because its support bucket is frequent but semantically diffuse.

