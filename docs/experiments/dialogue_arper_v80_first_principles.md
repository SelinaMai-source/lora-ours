# Dialogue ARPER v80 First-Principles Note

## Failure Chain

v79 improved Book and slightly improved global task-aware score, but Inform and NoBook matrix entries did not move. The remaining low-scoring Inform examples are often not exact train-support signature matches, so support-template selection cannot fire. The model then emits slot-listing fragments such as `the addr is ...`, which preserve SER but fail Dialogue NLG's natural-language objective.

## v80 Hypothesis

When an Inform output is a generic slot listing or prompt-like fragment, a deterministic slot-semantic template built only from the input DA/slot features is safer than leaving the fragment unchanged. This is not target leakage: the template uses no dev/test output, and it only fires when the model generation is already structurally risky.

## Implementation Boundary

- Keep v79 train-support semantic prototype selection.
- Add semantic template repair only for `Inform`.
- Require the replacement template to cover all expected slot tokens.
- Do not modify Book/NoBook replacement policy in v80.

