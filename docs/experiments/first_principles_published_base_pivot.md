# First-Principles Pivot: Published Base + Ours Increment

## Decision

Stop treating the generic LoRA/router generator as the main experimental base. For each suite, the base must be the strongest or most appropriate published method for that suite, and `ours` is an interpretable increment layered on top of that base.

## Dialogue NLG / ARPER WOZ3

- Published base: official ARPER SCLSTM pipeline and scorer.
- Current evidence: `arper_woz3_official_sclstm_formal_v66` reached BLEU4 `0.63231` and SER `4.817`.
- Strong prior: SCLSTM is purpose-built for dialogue-act-conditioned delexicalized NLG and already models DA/slot structure much better than the weak generic LoRA generator.
- Ours increment: train-support prototype/template selection, slot/SER safety, post-decode repair, and domain-act prototype routing around SCLSTM outputs or official SCLSTM training/evaluation flow.
- Failure chain addressed: v68-v80 LoRA route achieved low SER mostly through repair, but BLEU stayed far below the published SCLSTM base. The new route keeps the high-BLEU SCLSTM generator and applies interpretable safety/continual modules where they can improve SER or stability without replacing the base.
- Comparability: use official ARPER config, data split, checkpoint/scorer, and report raw SCLSTM plus overlay metrics separately. No test target routing or oracle selection.

Immediate Dialogue action: treat the existing post-decode repair overlay as a prototype only, then iterate it as `v81` on top of official SCLSTM with smoke first and formal only after a real gate.

## Standard T5-Large PEFT CL

- Published base: O-LoRA/LB-CL style PEFT continual-learning pipeline, not the generic ours runner as a replacement.
- Current evidence: v62 early-gate was directionally valid, but formal is blocked by persistent external SIGTERM before first eval.
- Strong prior: O-LoRA/LB-CL already encodes the correct orthogonal low-rank continual-learning bias.
- Ours increment: residual/router/replay/orthogonal-constraint modules layered on O-LoRA/LB-CL, with base metrics preserved.
- Failure chain addressed: avoid claiming gains from a weaker runner or from non-comparable ultrashort diagnostic jobs.
- Comparability: restore the formal runner or run an isolated neutral formal wrapper that invokes the same O-LoRA/LB-CL config without changing data, train budget, or evaluation.

Immediate Standard action: keep v62 SIGTERM as the blocker and prepare a neutral formal runner before more method changes.

## CITB / InstrDialog

- Published base: official CITB route, using the strongest valid published baseline/replay/AGEM/Multi-upper method only after the setting is audited.
- Current evidence: the available official script path completes as 500/50/50, while the paper setting says 500/50/100. The average-train-samples interpretation remains unresolved.
- Strong prior: CITB claims must follow its official task split and stage scripts; otherwise results are not main-claim comparable.
- Ours increment: only after the setting is resolved, layer residual/replay/router/constraint modules onto the best official baseline, not onto a private weak replacement.
- Failure chain addressed: prevents using an unresolved split as a headline claim.
- Comparability: if 500/50/100 cannot be reproduced without patched split code, record CITB as blocked or report only as diagnostic.

## Run Hygiene

- Weak-base `dialogue_nlg_arper_woz3_dialogue_act_seed1_ours_strict_v80_formal` was stopped because it conflicts with this pivot and occupied GPU.
- Preserve its smoke and partial formal evidence as diagnostics only.
- New runs must use tmux, W&B project `lora-ours` when the runner supports it, sentinel monitors, and explicit status files.

