# Dialogue ARPER v69 First-Principles Note

## Task Contract

Dialogue NLG receives a dialogue-act / slot-conditioned input and must generate a natural utterance that preserves required slot values. The comparable ARPER WOZ3 target is corpus BLEU-4 plus SER: good output needs non-empty fluent text, adequate length, lexical overlap with references, slot preservation, and task/domain-act conditioning.

## v68 Failure Chain

The v66 official SCLSTM baseline reaches BLEU4=0.63231 / SER=4.817 because it is a dialogue generator trained and decoded for the WOZ3 act-to-text contract. v68 Ours preserved SER=0.0 but failed the text-generation contract: final seen score was 0.001485, task-aware score 0.099747, BLEU mean 0.106024, ROUGE-L mean 0.130413, and prefix-1 match only 0.075. That means the model often avoided slot mistakes by producing poor or mismatched language rather than generating the target utterance style.

Routing also broke the continual-learning contract. Final routing selected b1 for 3951/6690 examples while oracle best branch counts favored b0 for 4813/6690; oracle agreement fell to 0.305. Forced task-aware branch assignment was not sufficient because the branch that originally received a task's updates was not always the branch that later best explained that task's training targets.

The official `BLEU-4` summary was null because postprocess expected `eval.arper_woz3_corpus_bleu4` or `eval.corpus_bleu4`, but v68 rows only carried sentence-level `eval.bleu_mean`.

## v69 Minimal Fix

1. Write corpus BLEU4 into eval extras and per-segment rows so the official Dialogue NLG summary reports a real BLEU-4 field.
2. Add a train-support NLL task fallback for evaluation: for a known eval task, choose the branch with the lowest NLL on that task's training targets, then generate test outputs with that branch. This uses train data only, not test references, and directly targets the causal failure where old tasks were routed to branches that no longer generated them well.
3. Use modest generation length controls in v69 configs: train-target length prior, `num_beams=2`, and no-repeat/repetition penalties. This addresses empty/too-short/degenerate language without changing the official data split or using test leakage.

v69 remains an adapted T5/Ours diagnostic row until comparability is explicitly reviewed; it must not be reported as SOTA unless the official-comparable setting is accepted.
