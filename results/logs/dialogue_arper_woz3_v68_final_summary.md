# Dialogue ARPER WOZ3 Ours v68 Final Summary

- Run ID: `dialogue_nlg_arper_woz3_dialogue_act_seed1_ours_strict_v68_formal`
- Status: completed, `exit_code=0`, phase `postprocess`
- Final segment: `12` / `welcome`
- Final metrics path: `/root/autodl-tmp/lora-baselines-run_v1/results/runs/dialogue_nlg_arper_woz3_dialogue_act_seed1_ours_strict_v68_formal/final_metrics.json`
- Process exit path: `/root/autodl-tmp/lora-baselines-run_v1/results/runs/dialogue_nlg_arper_woz3_dialogue_act_seed1_ours_strict_v68_formal/process_exit.json`

## Final Metrics

- `eval.seen_avg_score`: 0.0014852316424025657
- `eval.seen_avg_task_aware_score`: 0.0997469620812097
- `eval.bleu_mean`: 0.10602405455366919
- `eval.rouge_l_mean`: 0.13041270522926432
- `eval.slot_error_rate`: 0.0
- CCFA official Dialogue NLG `BLEU-4`: null in v68 postprocess
- CCFA official Dialogue NLG `SER`: 0.0

## Failure Chain

v68 did not meet the Dialogue target. It preserved slot values but failed the language generation objective: final prefix-1 match was 0.075186846, prefix-3 match was 0.012855007, and 6245/6690 examples had bad prefix mismatch.

Routing was also misaligned. Final routing selected `b1` for 3951/6690 examples, while oracle best branch counts favored `b0` for 4813/6690. Oracle agreement was 0.305231689, so many old-task examples were generated with branches that did not best explain their targets.

The official BLEU-4 field was null because v68 did not export `eval.arper_woz3_corpus_bleu4` or `eval.corpus_bleu4` into the per-segment table consumed by postprocess; only sentence-level `eval.bleu_mean` was present.
