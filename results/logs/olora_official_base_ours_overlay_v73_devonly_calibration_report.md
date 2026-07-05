# v73 Dev/Diagnostic-Only Calibration Report

## Artifact policy

- v69/v70b/v72 `predict_eval_predictions.jsonl` files are official predict/test surfaces. They are allowed for diagnosis and failure reporting only, not for choosing v73 rules.
- `amazon/dev.json` exists but is byte-identical to `amazon/test.json` (`md5=e8bac3951fb0fdd8c661f29be6864087` for both), so it is blocked as a calibration split.
- v73 non-test diagnostic therefore used only `amazon/train.json` through the isolated diagnostic runner, with `uses_test_json=false` in the manifest.

## Non-test diagnostic results

- Dev-as-predict probe: `olora_v69_final_adapter_amazon_sc_devdiag_v73`, exit 0, 7600 examples, EM 54.4079, but rejected for calibration because dev==test.
- Train-only probe: `olora_v69_final_adapter_amazon_sc_trainonly1000_v73`, exit 0, 1000 examples, EM 59.6. Gold labels were balanced, while predictions over-produced `very positive` and under-produced `positive`; this supports the ordinal-confusion diagnosis but does not justify more test-tuned lexical rules.

## v73 smoke decision

- Increment tested: isolated SC label verbalizer calibration on O-LoRA official-base + replay64 overlay. No replay multiplier, no balanced replay, no lexical repair.
- First launch failed before metrics due to `MAX_TRAIN_SAMPLES=-1` being passed to HF `Dataset.select(range(-1))`; recorded separately and not treated as a method result.
- Corrected smoke: `olora_official_base_ours_overlay_replay64_sc_calib_only_v73_smoke_r2_order1_seed1`. Round1 dbpedia EM 98.5, round2 amazon/SC EM 31.0.
- Gate decision: failed early; stopped before formal. This confirms the label verbalizer calibration alone is harmful under smoke and should not replace v69.

## Current best and next step

- Current Standard best remains v69 formal: EM 77.2566 / ROUGE-L 81.1919.
- Next Standard step should avoid prompt-only SC calibration, replay distribution rewrites, and test-derived lexical repair. A safer v74 direction is a train-only constrained-label decoding/scoring diagnostic, because it can use the known SC label set without inspecting test targets and can be validated first on train-only or a newly created train-heldout split.
