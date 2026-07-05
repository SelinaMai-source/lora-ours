# v73 SC Calibration Smoke Launch Failure

- Run: `olora_official_base_ours_overlay_replay64_sc_calib_only_v73_smoke_order1_seed1`
- Stage: round1 dbpedia, before amazon/SC gate.
- Outcome: launch/config failure, not a method-performance result.
- Cause: `MAX_TRAIN_SAMPLES=-1` was passed through to `--max_train_samples -1`, and HuggingFace `Dataset.select(range(-1))` raised `IndexError: index out of bounds`.
- Action: stop stale monitor and relaunch with a clean run name and no negative train cap.
- Leakage policy: v73 remains based on train-only diagnostics and generic SC label verbalizer calibration; no test confusion/targets are used to tune the rule.
