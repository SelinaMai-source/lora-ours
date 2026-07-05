# olora_official_base_ours_overlay_replay64_sc_calib_only_v73_smoke_r2_order1_seed1

- updated: 2026-07-05T19:52:39
- state: stopped
- reason: 
- selected base: O-LoRA official T5-large Standard CL order1 seed1
- overlay: Ours limited replay overlay on training config only
- replay_per_prior_task: 64
- sc_balanced_replay: 0
- sc_lexical_repair: 0
- label: v73_sc_calib_only_smoke_r2
- W&B project: lora-ours
- W&B group: published-base-standard-olora-plus-ours-overlay-v73-sc-calib-smoke
- log: /root/lora-ours/results/logs/olora_official_base_ours_overlay_replay64_sc_calib_only_v73_smoke_r2_order1_seed1.log
- manifest: /root/lora-ours/results/runs/olora_official_base_ours_overlay_replay64_sc_calib_only_v73_smoke_r2_order1_seed1/run_manifest.json
- overlay manifest: /root/lora-ours/results/runs/olora_official_base_ours_overlay_replay64_sc_calib_only_v73_smoke_r2_order1_seed1/ours_overlay_manifest.json
- setting: dbpedia -> amazon -> yahoo -> agnews; max_steps=20; max_predict_samples=200
- gradient_accumulation_steps: 8
- comparability: ours-overlay diagnostic; not an official-base result and not paper-comparable while smoke caps/single-GPU runtime are present
- preserved: official task order, official entry/scorer, T5-large, O-LoRA adapter chain, current-round dev/test configs, cumulative test metric surface

- stop_reason: early gate failed; round2 amazon/SC EM 31.0; no formal launch.
