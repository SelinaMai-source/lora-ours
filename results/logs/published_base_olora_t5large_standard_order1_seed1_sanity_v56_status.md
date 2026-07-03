# published_base_olora_t5large_standard_order1_seed1_sanity_v56

- updated: 2026-07-03T17:03:48
- state: running
- reason: 
- selected base: O-LoRA official T5-large Standard CL order1 seed1
- label: published-base O-LoRA Standard order1 sanity, not paper-comparable
- W&B project: lora-ours-published-base-standard
- W&B group: published-base-standard-olora-order1-sanity
- log: /root/lora-ours/results/logs/published_base_olora_t5large_standard_order1_seed1_sanity_v56.log
- manifest: /root/lora-ours/results/runs/published_base_olora_t5large_standard_order1_seed1_sanity_v56/run_manifest.json
- setting: dbpedia -> amazon -> yahoo -> agnews; max_steps=20; max_train_samples=160; max_predict_samples=200
- gradient_accumulation_steps: 1
- engineering notes: single-GPU diagnostic runtime; runtime copy only unblocks W&B env handling; sample/step caps are non-paper-comparable when present
