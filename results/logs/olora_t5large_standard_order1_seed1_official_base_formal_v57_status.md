# olora_t5large_standard_order1_seed1_official_base_formal_v57

- updated: 2026-07-03T18:03:50
- state: completed
- reason: 
- selected base: O-LoRA official T5-large Standard CL order1 seed1
- label: published_base_official_equivalent_single_gpu_formal_olora_order1
- W&B project: lora-ours
- W&B group: published-base-standard-olora-order1-formal-v57
- log: /root/lora-ours/results/logs/olora_t5large_standard_order1_seed1_official_base_formal_v57.log
- manifest: /root/lora-ours/results/runs/olora_t5large_standard_order1_seed1_official_base_formal_v57/run_manifest.json
- setting: dbpedia -> amazon -> yahoo -> agnews; max_steps=-1
- gradient_accumulation_steps: 8
- engineering notes: formal single-GPU runtime uses gradient accumulation to match the official 8-GPU global batch; runtime copy only unblocks W&B env handling; sample/step caps are non-paper-comparable when present
