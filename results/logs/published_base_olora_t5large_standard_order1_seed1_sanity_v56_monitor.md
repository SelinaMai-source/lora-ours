# published_base_olora_t5large_standard_order1_seed1_sanity_v56 monitor

- updated: 2026-07-03T17:04:45
- manifest_state: running
- label: published-base O-LoRA Standard order1 sanity, not paper-comparable
- note: published-base O-LoRA Standard order1 capped sanity: max_steps=20 train=160 predict=200; not paper-comparable
- task_order: dbpedia -> amazon -> yahoo -> agnews
- smoke_limits: {'max_steps': 20, 'max_train_samples': 160, 'max_predict_samples': 200}
- gpu: 0, 18314 MiB, 49140 MiB, 0 %
- tmux_present: False
- log_marker: 
- completed_rounds: 1/4
- observed_avg_exact: 96.0
- observed_avg_forgetting: None
- latest_task_exact: {'dbpedia': 96.0}

## Rounds

- 1-dbpedia: step=20, samples=200, exact=96.0, rougeL=96.0

## Comparability Boundary

- This monitor reports local diagnostic metrics only.
- A run with single-GPU runtime or reduced eval batch is not paper-comparable unless explicitly audited against the official setting.


AGENT_LOOP_WAKE_LORA_OURS standard_olora_stopped_incomplete state=running