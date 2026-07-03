# published_base_olora_t5large_standard_order1_seed1_sanity_v56 monitor

- updated: 2026-07-03T17:07:58
- manifest_state: completed
- label: published-base O-LoRA Standard order1 sanity, not paper-comparable
- note: published-base O-LoRA Standard order1 capped sanity completed: max_steps=20 train=160 predict=200; not paper-comparable; grouped forgetting is sample-cap biased
- task_order: dbpedia -> amazon -> yahoo -> agnews
- smoke_limits: {'max_steps': 20, 'max_train_samples': 160, 'max_predict_samples': 200}
- gpu: 0, 8523 MiB, 49140 MiB, 74 %
- tmux_present: True
- log_marker: 
- completed_rounds: 4/4
- round_avg_exact: 52.375
- round_avg_rougeL: 63.645825
- observed_avg_exact: 65.5
- observed_avg_forgetting: 6.5
- latest_task_exact: {'dbpedia': 96.0, 'amazon': 35.0}

## Rounds

- 1-dbpedia: step=20, samples=200, exact=96.0, rougeL=96.0
- 2-amazon: step=20, samples=200, exact=41.5, rougeL=63.0833
- 3-yahoo: step=20, samples=200, exact=37.0, rougeL=49.9167
- 4-agnews: step=20, samples=200, exact=35.0, rougeL=45.5833

## Comparability Boundary

- This monitor reports local diagnostic metrics only.
- A run with single-GPU runtime or reduced eval batch is not paper-comparable unless explicitly audited against the official setting.
