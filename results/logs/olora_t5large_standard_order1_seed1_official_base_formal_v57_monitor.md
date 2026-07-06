# olora_t5large_standard_order1_seed1_official_base_formal_v57 monitor

- updated: 2026-07-03T18:20:45
- manifest_state: completed
- label: published_base_official_equivalent_single_gpu_formal_olora_order1
- note: official_equivalent_single_gpu_gradaccum8_evalbatch16
- task_order: dbpedia -> amazon -> yahoo -> agnews
- smoke_limits: {'max_steps': -1, 'max_train_samples': None, 'max_predict_samples': None}
- gpu: 0, 751 MiB, 49140 MiB, 0 %
- tmux_present: True
- log_marker: 
- completed_rounds: 4/4
- round_avg_exact: 81.0518
- round_avg_rougeL: 84.465825
- observed_avg_exact: 76.80590000000001
- observed_avg_forgetting: 1.3377333333333326
- latest_task_exact: {'dbpedia': 98.1842, 'amazon': 50.9868, 'yahoo': 70.4605, 'agnews': 87.5921}

## Rounds

- 1-dbpedia: step=218, samples=7600, exact=98.8026, rougeL=98.8026
- 2-amazon: step=78, samples=15200, exact=74.1908, rougeL=79.8202
- 3-yahoo: step=156, samples=22800, exact=74.4079, rougeL=79.269
- 4-agnews: step=62, samples=30400, exact=76.8059, rougeL=79.9715

## Comparability Boundary

- This monitor reports local diagnostic metrics only.
- A run with single-GPU runtime or reduced eval batch is not paper-comparable unless explicitly audited against the official setting.
