# olora_t5large_standard_order1_seed1_official_base_formal_v55 monitor

- updated: 2026-07-03T17:02:00
- manifest_state: stopped_incomplete
- label: published-base official-runtime smoke, not ours improvement
- note: RED FLAG: FORMAL=1 single-GPU run stopped incomplete during round1; no all_results.json or reportable metric
- task_order: dbpedia -> amazon -> yahoo -> agnews
- smoke_limits: {'max_steps': -1, 'max_train_samples': None, 'max_predict_samples': None}
- gpu: 0, 395 MiB, 49140 MiB, 0 %
- completed_rounds: 0/4
- observed_avg_exact: None
- observed_avg_forgetting: None
- latest_task_exact: {}

## Rounds

- no completed round metrics yet

## Comparability Boundary

- This monitor reports local diagnostic metrics only.
- A run with single-GPU runtime or reduced eval batch is not paper-comparable unless explicitly audited against the official setting.
