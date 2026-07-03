# olora_t5large_standard_order1_seed1_official_base_formal_v55 monitor

- updated: 2026-07-03T16:59:50
- manifest_state: running
- label: published-base official-runtime smoke, not ours improvement
- note: RED FLAG: externally started FORMAL=1 single-GPU full-epoch run; not the requested v56 sanity and not paper-comparable until audited
- task_order: dbpedia -> amazon -> yahoo -> agnews
- smoke_limits: {'max_steps': -1, 'max_train_samples': None, 'max_predict_samples': None}
- gpu: 0, 22567 MiB, 49140 MiB, 18 %
- completed_rounds: 0/4
- observed_avg_exact: None
- observed_avg_forgetting: None
- latest_task_exact: {}

## Rounds

- no completed round metrics yet

## Comparability Boundary

- This monitor reports local diagnostic metrics only.
- A run with single-GPU runtime or reduced eval batch is not paper-comparable unless explicitly audited against the official setting.
