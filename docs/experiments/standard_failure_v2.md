# standard failure — ours-v2

Updated: 2026-07-12T06:59:22.879371+00:00

## Evidence

```json
{
  "version": "ours-v2",
  "suite": "standard",
  "dbpedia_em": 98.5,
  "amazon_em": 38.5,
  "reason": "early gate rejected round 2 amazon: EM 38.5 < 50",
  "assess_skip_amazon": true
}
```

## Diagnosis (single primary cause)

- **primary_cause:** current_task_underlearning_after_assess_pause
- **mechanism_suspect:** ssrg_class_coverage_or_replay_budget
- **recommended_next_delta:** replay_budget
- **rationale:** dbpedia EM 98.5 with amazon EM 38.5 FAIL indicates SC current-task underlearning after TC→SC transition, not retention collapse.

## Policy

- One mechanism only in next version.
- Do not change official scorer/split/task order/model.
- Published-base + minimal overlay only.
