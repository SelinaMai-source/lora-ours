# standard failure — ours-v3

Updated: 2026-07-12T12:22:01.936032+00:00

## Evidence

```json
{
  "version": "ours-v3",
  "suite": "standard",
  "reason": "early gate rejected round 2 amazon: EM 39.5 < 50",
  "assess_skip_amazon": true,
  "amazon_em": 39.5
}
```

## Diagnosis (single primary cause)

- **primary_cause:** current_task_underlearning_after_assess_pause
- **mechanism_suspect:** ssrg_class_coverage_or_replay_budget
- **recommended_next_delta:** replay_budget
- **rationale:** dbpedia EM None with amazon EM 39.5 FAIL indicates SC current-task underlearning after TC→SC transition, not retention collapse.

## Policy

- One mechanism only in next version.
- Do not change official scorer/split/task order/model.
- Published-base + minimal overlay only.
