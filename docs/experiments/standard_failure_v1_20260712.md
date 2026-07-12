# standard failure — ours-v1

Updated: 2026-07-12T06:34:50.519273+00:00

## Evidence

```json
{
  "version": "ours-v1",
  "suite": "standard",
  "dbpedia_em": 98.5,
  "amazon_em": 38.0,
  "reason": "early gate rejected round 2 amazon: EM 38.0 < 50"
}
```

## Diagnosis (single primary cause)

- **primary_cause:** current_task_underlearning
- **mechanism_suspect:** assess_retention_gate_interaction_under_smoke_caps
- **recommended_next_delta:** amazon_round_assess_pause
- **rationale:** dbpedia EM 98.5 PASS with amazon EM 38.0 FAIL indicates SC current-task underlearning after TC→SC transition, not retention collapse.

## Policy

- One mechanism only in next version.
- Do not change official scorer/split/task order/model.
- Published-base + minimal overlay only.
