# citb_instrdialog failure — ours-v1

Updated: 2026-07-12T12:28:48.075631+00:00

## Evidence

```json
{
  "version": "ours-v1",
  "suite": "citb_instrdialog",
  "reason": "formal 19/19 exit 0; AR 23.8947 < 53.31 and BWT -4.1111",
  "AR": 23.894736842105264,
  "AR_target": 53.31
}
```

## Diagnosis (single primary cause)

- **primary_cause:** insufficient_retention_or_replay_coverage
- **mechanism_suspect:** ssrg_ratio_or_replay_budget
- **recommended_next_delta:** replay_budget
- **rationale:** AR 23.894736842105264 < target 53.31; adjust SSRG/replay budget only (one delta).

## Policy

- One mechanism only in next version.
- Do not change official scorer/split/task order/model.
- Published-base + minimal overlay only.
