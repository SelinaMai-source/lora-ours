# Proposal ours-v2 — standard

Updated: 2026-07-12T06:34:56.148189+00:00

## Single mechanism delta

- **key:** `amazon_round_assess_pause`
- **description:** Pause Assess retention gate on amazon round only; keep class-coverage SSRG and O-LoRA base.
- **env / overlay:** `{"ASSESS_RETENTION_SKIP_TASKS": "amazon"}`

## Constraints

- Published-base + minimal overlay only
- Official scorer / split / task order / model unchanged
- One mechanism only (no stacking)

## Prior diagnosis

```json
{
  "primary_cause": "current_task_underlearning",
  "mechanism_suspect": "assess_retention_gate_interaction_under_smoke_caps",
  "recommended_next_delta": "amazon_round_assess_pause",
  "rationale": "dbpedia EM 98.5 PASS with amazon EM 38.0 FAIL indicates SC current-task underlearning after TC→SC transition, not retention collapse."
}
```
