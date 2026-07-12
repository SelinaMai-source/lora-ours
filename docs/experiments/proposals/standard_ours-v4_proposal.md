# Proposal ours-v4 — standard

Updated: 2026-07-12T12:22:01.988010+00:00

## Single mechanism delta

- **key:** `replay_budget`
- **description:** Increase replay budget one notch from ours-v3 (96→128); keep SSRG mode.
- **env / overlay:** `{"REPLAY_PER_TASK": "128"}`

## Constraints

- Published-base + minimal overlay only
- Official scorer / split / task order / model unchanged
- One mechanism only (no stacking)

## Prior diagnosis

```json
{
  "primary_cause": "current_task_underlearning_after_assess_pause",
  "mechanism_suspect": "ssrg_class_coverage_or_replay_budget",
  "recommended_next_delta": "replay_budget",
  "rationale": "dbpedia EM None with amazon EM 39.5 FAIL indicates SC current-task underlearning after TC→SC transition, not retention collapse."
}
```
