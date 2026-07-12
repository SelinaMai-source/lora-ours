# Proposal ours-v3 — standard

Updated: 2026-07-12T06:58:56.859641+00:00

## Single mechanism delta

- **key:** `replay_budget`
- **description:** Increase replay budget one notch; keep SSRG mode.
- **env / overlay:** `{"REPLAY_PER_TASK": "96"}`

## Constraints

- Published-base + minimal overlay only
- Official scorer / split / task order / model unchanged
- One mechanism only (no stacking)

## Prior diagnosis

```json
{
  "primary_cause": "unknown_or_infra",
  "mechanism_suspect": "queue_or_env",
  "recommended_next_delta": "replay_budget",
  "rationale": "Insufficient metric pattern; prefer smallest RP overlay knob (replay budget) after infra check."
}
```
