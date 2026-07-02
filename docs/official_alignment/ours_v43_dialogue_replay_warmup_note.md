# Ours v43 Dialogue Replay Warmup Note

## Why

V42 completed the five-segment smoke with eval-time target prototype conditioning disabled, so it removes the v41 leakage risk. The result still did not move `task574_air_dialogue_sentence_generation`: task-aware stayed at `0.03`.

The task574 debug outputs were not empty. They were routed to the current branch and produced fluent speaker-prefixed responses, but the content was generic flight-service boilerplate rather than slot-specific AIR dialogue turns. V42 also showed that continuation weighting and target-term content weighting were active on task574, so another weighting-only change is not the right next step.

## Smoke Design

V43 keeps prompts, targets, scoring, raw decoded predictions, and `target_prototype_conditioning.enabled=false` unchanged. It adds one train-only mechanism:

- Retrieve speaker-prefixed examples from the existing replay buffer using only previously seen training segments.
- Run a short active-adapter warmup before the task574 main training pass.
- Record retrieval and warmup metrics under `dialogue_replay_warmup.*`.

This is intended as broader generation-task transfer from allowed replay memory, not eval-time conditioning. No evaluation examples or current task target prototypes are inserted into eval prompts.

## Gate

Use v43 only as a five-segment smoke. It is healthy only if task574 rises materially above the repeated `0.03` task-aware blocker while preserving segment2 near `0.28` and task1714 near the v36b/v42 range. Do not launch full strict from v43 unless that gate is met.
