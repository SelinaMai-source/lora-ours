# Standard O-LoRA Overlay v85 Smoke — Failure (2026-07-06)

## Incident

| Field | Value |
|-------|-------|
| Run | `olora_official_base_ours_overlay_ssrg_v85_smoke_order1_seed1` |
| tmux | `lora-ours-standard-v85-smoke` |
| W&B round1 | `45amv0bn` |
| Failure point | round1 dbpedia, train step **14/20** (~70%) |

## Root cause

**GPU serial violation:** accidental CITB Replay(50) **formal** relaunch (no `SMOKE=1`) shared GPU0 with v85 T5-large smoke (~26GB combined). v85 process terminated mid-round without completing training log.

## Remediation

1. Stopped accidental CITB formal tmux (`lora-ours-citb-replay50-smoke`) — base smoke already **PASS** (`a2jh0n4y`).
2. Fixed queue script: `SMOKE=1` on citb-replay50 job.
3. Relaunch v85 smoke on idle GPU via serial queue.

## Gate status

**FAIL** (incomplete). Relaunch required before ARPER v86 / ToDCL / CITB ours v85.
