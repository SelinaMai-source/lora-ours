# Ours v1 Branch Compatibility And Coverage — 2026-07-08

## Branch Compatibility

- Date branch: `ours-v1-20260708` at `6449477`
- Exact compatibility branch: `ours-v1` pushed to the same commit.
- No exact `ours-v2` branch exists locally or on `origin` at this checkpoint.
- No force push or history rewrite was used.

## Training Status

| Suite | v1 smoke state | Evidence | Next action |
|-------|----------------|----------|-------------|
| Standard | launched, failed before amazon round train | `NameError: math is not defined` in O-LoRA runtime SSRG patch | fixed import injection; rerun when GPU free |
| ARPER | launched, failed before train | wrapper imported repo scripts from wrong parent path | fixed wrapper repo root/import path; rerun when GPU free |
| CITB | queue marked launched, no log found | launcher spawned a nested tmux and returned immediately | relaunch explicitly when GPU free |
| ToDCL | queue marked launched, no log found | overlay blocked until anchor checkpoint resolution | relaunch after confirming anchor checkpoint path |

Current GPU owner is an official ARPER strict retry (`arper_woz3_paper_aligned_exemplar250_formal_v88`). Do not duplicate v1 smokes while it is active.

## Official-Setting Coverage

### CITB

- Covered by v1: **InstrDialog** order1, script-strict `500/50/50`, Replay(50) anchor AR **39.98**.
- Gap: **InstrDialog++** is not covered by v1. Existing evidence shows public long-stream scripts/data do not cleanly satisfy paper `100/50/100`; prior v49/v50/v51 runs were diagnostic and must not be mixed into the v1 main table.
- Plan: prepare a separately labeled InstrDialog++ official-setting config/launcher after the current InstrDialog v1 smoke is stable. Use public long-stream evidence (`100/25/25`) or a clearly disclosed `100/50/100` CCF-A variant; do not fabricate paper-comparable results.

### Standard

- Operational anchor: **O-LoRA official-equivalent** v57, EM **76.81**.
- Reference only: **LB-CL** paper result, because no official runnable code is available in this repo. Do not mix LB-CL paper-only numbers with O-LoRA local overlay rows.
- v1 preserves official standard CL order, O-LoRA entry/scorer, T5-large, adapter chain, and cumulative test metric surface; smoke caps remain diagnostic only.

### Dialogue

- Official anchor now available: **ToDCL ADAPTER NLG** BLEU **22.6105**, EER **0.115031**, PASS.
- ARPER official Path B remains the only ARPER route; active v88 strict retry must finish before any duplicate ARPER overlay rerun.
- Non-official Path A/Core-train dialogue experiments are not main results.

## v2 Decision

Do not create `ours-v2` yet. Current v1 outcomes are launch/runtime failures, not validated mechanism failures or +1/3 misses. The next step is to rerun v1 smokes after the two small launcher fixes and current GPU owner completes.
