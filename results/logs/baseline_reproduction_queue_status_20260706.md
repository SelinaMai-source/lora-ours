# Baseline reproduction queue — 2026-07-06 (best-method-only)

Updated: 2026-07-06T17:17:22+08:00

## Scope
best-method-only: CITB Replay(50) → ToDCL ADAPTER → ARPER v87 (domain/exemplar500)

**Skipped (out of scope):** CITB L2/EWC/AGEM/Replay(10), Standard LFPT5/ProgPrompts/SeqLoRA/IncLoRA/Replay, LB-CL (paper_only; O-LoRA v57 completed).

## Phase
running: lora-ours-citb-replay50-formal

## Current GPU owner
CITB continual_instruct_tuning

## Next queued job


## Full priority queue
1. CITB Replay(50) formal 19-task — `lora-ours-citb-replay50-formal` (if not already done)
2. ToDCL ADAPTER NLG 37-domain anchor — `lora-ours-todcl-adapter-anchor`
3. ARPER paper-aligned repro (domain-wise, exemplar 500) — `lora-ours-arper-v87-formal`

## Tracker
results/tables/baseline_reproduction_tracker_20260706.md

## Flags
- citb_running: yes
- core.train_running: no
- gpu_idle: no
