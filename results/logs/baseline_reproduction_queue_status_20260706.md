# Baseline reproduction queue — 2026-07-06

Updated: 2026-07-06T17:05:00+08:00

## Phase
running: lora-ours-citb-replay50-formal (~6%, task 1/19)

## Current GPU owner
CITB continual_instruct_tuning (75% util, ~16.4 GiB)

## Progress
- Task 0/19 (`task848_pubmedqa_classification`): **done**
- Task 1/19 (`task611_mutual_multi_turn_dialogue`): training ~16% of 10395 steps
- ETA: ~9–12h (replay buffer grows each task)

## Next queued job
ToDCL ADAPTER NLG 37-domain anchor (after CITB Replay50 completes)

## Full priority queue
1. CITB Replay(50) formal 19-task — `lora-ours-citb-replay50-formal` **RUNNING**
2. ToDCL ADAPTER NLG 37-domain anchor — `lora-ours-todcl-adapter-anchor`
3. CITB L2 formal
4. CITB EWC formal
5. CITB AGEM(10) formal
6. CITB AGEM(50) formal
7. CITB Replay(10) formal
8. Standard LFPT5 / Progressive Prompts (audit blockers)
9. ARPER exemplar_500 + domain-vs-DA audit (after GPU free)

## Tracker
results/tables/baseline_reproduction_tracker_20260706.md

## Flags
- citb_running: yes
- core.train_running: no
- gpu_idle: no
- ours_v85_restarted: no (baselines first)
