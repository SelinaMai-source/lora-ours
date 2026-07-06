# Baseline reproduction GPU handoff — 2026-07-06

**Time (local):** 2026-07-06T16:53+08:00  
**Policy:** Official baseline paper reproduction takes priority over Ours overlay iteration.

## Incident

| Item | Detail |
|------|--------|
| Blocker | `lora-ours-standard-v85-smoke` — Standard v85 Ours overlay smoke, round2 **amazon** |
| Symptom | ~3h15m elapsed; GPU **0%** util, ~395 MiB resident; CPU ~99% on PID 8465 |
| Last log | Dataset download/prepare for `2-amazon` at 13:38:54; no training steps observed |
| Verdict | **Stuck / non-progressing** (>30 min at 0% GPU) |

## Action taken (v85 smoke only)

- Sent **SIGTERM** to v85 launcher PIDs **8457** / training **8465** (`run_uie_lora.py`, `max_steps=20`, round2 amazon).
- Confirmed GPU **idle** (0 MiB compute processes).
- **Not stopped:** `lora-ours-sentinel`, `lora-ours-monitor`, `lora-ours-sota-gpu-queue`, `official-gate-mihomo-proxy`.
- v85 tmux session ended with the killed job (session no longer listed after handoff).

## Baseline queue recovery

| Step | Result |
|------|--------|
| Prior `lora-ours-baseline-repro-queue` | **Dead** (bash zombie PID 35407; session absent) |
| Restart | `tmux new-session -s lora-ours-baseline-repro-queue` → `scripts/run_baseline_reproduction_queue.sh` at **16:53:18** |
| Next job | **CITB Replay(50) formal** launched **16:53:19** → tmux `lora-ours-citb-replay50-formal` |
| GPU after launch | **~61%** util, **~13 GiB** — `continual_learning/run_continual_instruct_tuning.py` (REPLAY, 19-task formal) |
| Log | `results/logs/citb_replay50_formal_20260706.log` |

## Operator notes

- Do not restart v85 smoke until baseline queue segment completes or user explicitly re-prioritizes Ours overlay.
- Queue status: `results/logs/baseline_reproduction_queue_status_20260706.md`
