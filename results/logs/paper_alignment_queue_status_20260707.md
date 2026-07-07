# Paper alignment queue — 20260707

Updated: 2026-07-07T16:45:08+08:00
Phase: ARPER v88 running (epoch 0+)

Priority:
1. ARPER v88 (config.cfg exact: exemplar 250, batch 128) — **running**
2. Post-v88 gate (`lora-ours-post-arper-v88-gate`) → v89 if FAIL → ToDCL ADAPTER
3. CITB Stage-1 seed50 + Replay v2 (optional; AR PASS on v56)

Standard O-LoRA v57: PASS — no relaunch.
CITB v56 AR matrix: PASS ±1 — Stage-1 optional.

Orchestrator: scripts/run_strict_paper_repro_iteration.sh
